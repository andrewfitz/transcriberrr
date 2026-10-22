mod audio_convert;
mod model_cache;
mod transcript;
use transcript::{
    DiarizationMetadata, RenderOptions, Segment, TranscriptDocument, TranscriptMetadata, Word,
    build_segments, is_word_char, normalize_word_timing, preserve_word_text, write_output,
};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use chrono::{DateTime, SecondsFormat, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(feature = "diarization")]
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
use qwen3_asr::forced_aligner::{ForcedAlignProcessor, Qwen3ForcedAligner};
use qwen3_asr::{AudioInput, Batch, LoadOptions, Qwen3Asr, TranscribeOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};
use walkdir::WalkDir;

#[cfg(feature = "parakeet")]
mod parakeet_engine;
#[cfg(feature = "parakeet")]
type ParakeetEngine = parakeet_engine::ParakeetEngine;
#[cfg(not(feature = "parakeet"))]
type ParakeetEngine = ();

const TARGET_SAMPLE_RATE: u32 = 16_000;
const AUDIO_EXTENSIONS: &[&str] = &[
    "aac", "aif", "aiff", "caf", "flac", "m4a", "m4b", "mp3", "mp4", "ogg", "opus", "wav", "webm",
];
/// Upper bound on Qwen output tokens per second of audio: fast speech is about 4 words (~6
/// tokens) per second, so this leaves headroom while still stopping a runaway generation.
const QWEN_TOKENS_PER_SECOND: f32 = 15.0;
/// Measured on Metal (BF16, aligner on): 30 s chunks peak near 8 GB for a 10-minute file and run
/// about twice as fast as 120 s chunks, with the same word error rate.
const QWEN_DEFAULT_CHUNK_SECONDS: f32 = 30.0;
#[cfg_attr(not(feature = "parakeet"), allow(dead_code))]
const PARAKEET_DEFAULT_CHUNK_SECONDS: f32 = 120.0;
/// qwen3_asr's MAX_FORCE_ALIGN_INPUT_SECONDS.
const QWEN_MAX_CHUNK_SECONDS: f32 = 180.0;

#[derive(Debug, Parser)]
#[command(
    name = "transcriberrr",
    version,
    about = "Offline Qwen3-ASR transcription on Apple Silicon",
    subcommand_negates_reqs = true,
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,
    #[arg(value_name = "AUDIO", required_unless_present_any = ["input_dir", "manifest", "model_status", "download_model", "cached_model"])]
    inputs: Vec<PathBuf>,
    #[arg(long, conflicts_with_all = ["inputs", "manifest"])]
    input_dir: Option<PathBuf>,
    /// JSON job manifest for per-track diarization/output settings.
    #[arg(long, conflicts_with_all = ["inputs", "input_dir"])]
    manifest: Option<PathBuf>,
    #[arg(long)]
    recursive: bool,
    /// Hugging Face model ID or local directory. Defaults to the newest 2026 1.7B model.
    #[arg(long, default_value = "Qwen/Qwen3-ASR-1.7B")]
    model: String,
    /// Recognition backend. Parakeet uses the latest English Unified ONNX export by default;
    /// pass an explicit legacy TDT model path with `--model` when needed.
    #[arg(long, value_enum, default_value_t = EngineKind::Qwen3)]
    engine: EngineKind,
    /// Root directory for downloaded model snapshots. Defaults to the macOS application-support cache.
    #[arg(long)]
    model_cache: Option<PathBuf>,
    /// Download one or more model IDs/aliases, then exit when no input is supplied.
    #[arg(long = "download-model")]
    download_model: Vec<String>,
    /// Print cached model status as JSON and exit when no input is supplied.
    #[arg(long)]
    model_status: bool,
    /// Report whether model IDs, aliases, or local paths are usable offline, as JSON, then exit.
    /// The app uses this so it never second-guesses the CLI's cache rules.
    #[arg(long = "cached-model", value_name = "ID", conflicts_with_all = ["inputs", "input_dir", "manifest"])]
    cached_model: Vec<String>,
    /// Qwen3 forced aligner ID or local directory for word timestamps.
    #[arg(long, default_value = "Qwen/Qwen3-ForcedAligner-0.6B")]
    forced_aligner: String,
    #[arg(long, value_enum, default_value_t = DeviceKind::Metal)]
    device: DeviceKind,
    #[arg(long)]
    diarization_model: Option<PathBuf>,
    #[arg(long)]
    diarize: bool,
    /// Limit diarization output to at most N speakers (Sortformer supports up to 8). Implies
    /// --diarize.
    #[arg(long, alias = "speaker-count", value_parser = speaker_count)]
    speakers: Option<usize>,
    #[arg(long, short = 'o', default_value = ".")]
    output_dir: PathBuf,
    /// Write output in a shared folder or next to each source file.
    #[arg(long, value_enum, default_value_t = OutputLocation::Folder)]
    output_location: OutputLocation,
    /// Output container. JSON is the default because it preserves word timestamps.
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
    #[arg(long, value_enum, default_value_t = TimestampGranularity::Words)]
    timestamps: TimestampGranularity,
    /// Seconds of audio per model call. Defaults to 30 for Qwen3 (memory grows faster than chunk
    /// length; 120 s chunks peaked above 20 GB) and 120 for Parakeet.
    #[arg(long, value_parser = positive_f32)]
    chunk_seconds: Option<f32>,
    #[arg(long)]
    overwrite: bool,
    /// With `--format txt --timestamps none`, write one `[Speaker N]` paragraph per speaker
    /// turn instead of the exact transcript text. Requires diarization to have any effect.
    #[arg(long)]
    speaker_turns: bool,
    /// Append JSONL status events so long queues can be resumed safely.
    #[arg(long, value_name = "PATH")]
    progress_file: Option<PathBuf>,
    /// Skip inputs already marked completed in --progress-file.
    #[arg(long, requires = "progress_file")]
    resume: bool,
    /// Stop at the first failed input instead of continuing a bulk queue.
    #[arg(long)]
    fail_fast: bool,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Convert a transcript JSON without loading audio or models.
    Convert(ConvertArgs),
}

#[derive(Debug, Args)]
struct ConvertArgs {
    #[arg(value_name = "TRANSCRIPT.json")]
    input: PathBuf,
    /// Destination file (defaults to INPUT.converted.EXT).
    #[arg(long, short = 'o')]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Txt)]
    format: OutputFormat,
    /// Sentence/segment timestamps by default; none produces plain text.
    #[arg(long, value_enum, default_value_t = TimestampGranularity::Sentences)]
    timestamps: TimestampGranularity,
    #[arg(long)]
    overwrite: bool,
    /// With `--format txt --timestamps none`, write one `[Speaker N]` paragraph per speaker
    /// turn instead of the exact transcript text.
    #[arg(long)]
    speaker_turns: bool,
}

fn convert_transcript(args: ConvertArgs) -> Result<()> {
    let input = expand_home(&args.input);
    let file = File::open(&input).with_context(|| format!("opening {}", input.display()))?;
    let mut doc: TranscriptDocument = serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("reading transcript {}", input.display()))?;
    doc.validate()?;
    if args.timestamps == TimestampGranularity::Words
        && doc.segments.iter().any(|segment| segment.words.is_empty())
    {
        bail!("word timestamps require a transcript with words in every segment");
    }
    if args.timestamps != TimestampGranularity::Words {
        doc.collapse_words();
    }
    let extension = match args.format {
        OutputFormat::Txt => "txt",
        OutputFormat::Json => "json",
        OutputFormat::Srt => "srt",
        OutputFormat::Vtt => "vtt",
    };
    let output = args
        .output
        .as_deref()
        .map(expand_home)
        .unwrap_or_else(|| input.with_extension(format!("converted.{extension}")));
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let content = transcript::render_output(
        &doc,
        RenderOptions {
            format: args.format,
            timestamps: args.timestamps,
            speaker_turns: args.speaker_turns,
        },
    )?;
    // create_new prevents accidental clobbering, including races with another converter.
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .create_new(!args.overwrite)
        .truncate(args.overwrite)
        .open(&output)
        .with_context(|| {
            format!(
                "creating {} (use --overwrite to replace an existing file)",
                output.display()
            )
        })?;
    file.write_all(content.as_bytes())?;
    eprintln!("{} -> {}", input.display(), output.display());
    Ok(())
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EngineKind {
    #[value(name = "qwen3")]
    Qwen3,
    #[value(name = "apple-native", alias = "apple")]
    AppleNative,
    #[value(name = "parakeet-v3", alias = "parakeet")]
    ParakeetV3,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceKind {
    Metal,
    Cpu,
    Coreml,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Txt,
    Json,
    Srt,
    Vtt,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum OutputLocation {
    Folder,
    #[value(
        name = "next-to-source",
        alias = "next_to_original",
        alias = "next-to-original",
        alias = "adjacent"
    )]
    NextToSource,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum TimestampGranularity {
    None,
    Words,
    #[value(alias = "segments")]
    Sentences,
}

#[derive(Debug)]
struct Audio {
    samples: Vec<f32>,
    duration_seconds: f32,
}

#[derive(Debug, Clone)]
struct Job {
    input: PathBuf,
    diarize: bool,
    speakers: Option<usize>,
    output: Option<PathBuf>,
    output_dir: PathBuf,
    output_location: OutputLocation,
}

struct InferenceModels<'a> {
    asr: Option<&'a Qwen3Asr>,
    aligner: Option<&'a Qwen3ForcedAligner>,
    parakeet: Option<&'a mut ParakeetEngine>,
    diarizer: &'a mut Option<Diarizer>,
    model_label: &'a str,
}

#[derive(Debug, Deserialize)]
struct JobManifest {
    items: Vec<ManifestItem>,
}

#[derive(Debug, Deserialize)]
struct ManifestItem {
    input: PathBuf,
    #[serde(default)]
    diarize: Option<bool>,
    #[serde(default)]
    speakers: Option<usize>,
    #[serde(default)]
    output: Option<PathBuf>,
    #[serde(default)]
    output_dir: Option<PathBuf>,
    #[serde(default)]
    output_location: Option<OutputLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProgressEvent {
    input: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    elapsed_seconds: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    progress: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<String>,
}

struct ProgressLog {
    writer: Option<BufWriter<File>>,
    latest: HashMap<String, ProgressEvent>,
}

impl ProgressLog {
    fn open(path: Option<&Path>, resume: bool) -> Result<Self> {
        let mut latest = HashMap::new();
        if resume {
            let path = path.context("--resume requires --progress-file")?;
            if path.exists() {
                let file = File::open(path)
                    .with_context(|| format!("opening progress file {}", path.display()))?;
                for (index, line) in BufReader::new(file).lines().enumerate() {
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    // A crash or kill can leave a torn final line. Losing one status event only
                    // means that input is processed again, so never let it block a resume.
                    match serde_json::from_str::<ProgressEvent>(&line) {
                        Ok(event) => {
                            latest.insert(event.input.clone(), event);
                        }
                        Err(error) => eprintln!(
                            "warning: ignoring invalid progress event at {}:{}: {error}",
                            path.display(),
                            index + 1
                        ),
                    }
                }
            }
        }
        let writer = path
            .map(|path| {
                if let Some(parent) = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                {
                    fs::create_dir_all(parent).with_context(|| {
                        format!("creating progress directory {}", parent.display())
                    })?;
                }
                let mut file = OpenOptions::new()
                    .create(true)
                    .read(true)
                    .append(true)
                    .open(path)
                    .with_context(|| format!("opening progress file {}", path.display()))?;
                // Terminate a torn final line so the next event starts on its own line instead
                // of being concatenated into the invalid fragment.
                if file.metadata()?.len() > 0 {
                    let mut last = [0u8; 1];
                    file.seek(SeekFrom::End(-1))?;
                    file.read_exact(&mut last)?;
                    if last[0] != b'\n' {
                        file.write_all(b"\n")?;
                    }
                }
                Ok::<_, anyhow::Error>(BufWriter::new(file))
            })
            .transpose()?;
        Ok(Self { writer, latest })
    }

    fn is_completed(&self, input: &Path) -> bool {
        self.latest
            .get(&input_key(input))
            .is_some_and(|event| event.status == "completed" || event.status == "skipped")
    }

    fn record(&mut self, event: ProgressEvent) -> Result<()> {
        self.latest.insert(event.input.clone(), event.clone());
        if let Some(writer) = self.writer.as_mut() {
            serde_json::to_writer(&mut *writer, &event)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        Ok(())
    }
}

#[cfg(feature = "diarization")]
type Diarizer = Sortformer;

#[cfg(not(feature = "diarization"))]
type Diarizer = ();

#[derive(Debug, Clone, Copy)]
struct SpeakerSpan {
    start: u64,
    end: u64,
    speaker_id: usize,
}

#[derive(Debug, Deserialize)]
struct TimestampEnvelope {
    items: Vec<AlignedWord>,
}

#[derive(Debug, Deserialize)]
struct AlignedWord {
    text: String,
    start_time: f32,
    end_time: f32,
}

fn main() -> Result<()> {
    install_cancel_handler();
    audio_convert::remove_stale_temp_files();
    let mut cli = Cli::parse();
    if let Some(CliCommand::Convert(args)) = cli.command.take() {
        return convert_transcript(args);
    }
    // Clap intentionally leaves path values untouched. Expand shell-style home paths here so
    // output/model paths behave the same way as input paths on macOS.
    cli.output_dir = expand_home(&cli.output_dir);
    cli.input_dir = cli.input_dir.as_deref().map(expand_home);
    cli.diarization_model = cli.diarization_model.as_deref().map(expand_home);
    cli.progress_file = cli.progress_file.as_deref().map(expand_home);
    if cli.model.starts_with('~') {
        cli.model = expand_home(Path::new(&cli.model))
            .to_string_lossy()
            .into_owned();
    }
    if cli.forced_aligner.starts_with('~') {
        cli.forced_aligner = expand_home(Path::new(&cli.forced_aligner))
            .to_string_lossy()
            .into_owned();
    }
    let model_cache = model_cache::ModelCache::new(cli.model_cache.clone())?;
    if !cli.cached_model.is_empty() {
        let report = cli
            .cached_model
            .iter()
            .map(|id| {
                let path = model_cache.lookup(id);
                serde_json::json!({
                    "id": id,
                    "cached": path.is_some(),
                    "path": path.map(|path| path.display().to_string()),
                })
            })
            .collect::<Vec<_>>();
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    if cli.model_status {
        println!("{}", serde_json::to_string_pretty(&model_cache.status())?);
        if cli.download_model.is_empty()
            && cli.inputs.is_empty()
            && cli.input_dir.is_none()
            && cli.manifest.is_none()
        {
            return Ok(());
        }
    }
    if !cli.download_model.is_empty() {
        for path in model_cache.download(&cli.download_model)? {
            println!("{}", path.display());
        }
        if cli.inputs.is_empty() && cli.input_dir.is_none() && cli.manifest.is_none() {
            return Ok(());
        }
    }
    // `--model` defaults to Qwen, and the app keeps the last Qwen model in its field. Any Qwen
    // ASR model (any size or alias) means "use the Parakeet default" for this engine; only
    // the exact default string used to be recognised, so Qwen 0.6B was loaded as Parakeet.
    if matches!(cli.engine, EngineKind::ParakeetV3)
        && model_cache::canonical_id(&cli.model).starts_with("Qwen/Qwen3-ASR")
    {
        #[cfg(feature = "parakeet")]
        {
            cli.model = parakeet_engine::DEFAULT_MODEL_ID.to_owned();
        }
    }
    // Keep the user-facing model identifier before replacing it with a local cache snapshot path.
    // Inference needs the resolved directory, while transcript metadata should remain stable
    // across cache revisions and machines.
    // Subtitles need cue timing. Without timestamps, Qwen loads no aligner and produced one
    // cue spanning the whole file; give subtitle formats sentence cues instead.
    if matches!(cli.format, OutputFormat::Srt | OutputFormat::Vtt)
        && cli.timestamps == TimestampGranularity::None
    {
        cli.timestamps = TimestampGranularity::Sentences;
    }
    let model_label = model_display_name(&cli.model);
    ensure_apple_silicon()?;
    let mut rejected = Vec::new();
    let jobs = collect_jobs(&cli, &mut rejected)?;
    if jobs.is_empty() && rejected.is_empty() {
        bail!("no supported audio files were found");
    }
    ensure_unique_outputs(&cli, &jobs)?;
    if jobs.iter().any(|job| job.diarize) && cli.diarization_model.is_none() {
        bail!("--diarize requires --diarization-model PATH");
    }
    if matches!(cli.engine, EngineKind::AppleNative) && wants_diarization_jobs(&jobs) {
        bail!(
            "Apple SpeechTranscriber does not provide diarization; use --engine qwen3 with --diarize"
        );
    }
    if matches!(cli.device, DeviceKind::Coreml) && !matches!(cli.engine, EngineKind::AppleNative) {
        bail!(
            "CoreML/ANE is not available for these Rust engines; use --device metal for native Apple GPU execution"
        );
    }
    #[cfg(not(feature = "parakeet"))]
    if matches!(cli.engine, EngineKind::ParakeetV3) {
        bail!("Parakeet support is not included in this build; rebuild with --features parakeet");
    }
    let wants_diarization = jobs.iter().any(|job| job.diarize);
    #[cfg(feature = "diarization")]
    let diarization_model_path = if wants_diarization {
        let value = cli.diarization_model.as_ref().expect("validated above");
        if value.is_file() {
            Some(value.clone())
        } else {
            Some(model_cache.ensure(&value.to_string_lossy())?)
        }
    } else {
        None
    };
    #[cfg(not(feature = "diarization"))]
    if wants_diarization {
        bail!(
            "diarization support is not included in this build; rebuild with --features diarization"
        );
    }

    fs::create_dir_all(&cli.output_dir)?;
    let mut progress = ProgressLog::open(cli.progress_file.as_deref(), cli.resume)?;
    for (input, error) in &rejected {
        eprintln!("error processing {}: {error}", input.display());
        progress.record(ProgressEvent {
            input: input_key(input),
            status: "failed".to_owned(),
            output: None,
            error: Some(error.clone()),
            elapsed_seconds: None,
            progress: None,
            phase: Some("Failed".to_owned()),
        })?;
    }
    if !rejected.is_empty() && (cli.fail_fast || jobs.is_empty()) {
        bail!("{} input file(s) failed", rejected.len());
    }
    if cli.resume
        && jobs
            .iter()
            .all(|job| progress.is_completed(&job.input) && output_path(&cli, job).exists())
    {
        for job in &jobs {
            progress.record(ProgressEvent {
                input: input_key(&job.input),
                status: "skipped".to_owned(),
                output: Some(output_path(&cli, job).display().to_string()),
                error: None,
                elapsed_seconds: None,
                progress: Some(1.0),
                phase: Some("Skipped".to_owned()),
            })?;
        }
        if !rejected.is_empty() {
            bail!("{} input file(s) failed", rejected.len());
        }
        return Ok(());
    }
    // The native Apple engine runs in Speech.framework and does not need a Candle device.
    // Avoid initializing Metal for that path so helper startup stays lightweight.
    let device = if matches!(cli.engine, EngineKind::AppleNative) {
        Device::Cpu
    } else {
        match cli.device {
            DeviceKind::Metal => Device::new_metal(0)?,
            DeviceKind::Cpu | DeviceKind::Coreml => Device::Cpu,
        }
    };
    let mut asr: Option<Qwen3Asr> = None;
    let mut aligner: Option<Qwen3ForcedAligner> = None;
    // The checkpoints are BF16. The crate's default F32 upcasts every weight, and Metal rounds
    // each buffer up to a power of two: default runs peaked at 21-26 GB. BF16 on Metal measured
    // ~2.5x less memory with the same transcript and word timings. Candle's CPU BF16 is slow,
    // so keep F32 there.
    let load_options = LoadOptions {
        dtype: if matches!(device, Device::Metal(_)) {
            DType::BF16
        } else {
            DType::F32
        },
        ..LoadOptions::default()
    };
    if matches!(cli.engine, EngineKind::Qwen3) {
        asr = Some(load_model(&model_cache, &cli.model, |path| {
            let path = path
                .to_str()
                .context("Qwen model cache path is not valid UTF-8")?;
            Qwen3Asr::from_pretrained(path, &device, &load_options)
                .with_context(|| format!("loading Qwen3-ASR model {}", cli.model))
        })?);
        // Speaker labels are assigned per word, so diarization needs word timings even
        // when the requested output has no timestamps.
        if cli.timestamps != TimestampGranularity::None || wants_diarization {
            aligner = Some(load_model(&model_cache, &cli.forced_aligner, |path| {
                let path = path
                    .to_str()
                    .context("forced aligner cache path is not valid UTF-8")?;
                Qwen3ForcedAligner::from_pretrained_with_processor(
                    path,
                    &device,
                    &load_options,
                    ForcedAlignProcessor::new(),
                )
                .with_context(|| format!("loading forced aligner {}", cli.forced_aligner))
            })?);
        }
    }
    #[cfg(feature = "parakeet")]
    let mut parakeet = if matches!(cli.engine, EngineKind::ParakeetV3) {
        Some(load_model(&model_cache, &cli.model, |path| {
            ParakeetEngine::from_pretrained(path, matches!(cli.device, DeviceKind::Metal))
        })?)
    } else {
        None
    };
    #[cfg(not(feature = "parakeet"))]
    let mut parakeet: Option<ParakeetEngine> = None;
    #[cfg(feature = "diarization")]
    let mut diarizer = if wants_diarization {
        let path = diarization_model_path.as_ref().expect("validated above");
        let provider = if matches!(cli.device, DeviceKind::Cpu) {
            parakeet_rs::ExecutionProvider::BurnCpu
        } else {
            parakeet_rs::ExecutionProvider::BurnWgpu
        };
        let execution_config =
            parakeet_rs::ExecutionConfig::new().with_execution_provider(provider);
        Some(
            Sortformer::with_config(path, Some(execution_config), DiarizationConfig::default())
                .with_context(|| format!("loading diarization model {}", path.display()))?,
        )
    } else {
        None
    };

    #[cfg(not(feature = "diarization"))]
    let mut diarizer: Option<Diarizer> = None;

    JOBS_STARTED.store(true, Ordering::SeqCst);
    let mut failures = rejected.len();
    for job in jobs {
        if cancel_requested() {
            break;
        }
        let key = input_key(&job.input);
        let predicted_output = output_path(&cli, &job);
        if cli.resume && progress.is_completed(&job.input) && predicted_output.exists() {
            progress.record(ProgressEvent {
                input: key,
                status: "skipped".to_owned(),
                output: Some(predicted_output.display().to_string()),
                error: None,
                elapsed_seconds: None,
                progress: Some(1.0),
                phase: Some("Skipped".to_owned()),
            })?;
            continue;
        }
        progress.record(ProgressEvent {
            input: key.clone(),
            status: "running".to_owned(),
            output: Some(predicted_output.display().to_string()),
            error: None,
            elapsed_seconds: None,
            progress: Some(0.0),
            phase: Some("Starting".to_owned()),
        })?;
        let started = Instant::now();
        let mut report = |value: f32, phase: &str| {
            check_cancelled()?;
            progress.record(ProgressEvent {
                input: key.clone(),
                status: "running".to_owned(),
                output: Some(predicted_output.display().to_string()),
                error: None,
                elapsed_seconds: Some(started.elapsed().as_secs_f32()),
                progress: Some(value),
                phase: Some(phase.to_owned()),
            })
        };
        match process_one(
            &cli,
            &job,
            InferenceModels {
                asr: asr.as_ref(),
                aligner: aligner.as_ref(),
                parakeet: parakeet.as_mut(),
                diarizer: &mut diarizer,
                model_label: &model_label,
            },
            &mut report,
        ) {
            Ok(output) => progress.record(ProgressEvent {
                input: key,
                status: "completed".to_owned(),
                output: Some(output.display().to_string()),
                error: None,
                elapsed_seconds: Some(started.elapsed().as_secs_f32()),
                progress: Some(1.0),
                phase: Some("Complete".to_owned()),
            })?,
            // Not a failure of this input: record it so the app shows the row as
            // canceled rather than failed, then stop.
            Err(_) if cancel_requested() => {
                progress.record(ProgressEvent {
                    input: key,
                    status: "canceled".to_owned(),
                    output: None,
                    error: None,
                    elapsed_seconds: Some(started.elapsed().as_secs_f32()),
                    progress: None,
                    phase: Some("Canceled".to_owned()),
                })?;
                break;
            }
            Err(error) => {
                eprintln!("error processing {}: {error:#}", job.input.display());
                failures += 1;
                progress.record(ProgressEvent {
                    input: key,
                    status: "failed".to_owned(),
                    output: Some(predicted_output.display().to_string()),
                    error: Some(format!("{error:#}")),
                    elapsed_seconds: Some(started.elapsed().as_secs_f32()),
                    progress: None,
                    phase: Some("Failed".to_owned()),
                })?;
                if cli.fail_fast {
                    return Err(error);
                }
            }
        }
    }
    check_cancelled()?;
    if failures > 0 {
        bail!("{failures} input file(s) failed");
    }
    Ok(())
}

/// Set by the first SIGTERM/SIGINT. The app cancels a run with SIGTERM; checking this at job
/// and chunk boundaries lets the CLI stop the Apple helper and drop temporary files instead of
/// dying mid-job and orphaning both.
static CANCELLED: AtomicBool = AtomicBool::new(false);
/// Set once jobs start. Before that (model downloads and loading) there is nothing to clean
/// up and no checkpoint to stop at, so a signal exits at once instead of after the app's
/// 20 s force-kill. hf-hub's download locks are flock(2) locks, released on exit.
static JOBS_STARTED: AtomicBool = AtomicBool::new(false);

extern "C" fn request_cancel(signal: libc::c_int) {
    if !JOBS_STARTED.load(Ordering::SeqCst) {
        // _exit is async-signal-safe; 128 + signal is the conventional status.
        unsafe { libc::_exit(128 + signal) }
    }
    CANCELLED.store(true, Ordering::SeqCst);
    // Restore the default action so a second signal (e.g. Ctrl-C twice) exits immediately.
    // signal(2) is async-signal-safe.
    unsafe {
        libc::signal(signal, libc::SIG_DFL);
    }
}

fn install_cancel_handler() {
    for signal in [libc::SIGTERM, libc::SIGINT] {
        unsafe {
            libc::signal(
                signal,
                request_cancel as extern "C" fn(libc::c_int) as libc::sighandler_t,
            );
        }
    }
}

fn cancel_requested() -> bool {
    CANCELLED.load(Ordering::SeqCst)
}

fn check_cancelled() -> Result<()> {
    if cancel_requested() {
        bail!("canceled");
    }
    Ok(())
}

/// Resolve and load a model. If a cached Hub snapshot fails to load, re-check it against the
/// repository once (fetching any missing files) and retry, so a partial download from an older
/// build repairs itself instead of failing every run.
fn load_model<T>(
    cache: &model_cache::ModelCache,
    id: &str,
    load: impl Fn(&Path) -> Result<T>,
) -> Result<T> {
    let path = cache.ensure(id)?;
    match load(&path) {
        Ok(model) => Ok(model),
        Err(error) => {
            eprintln!("warning: {error:#}; checking the {id} download for missing files");
            match cache.repair(id) {
                Ok(Some(path)) => load(&path),
                Ok(None) => Err(error),
                Err(repair_error) => Err(error.context(format!("repair failed: {repair_error:#}"))),
            }
        }
    }
}

fn render_options(cli: &Cli) -> RenderOptions {
    RenderOptions {
        format: cli.format,
        timestamps: cli.timestamps,
        speaker_turns: cli.speaker_turns,
    }
}

fn ensure_apple_silicon() -> Result<()> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        bail!("transcriberrr only supports Apple Silicon macOS (aarch64)");
    }
    Ok(())
}

fn wants_diarization_jobs(jobs: &[Job]) -> bool {
    jobs.iter().any(|job| job.diarize || job.speakers.is_some())
}

fn transcribe_qwen_chunks(
    cli: &Cli,
    asr: &Qwen3Asr,
    aligner: Option<&Qwen3ForcedAligner>,
    samples: &[f32],
    want_timestamps: bool,
    report: &mut dyn FnMut(f32, &str) -> Result<()>,
) -> Result<(String, String, Vec<AlignedWord>)> {
    // Above 180 s the crate splits the aligner input again and joins the pieces without a
    // space ("station?Samuel"), so never hand it longer chunks.
    let chunk_seconds = cli
        .chunk_seconds
        .unwrap_or(QWEN_DEFAULT_CHUNK_SECONDS)
        .min(QWEN_MAX_CHUNK_SECONDS);
    let chunk_samples = (chunk_seconds * TARGET_SAMPLE_RATE as f32).round().max(1.0) as usize;
    let chunks = chunk_ranges(samples, chunk_samples);
    let total = chunks.len().max(1);
    let mut text_parts = Vec::with_capacity(chunks.len());
    let mut language = "English".to_owned();
    let mut aligned = Vec::new();
    let file_peak = loudest_window_db(samples);
    for (index, range) in chunks.iter().enumerate() {
        // Qwen3-ASR invents text ("I'm not sure.", "Okay.") for audio with no speech, and
        // 30 s chunks make any long pause a chunk of its own. Skip chunks without speech.
        if is_speechless(loudest_window_db(&samples[range.clone()]), file_peak) {
            report(
                0.2 + 0.52 * (index + 1) as f32 / total as f32,
                &format!("Skipping silent chunk {}/{}", index + 1, total),
            )?;
            continue;
        }
        let audio_input = AudioInput::Waveform {
            samples: &samples[range.clone()],
            sample_rate: TARGET_SAMPLE_RATE,
        };
        let opts = TranscribeOptions {
            context: Batch::one(String::new()),
            language: Batch::one(Some("English".to_owned())),
            return_timestamps: want_timestamps,
            // 0 means the crate's default of 256 tokens, which silently truncates a chunk of
            // ordinary speech (~4 tokens/s) well before 120 s. Allow a generous rate instead.
            max_new_tokens: ((chunk_seconds * QWEN_TOKENS_PER_SECOND).ceil() as usize).max(256),
            max_batch_size: 1,
            chunk_max_sec: Some(chunk_seconds),
            bucket_by_length: false,
        };
        let result = if let Some(aligner) = aligner {
            asr.transcribe_with_forced_aligner(aligner, vec![audio_input], opts)?
        } else {
            asr.transcribe(vec![audio_input], opts)?
        }
        .into_iter()
        .next()
        .context("Qwen3 returned no transcription")?;
        language = result.language;
        if !result.text.trim().is_empty() {
            text_parts.push(result.text);
        }
        // Offset by the chunk's exact starting sample; chunk lengths vary with the cut points.
        let offset = range.start as f32 / TARGET_SAMPLE_RATE as f32;
        if let Some(value) = result.timestamps {
            let envelope = serde_json::from_value::<TimestampEnvelope>(value)
                .context("Qwen3 returned timestamps in an unexpected format")?;
            aligned.extend(envelope.items.into_iter().map(|mut word| {
                word.start_time += offset;
                word.end_time += offset;
                word
            }));
        }
        let progress = 0.2 + 0.52 * (index + 1) as f32 / total as f32;
        report(
            progress,
            &format!("Transcribing chunk {}/{}", index + 1, total),
        )?;
    }
    Ok((text_parts.join(" "), language, aligned))
}

/// Below this, a window is digital silence or near it, whatever the rest of the file does.
const SILENCE_FLOOR_DB: f32 = -90.0;
/// A chunk whose loudest 100 ms is this far below the file's loudest has no speech in it.
/// Measured: room tone sat ~60 dB under speech, while the quietest intelligible speech tested
/// was ~45 dB under the loudest in its file.
const SPEECHLESS_MARGIN_DB: f32 = 50.0;

/// Level of the loudest 100 ms window, in dBFS (RMS).
fn loudest_window_db(samples: &[f32]) -> f32 {
    let window = TARGET_SAMPLE_RATE as usize / 10;
    let loudest = samples
        .chunks(window)
        .map(|frame| frame.iter().map(|sample| sample * sample).sum::<f32>() / frame.len() as f32)
        .fold(0.0f32, f32::max);
    10.0 * loudest.max(1e-20).log10()
}

fn is_speechless(chunk_db: f32, file_db: f32) -> bool {
    chunk_db < SILENCE_FLOOR_DB || chunk_db < file_db - SPEECHLESS_MARGIN_DB
}

/// Split audio into chunks of at most `chunk_samples`, cutting each chunk at the quietest
/// 100 ms window in the preceding few seconds. Fixed-length cuts land mid-word, producing a
/// garbled or duplicated word on both sides of every boundary.
fn chunk_ranges(samples: &[f32], chunk_samples: usize) -> Vec<std::ops::Range<usize>> {
    let chunk_samples = chunk_samples.max(1);
    let window = (TARGET_SAMPLE_RATE as usize / 10).min(chunk_samples);
    let search = (5 * TARGET_SAMPLE_RATE as usize).min(chunk_samples / 2);
    // Never leave a sliver as the final chunk: Parakeet produces zero encoder frames for
    // under 10 ms of audio and fails the whole file.
    let min_tail = (TARGET_SAMPLE_RATE as usize).min(chunk_samples / 2);
    let mut ranges = Vec::new();
    let mut start = 0;
    while samples.len() - start > chunk_samples {
        let limit = (start + chunk_samples)
            .min(samples.len() - min_tail)
            .max(start + 1);
        let mut cut = limit;
        let first = limit.saturating_sub(search).max(start + 1);
        if search >= window && limit - first >= window {
            // Sliding sum of magnitudes over [position, position + window).
            let mut energy: f32 = samples[first..first + window].iter().map(|s| s.abs()).sum();
            let mut quietest = (energy, first);
            for position in first + 1..=limit - window {
                energy += samples[position + window - 1].abs() - samples[position - 1].abs();
                if energy < quietest.0 {
                    quietest = (energy, position);
                }
            }
            cut = quietest.1 + window / 2;
        }
        ranges.push(start..cut);
        start = cut;
    }
    ranges.push(start..samples.len());
    ranges
}

fn process_one(
    cli: &Cli,
    job: &Job,
    models: InferenceModels<'_>,
    report: &mut dyn FnMut(f32, &str) -> Result<()>,
) -> Result<PathBuf> {
    if matches!(cli.engine, EngineKind::AppleNative) {
        report(0.1, "Starting Apple SpeechTranscriber")?;
        let output = process_apple_native(cli, job)?;
        report(0.95, "Writing output")?;
        return Ok(output);
    }
    // Fail before decoding and inference, not after minutes of work.
    let destination = output_path(cli, job);
    if destination.exists() && !cli.overwrite {
        bail!("output exists (use --overwrite): {}", destination.display());
    }
    let started = Instant::now();
    report(0.05, "Decoding audio")?;
    let audio =
        decode_audio(&job.input).with_context(|| format!("decoding {}", job.input.display()))?;
    if audio.samples.is_empty() {
        bail!(
            "audio file contains no decodable samples: {}",
            job.input.display()
        );
    }
    report(0.2, "Transcribing")?;
    // Diarized jobs need word timings to attach speakers, whatever the output granularity.
    let want_timestamps = cli.timestamps != TimestampGranularity::None || job.diarize;
    let (text, language, aligned) = if let Some(parakeet) = models.parakeet {
        #[cfg(feature = "parakeet")]
        {
            let result = parakeet.transcribe_with_progress(
                &audio.samples,
                want_timestamps,
                cli.chunk_seconds.unwrap_or(PARAKEET_DEFAULT_CHUNK_SECONDS),
                report,
            )?;
            let parakeet_engine::ParakeetTranscript { text, words } = result;
            let aligned = words
                .into_iter()
                .map(|word| AlignedWord {
                    text: word.text,
                    start_time: word.start,
                    end_time: word.end,
                })
                .collect();
            (text, "English".to_owned(), aligned)
        }
        #[cfg(not(feature = "parakeet"))]
        {
            let _ = parakeet;
            unreachable!("Parakeet engine requires the diarization feature")
        }
    } else {
        let asr = models.asr.context("Qwen3 engine was not initialized")?;
        transcribe_qwen_chunks(
            cli,
            asr,
            models.aligner,
            &audio.samples,
            want_timestamps,
            report,
        )?
    };
    report(0.75, "Building timestamps")?;
    let speaker_segments = if job.diarize {
        report(0.82, "Diarizing")?;
        run_diarization(models.diarizer, &audio.samples, job.speakers.unwrap_or(8))?
    } else {
        None
    };
    let mut words = aligned
        .into_iter()
        .map(|item| Word {
            text: item.text,
            start: item.start_time,
            end: item.end_time,
            speaker: speaker_at(&speaker_segments, item.start_time, item.end_time),
        })
        // Parakeet times punctuation as its own token, often 0.3-1.4 s late and inside the
        // next sentence; as a word it became a separate segment (or the next speaker's).
        // preserve_word_text attaches it to the preceding word with that word's timing.
        .filter(|word| word.text.chars().any(is_word_char))
        .collect::<Vec<_>>();
    if want_timestamps && text.chars().any(is_word_char) && words.is_empty() {
        bail!("recognizer returned text without word timestamps");
    }
    normalize_word_timing(&mut words, audio.duration_seconds);
    preserve_word_text(&text, &mut words)?;
    let mut segments = if words.is_empty() {
        if text.trim().is_empty() {
            Vec::new()
        } else {
            vec![Segment {
                text: Some(text.clone()),
                start: 0.0,
                end: audio.duration_seconds,
                speaker: None,
                words: Vec::new(),
            }]
        }
    } else {
        build_segments(words)
    };
    if cli.timestamps != TimestampGranularity::Words {
        // Sentence timestamps retain segment boundaries but do not expose word items. This keeps
        // the JSON shape stable while honoring the requested granularity.
        for segment in &mut segments {
            segment.text = Some(segment.content());
            segment.words.clear();
        }
    }
    let model = models.model_label.to_owned();
    let doc = TranscriptDocument {
        source: job.input.display().to_string(),
        model: model.clone(),
        device: match cli.device {
            DeviceKind::Metal => "metal-gpu",
            DeviceKind::Cpu => "cpu",
            DeviceKind::Coreml => "coreml-ane",
        }
        .to_owned(),
        language,
        duration_seconds: audio.duration_seconds,
        segments,
        metadata: TranscriptMetadata {
            transcribed_with: "transcriberrr".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            source_created_at: source_timestamp(&job.input, FileTimestamp::Created),
            source_modified_at: source_timestamp(&job.input, FileTimestamp::Modified),
            transcribed_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
            model,
            diarization: job.diarize.then_some(DiarizationMetadata {
                enabled: true,
                speakers: job.speakers,
                detected_speakers: detected_speaker_count(&speaker_segments),
            }),
        },
    };
    let output_path = output_path(cli, job);
    report(0.95, "Writing output")?;
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    write_output(&doc, &output_path, render_options(cli), cli.overwrite)?;
    eprintln!(
        "{} -> {} ({:.2}s)",
        job.input.display(),
        output_path.display(),
        started.elapsed().as_secs_f32()
    );
    Ok(output_path)
}

fn process_apple_native(cli: &Cli, job: &Job) -> Result<PathBuf> {
    if job.diarize || job.speakers.is_some() {
        bail!(
            "Apple SpeechTranscriber does not provide diarization; use --engine qwen3 with --diarize"
        );
    }
    let helper = apple_native_helper().context(
        "Apple SpeechTranscriber helper is not bundled; set TRANSCRIBERRR_NATIVE or build the macOS app",
    )?;
    let output_path = output_path(cli, job);
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    if output_path.exists() && !cli.overwrite {
        bail!("output exists (use --overwrite): {}", output_path.display());
    }
    // AVFoundation cannot open Matroska/WebM; decode it here and hand the helper a WAV.
    let converted = if job
        .input
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("webm"))
    {
        let audio = decode_audio(&job.input)
            .with_context(|| format!("decoding {}", job.input.display()))?;
        let wav = audio_convert::TempPath::new("wav");
        audio_convert::write_float_wav(wav.path(), &audio.samples, TARGET_SAMPLE_RATE)?;
        Some(wav)
    } else {
        None
    };
    let helper_input = converted
        .as_ref()
        .map_or(job.input.as_path(), audio_convert::TempPath::path);
    let mut child = Command::new(&helper)
        .arg(helper_input)
        .args(["--format", "json", "--timestamps", "words"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| {
            format!(
                "launching Apple SpeechTranscriber helper {}",
                helper.display()
            )
        })?;
    // Drain stdout on a thread (a full pipe would block the helper) while polling for
    // cancellation, so a canceled run stops the helper instead of leaving it running.
    let mut stdout = child.stdout.take().context("helper stdout")?;
    let reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        stdout.read_to_end(&mut buffer).map(|_| buffer)
    });
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if cancel_requested() {
            let _ = child.kill();
            let _ = child.wait();
            bail!("canceled");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    let stdout = reader
        .join()
        .map_err(|_| anyhow::anyhow!("helper output reader panicked"))??;
    if !status.success() {
        bail!("Apple SpeechTranscriber failed with status {status}");
    }
    let mut doc: TranscriptDocument =
        serde_json::from_slice(&stdout).context("reading Apple SpeechTranscriber transcript")?;
    // Describe the user's file and this build, not a temporary WAV or the helper's own
    // version constant.
    doc.source = job.input.display().to_string();
    doc.metadata.version = env!("CARGO_PKG_VERSION").to_owned();
    doc.metadata.source_created_at = source_timestamp(&job.input, FileTimestamp::Created);
    doc.metadata.source_modified_at = source_timestamp(&job.input, FileTimestamp::Modified);
    doc.validate()?;
    if cli.timestamps != TimestampGranularity::Words {
        doc.collapse_words();
    }
    write_output(&doc, &output_path, render_options(cli), cli.overwrite)?;
    Ok(output_path)
}

fn apple_native_helper() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("TRANSCRIBERRR_NATIVE") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let directory = std::env::current_exe().ok()?.parent()?.to_path_buf();
    // Bundled: the helper sits beside the CLI in Contents/Helpers. Development: a cargo build
    // in target/release finds the SwiftPM build of the helper in the repository.
    let candidates = [
        directory.join("transcriberrr-native"),
        directory.join("../../ui/.build/out/Products/Release/transcriberrr-native"),
        directory.join("../../ui/.build/release/transcriberrr-native"),
    ];
    candidates.into_iter().find(|path| path.is_file())
}

/// Manifest items whose input is missing or unsupported are returned in `rejected` rather than
/// failing the run: a queue item deleted (or on an unmounted drive) since it was queued should
/// fail alone, not take the other thousands of tracks with it.
fn collect_jobs(cli: &Cli, rejected: &mut Vec<(PathBuf, String)>) -> Result<Vec<Job>> {
    if let Some(manifest) = &cli.manifest {
        let manifest_path = expand_home(manifest);
        let content = fs::read_to_string(&manifest_path)
            .with_context(|| format!("reading job manifest {}", manifest_path.display()))?;
        let manifest_data: JobManifest = serde_json::from_str(&content)
            .with_context(|| format!("parsing job manifest {}", manifest_path.display()))?;
        let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let mut jobs = Vec::with_capacity(manifest_data.items.len());
        for item in manifest_data.items {
            let input = resolve_manifest_path(base, &item.input);
            if let Err(error) = validate_input(&input) {
                rejected.push((input, format!("{error:#}")));
                continue;
            }
            // An explicit per-item `"diarize": false` wins over the global --speakers, which
            // would otherwise be inherited and switch diarization back on.
            let speakers = if item.diarize == Some(false) {
                None
            } else {
                item.speakers.or(cli.speakers)
            };
            let diarize = item.diarize.unwrap_or(cli.diarize) || speakers.is_some();
            if let Some(count) = speakers {
                speaker_count(&count.to_string()).map_err(|error| anyhow::anyhow!(error))?;
            }
            jobs.push(Job {
                input,
                diarize,
                speakers,
                output: item.output.map(|path| resolve_manifest_path(base, &path)),
                output_dir: item
                    .output_dir
                    .map(|path| resolve_manifest_path(base, &path))
                    .unwrap_or_else(|| cli.output_dir.clone()),
                output_location: item.output_location.unwrap_or(cli.output_location),
            });
        }
        return Ok(jobs);
    }
    if let Some(dir) = &cli.input_dir {
        let dir = expand_home(dir);
        let mut paths = Vec::new();
        if cli.recursive {
            for entry in WalkDir::new(dir) {
                let entry = entry?;
                // `path().is_file()` follows symlinks, matching the non-recursive branch.
                if entry.path().is_file() && is_audio_path(entry.path()) {
                    paths.push(entry.into_path());
                }
            }
        } else {
            for entry in fs::read_dir(&dir)
                .with_context(|| format!("reading input directory {}", dir.display()))?
            {
                let path = entry?.path();
                if path.is_file() && is_audio_path(&path) {
                    paths.push(path);
                }
            }
        }
        paths.sort();
        Ok(paths
            .into_iter()
            .map(|input| Job {
                input,
                diarize: cli.diarize || cli.speakers.is_some(),
                speakers: cli.speakers,
                output: None,
                output_dir: cli.output_dir.clone(),
                output_location: cli.output_location,
            })
            .collect())
    } else {
        let mut jobs = Vec::with_capacity(cli.inputs.len());
        for path in &cli.inputs {
            let path = expand_home(path);
            if let Err(error) = validate_input(&path) {
                rejected.push((path, format!("{error:#}")));
                continue;
            }
            jobs.push(Job {
                input: path,
                diarize: cli.diarize || cli.speakers.is_some(),
                speakers: cli.speakers,
                output: None,
                output_dir: cli.output_dir.clone(),
                output_location: cli.output_location,
            });
        }
        Ok(jobs)
    }
}

fn resolve_manifest_path(base: &Path, path: &Path) -> PathBuf {
    let path = expand_home(path);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn validate_input(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("input file does not exist: {}", path.display());
    }
    if !is_audio_path(path) {
        bail!("unsupported audio extension: {}", path.display());
    }
    Ok(())
}

fn decode_audio(path: &Path) -> Result<Audio> {
    match decode_with_symphonia(path) {
        // Symphonia has no Opus decoder; Core Audio does (see audio_convert).
        Err(error)
            if matches!(
                error.downcast_ref::<SymphoniaError>(),
                Some(SymphoniaError::Unsupported(_))
            ) =>
        {
            let wav = audio_convert::convert_with_core_audio(path)
                .with_context(|| format!("{error:#}; the Core Audio fallback also failed"))?;
            decode_with_symphonia(wav.path())
        }
        result => result,
    }
}

fn decode_with_symphonia(path: &Path) -> Result<Audio> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }
    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    // Not `default_track()`, which is just the first track: browser screen recordings put
    // the video track first, and it has no sample rate.
    let track = format
        .tracks()
        .iter()
        .find(|track| {
            track.codec_params.codec != CODEC_TYPE_NULL && track.codec_params.sample_rate.is_some()
        })
        .context("file has no audio track")?;
    let track_id = track.id;
    let container_rate = track
        .codec_params
        .sample_rate
        .context("audio stream has no sample rate")?;
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut mono = Vec::new();
    // Resample from the decoder's output rate. Apple's ALAC encoder writes 44100 in the MP4
    // sample entry for every file; the true rate is in the codec config, which the decoder
    // uses. Trusting the container stretched 48 kHz Voice Memos by 9%.
    let mut decoded_rate = None;
    let mut decoded_packets = 0usize;
    let mut skipped_packets = 0usize;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) if packet.track_id() == track_id => packet,
            Ok(_) => continue,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::IoError(error)) => return Err(error.into()),
            Err(error) => return Err(error.into()),
        };
        match decoder.decode(&packet) {
            Ok(buffer) => {
                decoded_rate.get_or_insert(buffer.spec().rate);
                append_decoded(buffer, &mut mono);
                decoded_packets += 1;
            }
            // A damaged frame (common in recovered MP3/AAC recordings) only loses that frame;
            // Symphonia documents decode errors as recoverable.
            Err(SymphoniaError::DecodeError(_)) => skipped_packets += 1,
            Err(error) => return Err(error.into()),
        }
    }
    if decoded_packets == 0 {
        bail!("audio stream has no decodable packets");
    }
    if skipped_packets > 0 {
        eprintln!(
            "warning: skipped {skipped_packets} undecodable packet(s) in {}",
            path.display()
        );
    }
    let sample_rate = decoded_rate.unwrap_or(container_rate);
    if sample_rate == 0 {
        bail!("audio stream has an invalid zero sample rate");
    }
    let samples = resample(mono, sample_rate, TARGET_SAMPLE_RATE);
    Ok(Audio {
        duration_seconds: samples.len() as f32 / TARGET_SAMPLE_RATE as f32,
        samples,
    })
}

/// Downmix each decoded buffer using its own channel count. Container metadata can be stale
/// (some AAC/M4A files), and the layout may change mid-stream, so a count taken from one buffer
/// must not be applied to another. Mixing per buffer also avoids holding the full interleaved
/// stream, which keeps peak RSS lower for long multichannel recordings.
fn append_decoded(buffer: AudioBufferRef<'_>, output: &mut Vec<f32>) {
    // Vorbis emits an empty first buffer; SampleBuffer panics copying zero frames of stereo.
    if buffer.frames() == 0 {
        return;
    }
    let channels = buffer.spec().channels.count().max(1);
    let mut sample_buffer = SampleBuffer::<f32>::new(buffer.frames() as u64, *buffer.spec());
    sample_buffer.copy_interleaved_ref(buffer);
    downmix_into(sample_buffer.samples(), channels, output);
}

fn downmix_into(interleaved: &[f32], channels: usize, output: &mut Vec<f32>) {
    if channels <= 1 {
        output.extend_from_slice(interleaved);
        return;
    }
    output.extend(
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32),
    );
}

/// Zero crossings on each side of the resampling kernel's centre.
const RESAMPLE_ZERO_CROSSINGS: f64 = 8.0;

/// Polyphase windowed-sinc resampler. Plain interpolation from 44.1/48 kHz folds 8–24 kHz
/// energy (sibilants, hiss) back into the 0–8 kHz speech band; the low-pass at the lower
/// Nyquist frequency prevents that. Sample rates share a large divisor in practice
/// (48 kHz → 16 kHz needs one phase, 44.1 kHz needs 160), so every kernel is precomputed.
fn resample(input: Vec<f32>, source_rate: u32, target_rate: u32) -> Vec<f32> {
    if input.is_empty() || source_rate == target_rate {
        return input;
    }
    let divisor = gcd(source_rate, target_rate);
    let up = (target_rate / divisor) as usize;
    let down = (source_rate / divisor) as usize;
    let cutoff = (target_rate as f64 / source_rate as f64).min(1.0) * 0.95;
    let half = (RESAMPLE_ZERO_CROSSINGS / cutoff).ceil() as usize;
    let kernels: Vec<Vec<f32>> = (0..up)
        .map(|phase| {
            let frac = phase as f64 / up as f64;
            let weights: Vec<f64> = (0..2 * half)
                .map(|tap| {
                    let distance = tap as f64 + 1.0 - half as f64 - frac;
                    let x = distance * cutoff;
                    let sinc = if x.abs() < 1e-9 {
                        1.0
                    } else {
                        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                    };
                    let window = if distance.abs() >= half as f64 {
                        0.0
                    } else {
                        0.5 + 0.5 * (std::f64::consts::PI * distance / half as f64).cos()
                    };
                    sinc * window
                })
                .collect();
            let total: f64 = weights.iter().sum();
            weights
                .iter()
                .map(|weight| (weight / total) as f32)
                .collect()
        })
        .collect();
    let output_len = ((input.len() as f64 * up as f64 / down as f64).round() as usize).max(1);
    (0..output_len)
        .map(|index| {
            let position = index * down;
            let base = position / up;
            let kernel = &kernels[position % up];
            // Taps cover input[base + 1 - half ..= base + half]. Renormalize over the taps that
            // fall inside the clip so the first and last few milliseconds are not attenuated.
            let first = (base + 1).saturating_sub(half);
            let skip = first + half - (base + 1);
            let (sum, weight) = input
                .get(first..)
                .unwrap_or_default()
                .iter()
                .zip(&kernel[skip..])
                .fold((0.0f32, 0.0f32), |(sum, total), (sample, weight)| {
                    (sum + sample * weight, total + weight)
                });
            if weight.abs() > 1e-6 {
                sum / weight
            } else {
                sum
            }
        })
        .collect()
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Assign the speaker whose span contains the word's midpoint, or else the nearest span.
/// Aligner and diarizer boundaries rarely agree exactly, and a word that falls in a short gap
/// between spans was still spoken by someone; leaving it unlabeled splits the segment.
fn speaker_at(segments: &Option<Vec<SpeakerSpan>>, start: f32, end: f32) -> Option<usize> {
    let midpoint = (start + end) * 0.5;
    segments
        .as_ref()?
        .iter()
        .map(|segment| {
            let span_start = segment.start as f32 / TARGET_SAMPLE_RATE as f32;
            let span_end = segment.end as f32 / TARGET_SAMPLE_RATE as f32;
            let distance = (span_start - midpoint).max(midpoint - span_end).max(0.0);
            (distance, segment.speaker_id)
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, speaker)| speaker)
}

/// Keep the `max_speakers` speakers with the most speaking time. Filtering by ID instead would
/// keep whichever speakers Sortformer happened to number first and drop a main speaker.
#[cfg_attr(not(feature = "diarization"), allow(dead_code))]
fn limit_speakers(mut spans: Vec<SpeakerSpan>, max_speakers: usize) -> Vec<SpeakerSpan> {
    let mut talk_time: HashMap<usize, u64> = HashMap::new();
    for span in &spans {
        *talk_time.entry(span.speaker_id).or_default() += span.end.saturating_sub(span.start);
    }
    if talk_time.len() <= max_speakers {
        return spans;
    }
    let mut ranked = talk_time.into_iter().collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
    let kept = ranked
        .into_iter()
        .take(max_speakers)
        .map(|(speaker, _)| speaker)
        .collect::<Vec<_>>();
    spans.retain(|span| kept.contains(&span.speaker_id));
    spans
}

fn detected_speaker_count(segments: &Option<Vec<SpeakerSpan>>) -> Option<usize> {
    let segments = segments.as_ref()?;
    let mut ids = segments
        .iter()
        .map(|segment| segment.speaker_id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    Some(ids.len())
}

#[cfg(feature = "diarization")]
fn run_diarization(
    diarizer: &mut Option<Diarizer>,
    samples: &[f32],
    max_speakers: usize,
) -> Result<Option<Vec<SpeakerSpan>>> {
    Ok(diarizer
        .as_mut()
        .map(|model| {
            model
                .diarize(samples.to_vec(), TARGET_SAMPLE_RATE, 1)
                .map(|segments| {
                    let spans = segments
                        .into_iter()
                        .map(|segment| SpeakerSpan {
                            start: segment.start,
                            end: segment.end,
                            speaker_id: segment.speaker_id,
                        })
                        .collect();
                    limit_speakers(spans, max_speakers)
                })
        })
        .transpose()?)
}

#[cfg(not(feature = "diarization"))]
fn run_diarization(
    _diarizer: &mut Option<Diarizer>,
    _samples: &[f32],
    _max_speakers: usize,
) -> Result<Option<Vec<SpeakerSpan>>> {
    Ok(None)
}

#[derive(Debug, Clone, Copy)]
enum FileTimestamp {
    Created,
    Modified,
}

fn source_timestamp(path: &Path, kind: FileTimestamp) -> Option<String> {
    let metadata = fs::metadata(path).ok()?;
    let timestamp = match kind {
        FileTimestamp::Created => metadata.created().ok()?,
        FileTimestamp::Modified => metadata.modified().ok()?,
    };
    Some(DateTime::<Utc>::from(timestamp).to_rfc3339_opts(SecondsFormat::Secs, true))
}

fn model_display_name(value: &str) -> String {
    Path::new(value)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != ".")
        .unwrap_or(value)
        .to_owned()
}

fn output_path(cli: &Cli, job: &Job) -> PathBuf {
    let extension = match cli.format {
        OutputFormat::Txt => "txt",
        OutputFormat::Json => "json",
        OutputFormat::Srt => "srt",
        OutputFormat::Vtt => "vtt",
    };
    if let Some(path) = &job.output {
        // Replace a transcript extension the caller supplied; otherwise append, so a dotted
        // name such as `out/meeting.2024` keeps its last component.
        let has_output_extension = path.extension().is_some_and(|existing| {
            ["txt", "json", "srt", "vtt"]
                .iter()
                .any(|known| existing.eq_ignore_ascii_case(known))
        });
        if has_output_extension {
            return path.with_extension(extension);
        }
        let mut name = path.as_os_str().to_os_string();
        name.push(".");
        name.push(extension);
        return PathBuf::from(name);
    }
    let parent = match job.output_location {
        OutputLocation::Folder => job.output_dir.clone(),
        OutputLocation::NextToSource => job
            .input
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    };
    if matches!(job.output_location, OutputLocation::Folder)
        && let Some(root) = cli.input_dir.as_deref()
        && let Ok(relative) = job.input.strip_prefix(root)
    {
        parent.join(relative).with_extension(extension)
    } else {
        // Append rather than `with_extension`: a dotted stem such as `meeting.2024` would
        // otherwise lose its last component and collide with `meeting.2025`.
        let mut name = job.input.file_stem().unwrap_or_default().to_os_string();
        name.push(".");
        name.push(extension);
        parent.join(name)
    }
}

/// Reject queues where two inputs (for example `talk.m4a` and `talk.wav`) would write the same
/// transcript. Without this, the second job either fails late or, with `--overwrite`, silently
/// replaces the first job's output.
fn ensure_unique_outputs(cli: &Cli, jobs: &[Job]) -> Result<()> {
    let mut seen: HashMap<String, &Path> = HashMap::new();
    for job in jobs {
        // APFS is case-insensitive by default, so compare case-folded paths.
        let key = output_path(cli, job).to_string_lossy().to_lowercase();
        if let Some(previous) = seen.insert(key, &job.input) {
            bail!(
                "{} and {} would both write {}; use a manifest with explicit outputs",
                previous.display(),
                job.input.display(),
                output_path(cli, job).display()
            );
        }
    }
    Ok(())
}

fn input_key(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn speaker_count(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| "must be an integer from 1 to 8".to_owned())?;
    if (1..=8).contains(&parsed) {
        Ok(parsed)
    } else {
        Err("must be an integer from 1 to 8".to_owned())
    }
}
fn is_audio_path(path: &Path) -> bool {
    // AppleDouble `._name.m4a` metadata files (exFAT, SMB) are not audio.
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("._"))
    {
        return false;
    }
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            AUDIO_EXTENSIONS
                .iter()
                .any(|allowed| ext.eq_ignore_ascii_case(allowed))
        })
        .unwrap_or(false)
}
fn expand_home(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if (raw == "~" || raw.starts_with("~/"))
        && let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return if raw == "~" {
            PathBuf::from(home)
        } else {
            PathBuf::from(home).join(raw.trim_start_matches("~/"))
        };
    }
    path.to_path_buf()
}
fn positive_f32(value: &str) -> std::result::Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| "must be a number".to_owned())?;
    if !parsed.is_finite() || parsed <= 0.0 {
        Err("must be a finite number greater than zero".to_owned())
    } else {
        Ok(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_is_path_free() {
        assert_eq!(model_display_name("Qwen/Qwen3-ASR-1.7B"), "Qwen3-ASR-1.7B");
        assert_eq!(
            model_display_name("/models/parakeet-unified"),
            "parakeet-unified"
        );
    }

    #[test]
    fn metadata_serializes_transcription_fields() {
        let metadata = TranscriptMetadata {
            transcribed_with: "transcriberrr".to_owned(),
            version: "0.1.0".to_owned(),
            source_created_at: Some("2026-10-15T12:00:00Z".to_owned()),
            source_modified_at: Some("2026-10-15T12:01:00Z".to_owned()),
            transcribed_at: "2026-10-15T12:02:00Z".to_owned(),
            model: "Qwen3-ASR-1.7B".to_owned(),
            diarization: Some(DiarizationMetadata {
                enabled: true,
                speakers: Some(2),
                detected_speakers: Some(2),
            }),
        };
        let value = serde_json::to_value(metadata).expect("metadata should serialize");
        assert_eq!(value["transcribed-with"], "transcriberrr");
        assert_eq!(value["version"], "0.1.0");
        assert_eq!(value["model"], "Qwen3-ASR-1.7B");
        assert_eq!(value["diarization"]["speakers"], 2);
        assert_eq!(value["diarization"]["detected_speakers"], 2);
    }

    #[test]
    fn detected_speaker_count_deduplicates_ids() {
        let spans = Some(vec![
            SpeakerSpan {
                start: 0,
                end: 1,
                speaker_id: 2,
            },
            SpeakerSpan {
                start: 1,
                end: 2,
                speaker_id: 1,
            },
            SpeakerSpan {
                start: 2,
                end: 3,
                speaker_id: 2,
            },
        ]);
        assert_eq!(detected_speaker_count(&spans), Some(2));
        assert_eq!(detected_speaker_count(&None), None);
    }

    #[test]
    fn resampling_keeps_nonempty_short_audio() {
        assert_eq!(resample(vec![0.5], 192_000, TARGET_SAMPLE_RATE), vec![0.5]);
        assert!(resample(Vec::new(), 192_000, TARGET_SAMPLE_RATE).is_empty());
    }

    fn tone(frequency: f32, rate: u32, seconds: f32) -> Vec<f32> {
        (0..(rate as f32 * seconds) as usize)
            .map(|index| (std::f32::consts::TAU * frequency * index as f32 / rate as f32).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn resampling_rejects_aliases_and_keeps_speech_band() {
        for rate in [44_100, 48_000] {
            // 12 kHz is above the 8 kHz output Nyquist; linear interpolation folds it to 4 kHz
            // at nearly full amplitude.
            let alias = resample(tone(12_000.0, rate, 1.0), rate, TARGET_SAMPLE_RATE);
            assert_eq!(alias.len(), TARGET_SAMPLE_RATE as usize);
            assert!(rms(&alias[800..15_200]) < 0.02, "{rate} Hz aliasing");
            let speech = resample(tone(1_000.0, rate, 1.0), rate, TARGET_SAMPLE_RATE);
            let level = rms(&speech[800..15_200]);
            assert!(
                (level - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02,
                "{level}"
            );
        }
        let upsampled = resample(vec![0.25; 800], 8_000, TARGET_SAMPLE_RATE);
        assert_eq!(upsampled.len(), 1_600);
        assert!(upsampled.iter().all(|sample| (sample - 0.25).abs() < 1e-3));
    }

    #[test]
    fn chunks_cut_in_silence_and_cover_all_samples() {
        let rate = TARGET_SAMPLE_RATE as usize;
        // 25 s of tone with a 200 ms pause starting at 8.5 s; 10 s chunks.
        let mut samples = tone(440.0, TARGET_SAMPLE_RATE, 25.0);
        samples[rate * 85 / 10..rate * 87 / 10].fill(0.0);
        let ranges = chunk_ranges(&samples, rate * 10);
        assert_eq!(ranges.first().map(|range| range.start), Some(0));
        assert_eq!(ranges.last().map(|range| range.end), Some(samples.len()));
        assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
        assert!(ranges.iter().all(|range| range.len() <= rate * 10));
        assert!((rate * 85 / 10..rate * 87 / 10).contains(&ranges[0].end));
        assert_eq!(chunk_ranges(&samples[..rate], rate * 10), vec![0..rate]);
        // 10.005 s with 10 s chunks must not end in an 80-sample chunk.
        let ranges = chunk_ranges(&samples[..rate * 10 + 80], rate * 10);
        assert!(ranges.last().is_some_and(|range| range.len() >= rate));
    }

    fn span(start_seconds: f32, end_seconds: f32, speaker_id: usize) -> SpeakerSpan {
        SpeakerSpan {
            start: (start_seconds * TARGET_SAMPLE_RATE as f32) as u64,
            end: (end_seconds * TARGET_SAMPLE_RATE as f32) as u64,
            speaker_id,
        }
    }

    #[test]
    fn words_between_diarization_spans_get_the_nearest_speaker() {
        let spans = Some(vec![span(0.0, 2.0, 0), span(3.0, 5.0, 1)]);
        assert_eq!(speaker_at(&spans, 1.0, 1.2), Some(0));
        assert_eq!(speaker_at(&spans, 2.1, 2.3), Some(0));
        assert_eq!(speaker_at(&spans, 2.7, 2.9), Some(1));
        assert_eq!(speaker_at(&spans, 9.0, 9.5), Some(1));
        assert_eq!(speaker_at(&None, 1.0, 1.2), None);
    }

    #[test]
    fn speaker_limit_keeps_the_most_talkative_speakers() {
        let spans = vec![
            span(0.0, 1.0, 0),
            span(1.0, 20.0, 2),
            span(20.0, 30.0, 1),
            span(30.0, 31.0, 0),
        ];
        let kept = limit_speakers(spans.clone(), 2);
        let mut ids = kept.iter().map(|span| span.speaker_id).collect::<Vec<_>>();
        ids.dedup();
        assert_eq!(ids, vec![2, 1]);
        assert_eq!(limit_speakers(spans, 3).len(), 4);
    }

    #[test]
    fn manifest_diarize_false_overrides_global_speakers() {
        let directory = audio_convert::TempPath::new("dir");
        fs::create_dir_all(directory.path()).unwrap();
        for name in ["solo.wav", "meeting.wav"] {
            audio_convert::write_float_wav(&directory.path().join(name), &[0.0; 16], 16_000)
                .unwrap();
        }
        let manifest = directory.path().join("jobs.json");
        fs::write(
            &manifest,
            r#"{"items":[{"input":"solo.wav","diarize":false},{"input":"meeting.wav"}]}"#,
        )
        .unwrap();
        let cli = Cli::parse_from([
            "transcriberrr".as_ref(),
            "--manifest".as_ref(),
            manifest.as_os_str(),
            "--speakers".as_ref(),
            "2".as_ref(),
        ]);
        let jobs = collect_jobs(&cli, &mut Vec::new()).unwrap();
        assert!(!jobs[0].diarize && jobs[0].speakers.is_none());
        assert!(jobs[1].diarize && jobs[1].speakers == Some(2));
        fs::remove_dir_all(directory.path()).unwrap();
    }

    #[test]
    fn speechless_chunks_are_detected_relative_to_the_file() {
        let speech = tone(300.0, TARGET_SAMPLE_RATE, 1.0); // ~-3 dBFS
        let room_tone = speech
            .iter()
            .map(|sample| sample * 0.000_3)
            .collect::<Vec<_>>(); // ~-73
        let quiet_speech = speech
            .iter()
            .map(|sample| sample * 0.01)
            .collect::<Vec<_>>(); // ~-43
        let file = loudest_window_db(&speech);
        assert!(is_speechless(loudest_window_db(&room_tone), file));
        assert!(!is_speechless(loudest_window_db(&quiet_speech), file));
        assert!(!is_speechless(file, file));
        // An all-silent file is speechless even relative to itself.
        let silence = vec![0.0; 16_000];
        assert!(is_speechless(
            loudest_window_db(&silence),
            loudest_window_db(&silence)
        ));
    }

    #[test]
    fn dotted_stems_keep_every_component() {
        let cli = Cli::parse_from(["transcriberrr", "--format", "txt", "-o", "out", "x.wav"]);
        let job = |input: &str| Job {
            input: PathBuf::from(input),
            diarize: false,
            speakers: None,
            output: None,
            output_dir: PathBuf::from("out"),
            output_location: OutputLocation::Folder,
        };
        assert_eq!(
            output_path(&cli, &job("rec/meeting.2024.m4a")),
            PathBuf::from("out/meeting.2024.txt")
        );
        assert_eq!(
            output_path(&cli, &job("rec/plain.m4a")),
            PathBuf::from("out/plain.txt")
        );
        assert!(ensure_unique_outputs(&cli, &[job("a/call.1.m4a"), job("a/call.2.m4a")]).is_ok());
        let error = ensure_unique_outputs(&cli, &[job("a/Talk.m4a"), job("b/talk.wav")])
            .expect_err("colliding outputs must be rejected");
        assert!(error.to_string().contains("would both write"));
    }

    #[test]
    fn resume_tolerates_a_torn_progress_line() {
        let directory = std::env::temp_dir().join(format!(
            "transcriberrr-progress-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("progress.jsonl");
        fs::write(
            &path,
            "{\"input\":\"/a.wav\",\"status\":\"completed\"}\n{\"input\":\"/b.wav\",\"sta",
        )
        .unwrap();
        let mut log = ProgressLog::open(Some(&path), true).expect("torn line must not abort");
        assert!(
            log.latest
                .get("/a.wav")
                .is_some_and(|event| event.status == "completed")
        );
        log.record(ProgressEvent {
            input: "/c.wav".to_owned(),
            status: "completed".to_owned(),
            output: None,
            error: None,
            elapsed_seconds: None,
            progress: None,
            phase: None,
        })
        .unwrap();
        drop(log);
        let reopened = ProgressLog::open(Some(&path), true).unwrap();
        assert!(reopened.latest.contains_key("/a.wav"));
        assert!(reopened.latest.contains_key("/c.wav"));
        fs::remove_dir_all(directory).unwrap();
    }

    /// Stereo Ogg Vorbis used to panic inside Symphonia, and Opus (Ogg or WebM) was rejected
    /// as an unsupported codec. Fixtures are generated with ffmpeg; skipped when it is absent.
    #[test]
    fn decodes_vorbis_and_opus_containers() {
        let Ok(ffmpeg) = which_ffmpeg() else {
            eprintln!("skipping: ffmpeg not found");
            return;
        };
        let source = audio_convert::TempPath::new("wav");
        audio_convert::write_float_wav(source.path(), &tone(440.0, 48_000, 1.0), 48_000).unwrap();
        for (extension, codec) in [("ogg", "vorbis"), ("ogg", "libopus"), ("webm", "libopus")] {
            let encoded = audio_convert::TempPath::new(extension);
            let status = Command::new(&ffmpeg)
                .args(["-loglevel", "error", "-y", "-i"])
                .arg(source.path())
                .args(["-ac", "2", "-strict", "-2", "-c:a", codec])
                .arg(encoded.path())
                .status()
                .unwrap();
            if !status.success() {
                eprintln!("skipping {codec}/{extension}: encoder unavailable");
                continue;
            }
            let audio = decode_audio(encoded.path()).unwrap_or_else(|error| {
                panic!("{codec} in {extension}: {error:#}");
            });
            assert!(
                (audio.duration_seconds - 1.0).abs() < 0.1,
                "{codec}/{extension}: {}",
                audio.duration_seconds
            );
            assert!(
                rms(&audio.samples) > 0.3,
                "{codec}/{extension} decoded silence"
            );
        }
    }

    fn which_ffmpeg() -> std::result::Result<PathBuf, ()> {
        ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
            .into_iter()
            .map(PathBuf::from)
            .find(|path| path.is_file())
            .ok_or(())
    }

    #[test]
    fn decodes_the_audio_track_of_a_video_webm() {
        let Ok(ffmpeg) = which_ffmpeg() else {
            eprintln!("skipping: ffmpeg not found");
            return;
        };
        let source = audio_convert::TempPath::new("wav");
        audio_convert::write_float_wav(source.path(), &tone(440.0, 48_000, 1.0), 48_000).unwrap();
        // Video first, as in browser screen recordings.
        let video = audio_convert::TempPath::new("webm");
        let status = Command::new(&ffmpeg)
            .args(["-loglevel", "error", "-y", "-f", "lavfi"])
            .args(["-i", "color=c=blue:s=64x64:d=1", "-i"])
            .arg(source.path())
            .args([
                "-map", "0:v", "-map", "1:a", "-c:v", "libvpx", "-c:a", "libopus",
            ])
            .arg(video.path())
            .status()
            .unwrap();
        if !status.success() {
            eprintln!("skipping: VP8/Opus encoders unavailable");
            return;
        }
        let audio = decode_audio(video.path()).unwrap();
        assert!((audio.duration_seconds - 1.0).abs() < 0.1);
        assert!(rms(&audio.samples) > 0.3);
    }

    #[test]
    fn missing_inputs_and_appledouble_files_do_not_abort_the_queue() {
        let directory = audio_convert::TempPath::new("dir");
        fs::create_dir_all(directory.path()).unwrap();
        let real = directory.path().join("real.wav");
        audio_convert::write_float_wav(&real, &[0.0; 16], 16_000).unwrap();
        fs::write(directory.path().join("._real.wav"), b"metadata").unwrap();
        let missing = directory.path().join("gone.wav");
        let cli = Cli::parse_from([
            "transcriberrr".as_ref(),
            real.as_os_str(),
            missing.as_os_str(),
        ]);
        let mut rejected = Vec::new();
        let jobs = collect_jobs(&cli, &mut rejected).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].0, missing);
        let cli = Cli::parse_from([
            "transcriberrr".as_ref(),
            "--input-dir".as_ref(),
            directory.path().as_os_str(),
        ]);
        let jobs = collect_jobs(&cli, &mut Vec::new()).unwrap();
        assert_eq!(
            jobs.iter().map(|job| job.input.clone()).collect::<Vec<_>>(),
            vec![real]
        );
        fs::remove_dir_all(directory.path()).unwrap();
    }

    #[test]
    fn explicit_outputs_keep_dotted_names() {
        let cli = Cli::parse_from(["transcriberrr", "--format", "json", "x.wav"]);
        let job = |output: &str| Job {
            input: PathBuf::from("x.wav"),
            diarize: false,
            speakers: None,
            output: Some(PathBuf::from(output)),
            output_dir: PathBuf::from("."),
            output_location: OutputLocation::Folder,
        };
        assert_eq!(
            output_path(&cli, &job("out/meeting.2024")),
            PathBuf::from("out/meeting.2024.json")
        );
        assert_eq!(
            output_path(&cli, &job("out/meeting.TXT")),
            PathBuf::from("out/meeting.json")
        );
        assert_eq!(
            output_path(&cli, &job("out/meeting.json")),
            PathBuf::from("out/meeting.json")
        );
    }

    #[test]
    fn chunk_seconds_rejects_non_finite_values() {
        assert!(positive_f32("NaN").is_err());
        assert!(positive_f32("inf").is_err());
        assert_eq!(positive_f32("1.5"), Ok(1.5));
    }

    #[test]
    fn segments_do_not_space_cjk_punctuation() {
        let segments = build_segments(vec![
            Word {
                text: "你好".to_owned(),
                start: 0.0,
                end: 0.5,
                speaker: None,
            },
            Word {
                text: "。".to_owned(),
                start: 0.5,
                end: 0.6,
                speaker: None,
            },
        ]);
        assert_eq!(segments[0].content(), "你好。");
    }
}
