mod model_cache;
mod transcript;
use transcript::{
    DiarizationMetadata, Segment, TranscriptDocument, TranscriptMetadata, Word, build_segments,
    preserve_word_text, write_output,
};

use anyhow::{Context, Result, bail};
use candle_core::Device;
use chrono::{DateTime, SecondsFormat, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(feature = "diarization")]
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
use qwen3_asr::forced_aligner::{ForcedAlignProcessor, Qwen3ForcedAligner};
use qwen3_asr::{AudioInput, Batch, LoadOptions, Qwen3Asr, TranscribeOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::DecoderOptions;
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
    "aac", "aif", "aiff", "caf", "flac", "m4a", "mp3", "ogg", "wav", "webm",
];

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
    #[arg(value_name = "AUDIO", required_unless_present_any = ["input_dir", "manifest", "model_status", "download_model"])]
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
    /// Qwen3 forced aligner ID or local directory for word timestamps.
    #[arg(long, default_value = "Qwen/Qwen3-ForcedAligner-0.6B")]
    forced_aligner: String,
    #[arg(long, value_enum, default_value_t = DeviceKind::Metal)]
    device: DeviceKind,
    #[arg(long)]
    diarization_model: Option<PathBuf>,
    #[arg(long)]
    diarize: bool,
    /// Limit diarization output to at most N speakers (Sortformer supports up to 8).
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
    #[arg(long, default_value_t = 120.0, value_parser = positive_f32)]
    chunk_seconds: f32,
    #[arg(long)]
    overwrite: bool,
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
    let content = transcript::render_output(&doc, args.format, args.timestamps)?;
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
                for line in BufReader::new(file).lines() {
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let event: ProgressEvent = serde_json::from_str(&line)
                        .with_context(|| format!("invalid progress event in {}", path.display()))?;
                    latest.insert(event.input.clone(), event);
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
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .with_context(|| format!("opening progress file {}", path.display()))
                    .map(BufWriter::new)
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
    if matches!(cli.engine, EngineKind::ParakeetV3) && cli.model == "Qwen/Qwen3-ASR-1.7B" {
        #[cfg(feature = "parakeet")]
        {
            cli.model = parakeet_engine::DEFAULT_MODEL_ID.to_owned();
        }
    }
    // Keep the user-facing model identifier before replacing it with a local cache snapshot path.
    // Inference needs the resolved directory, while transcript metadata should remain stable
    // across cache revisions and machines.
    let model_label = model_display_name(&cli.model);
    ensure_apple_silicon()?;
    let jobs = collect_jobs(&cli)?;
    if jobs.is_empty() {
        bail!("no supported audio files were found");
    }
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
    if matches!(cli.engine, EngineKind::Qwen3) && cli.timestamps != TimestampGranularity::None {
        cli.forced_aligner = model_cache
            .ensure(&cli.forced_aligner)?
            .to_string_lossy()
            .into_owned();
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
    let load_options = LoadOptions::default();
    if matches!(cli.engine, EngineKind::Qwen3) {
        let model_path = model_cache.ensure(&cli.model)?;
        let model_path = model_path
            .to_str()
            .context("Qwen model cache path is not valid UTF-8")?;
        asr = Some(
            Qwen3Asr::from_pretrained(model_path, &device, &load_options)
                .with_context(|| format!("loading Qwen3-ASR model {}", cli.model))?,
        );
        if cli.timestamps != TimestampGranularity::None {
            let aligner_path = model_cache.ensure(&cli.forced_aligner)?;
            let aligner_path = aligner_path
                .to_str()
                .context("forced aligner cache path is not valid UTF-8")?;
            aligner = Some(
                Qwen3ForcedAligner::from_pretrained_with_processor(
                    aligner_path,
                    &device,
                    &load_options,
                    ForcedAlignProcessor::new(),
                )
                .with_context(|| format!("loading forced aligner {}", cli.forced_aligner))?,
            );
        }
    }
    #[cfg(feature = "parakeet")]
    let mut parakeet = if matches!(cli.engine, EngineKind::ParakeetV3) {
        let model_path = model_cache.ensure(&cli.model)?;
        Some(ParakeetEngine::from_pretrained(
            &model_path,
            matches!(cli.device, DeviceKind::Metal),
        )?)
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

    let mut failures = 0usize;
    for job in jobs {
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
    if failures > 0 {
        bail!("{failures} input file(s) failed");
    }
    Ok(())
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
    let chunk_samples = (cli.chunk_seconds * TARGET_SAMPLE_RATE as f32)
        .round()
        .max(1.0) as usize;
    let chunks: Vec<&[f32]> = samples.chunks(chunk_samples).collect();
    let total = chunks.len().max(1);
    let mut text_parts = Vec::with_capacity(chunks.len());
    let mut language = "English".to_owned();
    let mut aligned = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let audio_input = AudioInput::Waveform {
            samples: chunk,
            sample_rate: TARGET_SAMPLE_RATE,
        };
        let opts = TranscribeOptions {
            context: Batch::one(String::new()),
            language: Batch::one(Some("English".to_owned())),
            return_timestamps: want_timestamps,
            max_new_tokens: 0,
            max_batch_size: 1,
            chunk_max_sec: Some(cli.chunk_seconds),
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
        // Offset by the actual number of samples in each chunk. Rounding the requested duration
        // to a sample count otherwise accumulates timestamp drift for non-integral durations.
        let chunk_duration = chunk_samples as f32 / TARGET_SAMPLE_RATE as f32;
        let offset = index as f32 * chunk_duration;
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
    let want_timestamps = cli.timestamps != TimestampGranularity::None;
    let (text, language, aligned) = if let Some(parakeet) = models.parakeet {
        #[cfg(feature = "parakeet")]
        {
            let result = parakeet.transcribe_with_progress(
                &audio.samples,
                want_timestamps,
                cli.chunk_seconds,
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
        .filter(|word| !word.text.trim().is_empty())
        .collect::<Vec<_>>();
    if want_timestamps && !text.trim().is_empty() && words.is_empty() {
        bail!("recognizer returned text without word timestamps");
    }
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
    if output_path.exists() && !cli.overwrite {
        bail!("output exists (use --overwrite): {}", output_path.display());
    }
    write_output(&doc, &output_path, cli.format, cli.timestamps)?;
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
    let result = Command::new(&helper)
        .arg(&job.input)
        .args(["--format", "json", "--timestamps", "words"])
        .stderr(std::process::Stdio::inherit())
        .output()
        .with_context(|| {
            format!(
                "launching Apple SpeechTranscriber helper {}",
                helper.display()
            )
        })?;
    if !result.status.success() {
        bail!(
            "Apple SpeechTranscriber failed with status {}",
            result.status
        );
    }
    let mut doc: TranscriptDocument = serde_json::from_slice(&result.stdout)
        .context("reading Apple SpeechTranscriber transcript")?;
    doc.validate()?;
    if cli.timestamps != TimestampGranularity::Words {
        doc.collapse_words();
    }
    write_output(&doc, &output_path, cli.format, cli.timestamps)?;
    Ok(output_path)
}

fn apple_native_helper() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("TRANSCRIBERRR_NATIVE") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let executable = std::env::current_exe().ok()?;
    let candidates = [
        executable.parent()?.join("transcriberrr-native"),
        executable
            .parent()?
            .join("../Resources/transcriberrr-native"),
    ];
    candidates.into_iter().find(|path| path.is_file())
}

fn collect_jobs(cli: &Cli) -> Result<Vec<Job>> {
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
            validate_input(&input)?;
            let speakers = item.speakers.or(cli.speakers);
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
                if entry.file_type().is_file() && is_audio_path(entry.path()) {
                    paths.push(entry.into_path());
                }
            }
        } else {
            for entry in fs::read_dir(dir)? {
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
        cli.inputs
            .iter()
            .map(|path| {
                let path = expand_home(path);
                validate_input(&path)?;
                Ok(Job {
                    input: path,
                    diarize: cli.diarize || cli.speakers.is_some(),
                    speakers: cli.speakers,
                    output: None,
                    output_dir: cli.output_dir.clone(),
                    output_location: cli.output_location,
                })
            })
            .collect()
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
    let track = format
        .default_track()
        .context("audio file has no default track")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("audio stream has no sample rate")?;
    if sample_rate == 0 {
        bail!("audio stream has an invalid zero sample rate");
    }
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut interleaved = Vec::new();
    // Use the decoder's actual channel count rather than trusting container metadata. Some
    // AAC/M4A files declare a stale layout, while the first decoded buffer has the real count.
    let mut decoded_channels = 0;
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
        append_decoded(
            decoder.decode(&packet)?,
            &mut interleaved,
            &mut decoded_channels,
        );
    }
    // Consume each intermediate buffer where possible.  Long recordings can contain
    // millions of samples; avoiding clones here keeps peak RSS substantially lower
    // before the model receives the waveform.
    if decoded_channels == 0 {
        bail!("audio stream has no decodable channel count");
    }
    let mono = downmix(interleaved, decoded_channels);
    let samples = resample_linear(mono, sample_rate, TARGET_SAMPLE_RATE);
    Ok(Audio {
        duration_seconds: samples.len() as f32 / TARGET_SAMPLE_RATE as f32,
        samples,
    })
}

fn append_decoded(buffer: AudioBufferRef<'_>, output: &mut Vec<f32>, channels: &mut usize) {
    if *channels == 0 {
        *channels = buffer.spec().channels.count();
    }
    let mut sample_buffer = SampleBuffer::<f32>::new(buffer.frames() as u64, *buffer.spec());
    sample_buffer.copy_interleaved_ref(buffer);
    output.extend_from_slice(sample_buffer.samples());
}

fn downmix(samples: Vec<f32>, channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples;
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

fn resample_linear(input: Vec<f32>, source_rate: u32, target_rate: u32) -> Vec<f32> {
    if input.is_empty() || source_rate == target_rate {
        return input;
    }
    let output_len = (((input.len() as f64 * target_rate as f64) / source_rate as f64)
        .round()
        .max(1.0)) as usize;
    (0..output_len)
        .map(|index| {
            let pos = index as f64 * source_rate as f64 / target_rate as f64;
            let left = pos.floor() as usize;
            let right = (left + 1).min(input.len().saturating_sub(1));
            let frac = (pos - left as f64) as f32;
            input.get(left).copied().unwrap_or(0.0) * (1.0 - frac)
                + input.get(right).copied().unwrap_or(0.0) * frac
        })
        .collect()
}

fn speaker_at(segments: &Option<Vec<SpeakerSpan>>, start: f32, end: f32) -> Option<usize> {
    let midpoint = (start + end) * 0.5;
    segments
        .as_ref()?
        .iter()
        .find(|segment| {
            midpoint >= segment.start as f32 / TARGET_SAMPLE_RATE as f32
                && midpoint <= segment.end as f32 / TARGET_SAMPLE_RATE as f32
        })
        .map(|segment| segment.speaker_id)
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
                    segments
                        .into_iter()
                        .filter(|segment| segment.speaker_id < max_speakers)
                        .map(|segment| SpeakerSpan {
                            start: segment.start,
                            end: segment.end,
                            speaker_id: segment.speaker_id,
                        })
                        .collect()
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
        return path.with_extension(extension);
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
        parent
            .join(job.input.file_stem().unwrap_or_default())
            .with_extension(extension)
    }
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
        assert_eq!(
            resample_linear(vec![0.5], 192_000, TARGET_SAMPLE_RATE),
            vec![0.5]
        );
        assert!(resample_linear(Vec::new(), 192_000, TARGET_SAMPLE_RATE).is_empty());
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
