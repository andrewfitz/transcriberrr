mod audio_convert;
mod bleed;
mod merge;
mod model_cache;
mod serve;
mod transcript;
mod vad;
use transcript::{
    DiarizationMetadata, RenderOptions, Segment, TranscriptDocument, TranscriptMetadata, Word,
    build_segments, drop_unplaced_words, is_word_char, normalize_word_timing, preserve_word_text,
    write_output,
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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer, Signal};
use symphonia::core::codecs::{CODEC_TYPE_AAC, CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};
use unicode_normalization::UnicodeNormalization;
use walkdir::WalkDir;

#[cfg(feature = "parakeet")]
mod parakeet_engine;
#[cfg(feature = "parakeet")]
type ParakeetEngine = parakeet_engine::ParakeetEngine;
#[cfg(not(feature = "parakeet"))]
type ParakeetEngine = ();

const TARGET_SAMPLE_RATE: u32 = 16_000;
const CLI_SCHEMA_VERSION: u32 = 1;
const AUDIO_EXTENSIONS: &[&str] = &[
    "aac", "aif", "aiff", "caf", "flac", "m4a", "m4b", "m4v", "mov", "mp3", "mp4", "ogg", "opus",
    "qta", "wav", "webm",
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

#[derive(Debug, Clone, Parser)]
#[command(
    name = "transcriberrr",
    version,
    about = "Local transcription and speaker-track merging on Apple Silicon",
    after_help = "For automation, start with `transcriberrr capabilities`, preview with --plan, then stream results with --jsonl.",
    subcommand_negates_reqs = true,
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,
    /// Audio/video files, Zoom meetings, or folders. Positional folders are searched recursively.
    #[arg(value_name = "INPUT", required_unless_present_any = ["input_dir", "manifest", "model_status", "download_model", "cached_model", "print_model_tag"])]
    inputs: Vec<PathBuf>,
    /// Scan a directory for recordings, including Zoom meetings in subfolders.
    #[arg(long, conflicts_with_all = ["inputs", "manifest"])]
    input_dir: Option<PathBuf>,
    /// JSON job manifest for per-track settings. Use `-` to read from stdin.
    #[arg(long, conflicts_with_all = ["inputs", "input_dir"])]
    manifest: Option<PathBuf>,
    /// With --input-dir, also include ordinary recordings from subfolders.
    #[arg(long)]
    recursive: bool,
    /// Print the planned inputs, Zoom meetings, and destinations as JSON without running models.
    #[arg(long, conflicts_with_all = ["model_status", "download_model", "cached_model", "print_model_tag"])]
    plan: bool,
    /// List inputs for the app before it assigns distinct output names.
    #[arg(long, hide = true, requires = "plan")]
    inventory_only: bool,
    /// Hugging Face model ID or local directory. Defaults to the newest 2026 1.7B model.
    #[arg(long, default_value = "Qwen/Qwen3-ASR-1.7B")]
    model: String,
    /// Recognition backend. Parakeet uses the latest English Unified ONNX export by default;
    /// pass an explicit legacy TDT model path with `--model` when needed.
    #[arg(long, value_enum, default_value_t = EngineKind::Qwen3)]
    engine: EngineKind,
    /// Run a model pass over the inputs, reusing existing results unless --overwrite is set.
    /// Repeat for multiple passes, using ENGINE or ENGINE=MODEL.
    #[arg(long = "pass", value_name = "ENGINE[=MODEL]", value_parser = parse_model_pass, conflicts_with_all = ["engine", "model", "model_status", "download_model", "cached_model", "print_model_tag", "inventory_only"])]
    passes: Vec<ModelPass>,
    /// Append a short model identifier to transcript filenames. Enabled automatically for --pass.
    #[arg(long)]
    model_suffix: bool,
    /// Print the filename identifier for --engine and --model, then exit.
    #[arg(long)]
    print_model_tag: bool,
    #[arg(skip)]
    output_tag: Option<String>,
    /// Root directory for downloaded model snapshots. Defaults to the macOS application-support cache.
    #[arg(long)]
    model_cache: Option<PathBuf>,
    /// Use only local model files and installed Apple speech assets. Disable downloads and cache repair.
    #[arg(long, conflicts_with = "download_model")]
    offline: bool,
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
    /// Sortformer ONNX file, or a model ID/alias. Defaults to `sortformer` when diarizing.
    #[arg(long)]
    diarization_model: Option<PathBuf>,
    /// Detect speakers in mixed audio. Separate speaker tracks use their own labels.
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
    /// Replace existing transcripts, including when --resume is supplied.
    #[arg(long)]
    overwrite: bool,
    /// With `--format txt --timestamps none`, write one labeled paragraph per speaker turn.
    /// Uses diarization labels or names from separate speaker tracks.
    #[arg(long)]
    speaker_turns: bool,
    /// Which inputs are per-speaker tracks: by default, Zoom's participant files
    /// (`Audio Record/audioAliceSmith21067067067.m4a`); with a bare `--iso-tracks`, every
    /// input; with `--iso-tracks=off`, none. Each session's tracks (one Zoom meeting, or one
    /// folder) are transcribed together with other speakers' bleed muted. For Zoom, only the
    /// merged transcript is written and it is reused on reruns unless --overwrite is supplied.
    #[arg(
        long,
        value_enum,
        default_value_t = IsoMode::Auto,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "all"
    )]
    iso_tracks: IsoMode,
    /// Skip inputs, and sessions, whose transcript is already there, so a growing folder can be
    /// run again for just its new recordings.
    #[arg(long, conflicts_with = "overwrite")]
    skip_existing: bool,
    /// Append JSONL status events so long queues can be resumed safely.
    #[arg(long, value_name = "PATH")]
    progress_file: Option<PathBuf>,
    /// Stream versioned JSON Lines progress and run results on stdout. Diagnostics stay on stderr.
    #[arg(long, conflicts_with_all = ["plan", "model_status", "cached_model", "download_model", "print_model_tag"])]
    jsonl: bool,
    /// Skip completed outputs recorded in --progress-file, if they still exist.
    #[arg(long, requires = "progress_file")]
    resume: bool,
    /// Stop at the first failed input instead of continuing a bulk queue.
    #[arg(long)]
    fail_fast: bool,
}

#[derive(Debug, Clone, Subcommand)]
enum CliCommand {
    /// Serve live single-stream and multi-track transcription over WebSocket.
    Serve(serve::ServeArgs),
    /// Print available engines, model defaults, formats and automation options as JSON.
    Capabilities,
    /// Convert a transcript JSON without loading audio or models.
    Convert(ConvertArgs),
    /// Merge transcripts of one session's per-speaker (iso) tracks into one transcript, each
    /// track labeled as its own speaker. Zoom participant files are named automatically.
    Merge(MergeArgs),
}

#[derive(Debug, Clone, Args)]
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
    /// Print a JSON result with the saved transcript's absolute path.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Args)]
struct MergeArgs {
    /// One transcript per speaker track, all on the same timeline; a Zoom participant's
    /// files from reconnecting are one speaker. A folder, or a single Zoom track, merges
    /// the transcripts in that folder from the same Zoom session.
    #[arg(value_name = "TRANSCRIPT.json|FOLDER", required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,
    /// Destination file.
    #[arg(long, short = 'o')]
    output: PathBuf,
    /// Speaker name for each input, in order; guessed from the track's file name otherwise.
    /// For a Zoom speaker with several files, naming any one of them names them all.
    #[arg(long = "name", value_name = "NAME")]
    names: Vec<String>,
    /// Zoom session ID (the digits at the end of the file names, after the file number)
    /// to merge from a folder.
    #[arg(long, value_name = "ID")]
    session: Option<String>,
    /// Select transcripts ending in .TAG.json, for example `--model-tag qw` in a multi-pass folder.
    #[arg(long, value_name = "TAG", value_parser = parse_model_tag)]
    model_tag: Option<String>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
    #[arg(long, value_enum, default_value_t = TimestampGranularity::Words)]
    timestamps: TimestampGranularity,
    #[arg(long)]
    overwrite: bool,
    /// With `--format txt --timestamps none`, write one `[Name]` paragraph per speaker turn
    /// instead of the exact transcript text.
    #[arg(long)]
    speaker_turns: bool,
    /// Print a JSON result with the saved transcript's absolute path.
    #[arg(long)]
    json: bool,
}

fn read_transcript(path: &Path) -> Result<TranscriptDocument> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let doc: TranscriptDocument = serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("reading transcript {}", path.display()))?;
    doc.validate()
        .with_context(|| format!("invalid transcript {}", path.display()))?;
    Ok(doc)
}

/// Keep word timings only when exporting them.
fn apply_timestamps(doc: &mut TranscriptDocument, timestamps: TimestampGranularity) -> Result<()> {
    if timestamps == TimestampGranularity::Words
        && doc.segments.iter().any(|segment| segment.words.is_empty())
    {
        bail!("word timestamps require a transcript with words in every segment");
    }
    if timestamps != TimestampGranularity::Words {
        doc.collapse_words();
    }
    Ok(())
}

fn merge_transcripts(args: MergeArgs) -> Result<()> {
    let paths = args
        .inputs
        .iter()
        .map(|path| expand_home(path))
        .collect::<Vec<_>>();
    let matches_tag = |path: &Path| {
        args.model_tag.as_ref().is_none_or(|tag| {
            path.file_stem()
                .and_then(|stem| Path::new(stem).extension())
                == Some(std::ffi::OsStr::new(tag))
        })
    };
    if let Some(path) = paths
        .iter()
        .find(|path| !path.is_dir() && !matches_tag(path))
    {
        bail!(
            "{} does not match --model-tag {}",
            path.display(),
            args.model_tag.as_deref().unwrap_or_default()
        );
    }
    // Folders and a lone track are searched for the rest of the session; listed files are
    // merged as given.
    let discover = paths.len() == 1 || paths.iter().any(|path| path.is_dir());
    let inputs = if discover {
        if !args.names.is_empty() {
            bail!("--name needs the transcripts listed one by one, in the order of the names");
        }
        let mut files = Vec::new();
        let mut anchor = None;
        for path in &paths {
            if path.is_dir() {
                files.extend(transcripts_in(path)?);
            } else {
                read_transcript(path)?;
                let folder = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty());
                files.extend(transcripts_in(folder.unwrap_or(Path::new(".")))?);
                anchor = Some(path.clone());
            }
        }
        // One entry per file, however it was reached.
        let mut seen = std::collections::HashSet::new();
        files.retain(|file| {
            matches_tag(file)
                && seen.insert(fs::canonicalize(file).unwrap_or_else(|_| file.clone()))
        });
        files.sort();
        let anchor = anchor.map(|anchor| {
            let canonical = fs::canonicalize(&anchor).ok();
            files
                .iter()
                .find(|file| fs::canonicalize(file).ok() == canonical)
                .unwrap_or(&anchor)
                .display()
                .to_string()
        });
        let mut candidates = Vec::new();
        for file in files {
            match read_transcript(&file) {
                Ok(doc) => candidates.push(merge::TrackInput {
                    doc,
                    path: file.display().to_string(),
                    name: None,
                    mixed: false,
                }),
                Err(error) => eprintln!("note: skipping {}: {error:#}", file.display()),
            }
        }
        let (tracks, notes) =
            merge::select_session(candidates, anchor.as_deref(), args.session.as_deref())?;
        for note in notes {
            eprintln!("note: {note}");
        }
        tracks
    } else {
        if args.session.is_some() {
            bail!("--session picks transcripts from a folder; pass the folder");
        }
        if args.names.len() > paths.len() {
            bail!(
                "{} names given for {} transcripts",
                args.names.len(),
                paths.len()
            );
        }
        let mut inputs = Vec::with_capacity(paths.len());
        for (index, path) in paths.iter().enumerate() {
            inputs.push(merge::TrackInput {
                doc: read_transcript(path)?,
                path: path.display().to_string(),
                name: args
                    .names
                    .get(index)
                    .map(|name| name.trim().to_owned())
                    .filter(|name| !name.is_empty()),
                mixed: false,
            });
        }
        inputs
    };
    let (mut doc, warnings) = merge::merge_tracks(inputs)?;
    for warning in warnings {
        eprintln!("warning: {warning}");
    }
    for track in &doc.metadata.merged_from {
        eprintln!("{} <- {}", doc.speaker_label(track.speaker), track.source);
    }
    apply_timestamps(&mut doc, args.timestamps)?;
    let output = expand_home(&args.output);
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    write_output(
        &doc,
        &output,
        RenderOptions {
            format: args.format,
            timestamps: args.timestamps,
            speaker_turns: args.speaker_turns,
        },
        args.overwrite,
    )?;
    eprintln!("-> {}", output.display());
    if args.json {
        write_json_line(&serde_json::json!({
            "schema_version": CLI_SCHEMA_VERSION, "output": output_key(&output),
        }))?;
    }
    Ok(())
}

fn parse_model_tag(tag: &str) -> std::result::Result<String, String> {
    if tag.is_empty()
        || !tag
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    {
        return Err("model identifiers use lowercase ASCII letters and digits".to_owned());
    }
    Ok(tag.to_owned())
}

/// The transcript JSON files directly inside a folder (not its subfolders).
fn transcripts_in(folder: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(folder).with_context(|| format!("reading {}", folder.display()))? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
            && path.is_file()
        {
            files.push(path);
        }
    }
    Ok(files)
}

fn convert_transcript(args: ConvertArgs) -> Result<()> {
    let input = expand_home(&args.input);
    let mut doc = read_transcript(&input)?;
    apply_timestamps(&mut doc, args.timestamps)?;
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
    // Written whole and then moved into place, so a failed write leaves the old file (which
    // may be the input itself) intact.
    write_output(
        &doc,
        &output,
        RenderOptions {
            format: args.format,
            timestamps: args.timestamps,
            speaker_turns: args.speaker_turns,
        },
        args.overwrite,
    )?;
    eprintln!("{} -> {}", input.display(), output.display());
    if args.json {
        write_json_line(&serde_json::json!({
            "schema_version": CLI_SCHEMA_VERSION, "output": output_key(&output),
        }))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EngineKind {
    #[value(name = "qwen3", alias = "qwen")]
    Qwen3,
    #[value(name = "apple-native", alias = "apple")]
    AppleNative,
    #[value(name = "parakeet-v3", alias = "parakeet")]
    ParakeetV3,
}

fn engine_name(engine: EngineKind) -> &'static str {
    match engine {
        EngineKind::Qwen3 => "qwen3",
        EngineKind::AppleNative => "apple-native",
        EngineKind::ParakeetV3 => "parakeet-v3",
    }
}

#[derive(Clone, Debug)]
struct ModelPass {
    engine: EngineKind,
    model: String,
}

const fn default_asr(engine: EngineKind) -> &'static str {
    match engine {
        EngineKind::Qwen3 => model_cache::QWEN3_ASR_17B,
        EngineKind::ParakeetV3 => model_cache::PARAKEET_UNIFIED_EN,
        EngineKind::AppleNative => "Apple SpeechTranscriber",
    }
}

fn parse_model_pass(value: &str) -> std::result::Result<ModelPass, String> {
    let (engine, model) = value
        .split_once('=')
        .map_or((value, None), |(engine, model)| (engine, Some(model)));
    let engine = EngineKind::from_str(engine, true)?;
    if model.is_some_and(|model| model.trim().is_empty()) {
        return Err("a model override must not be empty".to_owned());
    }
    if model.is_some() && matches!(engine, EngineKind::AppleNative) {
        return Err("apple-native uses the system model and takes no model override".to_owned());
    }
    if let Some(model) = model {
        let model = model_cache::canonical_id(model);
        if (matches!(engine, EngineKind::ParakeetV3) && model.starts_with("Qwen/Qwen3-ASR"))
            || (matches!(engine, EngineKind::Qwen3)
                && [
                    model_cache::PARAKEET_UNIFIED_EN,
                    model_cache::PARAKEET_TDT_V3,
                ]
                .contains(&model.as_str()))
        {
            return Err("the model override belongs to a different engine".to_owned());
        }
    }
    Ok(ModelPass {
        engine,
        model: model.unwrap_or(default_asr(engine)).to_owned(),
    })
}

fn model_tag(engine: EngineKind, model: &str) -> String {
    if matches!(engine, EngineKind::AppleNative) {
        return "ap".to_owned();
    }
    let model = model_cache::canonical_id(model);
    let prefix = match engine {
        EngineKind::Qwen3 => "qw",
        EngineKind::ParakeetV3 => "pk",
        EngineKind::AppleNative => unreachable!(),
    };
    if model == default_asr(engine) {
        return prefix.to_owned();
    }
    match model.as_str() {
        model_cache::QWEN3_ASR_06B if matches!(engine, EngineKind::Qwen3) => {
            return "qw06".to_owned();
        }
        model_cache::PARAKEET_TDT_V3 if matches!(engine, EngineKind::ParakeetV3) => {
            return "pkt".to_owned();
        }
        _ => {}
    }
    let local = Path::new(&model);
    let resolved = fs::canonicalize(local).ok();
    let identity = resolved.as_deref().unwrap_or(local);
    format!(
        "{prefix}{:06x}",
        stable_hash(identity.as_os_str().as_encoded_bytes().iter().copied()) & 0x00ff_ffff
    )
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

/// Which inputs are per-speaker tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum IsoMode {
    /// Zoom participant files, recognized by name.
    Auto,
    /// Every input.
    All,
    /// None.
    Off,
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
    /// Audio that could not be decoded, for the job's row.
    warning: Option<String>,
}

#[derive(Debug, Clone)]
struct Job {
    input: PathBuf,
    /// Later parts of the same recording (Zoom's `audio2…` after `audio1…` when the recording
    /// was stopped and started again), joined after `input` into one transcript.
    parts: Vec<PathBuf>,
    /// One speaker's track of a multi-track session.
    iso: bool,
    /// Zoom's mixed audio or video, used when participant tracks are unavailable.
    zoom_recording: bool,
    /// The Zoom audio-only export encoded identically to a video with a custom filename.
    zoom_audio_copy: Option<PathBuf>,
    diarize: bool,
    speakers: Option<usize>,
    output: Option<PathBuf>,
    /// An output already resolved by the recording planner, including any model suffix.
    planned_output: Option<PathBuf>,
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

impl InferenceModels<'_> {
    fn reborrow(&mut self) -> InferenceModels<'_> {
        InferenceModels {
            asr: self.asr,
            aligner: self.aligner,
            parakeet: self.parakeet.as_deref_mut(),
            diarizer: &mut *self.diarizer,
            model_label: self.model_label,
        }
    }
}

/// What the job loop runs: one transcription, or the speaker tracks of one session.
enum Unit {
    Single(Job),
    Session(IsoSession),
}

impl Unit {
    fn jobs(&self) -> &[Job] {
        match self {
            Self::Single(job) => std::slice::from_ref(job),
            Self::Session(session) => &session.jobs,
        }
    }

    fn writes_job_outputs(&self) -> bool {
        !matches!(self, Self::Session(session) if session.zoom.is_some())
    }
}

/// The speaker tracks of one recording session, transcribed together and merged.
struct IsoSession {
    label: String,
    /// The Zoom meeting ID, when the tracks are Zoom participant files.
    zoom: Option<String>,
    jobs: Vec<Job>,
    /// Each track's speaker, as the merge will name them.
    speakers: Vec<String>,
    /// The merged transcript.
    output: PathBuf,
}

/// A job's place in an iso-track session.
#[derive(Clone, Copy)]
struct Member<'a> {
    group: &'a str,
    speaker: &'a str,
    /// Other speakers' bleed to mute; nothing for the mixed recording.
    gate: &'a bleed::Mask,
    merged_only: bool,
}

struct Processed {
    output: Option<PathBuf>,
    warning: Option<String>,
    /// The word-level transcript, kept for a session's merge.
    words: Option<TranscriptDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JobManifest {
    items: Vec<ManifestItem>,
    discover_zoom: Option<bool>,
    #[serde(default)]
    excluded_inputs: Vec<PathBuf>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
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
    #[serde(default)]
    iso_track: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    /// A finished job whose transcript needs a look, e.g. one with no words in it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    warning: Option<String>,
    /// The iso-track session a track (or merged transcript) belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    group: Option<String>,
    /// The speaker a track's file name names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    speaker: Option<String>,
    /// `merge` for a session's merged transcript, whose `input` is the merged output path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
}

impl ProgressEvent {
    fn new(input: String, status: &str) -> Self {
        Self {
            input,
            status: status.to_owned(),
            ..Self::default()
        }
    }
}

struct ProgressLog {
    writer: Option<BufWriter<File>>,
    latest: HashMap<(String, Option<String>), ProgressEvent>,
    /// What earlier runs logged, without this run's own events: a session records events
    /// for its first track before that track's turn.
    earlier: HashMap<(String, Option<String>), ProgressEvent>,
    stream_pass: Option<serde_json::Value>,
}

impl ProgressLog {
    fn read(path: Option<&Path>, resume: bool) -> Result<Self> {
        let mut latest = HashMap::new();
        if resume {
            let path = path.context("--resume requires --progress-file")?;
            if path.exists() {
                let file = File::open(path)
                    .with_context(|| format!("opening progress file {}", path.display()))?;
                // Bytes, not `lines()`: a torn final line can end inside a multibyte character,
                // and invalid UTF-8 there must not stop the resume either.
                for (index, line) in BufReader::new(file).split(b'\n').enumerate() {
                    let line = line?;
                    if line.iter().all(u8::is_ascii_whitespace) {
                        continue;
                    }
                    // A crash or kill can leave a torn final line. Losing one status event only
                    // means that input is processed again, so never let it block a resume.
                    match serde_json::from_slice::<ProgressEvent>(&line) {
                        Ok(mut event) => {
                            normalize_event_output(&mut event);
                            latest.insert((event.input.clone(), event.output.clone()), event);
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
        Ok(Self {
            writer: None,
            earlier: latest.clone(),
            latest,
            stream_pass: None,
        })
    }

    fn open(path: Option<&Path>, resume: bool) -> Result<Self> {
        let mut log = Self::read(path, resume)?;
        log.writer = path
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
        Ok(log)
    }

    fn event_for_output(&self, key: &str, path: &Path) -> Option<&ProgressEvent> {
        self.latest.get(&(key.to_owned(), Some(output_key(path))))
    }

    /// Whether an earlier run logged anything for `key` writing to `path`.
    fn logged_before(&self, key: &str, path: &Path) -> bool {
        self.earlier
            .contains_key(&(key.to_owned(), Some(output_key(path))))
    }

    fn interrupted_write(&self, key: &str, path: &Path) -> bool {
        let earlier = self.earlier.get(&(key.to_owned(), Some(output_key(path))));
        earlier.is_some_and(|event| {
            event.status == "running" && event.phase.as_deref() == Some("Writing output")
        })
    }

    fn record(&mut self, mut event: ProgressEvent) -> Result<()> {
        normalize_event_output(&mut event);
        let key = (event.input.clone(), event.output.clone());
        if event.status == "skipped"
            && event.warning.is_none()
            && event.output.is_some()
            && let Some(earlier) = self.earlier.get(&key)
            && matches!(earlier.status.as_str(), "completed" | "skipped")
        {
            event.warning.clone_from(&earlier.warning);
        }
        if let Some(writer) = self.writer.as_mut() {
            serde_json::to_writer(&mut *writer, &event)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        if let Some(pass) = &self.stream_pass {
            let mut data = serde_json::to_value(&event)?;
            data["pass"] = pass.clone();
            data["event"] = "progress".into();
            data["schema_version"] = CLI_SCHEMA_VERSION.into();
            write_json_line(&data)?;
        }
        self.latest.insert(key, event);
        Ok(())
    }
}

fn normalize_event_output(event: &mut ProgressEvent) {
    if let Some(output) = &mut event.output {
        *output = output_key(Path::new(output));
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
    let cli = Cli::parse();
    if !matches!(cli.command, Some(CliCommand::Serve(_))) {
        install_cancel_handler();
    }
    if cli.jsonl {
        write_json_line(&serde_json::json!({
            "schema_version": CLI_SCHEMA_VERSION, "event": "run_started",
            "pass_count": cli.passes.len().max(1),
        }))?;
    }
    let stream = cli.jsonl;
    let result = run_command(cli);
    if stream {
        let event = write_finished_event("run_finished", None, &result);
        result.and(event)
    } else {
        result
    }
}

fn write_json_line(value: &serde_json::Value) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn write_finished_event(
    event: &str,
    pass: Option<&serde_json::Value>,
    result: &Result<()>,
) -> Result<()> {
    let mut data = serde_json::json!({
        "schema_version": CLI_SCHEMA_VERSION, "event": event,
        "status": if cancel_requested() { "canceled" } else if result.is_ok() { "completed" } else { "failed" },
    });
    if let Some(pass) = pass {
        data["pass"] = pass.clone();
    }
    if let Err(error) = result {
        data["error"] = format!("{error:#}").into();
    }
    write_json_line(&data)
}

fn print_capabilities() -> Result<()> {
    let engines = [
        EngineKind::Qwen3,
        EngineKind::ParakeetV3,
        EngineKind::AppleNative,
    ]
    .map(|engine| {
        let (alias, included, model_source, devices) = match engine {
            EngineKind::Qwen3 => ("qwen", true, "hugging-face", &["metal", "cpu"][..]),
            EngineKind::ParakeetV3 => (
                "parakeet",
                cfg!(feature = "parakeet"),
                "hugging-face",
                &["metal", "cpu"][..],
            ),
            EngineKind::AppleNative => ("apple", true, "macos", &[][..]),
        };
        let native = matches!(engine, EngineKind::AppleNative);
        serde_json::json!({
            "name": engine_name(engine), "aliases": [alias], "included": included,
            "default_model": default_asr(engine), "model_source": model_source,
            "model_tag": model_tag(engine, default_asr(engine)), "devices": devices,
            "diarization": !native && included && cfg!(feature = "diarization"),
            "realtime": serve::engine_capabilities(engine),
            "minimum_macos": if native { "26.0" } else { "14.0" },
            "helper_present": native.then(|| apple_native_helper().is_some()),
        })
    });
    write_json_line(&serde_json::json!({
        "schema_version": CLI_SCHEMA_VERSION, "version": env!("CARGO_PKG_VERSION"),
        "platform": "macos-arm64", "engines": engines, "models": model_cache::SUPPORTED_MODELS,
        "input_extensions": AUDIO_EXTENSIONS, "output_formats": ["json", "txt", "srt", "vtt"],
        "timestamps": ["words", "sentences", "none"],
        "automation": {
            "serve": ["serve"], "realtime_endpoint": "/v1/realtime?intent=transcription",
            "plan": "--plan", "events": "--jsonl", "offline": "--offline", "manifest_stdin": ["--manifest", "-"],
            "cache_status": "--model-status", "check_model": "--cached-model", "model_tag": "--print-model-tag",
            "convert": ["convert", "--json"], "merge": ["merge", "--json"],
        },
    }))
}

fn run_command(mut cli: Cli) -> Result<()> {
    match cli.command.take() {
        Some(CliCommand::Serve(args)) => return serve::run(args),
        Some(CliCommand::Capabilities) => return print_capabilities(),
        Some(CliCommand::Convert(args)) => return convert_transcript(args),
        Some(CliCommand::Merge(args)) => return merge_transcripts(args),
        None => {}
    }
    cli.output_dir = expand_home(&cli.output_dir);
    cli.input_dir = cli.input_dir.as_deref().map(expand_home);
    cli.diarization_model = cli.diarization_model.as_deref().map(expand_home);
    cli.progress_file = cli.progress_file.as_deref().map(expand_home);
    let passes = std::mem::take(&mut cli.passes);
    if passes.is_empty() {
        configure_model(&mut cli);
        if cli.print_model_tag {
            println!("{}", model_tag(cli.engine, &cli.model));
            return Ok(());
        }
        return run_pass(cli, None, 1);
    }

    let mut tags = HashSet::new();
    let mut jobs = Vec::with_capacity(passes.len());
    for pass in passes {
        let mut job = cli.clone();
        job.engine = pass.engine;
        job.model = pass.model;
        job.model_suffix = true;
        job.skip_existing = !job.overwrite;
        configure_model(&mut job);
        let tag = job.output_tag.as_deref().expect("model suffix enabled");
        if !tags.insert(tag.to_owned()) {
            bail!("model passes must have distinct identifiers; repeated identifier {tag}");
        }
        job.progress_file = cli
            .progress_file
            .as_deref()
            .map(|path| append_tag(path, tag));
        jobs.push(job);
    }
    let progress_paths = jobs
        .iter()
        .filter_map(|job| job.progress_file.as_deref())
        .collect::<Vec<_>>();
    // Snapshot the inputs once. In particular, avoid re-reading Zoom media to match exports
    // and give each pass the same tracks even if recordings arrive during transcription.
    let inputs = collect_plan_inputs(&cli)?;
    let plans = jobs
        .iter()
        .map(|job| {
            let plan = plan_inputs(job, inputs.clone())?;
            protect_plan_files(job, &plan.0, &plan.1, &plan.2, &progress_paths)?;
            Ok(plan)
        })
        .collect::<Result<Vec<_>>>()?;
    if cli.plan {
        let previews = jobs
            .iter()
            .zip(&plans)
            .map(|(job, (units, skips, rejected))| queue_plan(job, units, skips, rejected))
            .collect::<Result<Vec<_>>>()?;
        return write_json_line(&serde_json::json!({
            "schema_version": CLI_SCHEMA_VERSION, "passes": previews,
        }));
    }
    let mut failed = 0;
    for (index, (job, plan)) in jobs.into_iter().zip(plans).enumerate() {
        check_cancelled()?;
        JOBS_STARTED.store(false, Ordering::SeqCst);
        let tag = job.output_tag.clone().expect("model suffix enabled");
        eprintln!("Running model pass {tag} ({})", job.model);
        if let Err(error) = run_pass(job, Some(plan), index + 1) {
            check_cancelled()?;
            if cli.fail_fast {
                return Err(error.context(format!("model pass {tag}")));
            }
            eprintln!("error in model pass {tag}: {error:#}");
            failed += 1;
        }
    }
    if failed > 0 {
        bail!("{failed} model pass(es) failed");
    }
    Ok(())
}

fn run_pass(cli: Cli, plan: Option<Plan>, index: usize) -> Result<()> {
    if !cli.jsonl {
        return run_cli(cli, plan, None);
    }
    let pass = serde_json::json!({
        "index": index, "engine": engine_name(cli.engine), "model": cli.model,
        "model_tag": cli.output_tag, "progress_file": cli.progress_file.as_deref().map(output_key),
    });
    write_json_line(&serde_json::json!({
        "schema_version": CLI_SCHEMA_VERSION, "event": "pass_started", "pass": pass,
    }))?;
    let result = run_cli(cli, plan, Some(pass.clone()));
    let event = write_finished_event("pass_finished", Some(&pass), &result);
    result.and(event)
}

fn configure_model(cli: &mut Cli) {
    if matches!(cli.engine, EngineKind::AppleNative) {
        cli.model = default_asr(cli.engine).to_owned();
    }
    if cli.model.starts_with('~') {
        cli.model = expand_home(Path::new(&cli.model))
            .to_string_lossy()
            .into_owned();
    }
    // Preserve the original Parakeet default when a caller supplies a Qwen preset.
    if matches!(cli.engine, EngineKind::ParakeetV3)
        && model_cache::canonical_id(&cli.model).starts_with("Qwen/Qwen3-ASR")
    {
        cli.model = default_asr(cli.engine).to_owned();
    }
    cli.output_tag = cli.model_suffix.then(|| model_tag(cli.engine, &cli.model));
}

fn effective_timestamps(cli: &Cli) -> TimestampGranularity {
    if matches!(cli.format, OutputFormat::Srt | OutputFormat::Vtt)
        && cli.timestamps == TimestampGranularity::None
    {
        TimestampGranularity::Sentences
    } else {
        cli.timestamps
    }
}

struct RunRequirements {
    aligner: bool,
    diarization: bool,
}

impl RunRequirements {
    fn for_units<'a>(cli: &Cli, mut units: impl Iterator<Item = &'a Unit> + Clone) -> Self {
        let diarization = units
            .clone()
            .flat_map(|unit| unit.jobs())
            .any(|job| job.diarize);
        let aligner = matches!(cli.engine, EngineKind::Qwen3)
            && (effective_timestamps(cli) != TimestampGranularity::None
                || diarization
                || units.any(|unit| matches!(unit, Unit::Session(_))));
        Self {
            aligner,
            diarization,
        }
    }

    fn models<'a>(&self, cli: &'a Cli) -> Vec<(&'static str, &'a str)> {
        let mut models = Vec::new();
        if !matches!(cli.engine, EngineKind::AppleNative) {
            models.push(("asr", cli.model.as_str()));
        }
        if self.aligner {
            models.push(("aligner", cli.forced_aligner.as_str()));
        }
        if self.diarization {
            models.push((
                "diarization",
                cli.diarization_model
                    .as_deref()
                    .and_then(Path::to_str)
                    .unwrap_or("sortformer"),
            ));
        }
        models
    }

    fn validate(&self, cli: &Cli) -> Result<()> {
        ensure_apple_silicon()?;
        if matches!(cli.engine, EngineKind::AppleNative) {
            if self.diarization {
                bail!(
                    "Apple SpeechTranscriber does not provide diarization; use --engine qwen3 with --diarize"
                );
            }
            if apple_native_helper().is_none() {
                bail!(
                    "Apple SpeechTranscriber helper is not bundled; set TRANSCRIBERRR_NATIVE or build the macOS app"
                );
            }
        } else if matches!(cli.device, DeviceKind::Coreml) {
            bail!(
                "CoreML/ANE is not available for these Rust engines; use --device metal for native Apple GPU execution"
            );
        }
        if !cfg!(feature = "parakeet") && matches!(cli.engine, EngineKind::ParakeetV3) {
            bail!(
                "Parakeet support is not included in this build; rebuild with --features parakeet"
            );
        }
        if !cfg!(feature = "diarization") && self.diarization {
            bail!(
                "diarization support is not included in this build; rebuild with --features diarization"
            );
        }
        Ok(())
    }
}

fn preview_cli(cli: &Cli, progress_paths: &[&Path]) -> Result<serde_json::Value> {
    let (units, skips, rejected) = prepare_plan(cli)?;
    protect_plan_files(cli, &units, &skips, &rejected, progress_paths)?;
    queue_plan(cli, &units, &skips, &rejected)
}

fn run_cli(mut cli: Cli, plan: Option<Plan>, stream_pass: Option<serde_json::Value>) -> Result<()> {
    check_cancelled()?;
    if cli.forced_aligner.starts_with('~') {
        cli.forced_aligner = expand_home(Path::new(&cli.forced_aligner))
            .to_string_lossy()
            .into_owned();
    }
    if cli.plan {
        let progress_paths = cli.progress_file.as_deref().into_iter().collect::<Vec<_>>();
        return write_json_line(&preview_cli(&cli, &progress_paths)?);
    }
    let model_cache = model_cache::ModelCache::new(cli.model_cache.clone(), cli.offline);
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
    // Keep the user-facing model identifier before replacing it with a local cache snapshot path.
    // Inference needs the resolved directory, while transcript metadata should remain stable
    // across cache revisions and machines.
    // Subtitles need cue timing. Without timestamps, Qwen loads no aligner and produced one
    // cue spanning the whole file; give subtitle formats sentence cues instead.
    cli.timestamps = effective_timestamps(&cli);
    let model_label = model_display_name(&cli.model);
    let (units, skips, rejected) = match plan {
        Some(plan) => plan,
        None => prepare_plan(&cli)?,
    };
    if units.is_empty() && rejected.is_empty() {
        bail!("no supported audio files were found");
    }
    let progress_paths = cli.progress_file.as_deref().into_iter().collect::<Vec<_>>();
    protect_plan_files(&cli, &units, &skips, &rejected, &progress_paths)?;
    let mut progress = ProgressLog::open(cli.progress_file.as_deref(), cli.resume)?;
    progress.stream_pass = stream_pass;
    for (input, error) in &rejected {
        eprintln!("error processing {}: {error}", input.display());
        progress.record(ProgressEvent {
            error: Some(error.clone()),
            phase: Some("Failed".to_owned()),
            ..ProgressEvent::new(input_key(input), "failed")
        })?;
    }
    for skip in &skips {
        eprintln!(
            "note: skipping {}: {}",
            skip.job.input.display(),
            skip.reason
        );
        progress.record(ProgressEvent {
            progress: Some(1.0),
            phase: Some("Skipped".to_owned()),
            warning: Some(format!("Not transcribed: {}", skip.reason)),
            ..ProgressEvent::new(input_key(&skip.job.input), "skipped")
        })?;
    }
    if !rejected.is_empty() && (cli.fail_fast || units.is_empty()) {
        bail!("{} input file(s) failed", rejected.len());
    }
    // Report reusable outputs before model setup, which may fail or take time. Only the
    // remaining units can run afterward, so the loaded models cover every eligible job.
    let mut pending = Vec::new();
    for unit in units {
        check_cancelled()?;
        if unit_done(&cli, &progress, &unit) {
            record_skipped(&cli, &mut progress, &unit)?;
        } else {
            pending.push(unit);
        }
    }
    if pending.is_empty() {
        if !rejected.is_empty() {
            bail!("{} input file(s) failed", rejected.len());
        }
        return Ok(());
    }
    check_cancelled()?;
    let requirements = RunRequirements::for_units(&cli, pending.iter());
    requirements.validate(&cli)?;
    if cli.offline {
        for (_, id) in requirements.models(&cli) {
            if model_cache.lookup(id).is_none() {
                bail!(
                    "model {id:?} is not available locally (--offline); prefetch it with --download-model or supply a local model path"
                );
            }
        }
    }
    audio_convert::remove_stale_temp_files();
    // Resolved after the --resume early exit so an already-finished queue needs no model
    // (and no network to fetch one).
    #[cfg(feature = "diarization")]
    let diarization_model_path = if requirements.diarization {
        let value = cli
            .diarization_model
            .as_deref()
            .unwrap_or_else(|| Path::new("sortformer"));
        if value.is_file() {
            Some(value.to_path_buf())
        } else {
            Some(model_cache.ensure(&value.to_string_lossy())?)
        }
    } else {
        None
    };
    // Qwen uses Candle, while Parakeet and Sortformer initialize their own execution providers.
    let device = if matches!(cli.engine, EngineKind::Qwen3) {
        match cli.device {
            DeviceKind::Metal => Device::new_metal(0)?,
            DeviceKind::Cpu | DeviceKind::Coreml => Device::Cpu,
        }
    } else {
        Device::Cpu
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
        if requirements.aligner {
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
    let mut diarizer = if requirements.diarization {
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
    let mut models = InferenceModels {
        asr: asr.as_ref(),
        aligner: aligner.as_ref(),
        parakeet: parakeet.as_mut(),
        diarizer: &mut diarizer,
        model_label: &model_label,
    };
    for unit in pending {
        if cancel_requested() {
            // Caught between units, so no job recorded it: the app shows unrun rows as
            // canceled, rather than failed, only after a `canceled` event.
            for job in unit.jobs() {
                progress.record(ProgressEvent {
                    phase: Some("Canceled".to_owned()),
                    ..ProgressEvent::new(input_key(&job.input), "canceled")
                })?;
            }
            break;
        }
        if unit_done(&cli, &progress, &unit) {
            record_skipped(&cli, &mut progress, &unit)?;
            continue;
        }
        match unit {
            Unit::Single(job) => {
                run_job(
                    &cli,
                    &job,
                    models.reborrow(),
                    &mut progress,
                    None,
                    &mut failures,
                )?;
            }
            Unit::Session(session) => {
                run_session(&cli, &session, &mut models, &mut progress, &mut failures)?;
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
    CANCELLED.load(Ordering::SeqCst) || launcher_exited()
}

/// The app sets TRANSCRIBERRR_PARENT_PID. If it crashes or is force-quit, the CLI is
/// reparented and would keep the GPU busy, then die mid-write without recording the track.
/// Opt-in, so `nohup transcriberrr … &` from a closed terminal keeps running.
fn launcher_exited() -> bool {
    std::env::var("TRANSCRIBERRR_PARENT_PID")
        .ok()
        .and_then(|pid| pid.parse::<libc::pid_t>().ok())
        .is_some_and(|pid| unsafe { libc::getppid() } != pid)
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

fn format_extension(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Txt => "txt",
        OutputFormat::Json => "json",
        OutputFormat::Srt => "srt",
        OutputFormat::Vtt => "vtt",
    }
}

/// `input_key` for a file that may not exist yet: the canonical folder plus the name, so the
/// events recorded before and after writing it agree.
fn output_key(path: &Path) -> String {
    use std::path::Component;

    let mut suffix = Vec::new();
    let mut path_to_resolve = path;
    // The destination itself may be a symlink. Resolve only its ancestors.
    if let Some(parent) = path.parent() {
        suffix.extend(path.components().next_back());
        path_to_resolve = parent;
    }
    loop {
        let ancestor = if path_to_resolve.as_os_str().is_empty() {
            Path::new(".")
        } else {
            path_to_resolve
        };
        if let Ok(mut canonical) = fs::canonicalize(ancestor) {
            for component in suffix.into_iter().rev() {
                match component {
                    Component::Normal(name) => canonical.push(name),
                    Component::ParentDir => {
                        canonical.pop();
                    }
                    _ => {}
                }
            }
            return canonical.display().to_string();
        }
        let Some(parent) = path_to_resolve.parent() else {
            return path.display().to_string();
        };
        suffix.extend(path_to_resolve.components().next_back());
        path_to_resolve = parent;
    }
}

/// Zoom puts participant files in an `Audio Record` folder inside the meeting's folder,
/// beside the mixed recording; stand for the meeting's folder instead of that one.
fn meeting_folder(folder: PathBuf) -> PathBuf {
    let is_zoom_folder = folder
        .file_name()
        .is_some_and(|name| name.to_string_lossy().eq_ignore_ascii_case("audio record"));
    match folder.parent() {
        Some(meeting) if is_zoom_folder => meeting.to_path_buf(),
        _ => folder,
    }
}

/// The folder a track's recording session lives in.
fn session_folder(input: &Path) -> PathBuf {
    let input = fs::canonicalize(input).unwrap_or_else(|_| PathBuf::from(output_key(input)));
    meeting_folder(
        input
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    )
}

/// Video containers Zoom (and screen recorders) save beside an audio-only copy.
const VIDEO_EXTENSIONS: &[&str] = &["m4v", "mov", "mp4", "webm"];

/// A queued file the plan leaves out, and why.
struct Skip {
    job: Job,
    reason: String,
    /// The inputs transcribed instead; if all of them fail before running, so does this.
    instead: Vec<PathBuf>,
}

/// Fail the files that would write the same output, and the rest of their sessions.
fn drop_clashes(cli: &Cli, units: &mut Vec<Unit>, rejected: &mut Vec<(PathBuf, String)>) {
    let mut clashes = if cli.inventory_only {
        Vec::new()
    } else {
        output_clashes(units.iter().flat_map(|unit| {
            let job_outputs = unit
                .writes_job_outputs()
                .then_some(unit.jobs())
                .into_iter()
                .flatten()
                .map(|job| (job.input.as_path(), output_path(cli, job)));
            let merged_output = match unit {
                Unit::Session(session) => {
                    Some((session.jobs[0].input.as_path(), session.output.clone()))
                }
                Unit::Single(_) => None,
            };
            job_outputs.chain(merged_output)
        }))
    };
    for unit in units.iter().filter(|unit| !unit.writes_job_outputs()) {
        // Two encodings of the same participant export would duplicate its words.
        let mut stems = HashSet::new();
        for job in unit.jobs() {
            let stem = folded_path(&input_key(&job.input.with_extension("")));
            if !stems.insert(stem) {
                clashes.push((
                    job.input.clone(),
                    "duplicate Zoom participant export (same file name with a different extension)"
                        .to_owned(),
                ));
            }
        }
    }
    if !clashes.is_empty() {
        let clashing = clashes
            .iter()
            .map(|(input, _)| input.clone())
            .collect::<std::collections::HashSet<_>>();
        units.retain(|unit| {
            let jobs = unit.jobs();
            if !jobs.iter().any(|job| clashing.contains(&job.input)) {
                return true;
            }
            for job in jobs {
                if !clashing.contains(&job.input) {
                    rejected.push((
                        job.input.clone(),
                        "not transcribed: another file of its session clashes".to_owned(),
                    ));
                }
            }
            false
        });
        rejected.extend(clashes);
    }
}

/// Fail each skipped file whose replacements were all rejected before the run, which would
/// otherwise leave it shown as covered while nothing of it is transcribed.
fn reject_uncovered(skips: &mut Vec<Skip>, rejected: &mut Vec<(PathBuf, String)>) {
    let mut gone = rejected
        .iter()
        .map(|(input, _)| input.clone())
        .collect::<HashSet<_>>();
    loop {
        let previous_count = skips.len();
        skips.retain(|skip| {
            if skip.instead.is_empty() || !skip.instead.iter().all(|input| gone.contains(input)) {
                return true;
            }
            gone.insert(skip.job.input.clone());
            rejected.push((
                skip.job.input.clone(),
                format!(
                    "not transcribed: it was left out as {}, but those files cannot run",
                    skip.reason
                ),
            ));
            false
        });
        if skips.len() == previous_count {
            return;
        }
    }
}

/// The units to run, the files left out, and the files of sessions that cannot run.
type Plan = (Vec<Unit>, Vec<Skip>, Vec<(PathBuf, String)>);

#[derive(Clone)]
struct PlanInputs {
    jobs: Vec<Job>,
    missing: Vec<PathBuf>,
    rejected: Vec<(PathBuf, String)>,
}

fn collect_plan_inputs(cli: &Cli) -> Result<PlanInputs> {
    // `--iso-tracks off` takes `off` as a file and every input as a speaker track.
    if cli.iso_tracks == IsoMode::All
        && let Some(mode) = cli.inputs.iter().find(|input| {
            ["auto", "all", "off"].contains(&input.to_string_lossy().as_ref()) && !input.exists()
        })
    {
        bail!(
            "write --iso-tracks={}: without `=`, \"{}\" is read as a file",
            mode.display(),
            mode.display()
        );
    }
    let mut rejected = Vec::new();
    let mut missing = Vec::new();
    let mut jobs = collect_jobs(cli, &mut rejected, &mut missing)?;
    for job in &mut jobs {
        if job.iso {
            job.diarize = false;
            job.speakers = None;
        }
    }
    Ok(PlanInputs {
        jobs,
        missing,
        rejected,
    })
}

fn prepare_plan(cli: &Cli) -> Result<Plan> {
    plan_inputs(cli, collect_plan_inputs(cli)?)
}

fn plan_inputs(
    cli: &Cli,
    PlanInputs {
        jobs,
        missing,
        mut rejected,
    }: PlanInputs,
) -> Result<Plan> {
    let (mut units, mut skips) = plan_units(cli, jobs, &missing)?;
    reject_missing_zoom_parts(&mut units, &missing, &mut rejected);
    drop_clashes(cli, &mut units, &mut rejected);
    reject_uncovered(&mut skips, &mut rejected);
    Ok((units, skips, rejected))
}

/// A snapshot can name a track that disappeared after the queue was planned.
fn reject_missing_zoom_parts(
    units: &mut Vec<Unit>,
    missing: &[PathBuf],
    rejected: &mut Vec<(PathBuf, String)>,
) {
    let mut missing_tracks: HashMap<_, HashSet<_>> = HashMap::new();
    let mut missing_recordings: HashMap<_, Vec<_>> = HashMap::new();
    for path in missing {
        let name = merge::parse_path(path);
        if let Some(recording) = zoom_media(path, &name) {
            if !recording.legacy_copy {
                missing_recordings
                    .entry(session_folder(path))
                    .or_default()
                    .push(recording);
            }
        } else if let merge::TrackName::Zoom { meeting, .. } = name {
            missing_tracks
                .entry(session_folder(path))
                .or_default()
                .insert(meeting);
        }
    }
    if missing_tracks.is_empty() && missing_recordings.is_empty() {
        return;
    }
    units.retain(|unit| {
        let (folder, id, missing_segment) = match unit {
            Unit::Session(session) if session.zoom.is_some() => (
                session_folder(&session.jobs[0].input),
                session.zoom.clone(),
                false,
            ),
            Unit::Single(job) if job.zoom_recording => {
                let folder = session_folder(&job.input);
                let parts = std::iter::once(&job.input)
                    .chain(&job.parts)
                    .filter_map(|path| zoom_media(path, &merge::parse_path(path)))
                    .collect::<Vec<_>>();
                let id = parts.iter().find_map(|media| media.meeting.clone());
                let missing_segment = missing_recordings.get(&folder).is_some_and(|missing| {
                    missing.iter().any(|media| {
                        (media.meeting.is_none() || id.is_none() || media.meeting == id)
                            && !parts.iter().any(|part| part.segment == media.segment)
                    })
                });
                (folder, id, missing_segment)
            }
            _ => return true,
        };
        let missing_participant = missing_tracks
            .get(&folder)
            .is_some_and(|ids| id.is_none_or(|id| ids.contains(&id)));
        let reason = if missing_participant {
            "a selected participant track from this Zoom meeting is unavailable"
        } else if missing_segment {
            "a selected recording segment from this Zoom meeting is unavailable"
        } else {
            return true;
        };
        rejected.extend(
            unit.jobs()
                .iter()
                .map(|job| (job.input.clone(), reason.to_owned())),
        );
        false
    });
}

fn queue_plan(
    cli: &Cli,
    units: &[Unit],
    skips: &[Skip],
    rejected: &[(PathBuf, String)],
) -> Result<serde_json::Value> {
    let progress = ProgressLog::read(cli.progress_file.as_deref(), cli.resume)?;
    let mut files = Vec::new();
    let mut meetings = Vec::new();
    let mut meeting_members = HashMap::new();
    let mut pending = Vec::new();
    for unit in units {
        let done = unit_done(cli, &progress, unit);
        if !done {
            pending.push(unit);
        }
        match unit {
            Unit::Session(session) if let Some(id) = &session.zoom => {
                let folder = session_folder(&session.jobs[0].input);
                let key = format!("{}|{id}", folder.display());
                meetings.push(serde_json::json!({
                    "id": key, "label": session.label, "output": output_key(&session.output),
                    "done": done, "participants": session.speakers.iter().collect::<HashSet<_>>().len(),
                }));
                for (job, speaker) in session.jobs.iter().zip(&session.speakers) {
                    let path = input_key(&job.input);
                    meeting_members.insert(path.clone(), key.clone());
                    files.push(serde_json::json!({
                        "path": path, "meeting": key, "speaker": speaker, "iso": true, "done": done,
                    }));
                }
            }
            _ => {
                let (merge, group) = match unit {
                    Unit::Session(session) => (
                        Some(output_key(&session.output)),
                        Some(session.label.as_str()),
                    ),
                    Unit::Single(_) => (None, None),
                };
                for job in unit.jobs() {
                    files.push(serde_json::json!({
                        "path": input_key(&job.input), "output": output_key(&output_path(cli, job)),
                        "iso": job.iso, "diarize": job.diarize, "done": done,
                        "zoom_recording": job.zoom_recording,
                        "merge_output": merge, "group": group,
                    }));
                }
            }
        }
    }
    for skip in skips {
        let meeting = skip
            .instead
            .iter()
            .find_map(|path| meeting_members.get(&input_key(path)));
        files.push(serde_json::json!({
            "path": input_key(&skip.job.input), "meeting": meeting, "skip": skip.reason,
            "zoom_recording": skip.job.zoom_recording,
            "covered_by": skip.instead.iter().map(|path| input_key(path)).collect::<Vec<_>>(),
        }));
    }
    for (path, error) in rejected {
        files.push(serde_json::json!({"path": input_key(path), "error": error}));
    }
    let cache = model_cache::ModelCache::new(cli.model_cache.clone(), true);
    let mut issues = Vec::new();
    let mut required_models = Vec::new();
    if units.is_empty() && rejected.is_empty() {
        issues.push("no supported audio files were found".to_owned());
    }
    if !pending.is_empty() && !cli.inventory_only {
        let requirements = RunRequirements::for_units(cli, pending.iter().copied());
        if let Err(error) = requirements.validate(cli) {
            issues.push(error.to_string());
        }
        for (role, id) in requirements.models(cli) {
            let path = cache.lookup(id);
            if cli.offline && path.is_none() {
                issues.push(format!("model {id:?} is not available locally (--offline)"));
            }
            required_models.push(serde_json::json!({
                "role": role, "id": id, "cached": path.is_some(),
                "path": path.as_deref().map(output_key),
            }));
        }
    }
    Ok(serde_json::json!({
        "schema_version": CLI_SCHEMA_VERSION,
        "files": files, "meetings": meetings, "engine": engine_name(cli.engine),
        "model": cli.model, "model_tag": cli.output_tag,
        "progress_file": cli.progress_file.as_deref().map(output_key),
        "format": format_extension(cli.format),
        "timestamps": effective_timestamps(cli).to_possible_value().expect("CLI timestamp option").get_name(),
        "required_models": required_models, "issues": issues,
        "summary": {
            "pending_units": pending.len(), "existing_units": units.len() - pending.len(),
            "skipped_inputs": skips.len(), "rejected_inputs": rejected.len(),
        },
    }))
}

/// Group Zoom files by canonical meeting folder and recording ID, and manually marked
/// non-Zoom tracks by folder. Only associate an ID-less Zoom export when its folder
/// identifies one meeting.
fn plan_units(cli: &Cli, jobs: Vec<Job>, missing: &[PathBuf]) -> Result<(Vec<Unit>, Vec<Skip>)> {
    let names = merge::parse_paths(jobs.iter().map(|job| job.input.as_path()));
    let folders = jobs
        .iter()
        .map(|job| session_folder(&job.input))
        .collect::<Vec<_>>();
    let media = jobs
        .iter()
        .zip(&names)
        .map(|(job, name)| {
            if let Some(copy) = &job.zoom_audio_copy {
                return zoom_media(&job.input, &merge::parse_path(copy));
            }
            (!job.iso || matches!(name, merge::TrackName::ZoomMixed { .. }))
                .then(|| zoom_media(&job.input, name))
                .flatten()
        })
        .collect::<Vec<_>>();
    let mut ids_by_folder: HashMap<PathBuf, HashSet<String>> = HashMap::new();
    for ((name, folder), recording) in names.iter().zip(&folders).zip(&media) {
        if let Some(id) = zoom_id(name).or_else(|| {
            recording
                .as_ref()
                .and_then(|export| export.meeting.as_deref())
        }) {
            ids_by_folder
                .entry(folder.clone())
                .or_default()
                .insert(id.to_owned());
        }
    }
    for path in missing {
        if let Some(Some(id)) = zoom_candidate(path) {
            ids_by_folder
                .entry(session_folder(path))
                .or_default()
                .insert(id);
        }
    }
    let resolved_id = |folder: &Path, id: Option<&str>| {
        if let Some(id) = id {
            Some(id.to_owned())
        } else {
            ids_by_folder
                .get(folder)
                .filter(|ids| ids.len() == 1)
                .and_then(|ids| ids.iter().next())
                .cloned()
        }
    };
    let mut skips: Vec<Skip> = Vec::new();
    let mut taken = vec![false; jobs.len()];

    let mut audio_by_stem: HashMap<_, Vec<_>> = HashMap::new();
    let mut audio_by_input = HashMap::new();
    let mut iso_by_meeting: HashMap<_, Vec<_>> = HashMap::new();
    let stem_key = |index: usize| folded_path(&input_key(&jobs[index].input.with_extension("")));
    for (index, job) in jobs.iter().enumerate() {
        if is_video(&job.input) {
            continue;
        }
        audio_by_stem
            .entry(stem_key(index))
            .or_default()
            .push(index);
        audio_by_input.insert(input_key(&job.input), index);
        if let merge::TrackName::Zoom { meeting, .. } = &names[index]
            && job.iso
        {
            iso_by_meeting
                .entry((folders[index].as_path(), meeting.clone()))
                .or_default()
                .push(index);
        }
    }
    let track_paths = |indices: &[usize]| {
        indices
            .iter()
            .map(|&index| jobs[index].input.clone())
            .collect::<Vec<_>>()
    };
    for (index, job) in jobs.iter().enumerate() {
        let iso = media[index].as_ref().and_then(|recording| {
            let folder = folders[index].as_path();
            let id = resolved_id(folder, recording.meeting.as_deref())?;
            iso_by_meeting.get(&(folder, id))
        });
        let (reason, replacements) = if let Some(iso) = iso {
            let kind = if is_video(&job.input) {
                "video"
            } else {
                "mixed audio"
            };
            (
                format!("Zoom {kind} covered by the meeting's participant tracks"),
                iso.as_slice(),
            )
        } else if let Some(copy) = &job.zoom_audio_copy
            && let Some(index) = audio_by_input.get(&input_key(copy))
        {
            (
                "a renamed Zoom video whose identical audio-only export is used instead".to_owned(),
                std::slice::from_ref(index),
            )
        } else if !job.iso
            && is_video(&job.input)
            && let Some(audio) = audio_by_stem.get(&stem_key(index))
        {
            (
                "a video whose audio-only copy is transcribed instead".to_owned(),
                audio.as_slice(),
            )
        } else {
            continue;
        };
        taken[index] = true;
        skips.push(Skip {
            job: job.clone(),
            reason,
            instead: track_paths(replacements),
        });
    }

    let mut drafts = Vec::new();
    let mut draft_indices = HashMap::new();
    for (index, job) in jobs.iter().enumerate() {
        if !job.iso || taken[index] {
            continue;
        }
        let meeting = match &names[index] {
            merge::TrackName::Zoom { meeting, .. } if !is_video(&job.input) => {
                Some(meeting.clone())
            }
            merge::TrackName::ZoomMixed { .. } => continue,
            _ => None,
        };
        taken[index] = true;
        let draft = *draft_indices
            .entry((&folders[index], meeting.clone()))
            .or_insert_with(|| {
                drafts.push((meeting, Vec::new()));
                drafts.len() - 1
            });
        drafts[draft].1.push(index);
    }

    // Pick one mixed export for each recording segment. Use video for segments that
    // have no audio export and prefer the main video over alternate views.
    let mut recordings = HashMap::new();
    for (index, recording) in media.iter().enumerate() {
        let Some(recording) = recording else { continue };
        if taken[index] {
            continue;
        }
        taken[index] = true;
        let folder = folders[index].as_path();
        let id = resolved_id(folder, recording.meeting.as_deref());
        let copy = recording
            .legacy_copy
            .then(|| jobs[index].input.file_stem())
            .flatten();
        let segments = recordings
            .entry((folder, id, copy))
            .or_insert_with(BTreeMap::new);
        segments
            .entry(recording.segment)
            .or_insert_with(Vec::new)
            .push(index);
    }
    let mut singles: Vec<usize> = Vec::new();
    let mut joined = HashMap::new();
    for ((folder, id, copy), segments) in recordings {
        let mut parts = Vec::with_capacity(segments.len());
        let mut redundant = Vec::new();
        for mut candidates in segments.into_values() {
            candidates.sort_unstable_by_key(|&index| {
                (
                    media[index].as_ref().unwrap().kind,
                    input_key(&jobs[index].input),
                )
            });
            let (chosen, copies) = candidates
                .split_first()
                .expect("each segment has an export");
            parts.push(*chosen);
            redundant.extend_from_slice(copies);
        }
        let (first, rest) = parts.split_first().expect("each recording has a segment");
        let primary = &jobs[*first];
        for (indices, reason) in [
            (
                rest,
                format!(
                    "the recording continues {}; both are transcribed together",
                    primary
                        .input
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                ),
            ),
            (
                redundant.as_slice(),
                "another Zoom export covers the same recording segment".to_owned(),
            ),
        ] {
            for &index in indices {
                skips.push(Skip {
                    job: jobs[index].clone(),
                    reason: reason.clone(),
                    instead: vec![primary.input.clone()],
                });
            }
        }
        let output =
            (primary.zoom_recording && primary.output.is_none() && copy.is_none()).then(|| {
                let stable = fallback_output_path(cli, primary, folder, id.as_deref());
                if stable.exists() {
                    return stable;
                }
                if id.is_some()
                    && parts.iter().chain(&redundant).any(|&index| {
                        media[index]
                            .as_ref()
                            .is_some_and(|item| item.meeting.is_none())
                    })
                {
                    let legacy = fallback_output_path(cli, primary, folder, None);
                    if legacy.is_file() {
                        return legacy;
                    }
                }
                // Previous versions joined audio-only segments, or wrote each video separately.
                // Reuse their output where it represents the same planned recording.
                let old_paths = if parts.len() == 1 {
                    redundant.as_slice()
                } else {
                    &[]
                };
                if parts.len() == 1
                    || parts.iter().all(|&index| {
                        media[index]
                            .as_ref()
                            .is_some_and(|item| item.kind == ZoomMediaKind::Audio)
                    })
                {
                    return std::iter::once(first)
                        .chain(old_paths)
                        .map(|&index| &jobs[index])
                        .filter(|job| job.output.is_none())
                        .map(|job| output_path(cli, job))
                        .find(|path| path.is_file())
                        .unwrap_or(stable);
                }
                stable
            });
        singles.push(*first);
        joined.insert(*first, (rest.to_vec(), output));
    }
    for (index, taken) in taken.iter().enumerate() {
        if !taken {
            singles.push(index);
        }
    }

    // Place each unit where its first file was queued.
    let mut slots: Vec<Option<Job>> = jobs.into_iter().map(Some).collect();
    let mut take = |index: usize| slots[index].take().expect("each job is planned once");
    let mut planned: Vec<(usize, Unit)> = Vec::new();
    for (meeting, tracks) in drafts {
        let first = tracks[0];
        let mut tracks = tracks.into_iter().map(&mut take).collect::<Vec<_>>();
        let unit = if meeting.is_none() && tracks.len() == 1 {
            Unit::Single(tracks.remove(0))
        } else {
            Unit::Session(IsoSession {
                label: String::new(),
                zoom: meeting,
                jobs: tracks,
                speakers: Vec::new(),
                output: PathBuf::new(),
            })
        };
        planned.push((first, unit));
    }
    for index in singles {
        let mut job = take(index);
        if let Some((parts, output)) = joined.remove(&index) {
            job.parts = parts.into_iter().map(|index| take(index).input).collect();
            if let Some(output) = output {
                job.planned_output = Some(output);
            }
        }
        planned.push((index, Unit::Single(job)));
    }
    planned.sort_by_key(|(first, _)| *first);
    let mut units = planned
        .into_iter()
        .map(|(_, unit)| unit)
        .collect::<Vec<_>>();
    let progress = ProgressLog::read(cli.progress_file.as_deref(), cli.resume)?;
    name_sessions(cli, &progress, &mut units);
    Ok((units, skips))
}

/// Name each session after its source folder. Zoom uses a fixed destination so later runs
/// find the same transcript. Manual sessions keep their numbered collision handling.
fn name_sessions(cli: &Cli, progress: &ProgressLog, units: &mut [Unit]) {
    let base = |session: &IsoSession| {
        session_folder(&session.jobs[0].input)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "session".to_owned())
    };
    // Manual sessions put their merge beside the track transcripts.
    let folder = |session: &IsoSession| {
        meeting_folder(
            output_path(cli, &session.jobs[0])
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default(),
        )
    };
    let key = |folder: &Path, name: &str| folded_path(&folder.join(name).to_string_lossy());
    let mut taken = std::collections::HashSet::new();
    for unit in units {
        let Unit::Session(session) = unit else {
            continue;
        };
        let base = base(session);
        let extension = format_extension(cli.format);
        let label;
        if let Some(id) = &session.zoom {
            let job = &session.jobs[0];
            let source = session_folder(&job.input);
            let stem = &base[..base.floor_char_boundary(200)];
            let (folder, filename) = match job.output_location {
                OutputLocation::Folder => (zoom_destination(job, Some(id)), stem.to_owned()),
                OutputLocation::NextToSource => (source, format!("{stem} [{id}]")),
            };
            session.output =
                tagged_output(cli, folder.join(format!("{filename}.merged.{extension}")));
            label = base;
        } else {
            let folder = folder(session);
            let output = |label: &str| {
                tagged_output(cli, folder.join(format!("{label}.merged.{extension}")))
            };
            let mut candidate = base.clone();
            let mut number = 2;
            while !taken.insert(key(&folder, &candidate))
                || merged_elsewhere(&output(&candidate), session)
                || kept_merge(cli, progress, &output(&candidate), session)
            {
                candidate = format!("{base} {number}");
                number += 1;
            }
            session.output = output(&candidate);
            label = candidate;
        }
        session.speakers = merge::speaker_names(
            &session
                .jobs
                .iter()
                .map(|job| job.input.as_path())
                .collect::<Vec<_>>(),
        );
        session.label = label;
    }
}

/// Whether `--resume` may replace a manual session's existing merged transcript: one whose
/// write it logged as interrupted, or a JSON merge of this very session (a re-run).
fn resume_replaces_merge(
    cli: &Cli,
    progress: &ProgressLog,
    path: &Path,
    session: &IsoSession,
) -> bool {
    cli.resume
        && (progress.interrupted_write(&output_key(path), path)
            || merge_owner(path, session) == Some(true))
}

/// Whether a manual session's merged transcript exists and this run would neither skip the
/// session for it nor replace it: under `--resume`, a transcript that cannot tell which
/// session wrote it (not JSON). The session is merged under the next free name (`ep 2`)
/// instead of failing, as the app numbers re-run transcripts.
fn kept_merge(cli: &Cli, progress: &ProgressLog, path: &Path, session: &IsoSession) -> bool {
    if !cli.resume || cli.overwrite || cli.skip_existing || !path.exists() {
        return false;
    }
    let completed = progress
        .event_for_output(&output_key(path), path)
        .is_some_and(|event| matches!(event.status.as_str(), "completed" | "skipped"));
    !completed && !resume_replaces_merge(cli, progress, path, session)
}

/// Whether an existing merged transcript was merged from other recordings: those of another
/// session with this one's name (two "Episode 1" folders, run on different days into one
/// output folder). Only JSON tells; another format is taken to be this session's.
fn merged_elsewhere(path: &Path, session: &IsoSession) -> bool {
    merge_owner(path, session) == Some(false)
}

/// Whether a merged transcript at `path` is this session's (`Some(true)`), another's
/// (`Some(false)`), or cannot tell (`None`: missing, not JSON, or not a merge).
fn merge_owner(path: &Path, session: &IsoSession) -> Option<bool> {
    let doc = read_transcript(path).ok()?;
    if doc.metadata.merged_from.is_empty() {
        return None;
    }
    let inputs = session
        .jobs
        .iter()
        .map(|job| input_key(&job.input))
        .collect::<std::collections::HashSet<_>>();
    Some(
        doc.metadata
            .merged_from
            .iter()
            .any(|track| inputs.contains(&input_key(Path::new(&track.source)))),
    )
}

/// Whether a unit can be skipped: finished in an earlier run (`--resume`), or with its
/// transcript already there (`--skip-existing`).
fn unit_done(cli: &Cli, progress: &ProgressLog, unit: &Unit) -> bool {
    if cli.overwrite {
        return false;
    }
    let (exists, zoom) = match unit {
        Unit::Single(job) => (output_path(cli, job).is_file(), job.zoom_recording),
        Unit::Session(session) => (session.output.is_file(), session.zoom.is_some()),
    };
    if zoom {
        !cli.overwrite && exists
    } else {
        (cli.resume && unit_completed(cli, progress, unit)) || (cli.skip_existing && exists)
    }
}

fn unit_completed(cli: &Cli, progress: &ProgressLog, unit: &Unit) -> bool {
    let (key, output) = match unit {
        Unit::Single(job) => (input_key(&job.input), output_path(cli, job)),
        // A session counts as done only once merged; otherwise all its tracks run again,
        // since the merge needs every track's words.
        Unit::Session(session) => (output_key(&session.output), session.output.clone()),
    };
    output.is_file()
        && progress
            .event_for_output(&key, &output)
            .is_some_and(|event| matches!(event.status.as_str(), "completed" | "skipped"))
}

fn record_skipped(cli: &Cli, progress: &mut ProgressLog, unit: &Unit) -> Result<()> {
    let skipped = |input: String, output: &Path| ProgressEvent {
        output: Some(output.display().to_string()),
        progress: Some(1.0),
        phase: Some("Skipped".to_owned()),
        ..ProgressEvent::new(input, "skipped")
    };
    match unit {
        Unit::Single(job) => {
            let output = output_path(cli, job);
            if job.zoom_recording {
                eprintln!(
                    "note: skipping {}: transcript already exists",
                    output.display()
                );
            }
            progress.record(skipped(input_key(&job.input), &output))
        }
        Unit::Session(session) => record_skipped_session(cli, progress, session),
    }
}

fn record_skipped_session(
    cli: &Cli,
    progress: &mut ProgressLog,
    session: &IsoSession,
) -> Result<()> {
    eprintln!(
        "note: skipping {}: transcript already exists",
        session.output.display()
    );
    for (job, speaker) in session.jobs.iter().zip(&session.speakers) {
        progress.record(ProgressEvent {
            output: session
                .zoom
                .is_none()
                .then(|| output_path(cli, job))
                .filter(|path| path.is_file())
                .map(|path| path.display().to_string()),
            group: Some(session.label.clone()),
            speaker: Some(speaker.to_owned()),
            progress: Some(1.0),
            phase: Some("Merged transcript already exists".to_owned()),
            ..ProgressEvent::new(input_key(&job.input), "skipped")
        })?;
    }
    progress.record(ProgressEvent {
        output: Some(session.output.display().to_string()),
        progress: Some(1.0),
        phase: Some("Transcript already exists".to_owned()),
        group: Some(session.label.clone()),
        kind: Some("merge".to_owned()),
        ..ProgressEvent::new(output_key(&session.output), "skipped")
    })
}

/// Serialize Zoom runs targeting the same directory without leaving lock files behind.
/// Recheck for the transcript after the lock, since another run may have just written it.
fn lock_zoom_output(output: &Path) -> Result<File> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let directory = File::open(parent).with_context(|| format!("opening {}", parent.display()))?;
    loop {
        check_cancelled()?;
        match directory.try_lock() {
            Ok(()) => return Ok(directory),
            Err(fs::TryLockError::WouldBlock) => {
                std::thread::sleep(std::time::Duration::from_millis(100))
            }
            Err(fs::TryLockError::Error(error)) => {
                return Err(error)
                    .with_context(|| format!("locking output directory {}", parent.display()));
            }
        }
    }
}

/// Transcribe one job, recording its progress. `None` when it failed or was canceled.
fn run_job(
    cli: &Cli,
    job: &Job,
    models: InferenceModels<'_>,
    progress: &mut ProgressLog,
    member: Option<Member<'_>>,
    failures: &mut usize,
) -> Result<Option<Processed>> {
    let key = input_key(&job.input);
    // Before this run's own events.
    let logged = cli.resume && progress.interrupted_write(&key, &output_path(cli, job));
    let seen = cli.resume && progress.logged_before(&key, &output_path(cli, job));
    let predicted_output = (!member.is_some_and(|member| member.merged_only))
        .then(|| output_path(cli, job).display().to_string());
    let event = |status: &str| ProgressEvent {
        output: predicted_output.clone(),
        group: member.map(|member| member.group.to_owned()),
        speaker: member.map(|member| member.speaker.to_owned()),
        ..ProgressEvent::new(key.clone(), status)
    };
    progress.record(ProgressEvent {
        progress: Some(0.0),
        phase: Some("Starting".to_owned()),
        ..event("running")
    })?;
    let started = Instant::now();
    let mut report = |value: f32, phase: &str| {
        check_cancelled()?;
        progress.record(ProgressEvent {
            elapsed_seconds: Some(started.elapsed().as_secs_f32()),
            progress: Some(value),
            phase: Some(phase.to_owned()),
            ..event("running")
        })
    };
    let result = (|| {
        let _lock = if job.zoom_recording {
            let destination = output_path(cli, job);
            let lock = lock_zoom_output(&destination)?;
            if destination.is_file() && !cli.overwrite {
                return Ok(None);
            }
            Some(lock)
        } else {
            None
        };
        process_one(cli, job, models, member, (logged, seen), &mut report).map(Some)
    })();
    let elapsed_seconds = Some(started.elapsed().as_secs_f32());
    match result {
        Ok(Some(processed)) => {
            progress.record(ProgressEvent {
                output: processed
                    .output
                    .as_ref()
                    .map(|path| path.display().to_string()),
                elapsed_seconds,
                progress: Some(1.0),
                phase: Some("Complete".to_owned()),
                warning: processed.warning.clone(),
                ..event("completed")
            })?;
            Ok(Some(processed))
        }
        Ok(None) => {
            progress.record(ProgressEvent {
                progress: Some(1.0),
                phase: Some("Transcript already exists".to_owned()),
                ..event("skipped")
            })?;
            Ok(None)
        }
        // Not a failure of this input: record it so the app shows the row as canceled
        // rather than failed; the caller stops.
        Err(_) if cancel_requested() => {
            progress.record(ProgressEvent {
                output: None,
                elapsed_seconds,
                phase: Some("Canceled".to_owned()),
                ..event("canceled")
            })?;
            Ok(None)
        }
        Err(error) => {
            eprintln!("error processing {}: {error:#}", job.input.display());
            *failures += 1;
            progress.record(ProgressEvent {
                error: Some(format!("{error:#}")),
                elapsed_seconds,
                phase: Some("Failed".to_owned()),
                ..event("failed")
            })?;
            if cli.fail_fast {
                return Err(error);
            }
            Ok(None)
        }
    }
}

/// Where each track starts on its session's timeline, in seconds (see `merge::track_starts`).
fn track_starts(jobs: &[Job], durations: &[f32]) -> Vec<f32> {
    let names = merge::parse_paths(jobs.iter().map(|job| job.input.as_path()));
    merge::track_starts(&names, durations)
}

/// Transcribe a session's speaker tracks with each other's bleed muted, then merge them.
fn run_session(
    cli: &Cli,
    session: &IsoSession,
    models: &mut InferenceModels<'_>,
    progress: &mut ProgressLog,
    failures: &mut usize,
) -> Result<()> {
    let key = output_key(&session.output);
    let event = |status: &str| ProgressEvent {
        output: Some(session.output.display().to_string()),
        group: Some(session.label.clone()),
        kind: Some("merge".to_owned()),
        ..ProgressEvent::new(key.clone(), status)
    };
    let started = Instant::now();
    let fail =
        |progress: &mut ProgressLog, failures: &mut usize, error: anyhow::Error| -> Result<()> {
            if cancel_requested() {
                return progress.record(event("canceled"));
            }
            eprintln!("error merging {}: {error:#}", session.label);
            *failures += 1;
            progress.record(ProgressEvent {
                error: Some(format!("{error:#}")),
                elapsed_seconds: Some(started.elapsed().as_secs_f32()),
                ..event("failed")
            })?;
            if cli.fail_fast { Err(error) } else { Ok(()) }
        };
    let merged_only = session.zoom.is_some();
    let _lock = if merged_only {
        match lock_zoom_output(&session.output) {
            Ok(lock) => Some(lock),
            Err(error) => return fail(progress, failures, error),
        }
    } else {
        None
    };
    let overwrite = cli.overwrite
        || (!merged_only && resume_replaces_merge(cli, progress, &session.output, session));
    if session.output.is_file() && merged_only && !overwrite {
        return record_skipped_session(cli, progress, session);
    }
    if session.output.exists() && !overwrite {
        return fail(
            progress,
            failures,
            anyhow::anyhow!(
                "output exists (use --overwrite): {}",
                session.output.display()
            ),
        );
    }
    progress.record(ProgressEvent {
        phase: Some("Comparing speaker tracks".to_owned()),
        ..event("running")
    })?;
    // Shown on the first track's row, which is otherwise idle meanwhile.
    let first = &session.jobs[0];
    let mut levels = Vec::with_capacity(session.jobs.len());
    let mut durations = Vec::with_capacity(session.jobs.len());
    for (index, job) in session.jobs.iter().enumerate() {
        let measured = check_cancelled()
            .and_then(|()| {
                progress.record(ProgressEvent {
                    output: (!merged_only).then(|| output_path(cli, first).display().to_string()),
                    group: Some(session.label.clone()),
                    speaker: Some(session.speakers[0].clone()),
                    progress: Some(0.01),
                    phase: Some(format!(
                        "Comparing speaker tracks ({}/{})",
                        index + 1,
                        session.jobs.len()
                    )),
                    ..ProgressEvent::new(input_key(&first.input), "running")
                })
            })
            .and_then(|()| {
                let audio = decode_job(job)?;
                if merged_only && let Some(warning) = &audio.warning {
                    bail!("incomplete Zoom track {}: {warning}", job.input.display());
                }
                Ok(audio)
            });
        match measured {
            Ok(audio) => {
                durations.push(audio.duration_seconds);
                levels.push(bleed::frame_levels(&audio.samples));
            }
            Err(error) => {
                if cancel_requested() {
                    return progress.record(event("canceled"));
                }
                // No track can be judged without all of them; fail the session's files. They
                // are what failed, so the merge is not counted again.
                eprintln!("error merging {}: {error:#}", session.label);
                for job in &session.jobs {
                    *failures += 1;
                    progress.record(ProgressEvent {
                        output: (!merged_only).then(|| output_path(cli, job).display().to_string()),
                        error: Some(format!("{error:#}")),
                        phase: Some("Failed".to_owned()),
                        group: Some(session.label.clone()),
                        ..ProgressEvent::new(input_key(&job.input), "failed")
                    })?;
                }
                progress.record(ProgressEvent {
                    error: Some(format!("{error:#}")),
                    elapsed_seconds: Some(started.elapsed().as_secs_f32()),
                    ..event("failed")
                })?;
                return if cli.fail_fast { Err(error) } else { Ok(()) };
            }
        }
    }
    let masks = bleed::masks(&levels, &track_starts(&session.jobs, &durations));
    let mut docs = Vec::with_capacity(session.jobs.len());
    let previous_failures = *failures;
    for ((job, speaker), mask) in session.jobs.iter().zip(&session.speakers).zip(&masks) {
        if cancel_requested() {
            break;
        }
        eprintln!(
            "{}: muting {:.0} s of other speakers in {}",
            speaker,
            mask.muted_seconds(),
            job.input.display()
        );
        let member = Member {
            group: &session.label,
            speaker,
            gate: mask,
            merged_only,
        };
        match run_job(
            cli,
            job,
            models.reborrow(),
            progress,
            Some(member),
            failures,
        ) {
            Ok(Some(processed)) => {
                docs.push(processed.words.context("speaker tracks keep their words")?);
            }
            Ok(None) => {}
            Err(error) => {
                progress.record(ProgressEvent {
                    error: Some(format!("{error:#}")),
                    elapsed_seconds: Some(started.elapsed().as_secs_f32()),
                    phase: Some("Failed".to_owned()),
                    ..event("failed")
                })?;
                return Err(error);
            }
        }
    }
    if cancel_requested() {
        return progress.record(event("canceled"));
    }
    let expected = session.jobs.len();
    if docs.len() != expected {
        let error = format!("{} of {expected} tracks failed", expected - docs.len());
        if *failures == previous_failures {
            return fail(progress, failures, anyhow::anyhow!(error));
        }
        return progress.record(ProgressEvent {
            error: Some(error),
            elapsed_seconds: Some(started.elapsed().as_secs_f32()),
            phase: Some("Failed".to_owned()),
            ..event("failed")
        });
    }
    let inputs = docs
        .into_iter()
        .zip(&session.jobs)
        .map(|(mut doc, job)| {
            // The merged transcript lists its tracks by absolute path: `merged_elsewhere`
            // tells sessions apart by them, from whatever directory a later run starts in.
            doc.source = input_key(&job.input);
            doc
        })
        .zip(&session.jobs)
        .map(|(doc, job)| merge::TrackInput {
            doc,
            path: job.input.display().to_string(),
            name: None,
            mixed: false,
        })
        .collect();
    let merged = merge::merge_tracks(inputs).and_then(|(doc, mut warnings)| {
        if merged_only && doc.speaker_names.len() == 1 {
            warnings.push(
                "Zoom exported tracks for one participant; only their audio is included".to_owned(),
            );
        }
        progress.record(ProgressEvent {
            phase: Some("Writing output".to_owned()),
            ..event("running")
        })?;
        check_cancelled()?;
        write_job_output(cli, doc, &session.output, overwrite)?;
        Ok(warnings)
    });
    match merged {
        Ok(warnings) => {
            for warning in &warnings {
                eprintln!("warning: {}: {warning}", session.label);
            }
            eprintln!("{} -> {}", session.label, session.output.display());
            progress.record(ProgressEvent {
                elapsed_seconds: Some(started.elapsed().as_secs_f32()),
                progress: Some(1.0),
                phase: Some("Complete".to_owned()),
                warning: (!warnings.is_empty()).then(|| warnings.join("; ")),
                ..event("completed")
            })
        }
        Err(error) => fail(progress, failures, error),
    }
}

fn ensure_apple_silicon() -> Result<()> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        bail!("transcriberrr only supports Apple Silicon macOS (aarch64)");
    }
    Ok(())
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
    let mut skipped = 0;
    let detector = vad::VoiceDetector::new()?;
    // Judge chunks on a CPU thread (~0.2 s per 30 s chunk) while the GPU transcribes.
    // Allow only one queued verdict so the detector cannot run far ahead of inference.
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    std::thread::scope(|scope| -> Result<()> {
        let ranges = &chunks;
        scope.spawn(move || {
            for range in ranges {
                if cancel_requested()
                    || sender
                        .send(detector.has_speech(&samples[range.clone()]))
                        .is_err()
                {
                    break;
                }
            }
        });
        let receiver = receiver;
        for (index, range) in chunks.iter().enumerate() {
            // Qwen3-ASR invents text ("I'm not sure.", "Okay.") for audio with no speech.
            let has_speech = receiver.recv().unwrap_or(true);
            check_cancelled()?;
            if !has_speech {
                skipped += 1;
                report(
                    0.2 + 0.52 * (index + 1) as f32 / total as f32,
                    &format!("Skipping chunk {}/{} without speech", index + 1, total),
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
                // We already cut at the requested length. Handing the same fractional limit
                // to Qwen can split a rounded-up chunk again at a different boundary.
                chunk_max_sec: None,
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
        Ok(())
    })?;
    if skipped == chunks.len() && !samples.is_empty() {
        eprintln!(
            "warning: no speech detected in any of {} chunk(s); the transcript is empty",
            chunks.len()
        );
    } else if skipped > 0 {
        eprintln!(
            "note: skipped {skipped} of {} chunk(s) with no detectable speech",
            chunks.len()
        );
    }
    Ok((text_parts.join(" "), language, aligned))
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

/// Whether a job may replace its transcript: with `--overwrite`; with `--resume`, when the log
/// recorded writing this exact destination but not completion; or as a track of a session
/// rerun by `--skip-existing` or `--resume` because the merge needs the track's words again.
fn may_replace(cli: &Cli, gate: Option<&bleed::Mask>, logged: bool) -> bool {
    cli.overwrite || (cli.resume && logged) || (gate.is_some() && (cli.skip_existing || cli.resume))
}

/// Whether an existing transcript at a speaker track's output could be this track's own,
/// from an unfinished run of its session. A JSON transcript of another recording (a session
/// with the same folder and file names, elsewhere) is not. Other formats do not name their
/// source: one is this track's when the progress log has it (`seen`), and is taken to be
/// under `--skip-existing`, which skips a session whose merge exists.
fn own_track_output(cli: &Cli, destination: &Path, job: &Job, seen: bool) -> bool {
    let Ok(doc) = read_transcript(destination) else {
        // Unreadable as the JSON this run writes: a broken intermediate, nobody's transcript.
        return matches!(cli.format, OutputFormat::Json) || seen || cli.skip_existing;
    };
    if !doc.metadata.merged_from.is_empty() {
        return false;
    }
    let source = Path::new(&doc.source);
    if fs::canonicalize(source).is_ok() {
        return input_key(source) == input_key(&job.input);
    }
    // A relative source from a run started elsewhere does not resolve here.
    let same_name = source
        .file_name()
        .zip(job.input.file_name())
        .is_some_and(|(left, right)| {
            folded_path(&left.to_string_lossy()) == folded_path(&right.to_string_lossy())
        });
    same_name
        && doc.metadata.source_modified_at.is_some()
        && doc.metadata.source_modified_at == source_timestamp(&job.input, FileTimestamp::Modified)
}

/// Transcribe one job. With a `gate` (a speaker track), other speakers' bleed is muted
/// first and the word-level transcript is returned for the session's merge.
fn process_one(
    cli: &Cli,
    job: &Job,
    models: InferenceModels<'_>,
    member: Option<Member<'_>>,
    (logged, seen): (bool, bool),
    report: &mut dyn FnMut(f32, &str) -> Result<()>,
) -> Result<Processed> {
    let gate = member.map(|member| member.gate);
    let output = (!member.is_some_and(|member| member.merged_only)).then(|| output_path(cli, job));
    let overwrite = if job.zoom_recording {
        cli.overwrite
    } else {
        may_replace(cli, gate, logged)
            && (cli.overwrite
                || (cli.resume && logged)
                || output
                    .as_deref()
                    .is_none_or(|destination| own_track_output(cli, destination, job, seen)))
    };
    if let Some(destination) = &output
        && destination.exists()
        && !overwrite
    {
        bail!("output exists (use --overwrite): {}", destination.display());
    }
    if matches!(cli.engine, EngineKind::AppleNative) {
        return process_apple_native(cli, job, gate, output, overwrite, report);
    }
    let started = Instant::now();
    report(0.05, "Decoding audio")?;
    let mut audio = decode_job(job)?;
    if audio.samples.is_empty() {
        bail!(
            "audio file contains no decodable samples: {}",
            job.input.display()
        );
    }
    // A speaker track is transcribed without its long muted stretches (the others' turns),
    // which cost the recognizer as much as speech; its words are put back in place below.
    let mut timeline = None;
    if let Some(gate) = gate {
        gate.apply(&mut audio.samples);
        let (samples, cuts) = gate.cut(&audio.samples);
        audio.samples = samples;
        timeline = Some(cuts);
    }
    report(0.2, "Transcribing")?;
    // Diarized jobs need word timings to attach speakers, and speaker tracks to be merged,
    // whatever the output granularity.
    let want_timestamps =
        cli.timestamps != TimestampGranularity::None || job.diarize || gate.is_some();
    let (text, language, aligned) = if let Some(parakeet) = models.parakeet {
        #[cfg(feature = "parakeet")]
        {
            let result = parakeet.transcribe_with_progress(
                &audio.samples,
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
        .map(|item| {
            let (start, end) = match &timeline {
                Some(timeline) => timeline.original_span(item.start_time, item.end_time),
                None => (item.start_time, item.end_time),
            };
            Word {
                text: item.text,
                start,
                end,
                speaker: None,
            }
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
    // Speakers by the corrected timings: a word the aligner stretched over another speaker's
    // turn would otherwise take that speaker.
    let speakers = SpeakerLookup::new(speaker_segments.as_deref());
    for word in &mut words {
        word.speaker = speakers.at(word.start, word.end);
    }
    preserve_word_text(&text, &mut words)?;
    if let Some(gate) = gate {
        // Words timed wholly inside muted audio were made up from silence or leaked bleed.
        words.retain(|word| !gate.mutes(word.start, word.end));
        let dropped = drop_unplaced_words(&mut words);
        if dropped > 0 {
            eprintln!(
                "note: {}: dropped {dropped} words the aligner could not place in the audio",
                job.input.display()
            );
        }
    }
    let spoke = if gate.is_some() {
        !words.is_empty()
    } else {
        text.chars().any(is_word_char)
    };
    let segments = if words.is_empty() {
        // A speaker track's text without surviving words is not this speaker's.
        if text.trim().is_empty() || gate.is_some() {
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
        speaker_names: Vec::new(),
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
            merged_from: Vec::new(),
        },
    };
    let words = gate.is_some().then(|| doc.clone());
    if let Some(destination) = &output {
        report(0.95, "Writing output")?;
        write_job_output(cli, doc, destination, overwrite)?;
        eprintln!(
            "{} -> {} ({:.2}s)",
            job.input.display(),
            destination.display(),
            started.elapsed().as_secs_f32()
        );
    }
    // The app shows this on the finished row; otherwise an empty transcript looks like success.
    let warning = [
        (!spoke).then(|| "No speech detected; the transcript is empty".to_owned()),
        audio.warning,
    ]
    .into_iter()
    .flatten()
    .reduce(|first, second| format!("{first}; {second}"));
    Ok(Processed {
        output,
        warning,
        words,
    })
}

fn process_apple_native(
    cli: &Cli,
    job: &Job,
    gate: Option<&bleed::Mask>,
    output: Option<PathBuf>,
    overwrite: bool,
    report: &mut dyn FnMut(f32, &str) -> Result<()>,
) -> Result<Processed> {
    if job.diarize || job.speakers.is_some() {
        bail!(
            "Apple SpeechTranscriber does not provide diarization; use --engine qwen3 with --diarize"
        );
    }
    let helper = apple_native_helper().context(
        "Apple SpeechTranscriber helper is not bundled; set TRANSCRIBERRR_NATIVE or build the macOS app",
    )?;
    // Decode WebM and Ogg here for the helper. Normalize QTA too so every engine uses the same
    // playback-compatible track. Speaker tracks and split recordings also need a single WAV.
    let mut decode_warning = None;
    let converted = if gate.is_some()
        || !job.parts.is_empty()
        || job.input.extension().is_some_and(|extension| {
            ["webm", "ogg", "qta"]
                .iter()
                .any(|format| extension.eq_ignore_ascii_case(format))
        }) {
        report(0.05, "Decoding audio")?;
        let mut audio = decode_job(job)?;
        if audio.samples.is_empty() {
            bail!(
                "audio file contains no decodable samples: {}",
                job.input.display()
            );
        }
        if let Some(gate) = gate {
            gate.apply(&mut audio.samples);
        }
        decode_warning = audio.warning.take();
        let wav = audio_convert::TempPath::create("wav")?;
        audio_convert::write_float_wav(wav.path(), &audio.samples, TARGET_SAMPLE_RATE)?;
        Some(wav)
    } else {
        None
    };
    let helper_input = converted
        .as_ref()
        .map_or(job.input.as_path(), audio_convert::TempPath::path);
    report(0.2, "Transcribing with Apple SpeechTranscriber")?;
    let mut command = Command::new(&helper);
    command.arg(helper_input);
    if cli.offline {
        command.arg("--offline");
    }
    let mut child = command
        .args(["--format", "json", "--timestamps", "words"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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
    // Capture stderr too (still echoed) so a failure names the helper's reason, e.g. the
    // macOS version or a missing speech asset, rather than only an exit status.
    let mut stderr = child.stderr.take().context("helper stderr")?;
    let error_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stderr.read_to_end(&mut buffer);
        let text = String::from_utf8_lossy(&buffer).into_owned();
        eprint!("{text}");
        text
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
    let helper_errors = error_reader.join().unwrap_or_default();
    if !status.success() {
        match helper_errors
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
        {
            Some(reason) => bail!("Apple SpeechTranscriber failed: {}", reason.trim()),
            None => bail!("Apple SpeechTranscriber failed with status {status}"),
        }
    }
    let mut doc: TranscriptDocument =
        serde_json::from_slice(&stdout).context("reading Apple SpeechTranscriber transcript")?;
    // Describe the user's file and this build, not a temporary WAV or the helper's own
    // version constant.
    doc.source = job.input.display().to_string();
    doc.metadata.version = env!("CARGO_PKG_VERSION").to_owned();
    doc.metadata.source_created_at = source_timestamp(&job.input, FileTimestamp::Created);
    doc.metadata.source_modified_at = source_timestamp(&job.input, FileTimestamp::Modified);
    // Same labels as the other engines, whatever helper build wrote the transcript.
    doc.language = "English".to_owned();
    doc.metadata.transcribed_at = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
    doc.validate()?;
    if let Some(gate) = gate {
        let words = doc
            .segments
            .drain(..)
            .flat_map(|segment| segment.words)
            .filter(|word| !gate.mutes(word.start, word.end))
            .collect();
        doc.segments = build_segments(words);
    }
    let warning = [
        doc.segments
            .is_empty()
            .then(|| "No speech detected; the transcript is empty".to_owned()),
        decode_warning,
    ]
    .into_iter()
    .flatten()
    .reduce(|first, second| format!("{first}; {second}"));
    let words = gate.is_some().then(|| doc.clone());
    if let Some(destination) = &output {
        report(0.95, "Writing output")?;
        write_job_output(cli, doc, destination, overwrite)?;
    } else {
        check_cancelled()?;
    }
    Ok(Processed {
        output,
        warning,
        words,
    })
}

fn write_job_output(
    cli: &Cli,
    mut doc: TranscriptDocument,
    destination: &Path,
    overwrite: bool,
) -> Result<()> {
    if cli.timestamps != TimestampGranularity::Words {
        doc.collapse_words();
    }
    if let Some(parent) = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    write_output(&doc, destination, render_options(cli), overwrite)
}

fn apple_native_helper() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("TRANSCRIBERRR_NATIVE")
        .and_then(|path| std::path::absolute(path).ok())
        .filter(|path| path.is_file())
    {
        return Some(path);
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

fn is_video(path: &Path) -> bool {
    path.extension().is_some_and(|extension| {
        VIDEO_EXTENSIONS
            .iter()
            .any(|video| extension.eq_ignore_ascii_case(video))
    })
}

fn zoom_id(name: &merge::TrackName) -> Option<&str> {
    match name {
        merge::TrackName::Zoom { meeting, .. } => Some(meeting),
        merge::TrackName::ZoomMixed { meeting, .. } => meeting.as_deref(),
        _ => None,
    }
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum ZoomMediaKind {
    Audio,
    Video,
    AlternateVideo,
}

struct ZoomMedia {
    meeting: Option<String>,
    segment: u32,
    kind: ZoomMediaKind,
    /// Legacy `audio_only01`-style exports remain separate recordings.
    legacy_copy: bool,
}

fn zoom_media(path: &Path, name: &merge::TrackName) -> Option<ZoomMedia> {
    let media = zoom_media_by_name(path, name)?;
    // `video` and ten digits is also a Unix-time name (`video1700000000.mp4`, from screen
    // recorders and phones); only Zoom's folder around it makes it a meeting recording.
    if media.kind == ZoomMediaKind::Video
        && !matches!(name, merge::TrackName::ZoomMixed { .. })
        && let Some(meeting) = &media.meeting
        && !zoom_folder_evidence(path, meeting)
    {
        return None;
    }
    Some(media)
}

/// Whether the folder holding `path` looks like a Zoom recording of `meeting`: participant
/// tracks, a playlist, or another file of the same meeting (audio, a later segment, a view).
fn zoom_folder_evidence(path: &Path, meeting: &str) -> bool {
    let folder = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if folder.join("Audio Record").is_dir() || folder.join("playback.m3u").is_file() {
        return true;
    }
    let Ok(entries) = fs::read_dir(folder) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let other = entry.path();
        (name.eq_ignore_ascii_case("Audio Record") && other.is_dir())
            || (name.eq_ignore_ascii_case("playback.m3u") && other.is_file())
            || (Some(name.as_os_str()) != path.file_name()
                && is_audio_path(&other)
                && other.is_file()
                && zoom_media_by_name(&other, &merge::parse_path(&other))
                    .is_some_and(|media| media.meeting.as_deref() == Some(meeting)))
    })
}

fn zoom_media_by_name(path: &Path, name: &merge::TrackName) -> Option<ZoomMedia> {
    let video = is_video(path);
    let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();
    if let merge::TrackName::ZoomMixed { segment, meeting } = name {
        return Some(ZoomMedia {
            meeting: meeting.clone(),
            segment: *segment,
            kind: if video {
                ZoomMediaKind::Video
            } else {
                ZoomMediaKind::Audio
            },
            legacy_copy: !video && meeting.is_none() && stem != "audio_only",
        });
    }
    if !video {
        return None;
    }
    let prefix = stem.trim_end_matches(|c: char| c.is_ascii_digit());
    let tail = &stem[prefix.len()..];
    // Zoom numbers these `zoom_0`, `zoom_1`, …; a date such as `zoom_20240115` is someone's
    // own name, and taking it for a segment joined unrelated videos into one transcript.
    if prefix == "zoom_" && (1..=2).contains(&tail.len()) {
        return Some(ZoomMedia {
            meeting: None,
            segment: tail.parse().ok()?,
            kind: ZoomMediaKind::Video,
            legacy_copy: false,
        });
    }
    let kind = if matches!(prefix, "video" | "video_") {
        ZoomMediaKind::Video
    } else if ["shared_screen", "speaker_view", "gallery_view"]
        .iter()
        .any(|view| prefix.starts_with(view))
    {
        ZoomMediaKind::AlternateVideo
    } else {
        return None;
    };
    let segment = match tail.len() {
        9 => 0,
        10 => u32::from(tail.as_bytes()[0] - b'0'),
        _ => return None,
    };
    Some(ZoomMedia {
        meeting: Some(tail[tail.len() - 9..].to_owned()),
        segment,
        kind,
        legacy_copy: false,
    })
}

/// The Zoom meeting ID, with an outer option to distinguish a legacy export from other media.
fn zoom_candidate(path: &Path) -> Option<Option<String>> {
    let name = merge::parse_path(path);
    if !is_video(path)
        && let merge::TrackName::Zoom { meeting, .. } = name
    {
        return Some(Some(meeting));
    }
    zoom_media(path, &name).map(|media| media.meeting)
}

fn zoom_recording_id(path: &Path) -> Option<String> {
    zoom_media(path, &merge::parse_path(path)).and_then(|media| media.meeting)
}

fn input_files(dir: &Path, recursive: bool) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let walker = WalkDir::new(dir).max_depth(if recursive { usize::MAX } else { 1 });
    for entry in walker.into_iter().filter_entry(|entry| {
        let name = entry.file_name().to_string_lossy();
        entry.depth() == 0
            || !(name.starts_with('.')
                || (entry.file_type().is_dir()
                    && (name.starts_with('#')
                        || [".app", ".photoslibrary", ".imovielibrary", ".logicx"]
                            .iter()
                            .any(|suffix| name.to_ascii_lowercase().ends_with(suffix)))))
    }) {
        match entry {
            Ok(entry)
                if is_audio_path(entry.path())
                    && (entry.file_type().is_file()
                        || (entry.file_type().is_symlink() && entry.path().is_file())) =>
            {
                paths.push(entry.into_path())
            }
            Ok(entry)
                if entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("zoom")) =>
            {
                eprintln!(
                    "warning: skipping {}: Zoom has not converted this recording yet",
                    entry.path().display()
                );
            }
            Ok(_) => {}
            Err(error) => eprintln!("warning: skipping {error}"),
        }
    }
    paths.sort();
    paths
}

struct ZoomInventory {
    exports: Vec<(PathBuf, Option<String>)>,
    audio_copies: HashMap<String, PathBuf>,
    multiple_ids: bool,
    converting: bool,
}

fn zoom_inventory(folder: &Path) -> Result<ZoomInventory> {
    let read = |path: &Path| {
        fs::read_dir(path)
            .and_then(|entries| {
                entries
                    .map(|entry| entry.map(|entry| entry.path()))
                    .collect::<std::io::Result<Vec<_>>>()
            })
            .with_context(|| format!("reading {}", path.display()))
    };
    let mut files = read(folder)?;
    let mut participant_files = Vec::new();
    for directory in &files {
        if directory
            .file_name()
            .is_some_and(|name| name.eq_ignore_ascii_case("Audio Record"))
            && directory.is_dir()
        {
            participant_files.extend(read(directory)?);
        }
    }
    files.extend(participant_files);
    files.sort_unstable();

    let mut converting = false;
    let mut exports = Vec::new();
    let mut mixed_audio = Vec::new();
    let mut unknown_videos = Vec::new();
    for path in files {
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("zoom"))
            && path
                .file_name()
                .is_some_and(|name| !name.to_string_lossy().starts_with('.'))
            && path.is_file()
        {
            converting = true;
        }
        if !is_audio_path(&path) || !path.is_file() {
            continue;
        }
        if let Some(id) = zoom_candidate(&path) {
            if !is_video(&path)
                && matches!(merge::parse_path(&path), merge::TrackName::ZoomMixed { .. })
            {
                mixed_audio.push((path.clone(), id.clone()));
            }
            exports.push((path, id));
        } else if is_video(&path) {
            unknown_videos.push(path);
        }
    }

    let mut audio_copies = HashMap::new();
    for video in unknown_videos {
        let mut candidates = mixed_audio.iter().filter(|(audio, _)| {
            without_panic(|| audio_convert::same_aac_audio(audio, &video)).unwrap_or(false)
        });
        if let Some((audio, meeting)) = candidates.next()
            && candidates.next().is_none()
        {
            audio_copies.insert(input_key(&video), audio.clone());
            exports.push((video, meeting.clone()));
        }
    }
    let multiple_ids = exports
        .iter()
        .filter_map(|(_, id)| id.as_ref())
        .collect::<HashSet<_>>()
        .len()
        > 1;
    Ok(ZoomInventory {
        exports,
        audio_copies,
        multiple_ids,
        converting,
    })
}

/// A file selection expands only to recognized exports from the same Zoom recording.
/// Keep explicit jobs first so their settings win over those inherited by discovered files.
fn add_zoom_companions(
    jobs: &mut Vec<Job>,
    known: &mut HashSet<String>,
    opt_out: &HashSet<String>,
    discover: bool,
    rejected: &mut Vec<(PathBuf, String)>,
) {
    let mut inventories = HashMap::new();
    let mut expanded = HashSet::new();
    let mut companions = Vec::new();
    let mut unavailable = HashMap::new();
    for job in jobs.iter_mut() {
        let named_id = zoom_candidate(&job.input);
        if named_id.is_none() && !is_video(&job.input) {
            continue;
        }
        if opt_out.contains(&input_key(&job.input))
            || (job.iso
                && !is_iso_track(IsoMode::Auto, &job.input)
                && !matches!(
                    merge::parse_path(&job.input),
                    merge::TrackName::ZoomMixed { .. }
                ))
        {
            continue;
        }
        let folder = session_folder(&job.input);
        let inventory = match inventories
            .entry(folder.clone())
            .or_insert_with(|| zoom_inventory(&folder))
        {
            Ok(inventory) => inventory,
            Err(error) => {
                if named_id.is_some() {
                    unavailable.insert(job.input.clone(), format!("{error:#}"));
                }
                continue;
            }
        };
        let audio_copy = inventory.audio_copies.get(&input_key(&job.input));
        let Some(id) = named_id.or_else(|| audio_copy.and_then(|path| zoom_candidate(path))) else {
            continue;
        };
        if inventory.converting {
            unavailable.insert(
                job.input.clone(),
                format!(
                    "Zoom is still converting the recording in {}",
                    folder.display()
                ),
            );
            continue;
        }
        if let Some(audio) = audio_copy {
            job.zoom_audio_copy = Some(audio.clone());
            job.zoom_recording = true;
        }
        if id.is_none() && inventory.multiple_ids {
            unavailable.insert(
                job.input.clone(),
                "the Zoom export has no meeting ID and this folder contains several meetings"
                    .to_owned(),
            );
            continue;
        }
        if !discover || !expanded.insert((folder, id.clone())) {
            continue;
        }
        for (path, candidate) in &inventory.exports {
            let matches = match (&id, candidate) {
                (Some(id), Some(candidate)) => id == candidate,
                _ => !inventory.multiple_ids,
            };
            if !matches || !known.insert(input_key(path)) {
                continue;
            }
            let iso = is_iso_track(IsoMode::Auto, path);
            companions.push(Job {
                input: path.clone(),
                iso,
                zoom_recording: !iso,
                zoom_audio_copy: inventory.audio_copies.get(&input_key(path)).cloned(),
                output: None,
                planned_output: None,
                ..job.clone()
            });
        }
    }
    jobs.retain(|job| {
        if let Some(reason) = unavailable.get(&job.input) {
            rejected.push((job.input.clone(), reason.clone()));
            false
        } else {
            true
        }
    });
    jobs.extend(companions);
}

/// Missing inputs are rejected one at a time, so a stale queue row doesn't stop the library.
fn collect_jobs(
    cli: &Cli,
    rejected: &mut Vec<(PathBuf, String)>,
    missing: &mut Vec<PathBuf>,
) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    let mut known = HashSet::new();
    let mut opt_out = HashSet::new();
    let mut discover = true;
    let mut excluded = Vec::new();
    let mut add = |input: PathBuf, item: &ManifestItem, valid: bool| {
        let key = input_key(&input);
        if !known.insert(key.clone()) {
            return;
        }
        let auto_iso = is_audio_path(&input) && is_iso_track(IsoMode::Auto, &input);
        let iso = item
            .iso_track
            .unwrap_or_else(|| is_iso_track(cli.iso_tracks, &input));
        if auto_iso && !iso {
            opt_out.insert(key);
        }
        let zoom_recording = matches!(cli.iso_tracks, IsoMode::Auto)
            && !iso
            && !auto_iso
            && is_audio_path(&input)
            && zoom_candidate(&input).is_some();
        if !valid && let Err(error) = validate_input(&input) {
            if (auto_iso && iso) || zoom_recording {
                missing.push(input.clone());
            }
            rejected.push((input, format!("{error:#}")));
            return;
        }
        // An explicit diarize:false also overrides the global speaker count.
        let speakers = if item.diarize == Some(false) {
            None
        } else {
            item.speakers.or(cli.speakers)
        };
        if let Some(count) = speakers
            && let Err(error) = speaker_count(&count.to_string())
        {
            if (auto_iso && iso) || zoom_recording {
                missing.push(input.clone());
            }
            rejected.push((input, format!("invalid speakers {count}: {error}")));
            return;
        }
        jobs.push(Job {
            iso,
            zoom_recording,
            zoom_audio_copy: None,
            parts: Vec::new(),
            input,
            diarize: item.diarize.unwrap_or(cli.diarize) || speakers.is_some(),
            speakers,
            output: item.output.clone(),
            planned_output: None,
            output_dir: item
                .output_dir
                .clone()
                .unwrap_or_else(|| cli.output_dir.clone()),
            output_location: item.output_location.unwrap_or(cli.output_location),
        });
    };
    if let Some(manifest) = &cli.manifest {
        let manifest_path = expand_home(manifest);
        let manifest: JobManifest = if manifest_path == Path::new("-") {
            serde_json::from_reader(std::io::stdin().lock())
                .context("parsing job manifest from stdin")?
        } else {
            let file = File::open(&manifest_path)
                .with_context(|| format!("opening job manifest {}", manifest_path.display()))?;
            serde_json::from_reader(BufReader::new(file))
                .with_context(|| format!("parsing job manifest {}", manifest_path.display()))?
        };
        let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        discover = manifest.discover_zoom.unwrap_or(true);
        excluded = manifest
            .excluded_inputs
            .iter()
            .map(|path| input_key(&resolve_manifest_path(base, path)))
            .collect();
        for mut item in manifest.items {
            let input = resolve_manifest_path(base, &item.input);
            item.output = item.output.map(|path| resolve_manifest_path(base, &path));
            item.output_dir = item
                .output_dir
                .map(|path| resolve_manifest_path(base, &path));
            if input.is_dir() && item.output.is_none() {
                for path in input_files(&input, true) {
                    add(path, &item, true);
                }
            } else {
                add(input, &item, false);
            }
        }
    } else {
        let settings = ManifestItem::default();
        if let Some(dir) = &cli.input_dir {
            let dir = expand_home(dir);
            // Keep ordinary --input-dir non-recursive. Zoom libraries can contain
            // meetings at any depth, including the nested Audio Record directory.
            let auto = matches!(cli.iso_tracks, IsoMode::Auto);
            fs::read_dir(&dir)
                .with_context(|| format!("reading input directory {}", dir.display()))?;
            for path in input_files(&dir, cli.recursive || auto) {
                if cli.recursive
                    || path.parent() == Some(dir.as_path())
                    || (auto && zoom_candidate(&path).is_some())
                {
                    add(path, &settings, true);
                }
            }
        } else {
            for path in cli.inputs.iter().map(|path| expand_home(path)) {
                if path.is_dir() {
                    for file in input_files(&path, true) {
                        add(file, &settings, true);
                    }
                } else {
                    add(path, &settings, false);
                }
            }
        }
    }
    if matches!(cli.iso_tracks, IsoMode::Auto) {
        known.extend(excluded);
        add_zoom_companions(&mut jobs, &mut known, &opt_out, discover, rejected);
    }
    Ok(jobs)
}

fn is_iso_track(mode: IsoMode, path: &Path) -> bool {
    match mode {
        IsoMode::Auto => {
            !is_video(path) && matches!(merge::parse_path(path), merge::TrackName::Zoom { .. })
        }
        IsoMode::All => true,
        IsoMode::Off => false,
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

/// Decode a job's audio: its input, then any later parts end to end.
fn decode_job(job: &Job) -> Result<Audio> {
    let mut audio =
        decode_audio(&job.input).with_context(|| format!("decoding {}", job.input.display()))?;
    for part in &job.parts {
        let more = decode_audio(part).with_context(|| format!("decoding {}", part.display()))?;
        audio.samples.extend(more.samples);
        audio.duration_seconds += more.duration_seconds;
        audio.warning = audio.warning.or(more.warning);
    }
    Ok(audio)
}

fn decode_audio(path: &Path) -> Result<Audio> {
    match without_panic(|| decode_with_symphonia(path)) {
        // Symphonia has no Opus decoder, and it rejects files Apple's own encoders write
        // (afconvert ADTS AAC: "end of stream"; μ-law/A-law WAV: "malformed fmt chunk"). Core
        // Audio reads all of them (see audio_convert), so try it on any failure.
        Err(error) if path.is_file() => {
            let caf = without_panic(|| audio_convert::convert_with_core_audio(path))
                .with_context(|| format!("{error:#}; the Core Audio fallback also failed"))?;
            without_panic(|| decode_with_symphonia(caf.path()))
                .with_context(|| format!("{error:#}; decoding the Core Audio conversion failed"))
        }
        result => result,
    }
}

/// Symphonia panics on some damaged files (a truncated WebM whose index points past its
/// end); that must fail the attempt, not end the run with every later file unprocessed.
fn without_panic<T>(decode: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(decode)).unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<&str>()
            .map(|message| (*message).to_owned())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        bail!("the audio decoder crashed on this file (damaged or truncated?): {message}")
    })
}

/// The first audio track, with a decoder for it. Not `default_track()`, which is just the
/// first track: browser screen recordings put the video track first, and it has no sample rate.
fn audio_track(
    format: &dyn symphonia::core::formats::FormatReader,
    prefer_aac: bool,
) -> Result<(u32, Box<dyn symphonia::core::codecs::Decoder>)> {
    let mut audio = format.tracks().iter().filter(|track| {
        track.codec_params.codec != CODEC_TYPE_NULL && track.codec_params.sample_rate.is_some()
    });
    // A Voice Memo can carry alternate encodings of the same audio. AAC is the
    // compatibility track in Apple's spatial recordings.
    let compatible = prefer_aac
        .then(|| {
            audio
                .clone()
                .find(|track| track.codec_params.codec == CODEC_TYPE_AAC)
        })
        .flatten();
    let track = compatible
        .or_else(|| audio.next())
        .context("file has no audio track")?;
    let decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    Ok((track.id, decoder))
}

/// Decoded audio at one sample rate, resampled into the output when the rate changes.
#[derive(Default)]
struct Section {
    samples: Vec<f32>,
    rate: u32,
}

impl Section {
    fn flush(&mut self, output: &mut Vec<f32>) {
        let mut samples = std::mem::take(&mut self.samples);
        // Float WAV/CAF can hold NaN or infinity; one would spread through the resampler and
        // the bleed levels' filter state and silence everything after it.
        for sample in samples.iter_mut().filter(|sample| !sample.is_finite()) {
            *sample = 0.0;
        }
        let samples = resample(samples, self.rate, TARGET_SAMPLE_RATE);
        if output.is_empty() {
            *output = samples;
        } else {
            output.extend(samples);
        }
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
    let prefer_aac = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("qta"));
    let (mut track_id, mut decoder) = audio_track(format.as_ref(), prefer_aac)?;
    let mut output = Vec::new();
    // Resample from the decoder's output rate. Apple's ALAC encoder writes 44100 in the MP4
    // sample entry for every file; the true rate is in the codec config, which the decoder
    // uses. Trusting the container stretched 48 kHz Voice Memos by 9%.
    let mut section = Section::default();
    let mut scratch = None;
    let mut decoded_packets = 0usize;
    let mut skipped_packets = 0usize;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) if packet.track_id() == track_id => packet,
            Ok(_) => continue,
            // A chained Ogg file (streams joined end to end) starts a new stream, with new
            // track IDs and possibly another sample rate.
            Err(SymphoniaError::ResetRequired) => {
                (track_id, decoder) = audio_track(format.as_ref(), prefer_aac)?;
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
                let rate = buffer.spec().rate;
                if rate == 0 {
                    bail!("audio stream has an invalid zero sample rate");
                }
                if rate != section.rate {
                    if !section.samples.is_empty() {
                        section.flush(&mut output);
                    }
                    section.rate = rate;
                }
                append_decoded(buffer, &mut scratch, &mut section.samples);
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
    section.flush(&mut output);
    if output.is_empty() {
        bail!("audio stream has no decodable samples");
    }
    // Symphonia's MP3 decoder also rejects every frame after a change of sample rate or
    // channels (two files joined with `cat`), so many skipped packets mean lost speech.
    let warning = (skipped_packets > 0).then(|| {
        let message = format!(
            "{skipped_packets} of {} audio packets could not be decoded",
            decoded_packets + skipped_packets
        );
        eprintln!("warning: {}: {message}", path.display());
        if skipped_packets * 100 > decoded_packets + skipped_packets {
            format!("{message}; parts of the recording are missing from the transcript")
        } else {
            message
        }
    });
    Ok(Audio {
        duration_seconds: output.len() as f32 / TARGET_SAMPLE_RATE as f32,
        samples: output,
        warning,
    })
}

/// Downmix each decoded buffer using its own channel count. Container metadata can be stale
/// (some AAC/M4A files), and the layout may change mid-stream, so a count taken from one buffer
/// must not be applied to another. Mixing per buffer also avoids holding the full interleaved
/// stream, which keeps peak RSS lower for long multichannel recordings.
fn append_decoded(
    buffer: AudioBufferRef<'_>,
    scratch: &mut Option<SampleBuffer<f32>>,
    output: &mut Vec<f32>,
) {
    let frames = buffer.frames();
    let channels = buffer.spec().channels.count();
    // Vorbis emits an empty first buffer. Symphonia's interleaver panics on zero frames.
    if frames == 0 || channels == 0 {
        return;
    }
    // AAC, MP3, Vorbis and float PCM already decode into planar floats.
    if let AudioBufferRef::F32(ref planar) = buffer {
        match channels {
            1 => {
                output.extend_from_slice(planar.chan(0));
                return;
            }
            2 => {
                output.extend(
                    planar
                        .chan(0)
                        .iter()
                        .zip(planar.chan(1))
                        .map(|(left, right)| (left + right) * 0.5),
                );
                return;
            }
            _ => {}
        }
    }
    // Conversion depends on the current buffer's shape. A scratch allocation can be reused
    // whenever it has enough space for all of its samples.
    let sample_buffer = match scratch {
        Some(samples) if samples.capacity() >= frames * channels => samples,
        _ => scratch.insert(SampleBuffer::new(buffer.capacity() as u64, *buffer.spec())),
    };
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
            let taps = input.get(first..).unwrap_or_default();
            // Inside the clip every tap has a sample, and each kernel already sums to 1.
            if skip == 0 && taps.len() >= kernel.len() {
                return dot(&taps[..kernel.len()], kernel);
            }
            let (sum, weight) = taps
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

/// A dot product summed in eight lanes, which the compiler vectorizes; one running sum must
/// add in order and cannot be (the resampler ran 3-4x slower that way).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (a_chunks, a_rest) = a.as_chunks::<8>();
    let (b_chunks, b_rest) = b.as_chunks::<8>();
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..8 {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    let sum = lanes.iter().sum::<f32>();
    a_rest
        .iter()
        .zip(b_rest)
        .fold(sum, |sum, (x, y)| sum + x * y)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Find the speaker containing a word's midpoint, or the nearest span in a gap.
struct SpeakerLookup<'a> {
    spans: &'a [SpeakerSpan],
    chronological: bool,
}

impl<'a> SpeakerLookup<'a> {
    fn new(spans: Option<&'a [SpeakerSpan]>) -> Self {
        let spans = spans.unwrap_or_default();
        let chronological = spans.iter().all(|span| span.start <= span.end)
            && spans.windows(2).all(|pair| pair[0].end <= pair[1].start);
        Self {
            spans,
            chronological,
        }
    }

    fn at(&self, start: f32, end: f32) -> Option<usize> {
        let midpoint = (start + end) * 0.5;
        let seconds = |sample| sample as f32 / TARGET_SAMPLE_RATE as f32;
        let distance = |span: &SpeakerSpan| {
            (seconds(span.start) - midpoint)
                .max(midpoint - seconds(span.end))
                .max(0.0)
        };
        if self.chronological && midpoint.is_finite() {
            let next = self
                .spans
                .partition_point(|span| seconds(span.end) < midpoint);
            let (before, after) = self.spans.split_at(next);
            if let Some(previous) = before.last() {
                let previous_distance = distance(previous);
                if after
                    .first()
                    .is_none_or(|span| previous_distance <= distance(span))
                {
                    // Distances can round to the same value. Keep the earliest equally near span.
                    let nearest = before.partition_point(|span| distance(span) > previous_distance);
                    return Some(before[nearest].speaker_id);
                }
            }
            return after.first().map(|span| span.speaker_id);
        }
        self.spans
            .iter()
            .map(|span| (distance(span), span.speaker_id))
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, speaker)| speaker)
    }
}

/// Keep the `max_speakers` speakers with the most speaking time. Filtering by ID instead would
/// keep whichever speakers Sortformer happened to number first and drop a main speaker.
#[cfg_attr(not(feature = "diarization"), allow(dead_code))]
fn limit_speakers(mut spans: Vec<SpeakerSpan>, max_speakers: usize) -> Vec<SpeakerSpan> {
    let mut talk_time: HashMap<usize, u64> = HashMap::new();
    for span in &spans {
        *talk_time.entry(span.speaker_id).or_default() += span.end.saturating_sub(span.start);
    }
    if talk_time.len() > max_speakers {
        let mut ranked = talk_time.into_iter().collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
        let kept = ranked
            .into_iter()
            .take(max_speakers)
            .map(|(speaker, _)| speaker)
            .collect::<Vec<_>>();
        spans.retain(|span| kept.contains(&span.speaker_id));
    }
    // Number the speakers kept 0, 1, … by first appearance: transcripts label raw IDs, and
    // "Speaker 0" and "Speaker 3" for two speakers reads as if some were lost.
    let mut order = spans.clone();
    order.sort_by_key(|span| span.start);
    let mut ids: Vec<usize> = Vec::new();
    for span in &order {
        if !ids.contains(&span.speaker_id) {
            ids.push(span.speaker_id);
        }
    }
    for span in &mut spans {
        span.speaker_id = ids
            .iter()
            .position(|&id| id == span.speaker_id)
            .unwrap_or(span.speaker_id);
    }
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
            // A whole recording through the slice API uses the same mel extraction and
            // post-processing as diarize(), with no second copy of the mono recording.
            model.reset_state();
            model.diarize_chunk(samples).map(|segments| {
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

/// Use the canonical source folder plus the meeting ID so equal folder names in
/// different libraries get stable destinations even when processed in separate batches.
fn zoom_destination(job: &Job, meeting: Option<&str>) -> PathBuf {
    let source = session_folder(&job.input);
    let identity = source
        .as_os_str()
        .as_encoded_bytes()
        .iter()
        .copied()
        .chain([0])
        .chain(meeting.unwrap_or_default().bytes());
    let hash = stable_hash(identity);
    let name = source.file_name().unwrap_or_default().to_string_lossy();
    let name = name.trim();
    let name = &name[..name.floor_char_boundary(200)];
    job.output_dir.join(format!("{name} [{hash:016x}]"))
}

fn stable_hash(bytes: impl Iterator<Item = u8>) -> u64 {
    // FNV-1a is stable across runs and Rust versions.
    bytes.fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn append_tag(path: &Path, tag: &str) -> PathBuf {
    let mut name = path.file_stem().unwrap_or_default().to_os_string();
    name.push(".");
    name.push(tag);
    if let Some(extension) = path.extension() {
        name.push(".");
        name.push(extension);
    }
    path.with_file_name(name)
}

fn tagged_output(cli: &Cli, path: PathBuf) -> PathBuf {
    match cli.output_tag.as_deref() {
        Some(tag) => append_tag(&path, tag),
        None => path,
    }
}

fn fallback_output_path(cli: &Cli, job: &Job, folder: &Path, meeting: Option<&str>) -> PathBuf {
    let name = folder.file_name().unwrap_or_default().to_string_lossy();
    let name = &name[..name.floor_char_boundary(200)];
    let extension = format_extension(cli.format);
    let path = match job.output_location {
        OutputLocation::Folder => {
            zoom_destination(job, meeting).join(format!("{name}.zoom.{extension}"))
        }
        OutputLocation::NextToSource => {
            let name = meeting.map_or_else(|| name.to_owned(), |id| format!("{name} [{id}]"));
            folder.join(format!("{name}.zoom.{extension}"))
        }
    };
    tagged_output(cli, path)
}

fn output_path(cli: &Cli, job: &Job) -> PathBuf {
    if let Some(path) = &job.planned_output {
        return path.clone();
    }
    tagged_output(cli, base_output_path(cli, job))
}

fn base_output_path(cli: &Cli, job: &Job) -> PathBuf {
    let extension = format_extension(cli.format);
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
        OutputLocation::Folder if job.zoom_recording => {
            let id = zoom_recording_id(job.zoom_audio_copy.as_deref().unwrap_or(&job.input));
            return zoom_destination(job, id.as_deref())
                .join(job.input.file_name().unwrap_or_default())
                .with_extension(extension);
        }
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

/// Inputs that would write the same transcript as another (for example `talk.m4a` and
/// `talk.wav`), each with the reason. Without this check, the second job either fails late or,
/// with `--overwrite`, silently replaces the first job's output. Both are rejected, and the
/// rest of the queue runs.
fn output_clashes<'a>(
    outputs: impl IntoIterator<Item = (&'a Path, PathBuf)>,
) -> Vec<(PathBuf, String)> {
    let mut seen = HashMap::new();
    let mut clashes = HashMap::new();
    for (input, output) in outputs {
        // APFS is case-insensitive by default, so compare case-folded paths; canonical ones,
        // so `a/talk.json` and `./a/talk.json` meet.
        let key = folded_path(&output_key(&output));
        if let Some(previous) = seen.insert(key, input) {
            let error = || {
                format!(
                    "transcripts from {} and {} would both write {}",
                    previous.display(),
                    input.display(),
                    output.display(),
                )
            };
            clashes.entry(previous).or_insert_with(error);
            clashes.entry(input).or_insert_with(error);
        }
    }
    let mut clashes = clashes
        .into_iter()
        .map(|(input, error)| (input.to_path_buf(), error))
        .collect::<Vec<_>>();
    clashes.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    clashes
}

fn input_key(path: &Path) -> String {
    fs::canonicalize(path).map_or_else(|_| output_key(path), |path| path.display().to_string())
}

fn folded_path(path: &str) -> String {
    path.chars().flat_map(char::to_lowercase).nfc().collect()
}

/// A progress log is appended to before transcription starts. It must be a different
/// file from every source and destination, including through symlinks or hard links.
fn protect_plan_files(
    cli: &Cli,
    units: &[Unit],
    skips: &[Skip],
    rejected: &[(PathBuf, String)],
    progress_paths: &[&Path],
) -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    let manifest = cli
        .manifest
        .as_deref()
        .filter(|path| *path != Path::new("-"))
        .map(expand_home);
    if progress_paths.is_empty() && manifest.is_none() {
        return Ok(());
    }
    let inode = |path: &Path| fs::metadata(path).ok().map(|file| (file.dev(), file.ino()));
    let mut progress_keys = HashSet::new();
    let mut progress_inodes = HashSet::new();
    for &progress in progress_paths {
        if fs::symlink_metadata(progress).is_ok_and(|info| info.is_symlink()) {
            fs::canonicalize(progress)
                .with_context(|| format!("resolving progress file {}", progress.display()))?;
        }
        if !progress_keys.insert(folded_path(&input_key(progress)))
            || inode(progress).is_some_and(|identity| !progress_inodes.insert(identity))
        {
            bail!("model passes share a progress file: {}", progress.display());
        }
    }
    let check = |candidate: &Path| -> Result<()> {
        if progress_paths.is_empty() {
            return Ok(());
        }
        if progress_keys.contains(&folded_path(&input_key(candidate)))
            || (!progress_inodes.is_empty()
                && inode(candidate).is_some_and(|identity| progress_inodes.contains(&identity)))
        {
            bail!(
                "the progress file overlaps an input or transcript: {}",
                candidate.display()
            );
        }
        Ok(())
    };
    let manifest_identity = manifest
        .as_deref()
        .map(|path| (folded_path(&input_key(path)), inode(path)));
    let check_output = |path: &Path| -> Result<()> {
        if let Some((key, identity)) = &manifest_identity
            && (*key == folded_path(&input_key(path))
                || identity.is_some() && *identity == inode(path))
        {
            bail!(
                "transcript destination overlaps the input manifest: {}",
                path.display()
            );
        }
        check(path)
    };
    for unit in units {
        for job in unit.jobs() {
            check(&job.input)?;
            for part in &job.parts {
                check(part)?;
            }
            if unit.writes_job_outputs() {
                check_output(&output_path(cli, job))?;
            }
        }
        if let Unit::Session(session) = unit {
            check_output(&session.output)?;
        }
    }
    for skip in skips {
        check(&skip.job.input)?;
    }
    for (input, _) in rejected {
        check(input)?;
    }
    if let Some(manifest) = manifest {
        check(&manifest)?;
    }
    Ok(())
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
            merged_from: Vec::new(),
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
    fn downmixes_each_packet_with_its_own_layout() {
        use symphonia::core::audio::{AsAudioBufferRef, AudioBuffer, Channels, SignalSpec};

        let left = Channels::FRONT_LEFT;
        let stereo = left | Channels::FRONT_RIGHT;
        let mut mono = AudioBuffer::<i16>::new(2, SignalSpec::new(16_000, left));
        mono.render_silence(None);
        mono.chan_mut(0).copy_from_slice(&[16_384, -16_384]);

        let mut scratch = None;
        let mut samples = Vec::new();
        append_decoded(mono.as_audio_buffer_ref(), &mut scratch, &mut samples);

        let mut integer = AudioBuffer::<i16>::new(1, SignalSpec::new(16_000, stereo));
        integer.render_silence(None);
        integer.chan_mut(0)[0] = 16_384;
        append_decoded(integer.as_audio_buffer_ref(), &mut scratch, &mut samples);
        assert_eq!(scratch.as_ref().unwrap().capacity(), 2);

        let mut float = AudioBuffer::<f32>::new(2, SignalSpec::new(16_000, stereo));
        float.render_silence(None);
        float.chan_mut(0).copy_from_slice(&[1.0, -1.0]);
        float.chan_mut(1).copy_from_slice(&[0.0, 1.0]);
        append_decoded(float.as_audio_buffer_ref(), &mut scratch, &mut samples);

        assert_eq!(samples, [0.5, -0.5, 0.25, 0.5, 0.0]);
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
        let spans = [span(0.0, 2.0, 0), span(3.0, 5.0, 1)];
        let lookup = SpeakerLookup::new(Some(&spans));
        assert_eq!(lookup.at(1.0, 1.2), Some(0));
        assert_eq!(lookup.at(2.1, 2.3), Some(0));
        assert_eq!(lookup.at(2.5, 2.5), Some(0));
        assert_eq!(lookup.at(2.7, 2.9), Some(1));
        assert_eq!(lookup.at(9.0, 9.5), Some(1));
        assert_eq!(lookup.at(f32::MAX / 2.0, f32::MAX / 2.0), Some(0));
        assert_eq!(SpeakerLookup::new(None).at(1.0, 1.2), None);
    }

    #[test]
    fn overlapping_speakers_keep_the_diarizers_original_span_order() {
        let spans = [span(3.0, 5.0, 1), span(0.0, 4.0, 0)];
        let lookup = SpeakerLookup::new(Some(&spans));
        assert_eq!(lookup.at(3.5, 3.5), Some(1));
        assert_eq!(lookup.at(2.5, 2.5), Some(0));

        let touching = [span(0.0, 3.0, 0), span(3.0, 4.0, 1)];
        assert_eq!(SpeakerLookup::new(Some(&touching)).at(3.0, 3.0), Some(0));
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
        // Speakers 2 and 1 are kept, numbered by first appearance.
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(kept[0].end, span(1.0, 20.0, 0).end);
        let all = limit_speakers(spans, 3);
        assert_eq!(all.len(), 4);
        let ids = all.iter().map(|span| span.speaker_id).collect::<Vec<_>>();
        assert_eq!(ids, vec![0, 1, 2, 0]);
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
        let jobs = collect_jobs(&cli, &mut Vec::new(), &mut Vec::new()).unwrap();
        assert!(!jobs[0].diarize && jobs[0].speakers.is_none());
        assert!(jobs[1].diarize && jobs[1].speakers == Some(2));
        fs::remove_dir_all(directory.path()).unwrap();
    }

    #[test]
    fn speaker_tracks_group_by_zoom_session_and_folder() {
        let directory = audio_convert::TempPath::new("dir");
        let root = directory.path();
        let zoom = root.join("Meeting A/Audio Record");
        let podcast = root.join("Podcast");
        fs::create_dir_all(&zoom).unwrap();
        fs::create_dir_all(&podcast).unwrap();
        let files = [
            zoom.join("audioAliceSmith11067067067.wav"),
            root.join("Meeting A/audio1067067067.wav"),
            // Alice reconnected: her next file is part of the same session.
            zoom.join("audioAliceSmith31067067067.wav"),
            zoom.join("audioCara12062062062.wav"),
            root.join("solo.wav"),
            zoom.join("audioBob21067067067.wav"),
            zoom.join("audioDan12062062062.wav"),
            podcast.join("host.wav"),
            podcast.join("guest.wav"),
        ];
        for file in &files {
            audio_convert::write_float_wav(file, &[0.0; 16], 16_000).unwrap();
        }
        let items = files
            .iter()
            .map(|file| {
                serde_json::json!({
                    "input": file,
                    "iso_track": !file.ends_with("solo.wav"),
                })
            })
            .collect::<Vec<_>>();
        let manifest = root.join("jobs.json");
        fs::write(&manifest, serde_json::json!({ "items": items }).to_string()).unwrap();
        let out = root.join("out");
        let cli = Cli::parse_from([
            "transcriberrr".as_ref(),
            "--manifest".as_ref(),
            manifest.as_os_str(),
            "--output-dir".as_ref(),
            out.as_os_str(),
        ]);
        let (units, skips) = plan_units(
            &cli,
            collect_jobs(&cli, &mut Vec::new(), &mut Vec::new()).unwrap(),
            &[],
        )
        .unwrap();
        let summary = units
            .iter()
            .map(|unit| match unit {
                Unit::Single(job) => job
                    .input
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                Unit::Session(session) => format!(
                    "{} [{}] -> {}",
                    session.label,
                    session.speakers.join(", "),
                    session.output.file_name().unwrap().to_string_lossy()
                ),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                "Meeting A [Alice Smith, Alice Smith, Bob] -> Meeting A.merged.json",
                "Meeting A [Cara, Dan] -> Meeting A.merged.json",
                "solo.wav",
                "Podcast [Host, Guest] -> Podcast.merged.json",
            ]
        );
        let Unit::Session(second_meeting) = &units[1] else {
            panic!("not a session")
        };
        // The same folder can hold two Zoom meeting IDs.
        let Unit::Session(first_meeting) = &units[0] else {
            panic!("not a session")
        };
        assert_ne!(first_meeting.output, second_meeting.output);
        // The mixed recording belongs to no speaker, and the tracks cover it.
        assert_eq!(skips.len(), 1);
        assert!(skips[0].job.input.ends_with("audio1067067067.wav"));
        let Unit::Session(session) = &units[0] else {
            panic!("not a session")
        };
        // Alice's later file, like every file of the segment, starts with it.
        assert_eq!(
            track_starts(&session.jobs, &[600.0, 600.0, 600.0]),
            [0.0, 0.0, 0.0]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn plans_a_zoom_library_folder_like_wisdom() {
        let directory = audio_convert::TempPath::new("dir");
        let root = directory.path();
        let files = [
            // Two participants (one reconnected), the mixed recording, and the video.
            "Meeting A/audio1067067067.wav",
            "Meeting A/video1067067067.mp4",
            "Meeting A/shared_screen_with_speaker_view1067067067.mp4",
            "Meeting A/Audio Record/audioAliceSmith11067067067.m4a",
            "Meeting A/Audio Record/audioBob21067067067.m4a",
            "Meeting A/Audio Record/audioAliceSmith31067067067.m4a",
            // Stopped and started again, without participant files.
            "B/audio1656027999.m4a",
            "B/audio2656027999.m4a",
            // Only the host has a participant track.
            "C/audio1878787878.m4a",
            "C/audio2878787878.m4a",
            "C/Audio Record/audioCarlGrey21878787878.m4a",
            // 2018–2019 Zoom.
            "D/audio_only.m4a",
            "D/zoom_0.mp4",
            "D/Audio Record/audio_only_13572468_Dana_Reyes.m4a",
            "D/Audio Record/audio_only_13572468_Hana.m4a",
            // Not Zoom: a video beside its audio-only copy.
            "E/talk.m4a",
            "E/talk.mp4",
            "E/TALK.webm",
            // A renamed copy of the mixed recording is not its continuation.
            "G/audio_only.m4a",
            "G/audio_only01.m4a",
            // Names ending in a digit, not files 11 and 22.
            "H/Audio Record/audioGuest111444444444.m4a",
            "H/Audio Record/audioGuest221444444444.m4a",
            // Participant files from both halves of a restarted recording.
            "F/Audio Record/audioAlice11555555555.m4a",
            "F/Audio Record/audioBob21555555555.m4a",
            "F/Audio Record/audioAlice32555555555.m4a",
            "F/Audio Record/audioBob42555555555.m4a",
            "#recycle/old.m4a",
            "Notes/#42 Standup.m4a",
            "Notes/memo.m4a",
        ];
        for file in files {
            let path = root.join(file);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            audio_convert::write_float_wav(&path, &[0.0; 16], 16_000).unwrap();
        }
        let out = root.join("out");
        let cli = Cli::parse_from([
            "transcriberrr".as_ref(),
            "--input-dir".as_ref(),
            root.as_os_str(),
            "--recursive".as_ref(),
            "--output-dir".as_ref(),
            out.as_os_str(),
        ]);
        let (units, skips) = plan_units(
            &cli,
            collect_jobs(&cli, &mut Vec::new(), &mut Vec::new()).unwrap(),
            &[],
        )
        .unwrap();
        let name = |path: &Path| path.file_name().unwrap().to_string_lossy().into_owned();
        let summary = units
            .iter()
            .map(|unit| match unit {
                Unit::Single(job) => std::iter::once(&job.input)
                    .chain(&job.parts)
                    .map(|path| name(path))
                    .collect::<Vec<_>>()
                    .join(" + "),
                Unit::Session(session) => format!(
                    "{} [{}] -> {}",
                    session.label,
                    session.speakers.join(", "),
                    name(&session.output)
                ),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                "audio1656027999.m4a + audio2656027999.m4a",
                "C [Carl Grey] -> C.merged.json",
                "D [Dana Reyes, Hana] -> D.merged.json",
                "talk.m4a",
                "F [Alice, Alice, Bob, Bob] -> F.merged.json",
                "audio_only.m4a",
                "audio_only01.m4a",
                "H [Guest1, Guest2] -> H.merged.json",
                "Meeting A [Alice Smith, Alice Smith, Bob] -> Meeting A.merged.json",
                "#42 Standup.m4a",
                "memo.m4a",
            ]
        );
        let Unit::Session(one_track) = &units[1] else {
            panic!("not a session")
        };
        assert_eq!(one_track.jobs.len(), 1);
        let mut skipped = skips
            .iter()
            .map(|skip| name(&skip.job.input))
            .collect::<Vec<_>>();
        skipped.sort();
        assert_eq!(
            skipped,
            [
                "TALK.webm",
                "audio1067067067.wav",
                "audio1878787878.m4a",
                "audio2656027999.m4a",
                "audio2878787878.m4a",
                "audio_only.m4a",
                "shared_screen_with_speaker_view1067067067.mp4",
                "talk.mp4",
                "video1067067067.mp4",
                "zoom_0.mp4",
            ]
        );
        // A rerun skips what is there, but redoes a session that was never merged.
        let rerun = Cli::parse_from([
            "transcriberrr".as_ref(),
            "--input-dir".as_ref(),
            root.as_os_str(),
            "--recursive".as_ref(),
            "--skip-existing".as_ref(),
            "--output-dir".as_ref(),
            out.as_os_str(),
        ]);
        let progress = ProgressLog::open(None, false).unwrap();
        let Unit::Single(memo) = &units[10] else {
            panic!("not a file")
        };
        let Unit::Session(meeting) = &units[8] else {
            panic!("not a session")
        };
        for job in [memo, &meeting.jobs[0]] {
            let output = output_path(&rerun, job);
            fs::create_dir_all(output.parent().unwrap()).unwrap();
            fs::write(output, "{}").unwrap();
        }
        assert!(unit_done(&rerun, &progress, &units[10]));
        assert!(!unit_done(&rerun, &progress, &units[8]));
        assert!(may_replace(&rerun, Some(&bleed::Mask::default()), false));
        assert!(!may_replace(&rerun, None, false));
        let Unit::Session(restarted) = &units[4] else {
            panic!("not a session")
        };
        // Alice's files 1 and 3, then Bob's 2 and 4: the second segment starts where the
        // first one's longest file ends.
        assert_eq!(
            track_starts(&restarted.jobs, &[60.0, 30.0, 62.0, 30.0]),
            [0.0, 62.0, 0.0, 62.0]
        );
    }

    #[test]
    fn a_session_that_cannot_run_takes_its_skipped_files_with_it() {
        let directory = audio_convert::TempPath::new("dir");
        let root = directory.path();
        for file in [
            "M/audio1067067067.m4a",
            "M/video1067067067.mp4",
            // Two encodings of the same participant export.
            "M/Audio Record/audioAlice11067067067.m4a",
            "M/Audio Record/audioAlice11067067067.wav",
            "M/Audio Record/audioBob11067067067.m4a",
        ] {
            let path = root.join(file);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            audio_convert::write_float_wav(&path, &[0.0; 16], 16_000).unwrap();
        }
        let out = root.join("out");
        let cli = Cli::parse_from([
            "transcriberrr".as_ref(),
            "--input-dir".as_ref(),
            root.as_os_str(),
            "--recursive".as_ref(),
            "--output-dir".as_ref(),
            out.as_os_str(),
        ]);
        let (mut units, mut skips) = plan_units(
            &cli,
            collect_jobs(&cli, &mut Vec::new(), &mut Vec::new()).unwrap(),
            &[],
        )
        .unwrap();
        assert_eq!(skips.len(), 2, "the mixed recording and the video");
        let mut rejected = Vec::new();
        drop_clashes(&cli, &mut units, &mut rejected);
        reject_uncovered(&mut skips, &mut rejected);
        assert!(units.is_empty());
        assert!(skips.is_empty(), "nothing is shown as covered");
        let left_out = rejected
            .iter()
            .filter(|(_, error)| error.contains("left out as"))
            .count();
        assert_eq!(rejected.len(), 5);
        assert_eq!(left_out, 2);
    }

    #[test]
    fn session_names_do_not_depend_on_other_folders() {
        let directory = audio_convert::TempPath::new("dir");
        let root = directory.path();
        for file in [
            "2024/ep/host.m4a",
            "2024/ep/guest.m4a",
            "2025/ep/host.m4a",
            "2025/ep/guest.m4a",
        ] {
            let path = root.join(file);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            audio_convert::write_float_wav(&path, &[0.0; 16], 16_000).unwrap();
        }
        let labels = |input: &Path| {
            let cli = Cli::parse_from([
                "transcriberrr".as_ref(),
                "--iso-tracks".as_ref(),
                "--input-dir".as_ref(),
                input.as_os_str(),
                "--recursive".as_ref(),
                "--output-location".as_ref(),
                "next-to-source".as_ref(),
            ]);
            let (units, _) = plan_units(
                &cli,
                collect_jobs(&cli, &mut Vec::new(), &mut Vec::new()).unwrap(),
                &[],
            )
            .unwrap();
            units
                .iter()
                .map(|unit| match unit {
                    Unit::Session(session) => session.output.clone(),
                    Unit::Single(_) => panic!("not a session"),
                })
                .collect::<Vec<_>>()
        };
        let both = labels(root);
        assert_eq!(
            both,
            [
                root.join("2024/ep/ep.merged.json"),
                root.join("2025/ep/ep.merged.json")
            ]
        );
        assert_eq!(
            labels(&root.join("2025")),
            [root.join("2025/ep/ep.merged.json")]
        ); // Another show's "ep", merged there earlier, keeps its transcript.
        let merged = |source: &Path| {
            serde_json::json!({
                "source": "x", "model": "m", "device": "cpu", "language": "English",
                "duration_seconds": 1.0, "segments": [], "speaker_names": ["Host"],
                "metadata": {
                    "transcribed-with": "transcriberrr", "version": "0",
                    "source_created_at": null, "source_modified_at": null,
                    "transcribed_at": "2026-10-23T00:00:00Z", "model": "m",
                    "merged_from": [{"source": source, "speaker": 0, "offset_seconds": 0.0}]
                }
            })
            .to_string()
        };
        fs::write(
            root.join("2025/ep/ep.merged.json"),
            merged(Path::new("/elsewhere/host.m4a")),
        )
        .unwrap();
        assert_eq!(
            labels(&root.join("2025")),
            [root.join("2025/ep/ep 2.merged.json")]
        );
        // Its own earlier merge is replaced, and found by `--skip-existing`.
        fs::write(
            root.join("2025/ep/ep.merged.json"),
            merged(&root.join("2025/ep/host.m4a")),
        )
        .unwrap();
        assert_eq!(
            labels(&root.join("2025")),
            [root.join("2025/ep/ep.merged.json")]
        );
    }

    #[test]
    fn recursive_walk_skips_hidden_and_unreadable_folders() {
        use std::os::unix::fs::PermissionsExt;
        let directory = audio_convert::TempPath::new("dir");
        let root = directory.path();
        for folder in ["keep", ".Trashes", "locked"] {
            fs::create_dir_all(root.join(folder)).unwrap();
            audio_convert::write_float_wav(&root.join(folder).join("a.wav"), &[0.0; 16], 16_000)
                .unwrap();
        }
        fs::set_permissions(root.join("locked"), fs::Permissions::from_mode(0o300)).unwrap();
        let cli = Cli::parse_from([
            "transcriberrr".as_ref(),
            "--recursive".as_ref(),
            "--input-dir".as_ref(),
            root.as_os_str(),
        ]);
        let jobs = collect_jobs(&cli, &mut Vec::new(), &mut Vec::new());
        fs::set_permissions(root.join("locked"), fs::Permissions::from_mode(0o700)).unwrap();
        let inputs = jobs
            .unwrap()
            .into_iter()
            .map(|job| job.input)
            .collect::<Vec<_>>();
        assert_eq!(inputs, vec![root.join("keep/a.wav")]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_tolerates_a_line_torn_inside_a_multibyte_character() {
        let path = audio_convert::TempPath::new("jsonl");
        let mut bytes = b"{\"input\":\"/a.wav\",\"status\":\"completed\"}\n".to_vec();
        bytes.extend_from_slice(&"{\"input\":\"/é".as_bytes()[..13]);
        fs::write(path.path(), bytes).unwrap();
        let log = ProgressLog::open(Some(path.path()), true).expect("torn UTF-8 must not abort");
        assert!(log.latest.contains_key(&("/a.wav".to_owned(), None)));
    }

    #[test]
    fn dotted_stems_keep_every_component() {
        let cli = Cli::parse_from(["transcriberrr", "--format", "txt", "-o", "out", "x.wav"]);
        let job = |input: &str| Job {
            input: PathBuf::from(input),
            parts: Vec::new(),
            iso: false,
            zoom_recording: false,
            zoom_audio_copy: None,
            diarize: false,
            speakers: None,
            output: None,
            planned_output: None,
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
        let clashes = |inputs: [&str; 2]| {
            output_clashes(inputs.map(|input| (Path::new(input), output_path(&cli, &job(input)))))
        };
        assert!(clashes(["a/call.1.m4a", "a/call.2.m4a"]).is_empty());
        let clashes = clashes(["a/Talk.m4a", "b/talk.wav"]);
        assert_eq!(clashes.len(), 2, "both files fail");
        assert!(clashes[1].1.contains("would both write"));
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
                .get(&("/a.wav".to_owned(), None))
                .is_some_and(|event| event.status == "completed")
        );
        log.record(ProgressEvent::new("/c.wav".to_owned(), "completed"))
            .unwrap();
        drop(log);
        let reopened = ProgressLog::open(Some(&path), true).unwrap();
        assert!(reopened.latest.contains_key(&("/a.wav".to_owned(), None)));
        assert!(reopened.latest.contains_key(&("/c.wav".to_owned(), None)));
        fs::remove_dir_all(directory).unwrap();
    }

    /// A session records a "Comparing speaker tracks" event for its first track before that
    /// track runs; it must not hide what the earlier run logged for the track.
    #[test]
    fn earlier_runs_events_survive_this_runs_own() {
        let path = audio_convert::TempPath::new("jsonl");
        fs::write(
            path.path(),
            "{\"input\":\"/a.wav\",\"output\":\"/out/a.json\",\"status\":\"running\",\"phase\":\"Writing output\"}\n",
        )
        .unwrap();
        let mut log = ProgressLog::open(Some(path.path()), true).unwrap();
        let output = Path::new("/out/a.json");
        log.record(ProgressEvent {
            output: Some(output.display().to_string()),
            phase: Some("Comparing speaker tracks (1/2)".to_owned()),
            ..ProgressEvent::new("/a.wav".to_owned(), "running")
        })
        .unwrap();
        assert!(log.interrupted_write("/a.wav", output));
        assert!(log.logged_before("/a.wav", output));
        log.record(ProgressEvent {
            output: Some("/out/b.json".to_owned()),
            ..ProgressEvent::new("/b.wav".to_owned(), "running")
        })
        .unwrap();
        assert!(!log.logged_before("/b.wav", Path::new("/out/b.json")));
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
    fn decodes_apple_written_files_symphonia_rejects() {
        let source = audio_convert::TempPath::new("wav");
        audio_convert::write_float_wav(source.path(), &tone(440.0, 16_000, 1.0), 16_000).unwrap();
        // afconvert writes μ-law WAV with a fmt chunk Symphonia calls malformed, and ADTS AAC
        // with an MPEG-2 header it cannot read.
        for (extension, format, codec) in [("wav", "WAVE", "ulaw"), ("aac", "adts", "aac")] {
            let encoded = audio_convert::TempPath::new(extension);
            let status = Command::new("/usr/bin/afconvert")
                .args(["-f", format, "-d", codec])
                .arg(source.path())
                .arg(encoded.path())
                .status()
                .unwrap();
            assert!(status.success());
            let audio = decode_audio(encoded.path()).unwrap();
            // AAC adds up to ~0.2 s of encoder priming and padding.
            assert!(
                (audio.duration_seconds - 1.0).abs() < 0.25,
                "{codec}: {}",
                audio.duration_seconds
            );
        }
    }

    #[test]
    fn plans_qta_and_decodes_its_compatible_audio_track() {
        let Ok(ffmpeg) = which_ffmpeg() else {
            eprintln!("skipping: ffmpeg not found");
            return;
        };
        let source = audio_convert::TempPath::new("wav");
        audio_convert::write_float_wav(source.path(), &tone(440.0, 48_000, 1.0), 48_000).unwrap();
        let memo = audio_convert::TempPath::new("QTA");
        // Put a silent PCM track before the AAC compatibility track.
        let status = Command::new(&ffmpeg)
            .args(["-loglevel", "error", "-y", "-f", "lavfi"])
            .args(["-i", "anullsrc=r=48000:cl=mono:d=1", "-i"])
            .arg(source.path())
            .args([
                "-map",
                "0:a",
                "-map",
                "1:a",
                "-c:a:0",
                "pcm_s16le",
                "-c:a:1",
                "aac",
                "-f",
                "mov",
            ])
            .arg(memo.path())
            .status()
            .unwrap();
        assert!(status.success());

        let cli = Cli::parse_from(["transcriberrr".as_ref(), memo.path().as_os_str()]);
        let (units, skipped, rejected) = prepare_plan(&cli).unwrap();
        assert!(skipped.is_empty() && rejected.is_empty());
        assert!(matches!(&units[..], [Unit::Single(job)] if job.input == memo.path()));

        let audio = decode_with_symphonia(memo.path()).unwrap();
        assert!((audio.duration_seconds - 1.0).abs() < 0.1);
        assert!(rms(&audio.samples) > 0.3);
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
        let jobs = collect_jobs(&cli, &mut rejected, &mut Vec::new()).unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].0, missing);
        let cli = Cli::parse_from([
            "transcriberrr".as_ref(),
            "--input-dir".as_ref(),
            directory.path().as_os_str(),
        ]);
        let jobs = collect_jobs(&cli, &mut Vec::new(), &mut Vec::new()).unwrap();
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
            parts: Vec::new(),
            iso: false,
            zoom_recording: false,
            zoom_audio_copy: None,
            diarize: false,
            speakers: None,
            output: Some(PathBuf::from(output)),
            planned_output: None,
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

    #[test]
    fn dated_zoom_names_are_not_zoom_segments() {
        let media = |name: &str| {
            let path = Path::new(name);
            zoom_media(path, &merge::parse_path(path)).map(|media| media.segment)
        };
        assert_eq!(media("zoom_0.mp4"), Some(0));
        assert_eq!(media("zoom_12.mp4"), Some(12));
        assert_eq!(media("zoom_20240115.mp4"), None);
        assert_eq!(media("zoom_.mp4"), None);
    }

    #[test]
    fn a_unix_time_video_name_needs_a_zoom_folder() {
        let directory = audio_convert::TempPath::new("dir");
        let root = directory.path();
        let is_zoom = |name: &str| {
            let path = root.join(name);
            zoom_media(&path, &merge::parse_path(&path)).is_some()
        };
        for file in [
            "phone/video1700000000.mp4",
            "phone/talk.mp4",
            "call/video1067067067.mp4",
            "call/audio1067067067.m4a",
            "view/video1067067067.mp4",
            "view/shared_screen_with_speaker_view1067067067.mp4",
            "tracks/video1067067067.mp4",
            "tracks/Audio Record/audioAlice11067067067.m4a",
        ] {
            let path = root.join(file);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, []).unwrap();
        }
        assert!(!is_zoom("phone/video1700000000.mp4"));
        assert!(is_zoom("call/video1067067067.mp4"));
        assert!(is_zoom("view/video1067067067.mp4"));
        assert!(is_zoom("tracks/video1067067067.mp4"));
    }
}
