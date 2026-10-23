mod audio_convert;
mod bleed;
mod merge;
mod model_cache;
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
    "aac", "aif", "aiff", "caf", "flac", "m4a", "m4b", "m4v", "mov", "mp3", "mp4", "ogg", "opus",
    "wav", "webm",
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
    /// Sortformer ONNX file, or a model ID/alias. Defaults to `sortformer` when diarizing.
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
    /// Which inputs are per-speaker tracks: by default, Zoom's participant files
    /// (`Audio Record/audioAliceSmith21067067067.m4a`); with a bare `--iso-tracks`, every
    /// input; with `--iso-tracks=off`, none. Each session's tracks (one Zoom meeting, or one
    /// folder) are transcribed together with other speakers' bleed muted, then merged into one
    /// transcript beside the tracks' outputs; the meeting's mixed recording is then skipped.
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
    /// Merge transcripts of one session's per-speaker (iso) tracks into one transcript, each
    /// track labeled as its own speaker. Zoom participant files are named automatically.
    Merge(MergeArgs),
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

#[derive(Debug, Args)]
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
        files.retain(|file| seen.insert(fs::canonicalize(file).unwrap_or_else(|_| file.clone())));
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
    Ok(())
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
    fn jobs(&self) -> Vec<&Job> {
        match self {
            Self::Single(job) => vec![job],
            Self::Session(session) => session.jobs.iter().chain(&session.mixed).collect(),
        }
    }
}

/// The speaker tracks of one recording session, transcribed together and merged.
struct IsoSession {
    label: String,
    /// The Zoom meeting ID, when the tracks are Zoom participant files.
    zoom: Option<String>,
    jobs: Vec<Job>,
    /// The meeting's mixed recording, when only one participant has tracks: it supplies the
    /// words of everyone else.
    mixed: Option<Job>,
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
}

struct Processed {
    output: PathBuf,
    warning: Option<String>,
    /// The word-level transcript, kept for a session's merge.
    words: Option<TranscriptDocument>,
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
        self.completed_key(&input_key(input))
    }

    fn completed_key(&self, key: &str) -> bool {
        self.latest
            .get(key)
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
    match cli.command.take() {
        Some(CliCommand::Convert(args)) => return convert_transcript(args),
        Some(CliCommand::Merge(args)) => return merge_transcripts(args),
        None => {}
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
    let mut jobs = collect_jobs(&cli, &mut rejected)?;
    // Overlapping globs can name a file twice; it is transcribed once.
    let mut seen = std::collections::HashSet::new();
    jobs.retain(|job| seen.insert(input_key(&job.input)));
    if jobs.is_empty() && rejected.is_empty() {
        bail!("no supported audio files were found");
    }
    // A speaker track holds one speaker; the merge labels it, not Sortformer.
    for job in &mut jobs {
        if job.iso && (job.diarize || job.speakers.is_some()) {
            eprintln!(
                "note: not diarizing {}: it is a speaker track",
                job.input.display()
            );
            job.diarize = false;
            job.speakers = None;
        }
    }
    // A clash between two files fails those files (and their sessions), not a whole library.
    let (mut units, mut skips, unplanned) = plan_units(&cli, jobs)?;
    rejected.extend(unplanned);
    drop_clashes(&cli, &mut units, &mut rejected);
    reject_uncovered(&mut skips, &mut rejected);
    let jobs = units.iter().flat_map(Unit::jobs).collect::<Vec<_>>();
    // Merging needs every track's word timings, whatever the requested output.
    let wants_words = units.iter().any(|unit| matches!(unit, Unit::Session(_)));
    if jobs.iter().any(|job| job.diarize) && cli.diarization_model.is_none() {
        // The bundled-compatible Sortformer export; downloaded on first use like other models.
        cli.diarization_model = Some(PathBuf::from("sortformer"));
    }
    if matches!(cli.engine, EngineKind::AppleNative)
        && jobs.iter().any(|job| job.diarize || job.speakers.is_some())
    {
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
    if !rejected.is_empty() && (cli.fail_fast || jobs.is_empty()) {
        bail!("{} input file(s) failed", rejected.len());
    }
    if units.iter().all(|unit| unit_done(&cli, &progress, unit)) {
        for unit in &units {
            record_skipped(&cli, &mut progress, unit)?;
        }
        if !rejected.is_empty() {
            bail!("{} input file(s) failed", rejected.len());
        }
        return Ok(());
    }
    // Resolved after the --resume early exit so an already-finished queue needs no model
    // (and no network to fetch one).
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
        if cli.timestamps != TimestampGranularity::None || wants_diarization || wants_words {
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
    let mut models = InferenceModels {
        asr: asr.as_ref(),
        aligner: aligner.as_ref(),
        parakeet: parakeet.as_mut(),
        diarizer: &mut diarizer,
        model_label: &model_label,
    };
    for unit in units {
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
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    match (fs::canonicalize(parent), path.file_name()) {
        (Ok(parent), Some(name)) => parent.join(name).display().to_string(),
        _ => path.display().to_string(),
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
    let input = fs::canonicalize(input).unwrap_or_else(|_| input.to_path_buf());
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
    let clashes = output_clashes(cli, &units.iter().flat_map(Unit::jobs).collect::<Vec<_>>());
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
    loop {
        let gone = rejected
            .iter()
            .map(|(input, _)| input)
            .collect::<std::collections::HashSet<_>>();
        let Some(index) = skips.iter().position(|skip| {
            !skip.instead.is_empty() && skip.instead.iter().all(|input| gone.contains(input))
        }) else {
            return;
        };
        let skip = skips.remove(index);
        rejected.push((
            skip.job.input,
            format!(
                "not transcribed: it was left out as {}, but those files cannot run",
                skip.reason
            ),
        ));
    }
}

/// The units to run, the files left out, and the files of sessions that cannot run.
type Plan = (Vec<Unit>, Vec<Skip>, Vec<(PathBuf, String)>);

/// Plan the run from the queued files, as wisdom ingests a Zoom library:
/// - Speaker tracks are grouped into sessions: Zoom participant files by meeting folder and
///   meeting ID, other tracks by folder. A session's tracks run together, where its first
///   track was queued, and are merged.
/// - Zoom's mixed recording is left out when its meeting's tracks hold two or more speakers;
///   with one speaker's tracks, it supplies everyone else; without tracks, it is transcribed
///   on its own. A recording stopped and started again (`audio1…`, `audio2…`) is joined.
/// - A video is left out when its audio is queued too: beside an audio-only copy with the
///   same name, or in a Zoom meeting folder.
/// - A session's only track is transcribed on its own.
///
/// Also returns the files of sessions that cannot be planned, with the reason.
fn plan_units(cli: &Cli, jobs: Vec<Job>) -> Result<Plan> {
    let names = merge::parse_paths(jobs.iter().map(|job| job.input.as_path()));
    let folders = jobs
        .iter()
        .map(|job| session_folder(&job.input))
        .collect::<Vec<_>>();
    let is_zoom = |index: usize| {
        matches!(
            names[index],
            merge::TrackName::Zoom { .. } | merge::TrackName::ZoomMixed { .. }
        )
    };
    let is_video = |job: &Job| {
        job.input.extension().is_some_and(|extension| {
            VIDEO_EXTENSIONS
                .iter()
                .any(|video| extension.eq_ignore_ascii_case(video))
        })
    };
    let mut skips: Vec<Skip> = Vec::new();
    let mut taken = vec![false; jobs.len()];

    for (index, job) in jobs.iter().enumerate() {
        if job.iso || !is_video(job) || !matches!(names[index], merge::TrackName::Other(_)) {
            continue;
        }
        let stem = |job: &Job| {
            job.input
                .file_stem()
                .map(|stem| stem.to_string_lossy().to_lowercase())
        };
        let audio_copies = jobs
            .iter()
            .filter(|other| {
                !is_video(other)
                    && other.input.parent() == job.input.parent()
                    && stem(other) == stem(job)
            })
            .map(|other| other.input.clone())
            .collect::<Vec<_>>();
        let zoom_files = (0..jobs.len())
            .filter(|&other| is_zoom(other) && folders[other] == folders[index])
            .map(|other| jobs[other].input.clone())
            .collect::<Vec<_>>();
        if !audio_copies.is_empty() {
            skips.push(Skip {
                job: job.clone(),
                reason: "a video whose audio-only copy is transcribed instead".to_owned(),
                instead: audio_copies,
            });
            taken[index] = true;
        } else if !zoom_files.is_empty() {
            // Any video there, as in wisdom's ingest: `video…`, `zoom_0`, `shared_screen…`.
            skips.push(Skip {
                job: job.clone(),
                reason: "Zoom's video of the meeting, whose audio recording is transcribed instead"
                    .to_owned(),
                instead: zoom_files,
            });
            taken[index] = true;
        }
    }

    // Speaker tracks, by session.
    struct Draft {
        folder: PathBuf,
        meeting: Option<String>,
        zoom: bool,
        tracks: Vec<usize>,
        mixed: Vec<usize>,
    }
    let mut drafts: Vec<Draft> = Vec::new();
    for (index, job) in jobs.iter().enumerate() {
        if !job.iso || taken[index] {
            continue;
        }
        let (meeting, zoom) = match &names[index] {
            merge::TrackName::Zoom { meeting, .. } => (Some(meeting.clone()), true),
            merge::TrackName::ZoomMixed { .. } => continue,
            merge::TrackName::Other(_) => (None, false),
        };
        taken[index] = true;
        match drafts
            .iter_mut()
            .find(|draft| draft.folder == folders[index] && draft.meeting == meeting)
        {
            Some(draft) => draft.tracks.push(index),
            None => drafts.push(Draft {
                folder: folders[index].clone(),
                meeting,
                zoom,
                tracks: vec![index],
                mixed: Vec::new(),
            }),
        }
    }
    let speaker_count = |draft: &Draft| {
        draft
            .tracks
            .iter()
            .map(|&index| match &names[index] {
                merge::TrackName::Zoom { speaker, .. } => speaker.clone(),
                _ => jobs[index].input.display().to_string(),
            })
            .collect::<std::collections::HashSet<_>>()
            .len()
    };

    // Zoom's mixed recordings, by meeting, each in segment order.
    struct Recording {
        folder: PathBuf,
        meeting: Option<String>,
        /// (segment, job index)
        parts: Vec<(u32, usize)>,
    }
    let mut recordings: Vec<Recording> = Vec::new();
    for index in 0..jobs.len() {
        let merge::TrackName::ZoomMixed { segment, meeting } = &names[index] else {
            continue;
        };
        if taken[index] {
            continue;
        }
        taken[index] = true;
        // Two files of one segment (`audio_only` and its copy `audio_only01`, an m4a and a
        // wav) are not a continuation: each stays a recording of its own.
        match recordings.iter_mut().find(|recording| {
            recording.folder == folders[index]
                && recording.meeting == *meeting
                && recording.parts.iter().all(|(other, _)| other != segment)
        }) {
            Some(recording) => recording.parts.push((*segment, index)),
            None => recordings.push(Recording {
                folder: folders[index].clone(),
                meeting: meeting.clone(),
                parts: vec![(*segment, index)],
            }),
        }
    }
    let mut singles: Vec<usize> = Vec::new();
    let mut joined: HashMap<usize, Vec<usize>> = HashMap::new();
    for Recording {
        folder,
        meeting,
        mut parts,
    } in recordings
    {
        parts.sort_unstable();
        let parts = parts
            .into_iter()
            .map(|(_, index)| index)
            .collect::<Vec<_>>();
        let same_meeting = |draft: &Draft| {
            draft.zoom
                && draft.folder == folder
                && (meeting.is_none() || draft.meeting.is_none() || draft.meeting == meeting)
        };
        let tracks = drafts
            .iter()
            .filter(|draft| same_meeting(draft) && speaker_count(draft) > 1)
            .flat_map(|draft| draft.tracks.iter().map(|&index| jobs[index].input.clone()))
            .collect::<Vec<_>>();
        if !tracks.is_empty() {
            for &index in &parts {
                skips.push(Skip {
                    job: jobs[index].clone(),
                    reason: "Zoom's mixed recording; the meeting's speaker tracks are merged \
                             instead"
                        .to_owned(),
                    instead: tracks.clone(),
                });
            }
            continue;
        }
        let (first, rest) = parts.split_first().expect("a recording has a part");
        for &index in rest {
            skips.push(Skip {
                job: jobs[index].clone(),
                reason: format!(
                    "the recording continues {}; both are transcribed together",
                    jobs[*first]
                        .input
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                ),
                instead: vec![jobs[*first].input.clone()],
            });
        }
        if let Some(draft) = drafts
            .iter_mut()
            .find(|draft| same_meeting(draft) && draft.mixed.is_empty())
        {
            draft.mixed = parts;
            continue;
        }
        singles.push(*first);
        joined.insert(*first, rest.to_vec());
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
    for draft in drafts {
        let first = draft.tracks[0].min(draft.mixed.first().copied().unwrap_or(usize::MAX));
        let mixed = draft.mixed.split_first().map(|(&first, rest)| {
            let mut job = take(first);
            job.parts = rest.iter().map(|&index| take(index).input).collect();
            if job.diarize {
                eprintln!(
                    "note: not diarizing {}: its words are given to the speakers without a track",
                    job.input.display()
                );
                job.diarize = false;
                job.speakers = None;
            }
            job
        });
        let tracks = draft.tracks.into_iter().map(&mut take).collect::<Vec<_>>();
        if tracks.len() == 1 && mixed.is_none() {
            eprintln!(
                "note: {} is the only track of its session; transcribing it on its own",
                tracks[0].input.display()
            );
        }
        let unit = match (tracks, mixed) {
            (mut tracks, None) if tracks.len() == 1 => Unit::Single(tracks.remove(0)),
            (tracks, mixed) => Unit::Session(IsoSession {
                label: String::new(),
                zoom: draft.meeting,
                jobs: tracks,
                mixed,
                speakers: Vec::new(),
                output: PathBuf::new(),
            }),
        };
        planned.push((first, unit));
    }
    for index in singles {
        let mut job = take(index);
        job.parts = joined
            .remove(&index)
            .unwrap_or_default()
            .into_iter()
            .map(|index| take(index).input)
            .collect();
        planned.push((index, Unit::Single(job)));
    }
    planned.sort_by_key(|(first, _)| *first);
    let mut units = planned
        .into_iter()
        .map(|(_, unit)| unit)
        .collect::<Vec<_>>();
    let clashes = name_sessions(cli, &mut units);
    let mut rejected = Vec::new();
    for (index, error) in clashes.into_iter().rev() {
        for job in units.remove(index).jobs() {
            rejected.push((job.input.clone(), error.clone()));
        }
    }
    Ok((units, skips, rejected))
}

/// Name each session after its meeting folder, and place its merged transcript. Sessions
/// whose merged transcripts would share a folder and a name get their Zoom meeting ID (or a
/// number) added. Only those count, so a session keeps its name whatever else is queued, and
/// a rerun finds its transcript.
/// Returns the sessions whose merged transcript would replace another output, by index, with
/// the reason.
fn name_sessions(cli: &Cli, units: &mut [Unit]) -> Vec<(usize, String)> {
    let base = |session: &IsoSession| {
        session_folder(&session.jobs[0].input)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "session".to_owned())
    };
    // Beside the tracks' outputs, or beside the meeting's mixed recording when those mirror
    // Zoom's `Audio Record` folder.
    let folder = |session: &IsoSession| {
        meeting_folder(
            output_path(cli, &session.jobs[0])
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default(),
        )
    };
    let key = |folder: &Path, name: &str| folder.join(name).to_string_lossy().to_lowercase();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for unit in units.iter() {
        if let Unit::Session(session) = unit {
            *counts
                .entry(key(&folder(session), &base(session)))
                .or_default() += 1;
        }
    }
    let mut taken = std::collections::HashSet::new();
    let mut outputs = units
        .iter()
        .flat_map(Unit::jobs)
        .map(|job| output_path(cli, job).to_string_lossy().to_lowercase())
        .collect::<std::collections::HashSet<_>>();
    let mut clashes = Vec::new();
    for (index, unit) in units.iter_mut().enumerate() {
        let Unit::Session(session) = unit else {
            continue;
        };
        let base = base(session);
        let folder = folder(session);
        let mut label = match &session.zoom {
            Some(id) if counts[&key(&folder, &base)] > 1 => format!("{base} {id}"),
            _ => base.clone(),
        };
        let output =
            |label: &str| folder.join(format!("{label}.merged.{}", format_extension(cli.format)));
        let mut number = 2;
        while !taken.insert(key(&folder, &label)) || merged_elsewhere(&output(&label), session) {
            label = format!("{base} {number}");
            number += 1;
        }
        session.output = output(&label);
        if !outputs.insert(session.output.to_string_lossy().to_lowercase()) {
            clashes.push((
                index,
                format!(
                    "the merged transcript of {label} would replace another output: {}",
                    session.output.display()
                ),
            ));
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
    clashes
}

/// Whether an existing merged transcript was merged from other recordings: those of another
/// session with this one's name (two "Episode 1" folders, run on different days into one
/// output folder). Only JSON tells; another format is taken to be this session's.
fn merged_elsewhere(path: &Path, session: &IsoSession) -> bool {
    let Ok(doc) = read_transcript(path) else {
        return false;
    };
    let inputs = session
        .jobs
        .iter()
        .chain(&session.mixed)
        .map(|job| input_key(&job.input))
        .collect::<std::collections::HashSet<_>>();
    !doc.metadata.merged_from.is_empty()
        && !doc
            .metadata
            .merged_from
            .iter()
            .any(|track| inputs.contains(&input_key(Path::new(&track.source))))
}

/// Whether a unit can be skipped: finished in an earlier run (`--resume`), or with its
/// transcript already there (`--skip-existing`).
fn unit_done(cli: &Cli, progress: &ProgressLog, unit: &Unit) -> bool {
    let exists = match unit {
        Unit::Single(job) => output_path(cli, job).exists(),
        Unit::Session(session) => session.output.exists(),
    };
    (cli.resume && unit_completed(cli, progress, unit)) || (cli.skip_existing && exists)
}

fn unit_completed(cli: &Cli, progress: &ProgressLog, unit: &Unit) -> bool {
    match unit {
        Unit::Single(job) => progress.is_completed(&job.input) && output_path(cli, job).exists(),
        // A session counts as done only once merged; otherwise all its tracks run again,
        // since the merge needs every track's words.
        Unit::Session(session) => {
            progress.completed_key(&output_key(&session.output)) && session.output.exists()
        }
    }
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
            progress.record(skipped(input_key(&job.input), &output_path(cli, job)))
        }
        Unit::Session(session) => {
            let speakers = session
                .speakers
                .iter()
                .map(String::as_str)
                .chain(session.mixed.iter().map(|_| merge::OTHER_SPEAKERS));
            for (job, speaker) in session.jobs.iter().chain(&session.mixed).zip(speakers) {
                progress.record(ProgressEvent {
                    group: Some(session.label.clone()),
                    speaker: Some(speaker.to_owned()),
                    ..skipped(input_key(&job.input), &output_path(cli, job))
                })?;
            }
            progress.record(ProgressEvent {
                group: Some(session.label.clone()),
                kind: Some("merge".to_owned()),
                ..skipped(output_key(&session.output), &session.output)
            })
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
    let logged = progress.latest.contains_key(&key);
    let predicted_output = output_path(cli, job).display().to_string();
    let event = |status: &str| ProgressEvent {
        output: Some(predicted_output.clone()),
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
    let result = process_one(
        cli,
        job,
        models,
        member.map(|member| member.gate),
        logged,
        &mut report,
    );
    let elapsed_seconds = Some(started.elapsed().as_secs_f32());
    match result {
        Ok(processed) => {
            progress.record(ProgressEvent {
                output: Some(processed.output.display().to_string()),
                elapsed_seconds,
                progress: Some(1.0),
                phase: Some("Complete".to_owned()),
                warning: processed.warning.clone(),
                ..event("completed")
            })?;
            Ok(Some(processed))
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
    if let Some(parent) = session
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        && let Err(error) = fs::create_dir_all(parent)
    {
        let error = anyhow::Error::new(error)
            .context(format!("creating output directory {}", parent.display()));
        return fail(progress, failures, error);
    }
    // Fail before minutes of transcription, not at the end.
    // `--resume` reruns a session whose merge was not recorded as finished.
    if session.output.exists() && !(cli.overwrite || cli.resume) {
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
                    output: Some(output_path(cli, first).display().to_string()),
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
            .and_then(|()| decode_job(job));
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
                for job in session.jobs.iter().chain(&session.mixed) {
                    *failures += 1;
                    progress.record(ProgressEvent {
                        output: Some(output_path(cli, job).display().to_string()),
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
        };
        if let Some(processed) = run_job(
            cli,
            job,
            models.reborrow(),
            progress,
            Some(member),
            failures,
        )? {
            docs.push(processed.words.context("speaker tracks keep their words")?);
        }
    }
    if let Some(job) = &session.mixed
        && !cancel_requested()
        && docs.len() == session.jobs.len()
    {
        let member = Member {
            group: &session.label,
            speaker: merge::OTHER_SPEAKERS,
            gate: &bleed::Mask::default(),
        };
        if let Some(processed) = run_job(
            cli,
            job,
            models.reborrow(),
            progress,
            Some(member),
            failures,
        )? {
            docs.push(
                processed
                    .words
                    .context("the mixed recording keeps its words")?,
            );
        }
    } else if let Some(job) = &session.mixed
        && !cancel_requested()
    {
        // Its words would go to waste without the track's.
        *failures += 1;
        progress.record(ProgressEvent {
            output: Some(output_path(cli, job).display().to_string()),
            error: Some("not transcribed: the speaker track failed".to_owned()),
            phase: Some("Failed".to_owned()),
            group: Some(session.label.clone()),
            speaker: Some(merge::OTHER_SPEAKERS.to_owned()),
            ..ProgressEvent::new(input_key(&job.input), "failed")
        })?;
    }
    if cancel_requested() {
        return progress.record(event("canceled"));
    }
    let expected = session.jobs.len() + usize::from(session.mixed.is_some());
    if docs.len() < expected {
        // The failed tracks were counted already.
        return progress.record(ProgressEvent {
            error: Some(format!(
                "{} of {expected} recordings failed; not merged",
                expected - docs.len(),
            )),
            ..event("failed")
        });
    }
    let inputs = docs
        .into_iter()
        .zip(session.jobs.iter().chain(&session.mixed))
        .enumerate()
        .map(|(index, (doc, job))| {
            let mixed = index == session.jobs.len();
            merge::TrackInput {
                doc,
                path: job.input.display().to_string(),
                name: mixed.then(|| merge::OTHER_SPEAKERS.to_owned()),
                mixed,
            }
        })
        .collect();
    let merged = merge::merge_tracks(inputs).and_then(|(mut doc, warnings)| {
        apply_timestamps(&mut doc, cli.timestamps)?;
        write_output(
            &doc,
            &session.output,
            render_options(cli),
            cli.overwrite || cli.resume,
        )?;
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
    // Judge chunks on a CPU thread (~0.2 s per 30 s chunk) while the GPU transcribes, so voice
    // detection adds no wall-clock time. The thread stops when the receiver is dropped.
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| -> Result<()> {
        scope.spawn(|| {
            for range in &chunks {
                if sender
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
            if !receiver.recv().unwrap_or(true) {
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

/// Whether a job may replace its transcript: with `--overwrite`; with `--resume`, for an input
/// the progress file has seen started (`logged`) but not finished, since a kill can land
/// between the write and its event; or as a track of a session rerun by `--skip-existing` or
/// `--resume` because it was never merged, which needs the track's words again.
fn may_replace(cli: &Cli, gate: Option<&bleed::Mask>, logged: bool) -> bool {
    cli.overwrite || (cli.resume && logged) || (gate.is_some() && (cli.skip_existing || cli.resume))
}

/// Transcribe one job. With a `gate` (a speaker track), other speakers' bleed is muted
/// first and the word-level transcript is returned for the session's merge.
fn process_one(
    cli: &Cli,
    job: &Job,
    models: InferenceModels<'_>,
    gate: Option<&bleed::Mask>,
    logged: bool,
    report: &mut dyn FnMut(f32, &str) -> Result<()>,
) -> Result<Processed> {
    if matches!(cli.engine, EngineKind::AppleNative) {
        report(0.1, "Starting Apple SpeechTranscriber")?;
        // No progress report after this: the transcript is already written, and a cancel
        // caught there would record a finished job as canceled.
        return process_apple_native(cli, job, gate, logged);
    }
    // Fail before decoding and inference, not after minutes of work.
    let destination = output_path(cli, job);
    if destination.exists() && !may_replace(cli, gate, logged) {
        bail!("output exists (use --overwrite): {}", destination.display());
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
        .map(|item| {
            let (start, end) = match &timeline {
                Some(timeline) => (
                    timeline.original(item.start_time),
                    timeline.original(item.end_time),
                ),
                None => (item.start_time, item.end_time),
            };
            Word {
                text: item.text,
                start,
                end,
                speaker: speaker_at(&speaker_segments, start, end),
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
    let mut doc = TranscriptDocument {
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
    if cli.timestamps != TimestampGranularity::Words {
        // Sentence timestamps retain segment boundaries but do not expose word items. This keeps
        // the JSON shape stable while honoring the requested granularity.
        doc.collapse_words();
    }
    let output_path = output_path(cli, job);
    report(0.95, "Writing output")?;
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating output directory {}", parent.display()))?;
    }
    write_output(
        &doc,
        &output_path,
        render_options(cli),
        may_replace(cli, gate, logged),
    )?;
    eprintln!(
        "{} -> {} ({:.2}s)",
        job.input.display(),
        output_path.display(),
        started.elapsed().as_secs_f32()
    );
    // The app shows this on the finished row; otherwise an empty transcript looks like success.
    let warning = [
        (!spoke).then(|| "No speech detected; the transcript is empty".to_owned()),
        audio.warning,
    ]
    .into_iter()
    .flatten()
    .reduce(|first, second| format!("{first}; {second}"));
    Ok(Processed {
        output: output_path,
        warning,
        words,
    })
}

fn process_apple_native(
    cli: &Cli,
    job: &Job,
    gate: Option<&bleed::Mask>,
    logged: bool,
) -> Result<Processed> {
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
    if output_path.exists() && !may_replace(cli, gate, logged) {
        bail!("output exists (use --overwrite): {}", output_path.display());
    }
    // AVFoundation cannot open Matroska/WebM, and the helper hangs forever on FLAC in Ogg;
    // decode those here and hand the helper a WAV. A speaker track goes as a WAV with the
    // other speakers' bleed muted, and a recording in parts as one WAV of them all.
    let converted = if gate.is_some()
        || !job.parts.is_empty()
        || job.input.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("webm") || extension.eq_ignore_ascii_case("ogg")
        }) {
        let mut audio = decode_job(job)?;
        if let Some(gate) = gate {
            gate.apply(&mut audio.samples);
        }
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
    let words = gate.is_some().then(|| doc.clone());
    if cli.timestamps != TimestampGranularity::Words {
        doc.collapse_words();
    }
    write_output(
        &doc,
        &output_path,
        render_options(cli),
        may_replace(cli, gate, logged),
    )?;
    let warning = doc
        .segments
        .is_empty()
        .then(|| "No speech detected; the transcript is empty".to_owned());
    Ok(Processed {
        output: output_path,
        warning,
        words,
    })
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
            if let Some(count) = speakers
                && let Err(error) = speaker_count(&count.to_string())
            {
                rejected.push((input, format!("invalid speakers {count}: {error}")));
                continue;
            }
            jobs.push(Job {
                iso: item
                    .iso_track
                    .unwrap_or_else(|| is_iso_track(cli.iso_tracks, &input)),
                parts: Vec::new(),
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
            // Skip hidden entries (.Trashes, .Spotlight-V100, .fseventsd on volume roots hold
            // deleted or system files) and `#recycle`-style folders, and keep going past
            // unreadable folders: an error on any entry used to abort the whole run before a
            // single file was transcribed.
            let walker = WalkDir::new(&dir).into_iter().filter_entry(|entry| {
                let name = entry.file_name().to_string_lossy();
                // `#` only for folders: `#42 Standup.m4a` is a recording.
                entry.depth() == 0
                    || !(name.starts_with('.')
                        || (entry.file_type().is_dir() && name.starts_with('#')))
            });
            for entry in walker {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        eprintln!("warning: skipping {error}");
                        continue;
                    }
                };
                // `path().is_file()` follows symlinks, matching the non-recursive branch.
                if entry.path().is_file() && is_audio_path(entry.path()) {
                    paths.push(entry.into_path());
                } else if entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("zoom"))
                {
                    eprintln!(
                        "warning: skipping {}: Zoom has not converted this recording yet; open \
                         it in Zoom to convert it",
                        entry.path().display()
                    );
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
                iso: is_iso_track(cli.iso_tracks, &input),
                parts: Vec::new(),
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
                iso: is_iso_track(cli.iso_tracks, &path),
                parts: Vec::new(),
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

fn is_iso_track(mode: IsoMode, path: &Path) -> bool {
    match mode {
        IsoMode::Auto => matches!(merge::parse_path(path), merge::TrackName::Zoom { .. }),
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
            let wav = without_panic(|| audio_convert::convert_with_core_audio(path))
                .with_context(|| format!("{error:#}; the Core Audio fallback also failed"))?;
            without_panic(|| decode_with_symphonia(wav.path()))
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
) -> Result<(u32, Box<dyn symphonia::core::codecs::Decoder>)> {
    let track = format
        .tracks()
        .iter()
        .find(|track| {
            track.codec_params.codec != CODEC_TYPE_NULL && track.codec_params.sample_rate.is_some()
        })
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
    let (mut track_id, mut decoder) = audio_track(format.as_ref())?;
    let mut output = Vec::new();
    // Resample from the decoder's output rate. Apple's ALAC encoder writes 44100 in the MP4
    // sample entry for every file; the true rate is in the codec config, which the decoder
    // uses. Trusting the container stretched 48 kHz Voice Memos by 9%.
    let mut section = Section::default();
    let mut decoded_packets = 0usize;
    let mut skipped_packets = 0usize;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) if packet.track_id() == track_id => packet,
            Ok(_) => continue,
            // A chained Ogg file (streams joined end to end) starts a new stream, with new
            // track IDs and possibly another sample rate.
            Err(SymphoniaError::ResetRequired) => {
                (track_id, decoder) = audio_track(format.as_ref())?;
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
                if rate != section.rate {
                    if !section.samples.is_empty() {
                        section.flush(&mut output);
                    }
                    section.rate = rate;
                }
                append_decoded(buffer, &mut section.samples);
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
    if section.rate == 0 {
        bail!("audio stream has an invalid zero sample rate");
    }
    section.flush(&mut output);
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

/// Inputs that would write the same transcript as another (for example `talk.m4a` and
/// `talk.wav`), each with the reason. Without this check, the second job either fails late or,
/// with `--overwrite`, silently replaces the first job's output. Both are rejected, and the
/// rest of the queue runs.
fn output_clashes(cli: &Cli, jobs: &[&Job]) -> Vec<(PathBuf, String)> {
    let mut seen: HashMap<String, &Path> = HashMap::new();
    let mut clashes = Vec::new();
    for job in jobs {
        // APFS is case-insensitive by default, so compare case-folded paths; canonical ones,
        // so `a/talk.json` and `./a/talk.json` meet.
        let key = output_key(&output_path(cli, job)).to_lowercase();
        if let Some(previous) = seen.insert(key, &job.input) {
            let error = format!(
                "{} and {} would both write {}; use a manifest with explicit outputs",
                previous.display(),
                job.input.display(),
                output_path(cli, job).display()
            );
            // A third file on the same output fails itself, not the first one again.
            if !clashes.iter().any(|(input, _)| input == previous) {
                clashes.push((previous.to_path_buf(), error.clone()));
            }
            clashes.push((job.input.clone(), error));
        }
    }
    clashes
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
        let (units, skips, _) =
            plan_units(&cli, collect_jobs(&cli, &mut Vec::new()).unwrap()).unwrap();
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
                    session.output.strip_prefix(&out).unwrap().display()
                ),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                "Meeting A 067067067 [Alice Smith, Alice Smith, Bob] -> Meeting A 067067067.merged.json",
                "Meeting A 062062062 [Cara, Dan] -> Meeting A 062062062.merged.json",
                "solo.wav",
                "Podcast [Host, Guest] -> Podcast.merged.json",
            ]
        );
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
            // Only the host's track: the mixed recording has the client.
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
        let (units, skips, _) =
            plan_units(&cli, collect_jobs(&cli, &mut Vec::new()).unwrap()).unwrap();
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
                    "{} [{}]{} -> {}",
                    session.label,
                    session.speakers.join(", "),
                    session
                        .mixed
                        .as_ref()
                        .map(|job| format!(" + {}", name(&job.input)))
                        .unwrap_or_default(),
                    session.output.strip_prefix(&out).unwrap().display()
                ),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                "audio1656027999.m4a + audio2656027999.m4a",
                "C [Carl Grey] + audio1878787878.m4a -> C/C.merged.json",
                "D [Dana Reyes, Hana] -> D/D.merged.json",
                "talk.m4a",
                "F [Alice, Alice, Bob, Bob] -> F/F.merged.json",
                "audio_only.m4a",
                "audio_only01.m4a",
                "H [Guest1, Guest2] -> H/H.merged.json",
                "Meeting A [Alice Smith, Alice Smith, Bob] -> Meeting A/Meeting A.merged.json",
                "#42 Standup.m4a",
                "memo.m4a",
            ]
        );
        let Unit::Session(one_track) = &units[1] else {
            panic!("not a session")
        };
        let mixed = one_track.mixed.as_ref().unwrap();
        assert_eq!(
            mixed
                .parts
                .iter()
                .map(|path| name(path))
                .collect::<Vec<_>>(),
            ["audio2878787878.m4a"]
        );
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
            // Both write audioAlice11067067067.json.
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
        let (mut units, mut skips, mut rejected) =
            plan_units(&cli, collect_jobs(&cli, &mut Vec::new()).unwrap()).unwrap();
        assert_eq!(skips.len(), 2, "the mixed recording and the video");
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
            let (units, _, _) =
                plan_units(&cli, collect_jobs(&cli, &mut Vec::new()).unwrap()).unwrap();
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
        let jobs = collect_jobs(&cli, &mut Vec::new());
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
        assert!(log.latest.contains_key("/a.wav"));
    }

    #[test]
    fn dotted_stems_keep_every_component() {
        let cli = Cli::parse_from(["transcriberrr", "--format", "txt", "-o", "out", "x.wav"]);
        let job = |input: &str| Job {
            input: PathBuf::from(input),
            parts: Vec::new(),
            iso: false,
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
        assert!(output_clashes(&cli, &[&job("a/call.1.m4a"), &job("a/call.2.m4a")]).is_empty());
        let clashes = output_clashes(&cli, &[&job("a/Talk.m4a"), &job("b/talk.wav")]);
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
                .get("/a.wav")
                .is_some_and(|event| event.status == "completed")
        );
        log.record(ProgressEvent::new("/c.wav".to_owned(), "completed"))
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
            parts: Vec::new(),
            iso: false,
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
