//! Recorded-audio session contract. Recognition artifacts use track-local samples and can
//! be placed at a new start without rerunning a model.
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    Audio, Cli, EngineKind, InferenceModels, JOBS_STARTED, Job, OutputLocation, RunRequirements,
    TARGET_SAMPLE_RATE, cancel_requested, check_cancelled, configure_model, default_asr,
    engine_name, model_display_name, parse_model_pass, process_apple_native, transcribe_audio,
    transcript, with_inference_models, write_json_line,
};
use std::sync::atomic::Ordering;

mod native_result;
mod numbers;
mod reconcile;
pub(crate) mod shared_cache;
pub(crate) use native_result::NativeResult;
use reconcile::reconcile;

pub(crate) const VERSION: u8 = 1;
/// The one language the batch recognizers transcribe.
pub(crate) const RECOGNIZER_LANGUAGE: &str = "en";
const ACTIVITY_WINDOW: usize = TARGET_SAMPLE_RATE as usize / 50;

pub(crate) fn diarization_limit() -> Option<usize> {
    #[cfg(feature = "diarization")]
    {
        Some(parakeet_rs::sortformer::NUM_SPEAKERS)
    }
    #[cfg(not(feature = "diarization"))]
    {
        None
    }
}

#[derive(Debug, Clone, Args)]
pub(crate) struct SessionArgs {
    /// Versioned session manifest. Use - for stdin.
    #[arg(long)]
    manifest: PathBuf,
    /// Report start, track progress, track outcomes, and terminal status as JSON Lines.
    #[arg(long)]
    jsonl: bool,
    /// Use completed track artifacts without opening recognition models.
    #[arg(long, conflicts_with = "download")]
    reconcile_only: bool,
    /// Use installed models only (the default).
    #[arg(long, conflicts_with = "download")]
    offline: bool,
    /// Explicitly permit downloading missing models and Apple speech assets.
    #[arg(long)]
    download: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u8,
    job_id: String,
    fingerprint: String,
    engine: Option<EngineKind>,
    model: Option<String>,
    language: Option<String>,
    expected_speakers: Option<usize>,
    diarization: Option<bool>,
    output: PathBuf,
    tracks: Vec<Track>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Track {
    track_id: String,
    source_id: String,
    stream_evidence: Option<Value>,
    binding: Binding,
    channel: Option<u32>,
    name: String,
    role: Role,
    speaker: Option<String>,
    recognize: bool,
    start_frame: i64,
    audio: Option<PathBuf>,
    result: Option<PathBuf>,
    activity: Option<PathBuf>,
    recognition_key: Option<String>,
    unavailable: Option<String>,
    shared_cache: Option<shared_cache::Descriptor>,
    #[serde(default)]
    reuse: bool,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    stream_index: Option<u32>,
    origin: Time,
    session_start: Time,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Time {
    numerator: i64,
    denominator: i64,
}

impl Time {
    fn samples(self) -> Result<i64> {
        ensure!(self.denominator > 0, "clock denominator must be positive");
        let frames = i128::from(self.numerator) * i128::from(TARGET_SAMPLE_RATE);
        let scale = i128::from(self.denominator);
        i64::try_from(frames.signum() * ((frames.abs() + scale / 2) / scale))
            .context("clock cannot be expressed in signed samples")
    }
}

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Role {
    IsolatedMic,
    MixedAudio,
    Unknown,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Activity {
    start_sample: i64,
    end_sample: i64,
    level: f32,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ActivityArtifact<Frames = Vec<Activity>> {
    version: u8,
    sample_rate: u32,
    stream_evidence: Option<Value>,
    duration_samples: i64,
    activity: Frames,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LocalWord {
    id: String,
    utterance_id: String,
    text: String,
    start_sample: i64,
    end_sample: i64,
    speaker_id: Option<String>,
    precision: Precision,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Precision {
    Word,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrackArtifact {
    version: u8,
    sample_rate: u32,
    recognition_key: Option<String>,
    stream_evidence: Option<Value>,
    duration_samples: i64,
    words: Vec<LocalWord>,
    activity: Vec<Activity>,
    warning: Option<String>,
}

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Coverage {
    Recognized,
    Skipped,
    Failed,
    #[serde(rename = "coverage_only")]
    NoAudio,
}

impl Coverage {
    fn name(self) -> &'static str {
        match self {
            Self::Recognized => "recognized",
            Self::Skipped => "skipped",
            Self::Failed => "failed",
            Self::NoAudio => "coverage_only",
        }
    }
}

struct Outcome {
    coverage: Coverage,
    artifact: Option<TrackArtifact>,
    error: Option<String>,
}

#[derive(Serialize)]
struct SessionResult<'a> {
    version: u8,
    clock: &'static str,
    sample_rate: u32,
    job_id: &'a str,
    fingerprint: &'a str,
    tracks: Vec<TrackResult<'a>>,
    speakers: Vec<Speaker>,
    words: Vec<ConversationWord<'a>>,
}

#[derive(Serialize)]
struct TrackResult<'a> {
    id: &'a str,
    name: &'a str,
    source_id: &'a str,
    stream_index: Option<u32>,
    channel: Option<u32>,
    origin: Time,
    session_start: Time,
    stream_evidence: &'a Option<Value>,
    coverage: Coverage,
    role: Role,
    error: Option<&'a str>,
    warning: Option<&'a str>,
    activity: &'a [Activity],
    words: &'a [LocalWord],
    speakers: Vec<Speaker>,
}

#[derive(Deserialize, Serialize)]
struct Speaker {
    id: String,
    name: String,
}

#[derive(Serialize)]
struct ConversationWord<'a> {
    id: String,
    utterance_id: String,
    text: &'a str,
    start_sample: i64,
    end_sample: i64,
    speaker_id: Option<String>,
    precision: Precision,
    observations: Vec<Observation<'a>>,
}

#[derive(Serialize)]
struct Observation<'a> {
    track_id: &'a str,
    word_id: &'a str,
    text: &'a str,
    start_sample: i64,
    end_sample: i64,
    evidence: &'static str,
    basis: Option<&'static str>,
    speaker_id: Option<String>,
}

fn publish<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    check_cancelled()?;
    transcript::publish_atomic(path, true, |writer| {
        serde_json::to_writer(writer, value).map_err(std::io::Error::from)
    })
}

impl Manifest {
    fn diarization_enabled(&self) -> bool {
        self.expected_speakers.is_some()
            || self.diarization.unwrap_or(diarization_limit().is_some())
    }

    fn diarizes(&self, track: &Track) -> bool {
        track.role != Role::IsolatedMic && self.diarization_enabled()
    }

    fn validate(&self, input: &Path) -> Result<()> {
        ensure!(
            self.schema_version == VERSION,
            "unsupported session schema version {}",
            self.schema_version
        );
        ensure!(
            !self.job_id.is_empty() && !self.fingerprint.is_empty(),
            "job_id and fingerprint must be nonempty"
        );
        ensure!(!self.tracks.is_empty(), "session track inventory is empty");
        let mut ids = HashSet::new();
        let mut outputs = vec![self.output.as_path()];
        let mut inputs = Vec::new();
        if input != Path::new("-") {
            inputs.push(input);
        }
        for track in &self.tracks {
            ensure!(
                !track.track_id.is_empty() && ids.insert(&track.track_id),
                "empty or duplicate track ID {:?}",
                track.track_id
            );
            ensure!(
                !track.source_id.is_empty() && !track.name.trim().is_empty(),
                "track {:?} has no source ID or display name",
                track.track_id
            );
            track.binding.origin.samples()?;
            ensure!(
                track.binding.session_start.samples()? == track.start_frame,
                "track {:?}: start_frame disagrees with the exact session clock",
                track.track_id
            );
            if let Some(audio) = track.audio.as_deref() {
                inputs.push(audio);
            }
            if let Some(reason) = &track.unavailable {
                inputs.extend(track.activity.as_deref());
                inputs.extend(track.result.as_deref());
                ensure!(
                    !reason.trim().is_empty(),
                    "track {:?} has an empty unavailability reason",
                    track.track_id
                );
                continue;
            }
            if track.binding.stream_index.is_none() {
                ensure!(
                    !track.recognize
                        && !track.reuse
                        && track.audio.is_none()
                        && track.result.is_none()
                        && track.activity.is_none(),
                    "coverage-only track {:?} cannot have recognition or audio artifacts",
                    track.track_id
                );
                continue;
            }
            ensure!(
                track.stream_evidence.is_some(),
                "audio track {:?} has no stream evidence",
                track.track_id
            );
            let result = track
                .result
                .as_deref()
                .context("audio track has no result path")?;
            outputs.push(result);
            if let Some(activity) = track.activity.as_deref() {
                outputs.push(activity);
            }
        }
        protect_output_paths(&outputs, &inputs)?;
        Ok(())
    }

    fn event(&self, args: &SessionArgs, event: &str, status: &str, fields: Value) -> Result<()> {
        if !args.jsonl && event != "session_finished" {
            return Ok(());
        }
        let mut value = json!({
            "schema_version": VERSION, "job_id": self.job_id, "fingerprint": self.fingerprint,
            "event": event, "status": status,
        });
        if let Some(fields) = fields.as_object() {
            value
                .as_object_mut()
                .expect("event is a JSON object")
                .extend(fields.clone());
        }
        write_json_line(&value)
    }
}

pub(crate) fn run(mut cli: Cli, args: SessionArgs) -> Result<()> {
    let mut manifest: Manifest = if args.manifest == Path::new("-") {
        serde_json::from_reader(std::io::stdin().lock())
            .context("reading session manifest from stdin")?
    } else {
        read_json_file(&args.manifest)?
    };
    manifest.event(
        &args,
        "session_started",
        "running",
        json!({"track_count":manifest.tracks.len()}),
    )?;
    JOBS_STARTED.store(true, Ordering::SeqCst);
    let result = (|| {
        manifest.validate(&args.manifest)?;
        cli.engine = manifest.engine.unwrap_or(cli.engine);
        if let Some(model) = &manifest.model {
            if args.reconcile_only || model == default_asr(cli.engine) {
                cli.model.clone_from(model);
            } else {
                cli.model = parse_model_pass(&format!("{}={model}", engine_name(cli.engine)))
                    .map_err(anyhow::Error::msg)?
                    .model;
            }
        }
        configure_model(&mut cli);
        cli.offline = !args.download;
        let _cache_claims = shared_cache::hydrate(&cli, &args, &mut manifest)?;
        let outcomes = execute_tracks(&cli, &args, &manifest)?;
        check_cancelled()?;
        if manifest.tracks.iter().any(|track| track.recognize)
            && !outcomes
                .iter()
                .any(|outcome| outcome.coverage == Coverage::Recognized)
        {
            let reason = outcomes
                .iter()
                .find_map(|outcome| outcome.error.as_deref())
                .unwrap_or("no selected track produced a recognition artifact");
            anyhow::bail!("all selected tracks failed: {reason}");
        }
        let result = reconcile(&manifest, &outcomes)?;
        publish(&manifest.output, &result)?;
        Ok(outcomes)
    })();
    match result {
        Ok(outcomes) => {
            let failed = outcomes
                .iter()
                .filter(|item| item.coverage == Coverage::Failed)
                .count();
            manifest.event(
                &args,
                "session_finished",
                if failed == 0 { "completed" } else { "partial" },
                json!({"output":manifest.output,"failed_track_count":failed}),
            )
        }
        Err(error) => {
            manifest.event(
                &args,
                "session_finished",
                if cancel_requested() {
                    "canceled"
                } else {
                    "failed"
                },
                json!({"error":format!("{error:#}")}),
            )?;
            Err(error)
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ModelFailures<'a> {
    recognition: Option<&'a str>,
    diarization: Option<&'a str>,
}

fn execute_tracks(cli: &Cli, args: &SessionArgs, manifest: &Manifest) -> Result<Vec<Outcome>> {
    let needs_model = |track: &Track| {
        track.recognize && track.unavailable.is_none() && !track.reuse && !args.reconcile_only
    };
    if !manifest.tracks.iter().any(needs_model) {
        return process_tracks(cli, args, manifest, None, ModelFailures::default());
    }
    if let Some(language) = &manifest.language {
        ensure!(
            numbers::language_code(language).as_deref() == Some(RECOGNIZER_LANGUAGE),
            "this batch recognizer supports English; requested language was {language:?}"
        );
    }
    if let Some(count) = manifest.expected_speakers {
        ensure!(
            (1..=8).contains(&count),
            "expected_speakers must be between 1 and 8"
        );
    }
    let aligner = matches!(cli.engine, EngineKind::Qwen3);
    let model_label = model_display_name(&cli.model);
    let run = |diarization, failures: ModelFailures<'_>| {
        let requirements = RunRequirements {
            aligner,
            diarization,
        };
        // Nested result distinguishes an initialization failure from a run interrupted
        // after publishing recognized tracks.
        with_inference_models(cli, &requirements, &model_label, |models| {
            Ok(process_tracks(cli, args, manifest, Some(models), failures))
        })
    };
    let mut diarization_failure = None;
    if manifest
        .tracks
        .iter()
        .filter(|track| needs_model(track))
        .any(|track| manifest.diarizes(track))
    {
        match run(true, ModelFailures::default()) {
            Ok(result) => return result,
            Err(error) => {
                check_cancelled()?;
                diarization_failure =
                    Some(format!("requested diarization is unavailable: {error:#}"));
            }
        }
    }
    let mut failures = ModelFailures {
        diarization: diarization_failure.as_deref(),
        recognition: None,
    };
    if failures.diarization.is_some()
        && !manifest
            .tracks
            .iter()
            .filter(|track| needs_model(track))
            .any(|track| !manifest.diarizes(track))
    {
        return process_tracks(cli, args, manifest, None, failures);
    }
    match run(false, failures) {
        Ok(result) => result,
        Err(error) => {
            check_cancelled()?;
            let reason = format!("{error:#}");
            failures.recognition = Some(&reason);
            process_tracks(cli, args, manifest, None, failures)
        }
    }
}

fn process_tracks(
    cli: &Cli,
    args: &SessionArgs,
    manifest: &Manifest,
    mut models: Option<InferenceModels<'_>>,
    failures: ModelFailures<'_>,
) -> Result<Vec<Outcome>> {
    let mut outcomes = Vec::with_capacity(manifest.tracks.len());
    let mut completed: HashMap<&str, usize> = HashMap::new();
    for track in &manifest.tracks {
        check_cancelled()?;
        let report = &mut |progress: f32, phase: &str| {
            check_cancelled()?;
            manifest.event(
                args,
                "track_progress",
                "running",
                json!({
                    "track_id":track.track_id, "progress":progress, "phase":phase
                }),
            )
        };
        let prior = track
            .recognition_key
            .as_deref()
            .and_then(|key| completed.get(key))
            .and_then(|&index| outcomes.get(index))
            .and_then(|outcome: &Outcome| outcome.artifact.as_ref())
            .filter(|artifact| artifact.stream_evidence == track.stream_evidence);
        let result = if let Some(artifact) = prior.filter(|_| {
            track.recognize && track.unavailable.is_none() && !track.reuse && !args.reconcile_only
        }) {
            (|| {
                report(0.95, "Reusing identical track recognition")?;
                let mut artifact = artifact.clone();
                validate_artifact(track, &artifact)?;
                publish_activity(track, artifact.duration_samples, &artifact.activity)?;
                publish_recognition(track, &mut artifact)?;
                shared_cache::save(cli, track, &artifact)?;
                Ok(Outcome {
                    coverage: Coverage::Recognized,
                    artifact: Some(artifact),
                    error: None,
                })
            })()
        } else {
            let failure = failures.recognition.or_else(|| {
                manifest
                    .diarizes(track)
                    .then_some(failures.diarization)
                    .flatten()
            });
            process_track(
                cli,
                args,
                manifest,
                track,
                models.as_mut().map(InferenceModels::reborrow),
                failure,
                report,
            )
        };
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(error) => {
                if cancel_requested() {
                    manifest.event(
                        args,
                        "track_finished",
                        "canceled",
                        json!({"track_id":track.track_id}),
                    )?;
                    return Err(error);
                }
                Outcome {
                    coverage: Coverage::Failed,
                    artifact: load_failed_activity(track),
                    error: Some(format!("{error:#}")),
                }
            }
        };
        manifest.event(args, "track_finished", outcome.coverage.name(), json!({
            "track_id":track.track_id, "output":track.result, "activity":track.activity, "error":outcome.error,
            "warning":outcome.artifact.as_ref().and_then(|artifact| artifact.warning.as_deref()),
        }))?;
        if outcome.coverage == Coverage::Recognized
            && let Some(key) = track.recognition_key.as_deref()
        {
            completed.insert(key, outcomes.len());
        }
        outcomes.push(outcome);
    }
    Ok(outcomes)
}

fn process_track(
    cli: &Cli,
    args: &SessionArgs,
    manifest: &Manifest,
    track: &Track,
    models: Option<InferenceModels<'_>>,
    model_failure: Option<&str>,
    report: &mut dyn FnMut(f32, &str) -> Result<()>,
) -> Result<Outcome> {
    if let Some(reason) = &track.unavailable {
        return Ok(Outcome {
            coverage: Coverage::Failed,
            artifact: load_failed_activity(track),
            error: Some(reason.clone()),
        });
    }
    if track.binding.stream_index.is_none() {
        return Ok(Outcome {
            coverage: Coverage::NoAudio,
            artifact: None,
            error: None,
        });
    }
    let result = track.result.as_deref().context("no track result path")?;
    if track.recognize && (args.reconcile_only || track.reuse) {
        let mut artifact: TrackArtifact = read_json_file(result)?;
        validate_artifact(track, &artifact)?;
        if let Some(path) = track.activity.as_deref() {
            let activity = read_activity(track, path)?;
            ensure!(
                activity.duration_samples == artifact.duration_samples,
                "activity and recognition durations differ"
            );
            artifact.activity = activity.activity;
        }
        return Ok(Outcome {
            coverage: Coverage::Recognized,
            artifact: Some(artifact),
            error: None,
        });
    }
    if !track.recognize && (args.reconcile_only || track.audio.is_none()) {
        let artifact = load_skipped_activity(track)?;
        return Ok(Outcome {
            coverage: Coverage::Skipped,
            artifact,
            error: None,
        });
    }
    if track.recognize
        && track.audio.is_none()
        && let Some(reason) = model_failure
    {
        anyhow::bail!("{reason}");
    }
    report(0.0, "Reading extracted PCM")?;
    let path = track
        .audio
        .as_deref()
        .context("recognition needs an extracted PCM WAV")?;
    let audio = read_pcm(path)?;
    let duration_samples = i64::try_from(audio.samples.len())?;
    let activity = activity_levels(&audio.samples)?;
    publish_activity(track, duration_samples, &activity)?;
    let mut artifact = new_artifact(track, duration_samples, activity);
    if track.recognize && artifact.activity.iter().any(|frame| frame.level > 0.0) {
        if let Some(reason) = model_failure {
            return Ok(Outcome {
                coverage: Coverage::Failed,
                artifact: Some(artifact),
                error: Some(reason.to_owned()),
            });
        }
        let job = Job {
            input: path.to_owned(),
            parts: Vec::new(),
            iso: false,
            zoom_recording: false,
            zoom_audio_copy: None,
            diarize: manifest.diarizes(track),
            speakers: if track.role == Role::IsolatedMic {
                None
            } else {
                manifest.expected_speakers
            },
            output: None,
            planned_output: None,
            output_dir: PathBuf::new(),
            output_location: OutputLocation::Folder,
        };
        let (document, warning) = if matches!(cli.engine, EngineKind::AppleNative) {
            process_apple_native(
                cli,
                &job,
                None,
                models
                    .context("recognition models are unavailable")?
                    .diarizer,
                Some(audio),
                report,
            )?
        } else {
            transcribe_audio(
                cli,
                &job,
                models.context("recognition models are unavailable")?,
                None,
                audio,
                report,
            )?
        };
        artifact.words = local_words(&document, duration_samples)?;
        artifact.warning = warning;
    } else if track.recognize {
        artifact.warning = Some("No speech detected; the transcript is empty".to_owned());
    }
    validate_artifact(track, &artifact)?;
    if track.recognize {
        publish_recognition(track, &mut artifact)?;
        shared_cache::save(cli, track, &artifact)?;
    }
    Ok(Outcome {
        coverage: if track.recognize {
            Coverage::Recognized
        } else {
            Coverage::Skipped
        },
        artifact: Some(artifact),
        error: None,
    })
}

fn publish_activity(track: &Track, duration_samples: i64, activity: &[Activity]) -> Result<()> {
    if let Some(path) = track.activity.as_deref() {
        publish(
            path,
            &ActivityArtifact {
                version: VERSION,
                sample_rate: TARGET_SAMPLE_RATE,
                stream_evidence: track.stream_evidence.clone(),
                duration_samples,
                activity,
            },
        )?;
    }
    Ok(())
}

fn publish_recognition(track: &Track, artifact: &mut TrackArtifact) -> Result<()> {
    let result = track.result.as_deref().context("no track result path")?;
    if track.activity.is_some() {
        let activity = std::mem::take(&mut artifact.activity);
        let result = publish(result, artifact);
        artifact.activity = activity;
        result
    } else {
        publish(result, artifact)
    }
}

fn new_artifact(track: &Track, duration_samples: i64, activity: Vec<Activity>) -> TrackArtifact {
    TrackArtifact {
        version: VERSION,
        sample_rate: TARGET_SAMPLE_RATE,
        recognition_key: track.recognition_key.clone(),
        stream_evidence: track.stream_evidence.clone(),
        duration_samples,
        words: Vec::new(),
        activity,
        warning: None,
    }
}

fn load_skipped_activity(track: &Track) -> Result<Option<TrackArtifact>> {
    if let Some(path) = track.activity.as_deref().filter(|path| path.exists()) {
        let activity = read_activity(track, path)?;
        return Ok(Some(new_artifact(
            track,
            activity.duration_samples,
            activity.activity,
        )));
    }
    if let Some(path) = track.result.as_deref().filter(|path| path.exists()) {
        let mut artifact: TrackArtifact = read_json_file(path)?;
        validate_artifact(track, &artifact)?;
        artifact.words.clear();
        return Ok(Some(artifact));
    }
    Ok(None)
}

fn load_failed_activity(track: &Track) -> Option<TrackArtifact> {
    let artifact = read_activity(track, track.activity.as_deref()?).ok()?;
    Some(new_artifact(
        track,
        artifact.duration_samples,
        artifact.activity,
    ))
}

fn read_activity(track: &Track, path: &Path) -> Result<ActivityArtifact> {
    let artifact: ActivityArtifact = read_json_file(path)?;
    ensure!(
        artifact.version == VERSION
            && artifact.sample_rate == TARGET_SAMPLE_RATE
            && artifact.stream_evidence == track.stream_evidence,
        "activity artifact identity mismatch"
    );
    validate_activity(artifact.duration_samples, &artifact.activity)?;
    Ok(artifact)
}

fn validate_activity(duration: i64, activity: &[Activity]) -> Result<()> {
    ensure!(duration >= 0, "negative audio duration");
    let mut end = 0;
    for frame in activity {
        ensure!(
            frame.start_sample == end
                && end < duration
                && frame.end_sample == end.saturating_add(ACTIVITY_WINDOW as i64).min(duration)
                && (0.0..=1.0).contains(&frame.level),
            "invalid level frame"
        );
        end = frame.end_sample;
    }
    ensure!(end == duration, "incomplete activity frame coverage");
    Ok(())
}

fn validate_artifact(track: &Track, artifact: &TrackArtifact) -> Result<()> {
    ensure!(
        artifact.version == VERSION
            && artifact.sample_rate == TARGET_SAMPLE_RATE
            && artifact.recognition_key == track.recognition_key
            && artifact.stream_evidence == track.stream_evidence,
        "recognition artifact identity mismatch"
    );
    ensure!(
        artifact.duration_samples >= 0
            && track
                .start_frame
                .checked_add(artifact.duration_samples)
                .is_some(),
        "invalid artifact duration or session placement"
    );
    if track.activity.is_none() || !artifact.activity.is_empty() {
        validate_activity(artifact.duration_samples, &artifact.activity)?;
    }
    let mut last_start = 0;
    let mut ids = HashSet::new();
    for word in &artifact.words {
        ensure!(
            !word.id.is_empty()
                && ids.insert(&word.id)
                && !word.utterance_id.is_empty()
                && !word.text.is_empty()
                && word.speaker_id.as_ref().is_none_or(|id| !id.is_empty())
                && word.start_sample >= last_start
                && word.end_sample > word.start_sample
                && word.end_sample <= artifact.duration_samples,
            "invalid track word"
        );
        last_start = word.start_sample;
    }
    Ok(())
}

fn local_words(
    document: &transcript::TranscriptDocument,
    duration_samples: i64,
) -> Result<Vec<LocalWord>> {
    document.validate()?;
    let mut words = Vec::new();
    for (segment_index, segment) in document.segments.iter().enumerate() {
        ensure!(
            segment
                .text
                .as_deref()
                .is_none_or(|text| text.trim().is_empty()),
            "recognizer produced segment-only speech"
        );
        for word in &segment.words {
            let sample =
                |seconds: f32| (f64::from(seconds) * f64::from(TARGET_SAMPLE_RATE)).round() as i64;
            let start_sample =
                sample(word.start).clamp(0, duration_samples.saturating_sub(1).max(0));
            let end_sample =
                sample(word.end).clamp((start_sample + 1).min(duration_samples), duration_samples);
            ensure!(
                end_sample > start_sample,
                "recognizer returned a word on empty audio"
            );
            words.push(LocalWord {
                id: format!("w{}", words.len() + 1),
                utterance_id: format!("u{}", segment_index + 1),
                text: word.text.clone(),
                start_sample,
                end_sample,
                speaker_id: word.speaker.map(|id| format!("s{id}")),
                precision: Precision::Word,
            });
        }
    }
    words.sort_by_key(|word| word.start_sample);
    Ok(words)
}

fn activity_levels(samples: &[f32]) -> Result<Vec<Activity>> {
    let mut activity = Vec::with_capacity(samples.len().div_ceil(ACTIVITY_WINDOW));
    for (index, slice) in samples.chunks(ACTIVITY_WINDOW).enumerate() {
        if index % 500 == 0 {
            check_cancelled()?;
        }
        let start = index * ACTIVITY_WINDOW;
        let sum: f32 = slice.iter().map(|sample| sample * sample).sum();
        activity.push(Activity {
            start_sample: start as i64,
            end_sample: (start + slice.len()) as i64,
            level: (sum / slice.len() as f32).sqrt(),
        });
    }
    Ok(activity)
}

fn read_pcm(path: &Path) -> Result<Audio> {
    let mut input = BufReader::new(
        File::open(path).with_context(|| format!("opening PCM {}", path.display()))?,
    );
    let mut header = [0; 12];
    input.read_exact(&mut header)?;
    ensure!(
        &header[..4] == b"RIFF" && &header[8..] == b"WAVE",
        "expected a RIFF PCM WAV: {}",
        path.display()
    );
    let end = 8 + u64::from(u32::from_le_bytes(header[4..8].try_into()?));
    ensure!(
        end <= input.get_ref().metadata()?.len(),
        "truncated PCM WAV: {}",
        path.display()
    );
    let mut format = false;
    let mut data = None;
    while input.stream_position()? + 8 <= end {
        check_cancelled()?;
        let mut chunk = [0; 8];
        input.read_exact(&mut chunk)?;
        let size = u64::from(u32::from_le_bytes(chunk[4..].try_into()?));
        let start = input.stream_position()?;
        ensure!(start + size + (size % 2) <= end, "invalid WAV chunk size");
        match &chunk[..4] {
            b"fmt " => {
                ensure!(!format && size >= 16, "invalid WAV format chunk");
                let mut fmt = [0; 16];
                input.read_exact(&mut fmt)?;
                ensure!(
                    u16::from_le_bytes(fmt[0..2].try_into()?) == 1
                        && u16::from_le_bytes(fmt[2..4].try_into()?) == 1
                        && u32::from_le_bytes(fmt[4..8].try_into()?) == TARGET_SAMPLE_RATE
                        && u32::from_le_bytes(fmt[8..12].try_into()?) == TARGET_SAMPLE_RATE * 2
                        && u16::from_le_bytes(fmt[12..14].try_into()?) == 2
                        && u16::from_le_bytes(fmt[14..16].try_into()?) == 16,
                    "expected 16 kHz mono signed 16-bit PCM WAV"
                );
                format = true;
            }
            b"data" => {
                ensure!(data.is_none() && size % 2 == 0, "invalid WAV sample data");
                data = Some((start, size as usize));
            }
            _ => (),
        }
        input.seek(SeekFrom::Start(start + size + (size % 2)))?;
    }
    ensure!(format, "WAV has no format chunk");
    let (start, size) = data.context("WAV has no sample data")?;
    input.seek(SeekFrom::Start(start))?;
    let mut samples = Vec::with_capacity(size / 2);
    let mut block = [0; 32_768];
    let mut remaining = size;
    while remaining > 0 {
        check_cancelled()?;
        let count = block.len().min(remaining);
        input.read_exact(&mut block[..count])?;
        samples.extend(
            block[..count]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&pair| f32::from(i16::from_le_bytes(pair)) / 32768.0),
        );
        remaining -= count;
    }
    let duration_seconds = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
    Ok(Audio {
        samples,
        duration_seconds,
        warning: None,
    })
}

fn read_json_file<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let read = || -> Result<T> { Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?) };
    read().with_context(|| format!("reading {}", path.display()))
}

fn protect_output_paths(outputs: &[&Path], inputs: &[&Path]) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let identity = |path: &Path| fs::metadata(path).ok().map(|info| (info.dev(), info.ino()));
    let key = |path: &Path| crate::folded_path(&crate::input_key(path));
    let mut keys: HashSet<_> = inputs.iter().map(|&path| key(path)).collect();
    let mut identities: HashSet<_> = inputs.iter().filter_map(|&path| identity(path)).collect();
    for &path in outputs {
        ensure!(
            path.file_name().is_some()
                && keys.insert(key(path))
                && identity(path).is_none_or(|id| identities.insert(id)),
            "session output overlaps another input or output: {}",
            path.display()
        );
    }
    Ok(())
}
