//! Per-track cache exchange for MPCut's standalone handoff. The provider owns the
//! effective recognition key, so staging a manifest never needs to invoke ASR.
use super::*;
use crate::{DeviceKind, TimestampGranularity, apple_native_helper, model_cache};
use base64::{Engine, engine::general_purpose::STANDARD};
use ring::digest::{Context as DigestContext, SHA256, digest};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

const ALIGNER: &str = "Qwen/Qwen3-ForcedAligner-0.6B";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Descriptor {
    source: Snapshot,
    audio: Option<Snapshot>,
    scope: Option<String>,
    directory: Option<PathBuf>,
    write: bool,
    restart: bool,
    #[serde(skip)]
    exchange: Option<PathBuf>,
    #[serde(skip)]
    model_revisions: Vec<(String, String)>,
    #[serde(skip)]
    settings: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    path: PathBuf,
    stat_values: Vec<String>,
}

impl Snapshot {
    fn is_current(&self) -> bool {
        std::fs::metadata(&self.path)
            .ok()
            .filter(std::fs::Metadata::is_file)
            .is_some_and(|metadata| stat_values(&metadata) == self.stat_values)
    }
}

fn stat_values(metadata: &std::fs::Metadata) -> Vec<String> {
    [
        metadata.dev().to_string(),
        metadata.ino().to_string(),
        metadata.size().to_string(),
        metadata.mtime().to_string(),
        metadata.mtime_nsec().to_string(),
        metadata.ctime().to_string(),
        metadata.ctime_nsec().to_string(),
    ]
    .into()
}

fn hex(data: &[u8]) -> String {
    data.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hash(data: &[u8]) -> String {
    hex(digest(&SHA256, data).as_ref())
}

fn executable_revision(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = vec![0; 128 * 1024];
    loop {
        check_cancelled()?;
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            return Ok(hex(context.finish().as_ref()));
        }
        context.update(&buffer[..count]);
    }
}

pub(crate) fn provider_revision() -> Option<String> {
    executable_revision(&std::env::current_exe().ok()?).ok()
}

pub(crate) fn apple_revision() -> Option<String> {
    let helper = apple_native_helper()?;
    let build = std::process::Command::new("/usr/bin/sw_vers")
        .arg("-buildVersion")
        .output()
        .ok()?;
    let os = std::str::from_utf8(&build.stdout).ok()?.trim();
    if !build.status.success() || os.is_empty() {
        return None;
    }
    let helper = executable_revision(&helper).ok()?;
    Some(format!("{os}:{helper}"))
}

/// Installed-model revision from the actual paths resolved by the provider.
/// Including target identity and ctime also invalidates an edited Hub snapshot.
pub(crate) fn revision(path: &Path) -> Result<String> {
    let mut inventory = Vec::new();
    let resolved = path.canonicalize()?;
    if resolved.is_file() {
        inventory.push((resolved.clone(), stat_values(&resolved.metadata()?)));
    } else {
        for entry in walkdir::WalkDir::new(&resolved).follow_links(true) {
            check_cancelled()?;
            let entry = entry?;
            if entry.file_type().is_file() {
                inventory.push((entry.path().to_owned(), stat_values(&entry.metadata()?)));
            }
        }
    }
    ensure!(!inventory.is_empty(), "model contains no readable files");
    inventory.sort_unstable();
    Ok(hash(&serde_json::to_vec(&inventory)?))
}

fn model_revision(cache: &model_cache::ModelCache, id: &str) -> Option<String> {
    revision(&cache.lookup(id)?).ok()
}

fn models_are_current(cache: &model_cache::ModelCache, descriptor: &Descriptor) -> bool {
    descriptor
        .model_revisions
        .iter()
        .all(|(id, expected)| model_revision(cache, id).as_ref() == Some(expected))
}

fn snapshots_are_current(descriptor: &Descriptor, track: &Track) -> bool {
    descriptor.source.is_current()
        && descriptor.audio.as_ref().is_none_or(|audio| {
            track.audio.as_deref() == Some(audio.path.as_path())
                && (audio.is_current() || !audio.path.exists())
        })
}

#[derive(Deserialize, Serialize)]
struct Bundle {
    result: String,
    activity: Option<String>,
    digest: String,
}

#[derive(Deserialize, Serialize)]
struct Provenance {
    settings: String,
    recognition_key: String,
    digest: String,
}

fn artifacts_digest(result: &[u8], activity: Option<&[u8]>) -> String {
    let mut hash = DigestContext::new(&SHA256);
    hash.update(result);
    hash.update(&[0]);
    if let Some(activity) = activity {
        hash.update(activity);
    }
    hex(hash.finish().as_ref())
}

fn provenance_path(track: &Track) -> Result<PathBuf> {
    Ok(track
        .result
        .as_ref()
        .context("missing result path")?
        .with_added_extension("provenance"))
}

fn artifact_bytes(track: &Track) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    Ok((
        fs::read(
            track
                .result
                .as_deref()
                .context("missing recognition output")?,
        )?,
        track.activity.as_ref().map(fs::read).transpose()?,
    ))
}

fn local_key(track: &Track, settings: &str, expected_key: Option<&str>) -> Result<String> {
    let receipt: Provenance = read_json_file(&provenance_path(track)?)?;
    ensure!(
        receipt.settings == settings
            && valid_digest(&receipt.recognition_key)
            && expected_key.is_none_or(|key| key == receipt.recognition_key),
        "local recognition provenance changed"
    );
    let (result, activity) = artifact_bytes(track)?;
    ensure!(
        artifacts_digest(&result, activity.as_deref()) == receipt.digest,
        "local recognition artifact changed"
    );
    Ok(receipt.recognition_key)
}

fn save_provenance(track: &Track, settings: &str, digest: String) -> Result<()> {
    let recognition_key = track
        .recognition_key
        .clone()
        .context("missing recognition key")?;
    publish(
        &provenance_path(track)?,
        &Provenance {
            settings: settings.to_owned(),
            recognition_key,
            digest,
        },
    )
}

/// Dropping the file handles releases every cross-process claim, including when
/// session execution is canceled. Sort by ID to match MPCut's lock order.
pub(super) fn hydrate(cli: &Cli, args: &SessionArgs, manifest: &mut Manifest) -> Result<Vec<File>> {
    if !manifest
        .tracks
        .iter()
        .any(|track| track.shared_cache.is_some())
    {
        return Ok(Vec::new());
    }
    let mut locks = Vec::new();
    let mut paths = Vec::new();
    let provider_revision = provider_revision();
    let apple_revision = matches!(cli.engine, EngineKind::AppleNative)
        .then(apple_revision)
        .flatten();
    let defaults = matches!(cli.device, DeviceKind::Metal)
        && cli.timestamps == TimestampGranularity::Words
        && cli.chunk_seconds.is_none()
        && cli.forced_aligner == ALIGNER
        && cli.diarization_model.is_none()
        && !cli.diarize
        && cli.speakers.is_none();
    let cache = model_cache::ModelCache::new(cli.model_cache.clone(), true);
    let mut revisions = HashMap::<String, Option<String>>::new();
    let diarization_enabled = manifest.diarization_enabled();
    for track in &mut manifest.tracks {
        let Some(descriptor) = track.shared_cache.as_ref() else {
            continue;
        };
        if !snapshots_are_current(descriptor, track) {
            track.unavailable =
                Some("Media or staged audio changed. Prepare the session again in MPCut.".into());
            continue;
        }
        let Some(provider_revision) = provider_revision
            .as_deref()
            .filter(|_| defaults && !args.download)
        else {
            continue;
        };
        if manifest.language.as_ref().is_some_and(|language| {
            super::numbers::language_code(language).as_deref() != Some(super::RECOGNIZER_LANGUAGE)
        }) {
            continue;
        }
        let effective_diarization = track.role != Role::IsolatedMic && diarization_enabled;
        let requirements = RunRequirements {
            aligner: matches!(cli.engine, EngineKind::Qwen3),
            diarization: effective_diarization,
        };
        // A valid cache hit still has to obey the selected engine's capabilities.
        if requirements.validate(cli).is_err() {
            continue;
        }
        let Some(evidence) = track.stream_evidence.as_ref().and_then(Value::as_str) else {
            continue;
        };
        let versions = requirements
            .models(cli)
            .into_iter()
            .map(|(_, id)| {
                revisions
                    .entry(id.to_owned())
                    .or_insert_with(|| model_revision(&cache, id))
                    .clone()
                    .map(|value| (id.to_owned(), value))
            })
            .collect::<Option<Vec<_>>>();
        let platform = if matches!(cli.engine, EngineKind::AppleNative) {
            let Some(value) = apple_revision.as_deref() else {
                continue;
            };
            value
        } else {
            ""
        };
        let speakers = manifest
            .expected_speakers
            .filter(|_| effective_diarization)
            .map(|count| count.to_string())
            .unwrap_or_default();
        let channel = track
            .channel
            .map(|number| number.to_string())
            .unwrap_or_default();
        let stream = track
            .binding
            .stream_index
            .map(|number| number.to_string())
            .unwrap_or_default();
        let mut values = vec![
            "pcm-v1",
            "session-v1",
            provider_revision,
            engine_name(cli.engine),
            &cli.model,
            "en",
            if effective_diarization {
                "true"
            } else {
                "false"
            },
            &speakers,
            platform,
            evidence,
            &channel,
            &stream,
        ];
        let settings = hash(values.join("\0").as_bytes());
        let key = if let Some(versions) = versions.as_ref() {
            values.extend(versions.iter().map(|(_, value)| value.as_str()));
            hash(values.join("\0").as_bytes())
        } else if args.reconcile_only && !descriptor.restart {
            let Ok(key) = local_key(track, &settings, None) else {
                continue;
            };
            key
        } else {
            continue;
        };
        track.recognition_key = Some(key.clone());
        let descriptor = track.shared_cache.as_mut().expect("standalone descriptor");
        descriptor.settings = Some(settings);
        descriptor.model_revisions = versions.unwrap_or_default();
        if let (Some(scope), Some(directory)) = (&descriptor.scope, &descriptor.directory) {
            ensure!(
                valid_digest(scope) && directory.is_absolute(),
                "invalid recognition cache location"
            );
            let id = hash(format!("{scope}\0{key}").as_bytes());
            let lock_path = directory.join(&id);
            let exchange = directory.join(format!("{id}.json"));
            descriptor.exchange = Some(exchange);
            paths.push(lock_path);
        }
    }
    paths.sort_unstable();
    paths.dedup();
    for path in paths {
        check_cancelled()?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&path)
            .context("opening shared recognition claim")?;
        while unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = std::io::Error::last_os_error();
            if !matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ) {
                return Err(error).context("claiming shared recognition cache");
            }
            check_cancelled()?;
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        locks.push(lock);
    }
    for track in &mut manifest.tracks {
        check_cancelled()?;
        let Some(descriptor) = track.shared_cache.as_ref() else {
            continue;
        };
        if !snapshots_are_current(descriptor, track) {
            track.unavailable =
                Some("Media or staged audio changed. Prepare the session again in MPCut.".into());
            continue;
        }
        if track.recognition_key.is_some() && !models_are_current(&cache, descriptor) {
            track.unavailable = Some(
                "An installed model changed. Rerun the session with the current models.".into(),
            );
            continue;
        }
        if track.unavailable.is_some() {
            continue;
        }
        if let (false, Some(settings), Some(key)) = (
            descriptor.restart,
            descriptor.settings.as_deref(),
            track.recognition_key.as_deref(),
        ) {
            if local_key(track, settings, Some(key)).is_ok() && validated_local(track).is_ok() {
                track.reuse = true;
                continue;
            }
            if let Some(path) = descriptor.exchange.as_deref()
                && let Ok(bundle) = read_json_file::<Bundle>(path)
                && let Ok((result, activity)) = validated_bundle(track, &bundle)
            {
                let output = track
                    .result
                    .as_deref()
                    .context("missing recognition output")?;
                transcript::publish_atomic(output, true, |writer| {
                    std::io::Write::write_all(writer, &result)
                })?;
                if let (Some(path), Some(activity)) = (&track.activity, activity.as_deref()) {
                    transcript::publish_atomic(path, true, |writer| {
                        std::io::Write::write_all(writer, activity)
                    })?;
                }
                // The exchange may include levels even when this session only
                // requested a transcript. Record exactly the files restored here.
                let digest = if track.activity.is_none() && activity.is_some() {
                    artifacts_digest(&result, None)
                } else {
                    bundle.digest
                };
                save_provenance(track, settings, digest)?;
                track.reuse = true;
                continue;
            }
        }
        if args.reconcile_only {
            track.unavailable = Some("No verified recognition artifact for this track. Run the prepared offline command first.".into());
        }
    }
    Ok(locks)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validated_local(track: &Track) -> Result<TrackArtifact> {
    let artifact: TrackArtifact =
        read_json_file(track.result.as_deref().context("missing result")?)?;
    validate_artifact(track, &artifact)?;
    if let Some(path) = &track.activity {
        let activity = read_activity(track, path)?;
        ensure!(
            activity.duration_samples == artifact.duration_samples,
            "recognition and levels have different durations"
        );
    }
    Ok(artifact)
}

fn validated_bundle(track: &Track, bundle: &Bundle) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    let result = STANDARD.decode(&bundle.result)?;
    let activity = bundle
        .activity
        .as_ref()
        .map(|encoded| STANDARD.decode(encoded))
        .transpose()?;
    ensure!(
        artifacts_digest(&result, activity.as_deref()) == bundle.digest,
        "cache artifact changed"
    );
    let artifact: TrackArtifact = serde_json::from_slice(&result)?;
    validate_artifact(track, &artifact)?;
    if track.activity.is_some() {
        let activity: ActivityArtifact =
            serde_json::from_slice(activity.as_deref().context("cache entry lacks levels")?)?;
        ensure!(
            activity.version == VERSION
                && activity.sample_rate == TARGET_SAMPLE_RATE
                && activity.stream_evidence == track.stream_evidence
                && activity.duration_samples == artifact.duration_samples,
            "cache levels have different provenance"
        );
        validate_activity(activity.duration_samples, &activity.activity)?;
    }
    Ok((result, activity))
}

pub(super) fn save(cli: &Cli, track: &Track, _artifact: &TrackArtifact) -> Result<()> {
    let Some(descriptor) = track.shared_cache.as_ref() else {
        return Ok(());
    };
    let Some(settings) = descriptor.settings.as_deref() else {
        return Ok(());
    };
    if track.reuse
        || !descriptor.source.is_current()
        || !descriptor.audio.as_ref().is_some_and(Snapshot::is_current)
        || !models_are_current(
            &model_cache::ModelCache::new(cli.model_cache.clone(), true),
            descriptor,
        )
    {
        return check_cancelled();
    }
    let result: Result<()> = (|| {
        validated_local(track)?;
        let (result, activity) = artifact_bytes(track)?;
        let digest = artifacts_digest(&result, activity.as_deref());
        save_provenance(track, settings, digest.clone())?;
        if let Some(path) = descriptor.exchange.as_deref().filter(|_| descriptor.write) {
            if let Ok(bundle) = read_json_file::<Bundle>(path)
                && validated_bundle(track, &bundle).is_ok()
            {
                return Ok(());
            }
            publish(
                path,
                &Bundle {
                    result: STANDARD.encode(result),
                    activity: activity.map(|value| STANDARD.encode(value)),
                    digest,
                },
            )?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        check_cancelled()?;
        eprintln!(
            "warning: could not share recognition for {}: {error:#}",
            track.name
        );
    }
    Ok(())
}
