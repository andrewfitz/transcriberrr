//! Local model cache and Hugging Face model acquisition.
//!
//! Model weights are intentionally kept outside the executable and source tree.  The cache uses
//! the standard Hugging Face snapshot layout, so downloads are resumable and shared by CLI/UI
//! processes without duplicating files.

use anyhow::{Context, Result, bail};
use hf_hub::api::Progress;
use hf_hub::api::sync::{Api, ApiBuilder, ApiError, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};
use ring::digest::{Context as DigestContext, SHA1_FOR_LEGACY_USE_ONLY, SHA256};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const QWEN3_ASR_17B: &str = "Qwen/Qwen3-ASR-1.7B";
pub const QWEN3_ASR_06B: &str = "Qwen/Qwen3-ASR-0.6B";
pub const QWEN3_ALIGNER_06B: &str = "Qwen/Qwen3-ForcedAligner-0.6B";
/// The repository that publishes the ONNX exports parakeet-rs generates its Burn graphs from.
/// A Burn graph loads only the export it was generated from, so every Parakeet model comes
/// from one revision of this repository. Change the revision together with the parakeet-rs
/// version in `Cargo.toml`, then run `examples/parakeet_parity.rs`.
const PARAKEET_RS_REPO: &str = "altunenes/parakeet-rs";
const PARAKEET_RS_REVISION: &str = "26684938ff242cb4f7a65fad712476c6015a4477";
/// English Parakeet Unified RNN-T.
pub const PARAKEET_UNIFIED_EN: &str = "altunenes/parakeet-rs/parakeet-unified-en-0.6b";
/// Multilingual Parakeet TDT v3, for explicit `--model` overrides.
pub const PARAKEET_TDT_V3: &str = "altunenes/parakeet-rs/parakeet-tdt-0.6b-v3";
/// Sortformer v3 (Nemotron-3) diarization.
pub const SORTFORMER_4SPK: &str = "altunenes/parakeet-rs/nemotron-3-diarization";

/// One model folder of the pinned parakeet-rs repository. Its ID is the repository followed
/// by the folder, so several models share the repository's cache directory.
struct PinnedModel {
    id: &'static str,
    folder: &'static str,
    files: &'static [&'static str],
    /// The file inference loads, or `None` when it loads the folder.
    entry: Option<&'static str>,
}

const PINNED_MODELS: &[PinnedModel] = &[
    PinnedModel {
        id: PARAKEET_UNIFIED_EN,
        folder: "parakeet-unified-en-0.6b",
        files: &[
            "encoder.onnx",
            "encoder.onnx.data",
            "decoder_joint.onnx",
            "tokenizer.model",
        ],
        entry: None,
    },
    PinnedModel {
        id: PARAKEET_TDT_V3,
        folder: "parakeet-tdt-0.6b-v3",
        files: &[
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
            "vocab.txt",
        ],
        entry: None,
    },
    PinnedModel {
        id: SORTFORMER_4SPK,
        folder: "nemotron-3-diarization",
        files: &["nemotron3_diar_v3.onnx"],
        entry: Some("nemotron3_diar_v3.onnx"),
    },
];

fn pinned_model(id: &str) -> Option<&'static PinnedModel> {
    PINNED_MODELS.iter().find(|model| model.id == id)
}

const SNAPSHOT_MANIFEST: &str = ".transcriberrr-snapshot.json";
const INCOMPLETE_MARKER: &str = ".transcriberrr-incomplete";

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ModelSpec {
    pub id: &'static str,
    pub kind: &'static str,
    pub description: &'static str,
}

pub const SUPPORTED_MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: QWEN3_ASR_17B,
        kind: "asr",
        description: "Qwen3 English ASR 1.7B (recommended)",
    },
    ModelSpec {
        id: QWEN3_ASR_06B,
        kind: "asr",
        description: "Qwen3 English ASR 0.6B (smaller/faster)",
    },
    ModelSpec {
        id: QWEN3_ALIGNER_06B,
        kind: "aligner",
        description: "Qwen3 word-level forced aligner 0.6B",
    },
    ModelSpec {
        id: PARAKEET_UNIFIED_EN,
        kind: "asr",
        description: "Parakeet Unified English 0.6B ONNX",
    },
    ModelSpec {
        id: PARAKEET_TDT_V3,
        kind: "asr",
        description: "Parakeet TDT v3 0.6B ONNX (multilingual)",
    },
    ModelSpec {
        id: SORTFORMER_4SPK,
        kind: "diarization",
        description: "Sortformer streaming 4-speaker diarization",
    },
];

/// A cache manager. `root` is the Hugging Face `hub` directory itself.
#[derive(Debug, Clone)]
pub struct ModelCache {
    root: PathBuf,
    offline: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub kind: String,
    pub description: String,
    pub cached: bool,
    pub path: Option<String>,
}

impl ModelCache {
    pub fn new(root: Option<PathBuf>, offline: bool) -> Self {
        let root = root.unwrap_or_else(default_cache_dir);
        Self {
            root: expand_home(&root),
            offline,
        }
    }

    /// Resolve a model ID or local directory.  A remote ID is downloaded into this cache and the
    /// local immutable snapshot path is returned to inference libraries.
    pub fn ensure(&self, id_or_path: &str) -> Result<PathBuf> {
        // Accept shell-style home paths even when the value came from a repeated
        // --download-model argument or a manifest rather than a top-level CLI path.
        let expanded = expand_home(Path::new(id_or_path));
        if expanded.exists() {
            if expanded.is_dir() {
                return Ok(expanded);
            }
            bail!("model path is not a directory: {}", expanded.display());
        }

        let id = canonical_id(&expanded.to_string_lossy());
        if let Some(snapshot) = self.cached_path(&id) {
            return Ok(snapshot);
        }
        self.fetch(&id, false)
    }

    /// Re-check a Hub model against the repository listing and download missing or damaged files.
    /// Returns `None` for local models and offline runs.
    pub fn repair(&self, id_or_path: &str) -> Result<Option<PathBuf>> {
        let expanded = expand_home(Path::new(id_or_path));
        if self.offline || expanded.exists() {
            return Ok(None);
        }
        self.fetch(&canonical_id(&expanded.to_string_lossy()), true)
            .map(Some)
    }

    /// Offline lookup used by `--cached-model` so the app asks the CLI instead of
    /// re-implementing aliases, markers, and snapshot resolution.
    pub fn lookup(&self, id_or_path: &str) -> Option<PathBuf> {
        let expanded = expand_home(Path::new(id_or_path));
        if expanded.exists() {
            return Some(expanded);
        }
        self.cached_path(&canonical_id(&expanded.to_string_lossy()))
    }

    /// Download the selected assets from one immutable revision. Publish `main` only after
    /// validating that revision, so a repository update cannot mix files from two snapshots.
    fn fetch(&self, id: &str, repair: bool) -> Result<PathBuf> {
        if !is_model_id(id) {
            bail!("model path does not exist or model ID is invalid: {id:?}");
        }
        if self.offline {
            bail!(
                "model {id:?} is not available locally (--offline); prefetch it with --download-model or supply a local model path"
            );
        }

        let directory = self.repo_dir(id);
        let _lock = lock_repository(&directory, id)?;
        if !repair && let Some(snapshot) = self.cached_path(id) {
            return Ok(snapshot);
        }

        let api = ApiBuilder::from_cache(Cache::new(self.root.clone()))
            .build()
            .context("creating Hugging Face client")?;
        self.download_snapshot(id, repair, &directory, &api)
    }

    fn download_snapshot(
        &self,
        id: &str,
        repair: bool,
        directory: &Path,
        api: &Api,
    ) -> Result<PathBuf> {
        if let Some(model) = pinned_model(id) {
            let repo = api.repo(Repo::with_revision(
                PARAKEET_RS_REPO.to_owned(),
                RepoType::Model,
                PARAKEET_RS_REVISION.to_owned(),
            ));
            let snapshot = directory.join("snapshots").join(PARAKEET_RS_REVISION);
            let files: Vec<String> = model
                .files
                .iter()
                .map(|file| format!("{}/{file}", model.folder))
                .collect();
            let files: Vec<&str> = files.iter().map(String::as_str).collect();
            // Each folder has its own marker and manifest, so fetching one model leaves
            // the repository's other models usable.
            let state = snapshot.join(model.folder);
            return download_files(&repo, id, &snapshot, &state, &files, repair);
        }
        let info = api
            .repo(Repo::new(id.to_owned(), RepoType::Model))
            .info()
            .with_context(|| format!("reading Hugging Face repository {id:?}"))?;
        if info.sha.len() != 40 || !info.sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("Hugging Face repository {id:?} returned an invalid revision");
        }
        let files: Vec<&str> = info
            .siblings
            .iter()
            .map(|file| file.rfilename.as_str())
            .filter(|file| is_model_asset(file))
            .collect();
        if files.is_empty() || files.iter().any(|file| !is_relative_asset(file)) {
            bail!("Hugging Face repository {id:?} has no usable model file listing");
        }

        let revision = info.sha;
        let repo = api.repo(Repo::with_revision(
            id.to_owned(),
            RepoType::Model,
            revision.clone(),
        ));
        let snapshot = directory.join("snapshots").join(&revision);
        let entry = download_files(&repo, id, &snapshot, &snapshot, &files, repair)?;
        atomic_write(&directory.join("refs/main"), revision.as_bytes())?;
        // Older builds used one marker for the whole repository. Clear it only once
        // the new, validated snapshot is published.
        match fs::remove_file(directory.join(INCOMPLETE_MARKER)) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("clearing legacy model download marker"),
        }
        Ok(entry)
    }

    /// Download one or more known or arbitrary Hugging Face model IDs and return local snapshots.
    pub fn download(&self, ids: &[String]) -> Result<Vec<PathBuf>> {
        ids.iter().map(|id| self.ensure(id)).collect()
    }

    /// Read-only status; this never performs network access.
    pub fn status(&self) -> Vec<ModelStatus> {
        SUPPORTED_MODELS
            .iter()
            .map(|spec| {
                let path = self.cached_path(spec.id);
                ModelStatus {
                    id: spec.id.to_owned(),
                    kind: spec.kind.to_owned(),
                    description: spec.description.to_owned(),
                    cached: path.is_some(),
                    path: path.map(|p| p.display().to_string()),
                }
            })
            .collect()
    }

    fn repo_dir(&self, id: &str) -> PathBuf {
        let repo = pinned_model(id).map_or(id, |_| PARAKEET_RS_REPO);
        self.root
            .join(Repo::new(repo.to_owned(), RepoType::Model).folder_name())
    }

    fn cached_path(&self, id: &str) -> Option<PathBuf> {
        if !is_model_id(id) {
            return None;
        }
        let directory = self.repo_dir(id);
        if let Some(model) = pinned_model(id) {
            // Only the pinned revision counts, and only once its manifest is written.
            let snapshot = directory.join("snapshots").join(PARAKEET_RS_REVISION);
            let state = snapshot.join(model.folder);
            return (state.join(SNAPSHOT_MANIFEST).exists() && manifest_matches(&snapshot, &state))
                .then(|| snapshot_entry(id, &snapshot))
                .flatten();
        }
        if directory.join(INCOMPLETE_MARKER).exists() {
            return None;
        }
        let revision = fs::read_to_string(directory.join("refs/main")).ok()?;
        let revision = revision.trim();
        if !is_single_component(revision) {
            return None;
        }
        let snapshot = directory.join("snapshots").join(revision);
        if !manifest_matches(&snapshot, &snapshot) {
            return None;
        }
        snapshot_entry(id, &snapshot)
    }
}

/// Whether the download recorded in `state` finished and its files are still whole. A snapshot
/// from before manifests existed has none and passes.
fn manifest_matches(snapshot: &Path, state: &Path) -> bool {
    if state.join(INCOMPLETE_MARKER).exists() {
        return false;
    }
    let manifest_path = state.join(SNAPSHOT_MANIFEST);
    if !manifest_path.exists() {
        return true;
    }
    read_manifest(&manifest_path).is_some_and(|files| {
        !files.is_empty()
            && files.iter().all(|(file, size)| {
                is_relative_asset(file) && file_size(&snapshot.join(file)) == Some(*size)
            })
    })
}

/// Download `files`, named relative to `snapshot`, from one revision and validate the model.
/// `state` holds the download's marker and manifest: the snapshot itself, or one model's
/// folder when several models share the snapshot.
fn download_files(
    repo: &ApiRepo,
    id: &str,
    snapshot: &Path,
    state: &Path,
    files: &[&str],
    repair: bool,
) -> Result<PathBuf> {
    fs::create_dir_all(state)?;
    let incomplete = state.join(INCOMPLETE_MARKER);
    fs::write(&incomplete, b"").with_context(|| format!("writing {}", incomplete.display()))?;
    let manifest_path = state.join(SNAPSHOT_MANIFEST);
    let previous = read_manifest(&manifest_path).unwrap_or_default();
    let mut downloaded = BTreeMap::new();
    for file in files {
        let path = snapshot.join(file);
        let size = get_file(repo, id, file, &path, previous.get(*file).copied(), repair)
            .with_context(|| format!("downloading {id}/{file}"))?;
        downloaded.insert(*file, size);
    }

    let entry = snapshot_entry(id, snapshot)
        .with_context(|| format!("model {id:?} has missing or invalid inference files"))?;
    crate::check_cancelled()?;
    atomic_write(&manifest_path, &serde_json::to_vec(&downloaded)?)?;
    fs::remove_file(&incomplete).with_context(|| format!("removing {}", incomplete.display()))?;
    Ok(entry)
}

fn lock_repository(directory: &Path, id: &str) -> Result<File> {
    fs::create_dir_all(directory)
        .with_context(|| format!("creating model cache {}", directory.display()))?;
    let path = directory.join(".transcriberrr-lock");
    let lock = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut announced = false;
    loop {
        crate::check_cancelled()?;
        match lock.try_lock() {
            Ok(()) => return Ok(lock),
            Err(TryLockError::WouldBlock) => {
                if !announced {
                    eprintln!("Waiting for another download of {id}");
                    announced = true;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            Err(TryLockError::Error(error)) => {
                return Err(error).with_context(|| format!("locking {}", path.display()));
            }
        }
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("cache path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let temporary = path.with_extension("transcriberrr-tmp");
    fs::write(&temporary, bytes).with_context(|| format!("writing {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("publishing {}", path.display()))
}

fn read_manifest(path: &Path) -> Option<BTreeMap<String, u64>> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn file_size(path: &Path) -> Option<u64> {
    let metadata = fs::metadata(path).ok()?;
    metadata.is_file().then_some(metadata.len())
}

/// Reuse valid paths in the pinned snapshot. hf-hub leaves existing snapshot pointers alone and
/// cannot replace a dangling symlink, so remove an invalid pointer before handing it the file.
fn get_file(
    repo: &ApiRepo,
    id: &str,
    filename: &str,
    path: &Path,
    expected: Option<u64>,
    repair: bool,
) -> Result<u64> {
    let mut announced = false;
    loop {
        crate::check_cancelled()?;
        if let Some(size) = cached_asset_size(path, expected)
            && (!repair || matches_blob_hash(path, size)?)
        {
            return Ok(size);
        }
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("removing {}", path.display()));
            }
        }
        let mut progress = DownloadProgress {
            terminal: io::stderr().is_terminal(),
            ..DownloadProgress::default()
        };
        let result = repo.download_with_progress(filename, &mut progress);
        match result {
            Ok(downloaded) => {
                anyhow::ensure!(
                    downloaded == path,
                    "Hugging Face returned files from a different revision"
                );
                let size = file_size(path);
                if size == Some(progress.total as u64) {
                    return Ok(progress.total as u64);
                }
                fs::remove_file(path).with_context(|| {
                    format!("removing incomplete model file {}", path.display())
                })?;
                bail!("incomplete model file {}", path.display());
            }
            Err(ApiError::LockAcquisition(_)) => {
                if !announced {
                    eprintln!("Waiting for another download of {id}/{filename}");
                    announced = true;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

/// Hub pointers name the underlying Git blob or LFS object by its content hash. Verify it only
/// after inference fails, when the extra read of healthy multi-gigabyte weights is warranted.
/// A copied snapshot file has no trustworthy original hash, so fetch it again on repair.
fn matches_blob_hash(path: &Path, size: u64) -> Result<bool> {
    let Ok(target) = fs::read_link(path) else {
        return Ok(false);
    };
    let Some(hash) = target.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    if !target
        .parent()
        .is_some_and(|parent| parent.ends_with("blobs"))
        || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Ok(false);
    }
    let verified = match hash.len() {
        40 => hash_file(
            DigestContext::new(&SHA1_FOR_LEGACY_USE_ONLY),
            path,
            format!("blob {size}\0").as_bytes(),
            hash,
        ),
        64 => hash_file(DigestContext::new(&SHA256), path, &[], hash),
        _ => return Ok(false),
    };
    verified.with_context(|| format!("verifying {}", path.display()))
}

fn hash_file(
    mut digest: DigestContext,
    path: &Path,
    prefix: &[u8],
    expected: &str,
) -> Result<bool> {
    digest.update(prefix);
    let mut file = File::open(path)?;
    let mut buffer = [0; 64 * 1024];
    loop {
        crate::check_cancelled()?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let calculated = digest
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(calculated.eq_ignore_ascii_case(expected))
}

fn cached_asset_size(path: &Path, expected: Option<u64>) -> Option<u64> {
    let size = file_size(path)?;
    if let Some(expected) = expected {
        return (size == expected).then_some(size);
    }
    if size == 0 {
        return None;
    }
    match path.file_name()?.to_str()? {
        "model.safetensors.index.json" | "pytorch_model.bin.index.json" => {
            valid_shards(path, &|_| true).then_some(size)
        }
        "config.json" | "tokenizer.json" | "vocab.json" => {
            let bytes = fs::read(path).ok()?;
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()?
                .is_object()
                .then_some(size)
        }
        _ => Some(size),
    }
}

fn snapshot_entry(id: &str, snapshot: &Path) -> Option<PathBuf> {
    let has = |file: &str| file_size(&snapshot.join(file)).is_some_and(|size| size > 0);
    if let Some(model) = pinned_model(id) {
        let folder = snapshot.join(model.folder);
        return model
            .files
            .iter()
            .all(|file| has(&format!("{}/{file}", model.folder)))
            .then(|| model.entry.map_or(folder.clone(), |file| folder.join(file)));
    }
    if matches!(id, QWEN3_ASR_17B | QWEN3_ASR_06B | QWEN3_ALIGNER_06B) {
        // The Qwen loader prefers a single safetensors file, then a sharded index.
        // ONNX and PyTorch entries in a shared Hub snapshot cannot satisfy it.
        let usable_tokenizer = if snapshot.join("tokenizer.json").exists() {
            has("tokenizer.json")
        } else {
            has("vocab.json")
        };
        let usable_weights = if snapshot.join("model.safetensors").exists() {
            has("model.safetensors")
        } else {
            valid_shards(&snapshot.join("model.safetensors.index.json"), &has)
        };
        return (has("config.json") && usable_tokenizer && usable_weights)
            .then(|| snapshot.to_path_buf());
    }

    let indices = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ];
    for index in indices {
        let path = snapshot.join(index);
        if fs::symlink_metadata(&path).is_ok() && !valid_shards(&path, &has) {
            return None;
        }
    }
    let has_weights = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.bin",
    ]
    .into_iter()
    .any(&has);
    let has_entry = has("model.onnx")
        || has("encoder-model.onnx")
        || has("diar_streaming_sortformer_4spk-v2.1.nemo");
    (has_entry || (has("config.json") && has_weights)).then(|| snapshot.to_path_buf())
}

fn valid_shards(index: &Path, has: &impl Fn(&str) -> bool) -> bool {
    #[derive(serde::Deserialize)]
    struct WeightIndex {
        weight_map: BTreeMap<String, String>,
    }

    let Some(index) = fs::read(index)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<WeightIndex>(&bytes).ok())
    else {
        return false;
    };
    let shards: HashSet<_> = index.weight_map.values().map(String::as_str).collect();
    !shards.is_empty()
        && shards
            .into_iter()
            .all(|name| is_relative_asset(name) && has(name))
}

fn is_single_component(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\'])
}

fn is_relative_asset(name: &str) -> bool {
    name.split('/').all(is_single_component)
}

fn is_model_id(id: &str) -> bool {
    let valid = |part: &str| {
        !part.is_empty()
            && part
                .chars()
                .all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.'))
            && !part.starts_with(['.', '-'])
            && !part.ends_with(['.', '-'])
            && !part.contains("..")
            && !part.contains("--")
    };
    if pinned_model(id).is_some() {
        return true;
    }
    match id.split_once('/') {
        Some((owner, name)) => valid(owner) && valid(name),
        None => valid(id),
    }
}

/// Record the server's expected size and report download status to the CLI or app.
#[derive(Default)]
struct DownloadProgress {
    terminal: bool,
    filename: String,
    total: usize,
    received: usize,
    reported: Option<usize>,
}

impl Progress for &mut DownloadProgress {
    fn init(&mut self, size: usize, filename: &str) {
        // Each transfer attempt starts with an update for the resume offset.
        self.received = 0;
        if self.filename != filename || self.total != size {
            self.filename = filename.to_owned();
            self.total = size;
            eprintln!("Downloading {filename} ({:.1} MB)", size as f64 / 1e6);
        }
    }

    fn update(&mut self, size: usize) {
        self.received = self.received.saturating_add(size);
        let percent = self
            .received
            .saturating_mul(100)
            .checked_div(self.total)
            .unwrap_or(100)
            .min(100);
        if self.reported != Some(percent) {
            self.reported = Some(percent);
            if self.terminal {
                eprint!("\rDownloading {}: {percent:3}%", self.filename);
            } else {
                eprintln!("Downloading {}: {percent}%", self.filename);
            }
        }
    }

    fn finish(&mut self) {
        if self.terminal {
            eprintln!();
        }
    }
}

fn is_model_asset(name: &str) -> bool {
    matches!(
        Path::new(name).extension().and_then(|ext| ext.to_str()),
        Some("json")
            | Some("bin")
            | Some("safetensors")
            | Some("onnx")
            | Some("data")
            | Some("txt")
            | Some("nemo")
            | Some("model")
    )
}

pub fn default_cache_dir() -> PathBuf {
    if let Ok(path) = env::var("TRANSCRIBERRR_MODEL_CACHE")
        && !path.trim().is_empty()
    {
        return PathBuf::from(path);
    }
    if let Ok(home) = env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home).join("Library/Application Support/Transcriberrr/models/hub");
    }
    PathBuf::from(".transcriberrr/models/hub")
}

pub fn canonical_id(value: &str) -> String {
    match value {
        "qwen3-asr-1.7b" | "qwen-1.7b" => QWEN3_ASR_17B.to_owned(),
        "qwen3-asr-0.6b" | "qwen-0.6b" => QWEN3_ASR_06B.to_owned(),
        "qwen3-forced-aligner-0.6b" | "forced-aligner" => QWEN3_ALIGNER_06B.to_owned(),
        "parakeet-v3"
        | "parakeet-unified"
        | "parakeet-unified-en"
        | "nvidia/parakeet-unified-en-0.6b" => PARAKEET_UNIFIED_EN.to_owned(),
        "parakeet-tdt-v3" => PARAKEET_TDT_V3.to_owned(),
        "sortformer-4spk" | "sortformer" => SORTFORMER_4SPK.to_owned(),
        other => other.to_owned(),
    }
}

fn expand_home(path: &Path) -> PathBuf {
    let value = path.to_string_lossy();
    if (value == "~" || value.starts_with("~/"))
        && let Ok(home) = env::var("HOME")
        && !home.trim().is_empty()
    {
        return if value == "~" {
            PathBuf::from(home)
        } else {
            PathBuf::from(home).join(value.trim_start_matches("~/"))
        };
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_are_stable() {
        assert_eq!(canonical_id("parakeet-v3"), PARAKEET_UNIFIED_EN);
        assert_eq!(
            canonical_id("nvidia/parakeet-unified-en-0.6b"),
            PARAKEET_UNIFIED_EN
        );
        assert_eq!(canonical_id("qwen3-asr-1.7b"), QWEN3_ASR_17B);
    }

    #[test]
    fn interrupted_download_is_not_reported_as_cached() {
        let root = env::temp_dir().join(format!("transcriberrr-cache-{}", std::process::id()));
        let snapshot = root.join("models--Org--Model/snapshots/abc");
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(root.join("models--Org--Model/refs")).unwrap();
        fs::write(root.join("models--Org--Model/refs/main"), "abc").unwrap();
        fs::write(snapshot.join("config.json"), "{}").unwrap();
        fs::write(snapshot.join("model.safetensors"), "weight").unwrap();
        let cache = ModelCache::new(Some(root.clone()), true);
        assert_eq!(cache.cached_path("Org/Model"), Some(snapshot.clone()));
        let legacy_marker = cache.repo_dir("Org/Model").join(INCOMPLETE_MARKER);
        fs::write(&legacy_marker, "").unwrap();
        assert_eq!(cache.cached_path("Org/Model"), None);
        fs::remove_file(legacy_marker).unwrap();

        // Pre-marker partial download: the index names a shard that never arrived.
        fs::write(
            snapshot.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001.safetensors","b":"model-00002.safetensors"}}"#,
        )
        .unwrap();
        fs::write(snapshot.join("model-00001.safetensors"), "weight").unwrap();
        assert_eq!(cache.cached_path("Org/Model"), None);
        fs::write(snapshot.join("model-00002.safetensors"), "weight").unwrap();
        assert_eq!(cache.cached_path("Org/Model"), Some(snapshot.clone()));
        assert_eq!(cache.lookup("Org/Model"), Some(snapshot));
        assert_eq!(
            cache.lookup(&root.display().to_string()),
            Some(root.clone())
        );
        fs::remove_dir_all(root).unwrap();
    }

    fn publish_pinned(snapshot: &Path, model: &PinnedModel) {
        let mut manifest = BTreeMap::new();
        for file in model.files {
            let name = format!("{}/{file}", model.folder);
            atomic_write(&snapshot.join(&name), b"weights").unwrap();
            manifest.insert(name, 7_u64);
        }
        atomic_write(
            &snapshot.join(model.folder).join(SNAPSHOT_MANIFEST),
            &serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn pinned_models_share_one_repository_and_cache_separately() {
        let root =
            env::temp_dir().join(format!("transcriberrr-pinned-cache-{}", std::process::id()));
        let cache = ModelCache::new(Some(root.clone()), true);
        let directory = cache.repo_dir(PARAKEET_UNIFIED_EN);
        assert_eq!(directory, root.join("models--altunenes--parakeet-rs"));
        assert_eq!(directory, cache.repo_dir(SORTFORMER_4SPK));
        let snapshot = directory.join("snapshots").join(PARAKEET_RS_REVISION);
        let diarization = snapshot.join("nemotron-3-diarization/nemotron3_diar_v3.onnx");
        let unified = snapshot.join("parakeet-unified-en-0.6b");

        // A revision other than the pinned one is never used, whatever `refs/main` names.
        atomic_write(&directory.join("refs/main"), b"older").unwrap();
        atomic_write(
            &directory.join("snapshots/older/nemotron-3-diarization/nemotron3_diar_v3.onnx"),
            b"old graph",
        )
        .unwrap();
        assert_eq!(cache.lookup("sortformer"), None);

        publish_pinned(&snapshot, pinned_model(SORTFORMER_4SPK).unwrap());
        assert_eq!(cache.lookup("sortformer"), Some(diarization.clone()));
        assert_eq!(cache.lookup("parakeet-v3"), None);

        // A second model arriving in the same snapshot is hidden until it completes, and
        // does not hide the first.
        atomic_write(&unified.join(INCOMPLETE_MARKER), b"").unwrap();
        publish_pinned(&snapshot, pinned_model(PARAKEET_UNIFIED_EN).unwrap());
        assert_eq!(cache.lookup("parakeet-v3"), None);
        assert_eq!(cache.lookup("sortformer"), Some(diarization.clone()));
        fs::remove_file(unified.join(INCOMPLETE_MARKER)).unwrap();
        assert_eq!(cache.lookup("parakeet-v3"), Some(unified.clone()));

        // Offline, a cached model resolves and a missing one fails.
        assert_eq!(cache.ensure("parakeet-v3").unwrap(), unified);
        assert_eq!(cache.ensure("sortformer").unwrap(), diarization);
        let missing = cache.ensure("parakeet-tdt-v3").unwrap_err();
        assert!(format!("{missing:#}").contains("--offline"));

        // A truncated file un-caches only its own model.
        fs::write(unified.join("tokenizer.model"), b"x").unwrap();
        assert_eq!(cache.lookup("parakeet-v3"), None);
        assert_eq!(cache.lookup("sortformer"), Some(diarization));
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn pinned_download_fetches_one_folder_of_the_pinned_revision() {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;

        const GRAPH_SHA: &str = "eef93e1d14482804277fca0172464032d1a4fdbcc338524059fa1e861454ad4d";
        let root = env::temp_dir().join(format!(
            "transcriberrr-pinned-hub-cache-{}",
            std::process::id()
        ));
        let cache = ModelCache::new(Some(root.clone()), false);
        let directory = cache.repo_dir(SORTFORMER_4SPK);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            // The revision is pinned, so there is no repository listing request: only the
            // size probe and the file itself.
            for probe in [true, false] {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&mut stream);
                let mut request = String::new();
                reader.read_line(&mut request).unwrap();
                loop {
                    let mut header = String::new();
                    if reader.read_line(&mut header).unwrap() == 0 || header == "\r\n" {
                        break;
                    }
                }
                assert!(request.starts_with(&format!(
                    "GET /altunenes/parakeet-rs/resolve/{PARAKEET_RS_REVISION}/nemotron-3-diarization/nemotron3_diar_v3.onnx "
                )));
                let data = if probe { "g" } else { "graph" };
                let response = format!(
                    "HTTP/1.1 206 Partial Content\r\nX-Repo-Commit: {PARAKEET_RS_REVISION}\r\nETag: {GRAPH_SHA}\r\nContent-Range: bytes 0-{}/5\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}",
                    data.len() - 1,
                    data.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        let api = ApiBuilder::from_cache(Cache::new(root.clone()))
            .with_token(None)
            .with_endpoint(endpoint)
            .build()
            .unwrap();
        let entry = cache
            .download_snapshot(SORTFORMER_4SPK, false, &directory, &api)
            .unwrap();
        server.join().unwrap();
        let snapshot = directory.join("snapshots").join(PARAKEET_RS_REVISION);
        assert_eq!(
            entry,
            snapshot.join("nemotron-3-diarization/nemotron3_diar_v3.onnx")
        );
        assert_eq!(fs::read(&entry).unwrap(), b"graph");
        assert_eq!(cache.cached_path(SORTFORMER_4SPK), Some(entry));
        assert_eq!(cache.cached_path(PARAKEET_UNIFIED_EN), None);
        assert!(!directory.join("refs/main").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn qwen_requires_its_own_config_tokenizer_and_weights() {
        let root = env::temp_dir().join(format!("transcriberrr-qwen-cache-{}", std::process::id()));
        atomic_write(&root.join("model.onnx"), b"graph").unwrap();
        atomic_write(&root.join("tokenizer.json"), b"{}").unwrap();
        assert_eq!(snapshot_entry(QWEN3_ASR_17B, &root), None);
        atomic_write(&root.join("config.json"), b"{}").unwrap();
        atomic_write(&root.join("pytorch_model.bin"), b"weights").unwrap();
        assert_eq!(snapshot_entry(QWEN3_ASR_17B, &root), None);
        atomic_write(&root.join("model.safetensors"), b"weights").unwrap();
        assert_eq!(snapshot_entry(QWEN3_ASR_17B, &root), Some(root.clone()));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_onnx_entry_does_not_require_transformer_config() {
        let root = env::temp_dir().join(format!("transcriberrr-onnx-cache-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        assert_eq!(snapshot_entry("Org/Model", &root), None);
        fs::write(root.join("encoder-model.onnx"), "graph").unwrap();
        assert_eq!(snapshot_entry("Org/Model", &root), Some(root.clone()));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_detects_truncated_optional_asset() {
        let root = env::temp_dir().join(format!(
            "transcriberrr-manifest-cache-{}",
            std::process::id()
        ));
        let cache = ModelCache::new(Some(root.clone()), true);
        let repo = cache.repo_dir("Org/Model");
        let snapshot = repo.join("snapshots/abc");
        atomic_write(&repo.join("refs/main"), b"abc").unwrap();
        atomic_write(&snapshot.join("model.onnx"), b"graph").unwrap();
        atomic_write(&snapshot.join("tokenizer.json"), b"tokens").unwrap();
        atomic_write(
            &snapshot.join(SNAPSHOT_MANIFEST),
            br#"{"model.onnx": 5, "tokenizer.json": 6}"#,
        )
        .unwrap();
        assert_eq!(cache.lookup("Org/Model"), Some(snapshot.clone()));
        fs::write(snapshot.join("tokenizer.json"), b"tok").unwrap();
        assert_eq!(cache.lookup("Org/Model"), None);
        fs::remove_dir_all(root).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn pinned_download_repairs_bad_pointers_and_reuses_verified_blobs() {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;

        const REVISION: &str = "0123456789012345678901234567890123456789";
        const GRAPH_SHA: &str = "eef93e1d14482804277fca0172464032d1a4fdbcc338524059fa1e861454ad4d";
        const TOKENIZER_SHA: &str = "daa5053ecf5f9a37b2de733d0751cc1ab53ac010";
        let root = env::temp_dir().join(format!("transcriberrr-hub-cache-{}", std::process::id()));
        let cache = ModelCache::new(Some(root.clone()), false);
        let directory = cache.repo_dir("Org/Model");
        let snapshot = directory.join("snapshots").join(REVISION);
        let main_ref = directory.join("refs/main");
        atomic_write(&main_ref, b"prior").unwrap();
        let prior_snapshot = directory.join("snapshots/prior");
        atomic_write(&prior_snapshot.join("model.onnx"), b"usable graph").unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        std::os::unix::fs::symlink("../../blobs/missing", snapshot.join("model.onnx")).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let reference = main_ref.clone();
        let server = std::thread::spawn(move || {
            // The healthy sibling is reused on each repair, including validation against its
            // Git or LFS hash. An unexpected request is detected by the expected URL sequence.
            let requests = [
                None,
                Some(("model.onnx", GRAPH_SHA, "graph", true)),
                Some(("model.onnx", GRAPH_SHA, "graph", true)),
                None,
                Some(("model.onnx", GRAPH_SHA, "graph", true)),
                Some(("model.onnx", GRAPH_SHA, "graph", false)),
                Some(("tokenizer.json", TOKENIZER_SHA, r#"{"a":1}"#, true)),
                Some(("tokenizer.json", TOKENIZER_SHA, r#"{"a":1}"#, false)),
                None,
                Some(("model.onnx", GRAPH_SHA, "graph", true)),
                Some(("model.onnx", GRAPH_SHA, "graph", false)),
                None,
                Some(("tokenizer.json", TOKENIZER_SHA, r#"{"a":1}"#, true)),
                Some(("tokenizer.json", TOKENIZER_SHA, r#"{"a":1}"#, false)),
            ];
            for (index, expected) in requests.into_iter().enumerate() {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&mut stream);
                let mut request = String::new();
                reader.read_line(&mut request).unwrap();
                loop {
                    let mut header = String::new();
                    if reader.read_line(&mut header).unwrap() == 0 || header == "\r\n" {
                        break;
                    }
                }
                let response = match expected {
                    None => {
                        assert!(request.starts_with("GET /api/models/Org/Model/revision/main "));
                        let json = format!(
                            r#"{{"sha":"{REVISION}","siblings":[{{"rfilename":"model.onnx"}},{{"rfilename":"tokenizer.json"}}]}}"#
                        );
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                            json.len()
                        )
                    }
                    Some((file, hash, body, probe)) => {
                        assert!(
                            request
                                .starts_with(&format!("GET /Org/Model/resolve/{REVISION}/{file} "))
                        );
                        let data = if probe { &body[..1] } else { body };
                        format!(
                            "HTTP/1.1 206 Partial Content\r\nX-Repo-Commit: {REVISION}\r\nETag: {hash}\r\nContent-Range: bytes 0-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}",
                            data.len() - 1,
                            body.len(),
                            data.len()
                        )
                    }
                };
                if index < 8 {
                    assert_eq!(fs::read_to_string(&reference).unwrap(), "prior");
                }
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        let api = ApiBuilder::from_cache(Cache::new(root.clone()))
            .with_token(None)
            .with_endpoint(endpoint)
            .build()
            .unwrap();
        let short_transfer = cache
            .download_snapshot("Org/Model", false, &directory, &api)
            .unwrap_err();
        assert!(format!("{short_transfer:#}").contains("incomplete model file"));
        assert_eq!(fs::read_to_string(&main_ref).unwrap(), "prior");
        assert!(snapshot.join(INCOMPLETE_MARKER).exists());
        assert_eq!(cache.cached_path("Org/Model"), Some(prior_snapshot));
        assert_eq!(
            cache
                .download_snapshot("Org/Model", false, &directory, &api)
                .unwrap(),
            snapshot
        );
        assert_eq!(fs::read_to_string(&main_ref).unwrap(), REVISION);
        assert_eq!(cache.cached_path("Org/Model"), Some(snapshot.clone()));

        fs::write(snapshot.join("model.onnx"), b"x").unwrap();
        assert_eq!(cache.cached_path("Org/Model"), None);
        cache
            .download_snapshot("Org/Model", true, &directory, &api)
            .unwrap();
        assert_eq!(fs::read(snapshot.join("model.onnx")).unwrap(), b"graph");
        assert_eq!(cache.cached_path("Org/Model"), Some(snapshot.clone()));

        fs::remove_file(snapshot.join(SNAPSHOT_MANIFEST)).unwrap();
        fs::write(snapshot.join("tokenizer.json"), br#"{"a":2}"#).unwrap();
        cache
            .download_snapshot("Org/Model", true, &directory, &api)
            .unwrap();
        assert_eq!(
            fs::read(snapshot.join("tokenizer.json")).unwrap(),
            br#"{"a":1}"#
        );
        server.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
