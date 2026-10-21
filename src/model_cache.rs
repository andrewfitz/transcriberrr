//! Local model cache and Hugging Face model acquisition.
//!
//! Model weights are intentionally kept outside the executable and source tree.  The cache uses
//! the standard Hugging Face snapshot layout, so downloads are resumable and shared by CLI/UI
//! processes without duplicating files.

use anyhow::{Context, Result, bail};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};
use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub const QWEN3_ASR_17B: &str = "Qwen/Qwen3-ASR-1.7B";
pub const QWEN3_ASR_06B: &str = "Qwen/Qwen3-ASR-0.6B";
pub const QWEN3_ALIGNER_06B: &str = "Qwen/Qwen3-ForcedAligner-0.6B";
/// Latest English Parakeet Unified RNNT ONNX export compatible with parakeet-rs.
pub const PARAKEET_UNIFIED_EN: &str = "eschmidbauer/parakeet-unified-en-0.6b-onnx";
/// Legacy multilingual TDT v3 ONNX export, retained for explicit `--model` overrides.
pub const PARAKEET_TDT_V3: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";
/// Sortformer v3 ONNX export compatible with parakeet-rs. The repository also contains unrelated
/// exports, so the downloader below fetches only the compatible graph.
pub const SORTFORMER_4SPK: &str = "altunenes/parakeet-rs";
const SORTFORMER_FILE: &str = "nemotron-3-diarization/nemotron3_diar_v3.onnx";

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
        description: "Parakeet Unified English 0.6B ONNX (latest)",
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
    pub fn new(root: Option<PathBuf>) -> Result<Self> {
        let root = root.unwrap_or_else(default_cache_dir);
        let root = expand_home(&root);
        fs::create_dir_all(&root)
            .with_context(|| format!("creating model cache {}", root.display()))?;
        Ok(Self { root })
    }

    /// Resolve a model ID or local directory.  A remote ID is downloaded into this cache and the
    /// local immutable snapshot path is returned to inference libraries.
    pub fn ensure(&self, id_or_path: &str) -> Result<PathBuf> {
        // Accept shell-style home paths even when the value came from a repeated
        // --download-model argument or a manifest rather than a top-level CLI path.
        let expanded = expand_home(Path::new(id_or_path));
        let id_or_path = canonical_id(&expanded.to_string_lossy());
        let path = Path::new(&id_or_path);
        if path.exists() {
            if path.is_dir() {
                return Ok(path.to_path_buf());
            }
            bail!("model path is not a directory: {}", path.display());
        }

        // A complete snapshot is already available locally. Avoid metadata requests on every
        // transcription startup and keep offline runs genuinely offline.
        if let Some(snapshot) = self.cached_path(&id_or_path) {
            return Ok(snapshot);
        }
        self.fetch(&id_or_path)
    }

    /// Re-check a Hub model against the repository listing and download any missing files.
    /// Used when a cached snapshot fails to load: caches interrupted before the incomplete
    /// marker existed still look complete offline. Returns `None` for local model paths.
    pub fn repair(&self, id_or_path: &str) -> Result<Option<PathBuf>> {
        let expanded = expand_home(Path::new(id_or_path));
        let id = canonical_id(&expanded.to_string_lossy());
        if Path::new(&id).exists() {
            return Ok(None);
        }
        self.fetch(&id).map(Some)
    }

    /// Offline lookup used by `--cached-model` so the app asks the CLI instead of
    /// re-implementing aliases, markers, and snapshot resolution.
    pub fn lookup(&self, id_or_path: &str) -> Option<PathBuf> {
        let expanded = expand_home(Path::new(id_or_path));
        let id = canonical_id(&expanded.to_string_lossy());
        let path = Path::new(&id);
        if path.exists() {
            return Some(path.to_path_buf());
        }
        self.cached_path(&id)
    }

    /// Download `id` into the cache; files already present are not fetched again.
    fn fetch(&self, id_or_path: &str) -> Result<PathBuf> {
        let id_or_path = id_or_path.to_owned();
        let api = ApiBuilder::from_cache(Cache::new(self.root.clone()))
            .with_progress(true)
            .build()
            .context("creating Hugging Face client")?;
        let repo = api.repo(Repo::new(id_or_path.clone(), RepoType::Model));
        if id_or_path == SORTFORMER_4SPK {
            return repo
                .get(SORTFORMER_FILE)
                .with_context(|| format!("downloading {id_or_path}/{SORTFORMER_FILE}"));
        }
        if id_or_path == PARAKEET_UNIFIED_EN {
            let files = [
                "onnx_fp32/encoder.onnx",
                "onnx_fp32/encoder.onnx.data",
                "onnx_fp32/decoder_joint.onnx",
                "onnx_fp32/decoder_joint.onnx.data",
                "onnx_fp32/tokenizer.model",
            ];
            for file in files {
                repo.get(file)
                    .with_context(|| format!("downloading {id_or_path}/{file}"))?;
            }
            return repo
                .get("onnx_fp32/encoder.onnx")
                .map(|file| file.parent().map(Path::to_path_buf).unwrap_or(file))
                .with_context(|| format!("model {id_or_path:?} has no unified ONNX entry point"));
        }
        let info = repo
            .info()
            .with_context(|| format!("reading Hugging Face repository {id_or_path:?}"))?;
        if info.siblings.is_empty() {
            bail!("Hugging Face repository {id_or_path:?} contains no files");
        }
        // Files arrive in name order, so the cached_path markers (config.json plus e.g.
        // model.safetensors) can exist long before tokenizer files or `.onnx.data` weights. Mark
        // the repository incomplete until every file is present; an interrupted download then
        // resumes on the next run instead of loading a partial snapshot forever.
        let incomplete = self.incomplete_marker(&id_or_path);
        if let Some(parent) = incomplete.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating model cache {}", parent.display()))?;
        }
        fs::write(&incomplete, b"").with_context(|| format!("writing {}", incomplete.display()))?;
        // Download every repository file.  Model indexes reference shard names, tokenizer files
        // are loaded separately, and optional config files vary between engines. Skip README,
        // image, and metadata assets so a fresh install downloads only inference inputs.
        for sibling in info.siblings {
            if !is_model_asset(&sibling.rfilename) {
                continue;
            }
            repo.get(&sibling.rfilename)
                .with_context(|| format!("downloading {id_or_path}/{}", sibling.rfilename))?;
        }
        fs::remove_file(&incomplete)
            .with_context(|| format!("removing {}", incomplete.display()))?;
        repo.get("config.json")
            .or_else(|_| repo.get("model.onnx"))
            .or_else(|_| repo.get("encoder-model.onnx"))
            .or_else(|_| repo.get("diar_streaming_sortformer_4spk-v2.1.nemo"))
            .map(snapshot_dir)
            .with_context(|| format!("model {id_or_path:?} has no recognised entry point"))
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

    /// Present while `ensure` is downloading a full repository. The Swift app checks the same
    /// file name, so keep them in sync.
    fn incomplete_marker(&self, id: &str) -> PathBuf {
        self.root
            .join(Repo::new(id.to_owned(), RepoType::Model).folder_name())
            .join(".transcriberrr-incomplete")
    }

    fn cached_path(&self, id: &str) -> Option<PathBuf> {
        if self.incomplete_marker(id).exists() {
            return None;
        }
        let repo = Cache::new(self.root.clone()).repo(Repo::new(id.to_owned(), RepoType::Model));
        let config = repo.get("config.json");
        // Require a configuration plus at least one weight/entry point for normal models. This
        // prevents a network interruption after config.json from being mistaken for a complete
        // snapshot on the next run.
        // Sharded weights: the index names every shard, so a partial download is detectable
        // offline even without the incomplete marker.
        if let Some(index) = repo.get("model.safetensors.index.json") {
            let shards = fs::read_to_string(&index)
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| {
                    value["weight_map"].as_object().map(|map| {
                        map.values()
                            .filter_map(|shard| shard.as_str().map(str::to_owned))
                            .collect::<Vec<_>>()
                    })
                });
            match shards {
                Some(shards) if shards.iter().all(|shard| repo.get(shard).is_some()) => {}
                _ => return None,
            }
        }
        if config.is_some() {
            for marker in [
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "model.bin",
                "model.onnx",
                "encoder-model.onnx",
            ] {
                if let Some(file) = repo.get(marker) {
                    return file.parent().map(Path::to_path_buf);
                }
            }
        }
        if id == SORTFORMER_4SPK
            && let Some(file) = repo.get(SORTFORMER_FILE)
        {
            return Some(file);
        }
        if id == PARAKEET_UNIFIED_EN {
            let files = [
                "onnx_fp32/encoder.onnx",
                "onnx_fp32/encoder.onnx.data",
                "onnx_fp32/decoder_joint.onnx",
                "onnx_fp32/decoder_joint.onnx.data",
                "onnx_fp32/tokenizer.model",
            ];
            if files.iter().all(|file| repo.get(file).is_some()) {
                return repo
                    .get("onnx_fp32/encoder.onnx")
                    .and_then(|file| file.parent().map(Path::to_path_buf));
            }
        }
        None
    }
}

fn snapshot_dir(entry_point: PathBuf) -> PathBuf {
    // Inference backends consume the snapshot directory, not the marker file. Returning
    // config.json here makes a subsequent ensure() reject the path as a file and prevents
    // Qwen/aligner models from loading.
    entry_point
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(entry_point)
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
        fs::write(snapshot.join("model.safetensors"), "").unwrap();
        let cache = ModelCache::new(Some(root.clone())).unwrap();
        assert_eq!(cache.cached_path("Org/Model"), Some(snapshot.clone()));
        fs::write(cache.incomplete_marker("Org/Model"), "").unwrap();
        assert_eq!(cache.cached_path("Org/Model"), None);
        fs::remove_file(cache.incomplete_marker("Org/Model")).unwrap();

        // Pre-marker partial download: the index names a shard that never arrived.
        fs::write(
            snapshot.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001.safetensors","b":"model-00002.safetensors"}}"#,
        )
        .unwrap();
        fs::write(snapshot.join("model-00001.safetensors"), "").unwrap();
        assert_eq!(cache.cached_path("Org/Model"), None);
        fs::write(snapshot.join("model-00002.safetensors"), "").unwrap();
        assert_eq!(cache.cached_path("Org/Model"), Some(snapshot.clone()));
        assert_eq!(cache.lookup("Org/Model"), Some(snapshot));
        assert_eq!(
            cache.lookup(&root.display().to_string()),
            Some(root.clone())
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ensure_entry_point_resolves_to_snapshot_directory() {
        let entry = PathBuf::from("/tmp/hub/models--Qwen--Qwen3/snapshots/abc/config.json");
        assert_eq!(
            snapshot_dir(entry),
            PathBuf::from("/tmp/hub/models--Qwen--Qwen3/snapshots/abc")
        );
    }
}
