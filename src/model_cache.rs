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
/// ONNX Parakeet v3 model distributed by the community for parakeet-rs.
pub const PARAKEET_V3: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";
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
        id: PARAKEET_V3,
        kind: "asr",
        description: "Parakeet TDT v3 ONNX (optional engine)",
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
        let id_or_path = canonical_id(id_or_path);
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
        let info = repo
            .info()
            .with_context(|| format!("reading Hugging Face repository {id_or_path:?}"))?;
        if info.siblings.is_empty() {
            bail!("Hugging Face repository {id_or_path:?} contains no files");
        }
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
        repo.get("config.json")
            .or_else(|_| repo.get("model.onnx"))
            .or_else(|_| repo.get("encoder-model.onnx"))
            .or_else(|_| repo.get("diar_streaming_sortformer_4spk-v2.1.nemo"))
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

    fn cached_path(&self, id: &str) -> Option<PathBuf> {
        let repo = Cache::new(self.root.clone()).repo(Repo::new(id.to_owned(), RepoType::Model));
        let config = repo.get("config.json");
        // Require a configuration plus at least one weight/entry point for normal models. This
        // prevents a network interruption after config.json from being mistaken for a complete
        // snapshot on the next run.
        if config.is_some() {
            for marker in [
                "model.safetensors",
                "model.safetensors.index.json",
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
        None
    }
}

fn is_model_asset(name: &str) -> bool {
    matches!(
        Path::new(name).extension().and_then(|ext| ext.to_str()),
        Some("json")
            | Some("safetensors")
            | Some("onnx")
            | Some("data")
            | Some("txt")
            | Some("nemo")
            | Some("model")
    )
}

pub fn default_cache_dir() -> PathBuf {
    if let Ok(path) = env::var("TRANSCRIBERRR_MODEL_CACHE") {
        return PathBuf::from(path);
    }
    if let Ok(home) = env::var("HOME") {
        return PathBuf::from(home).join("Library/Application Support/Transcriberrr/models/hub");
    }
    PathBuf::from(".transcriberrr/models/hub")
}

pub fn canonical_id(value: &str) -> String {
    match value {
        "qwen3-asr-1.7b" | "qwen-1.7b" => QWEN3_ASR_17B.to_owned(),
        "qwen3-asr-0.6b" | "qwen-0.6b" => QWEN3_ASR_06B.to_owned(),
        "qwen3-forced-aligner-0.6b" | "forced-aligner" => QWEN3_ALIGNER_06B.to_owned(),
        "parakeet-v3" | "parakeet-tdt-v3" => PARAKEET_V3.to_owned(),
        "sortformer-4spk" | "sortformer" => SORTFORMER_4SPK.to_owned(),
        other => other.to_owned(),
    }
}

fn expand_home(path: &Path) -> PathBuf {
    let value = path.to_string_lossy();
    if (value == "~" || value.starts_with("~/"))
        && let Ok(home) = env::var("HOME")
    {
        return PathBuf::from(home).join(value.trim_start_matches("~/"));
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_are_stable() {
        assert_eq!(canonical_id("parakeet-v3"), PARAKEET_V3);
        assert_eq!(canonical_id("qwen3-asr-1.7b"), QWEN3_ASR_17B);
    }
}
