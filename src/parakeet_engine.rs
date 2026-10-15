//! Parakeet v3 (FastConformer-TDT) backend.
//!
//! This module deliberately keeps model loading separate from the CLI.  Parakeet's
//! model files are downloaded into the user's cache at runtime (rather than being
//! embedded in the executable) and are then opened by `parakeet-rs`.  The crate's
//! `metal` feature selects Burn's native WGPU/Metal backend on Apple Silicon.

#![allow(dead_code)]

use anyhow::{Context, Result};
use std::path::Path;

use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, Transcriber};

/// NVIDIA Parakeet TDT v3 ONNX export.  The directory must contain
/// `encoder-model.onnx`, `decoder_joint-model.onnx`, and `vocab.txt`.
pub const DEFAULT_MODEL_ID: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";

/// A timestamped item returned by Parakeet.  Times are seconds from the start of
/// the input and are suitable for the common JSON/SRT/VTT writers.
#[derive(Debug, Clone)]
pub struct ParakeetWord {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone)]
pub struct ParakeetTranscript {
    pub text: String,
    pub words: Vec<ParakeetWord>,
}

pub struct ParakeetEngine {
    model: ParakeetTDT,
}

impl ParakeetEngine {
    /// Load a v3 TDT export from a local cache directory.
    ///
    /// `use_metal` selects Burn's WGPU backend (Metal on macOS).  The CPU
    /// fallback is useful for diagnostics and machines without a GPU.
    pub fn from_pretrained(path: &Path, use_metal: bool) -> Result<Self> {
        let provider = if use_metal {
            ExecutionProvider::BurnWgpu
        } else {
            ExecutionProvider::BurnCpu
        };
        let config = ExecutionConfig::new().with_execution_provider(provider);
        let model = ParakeetTDT::from_pretrained(path, Some(config))
            .with_context(|| format!("loading Parakeet v3 model from {}", path.display()))?;
        Ok(Self { model })
    }

    /// Transcribe 16 kHz mono samples.  Parakeet's word grouping is performed
    /// by `parakeet-rs`; no secondary aligner model is necessary.
    pub fn transcribe(&mut self, samples: &[f32], timestamps: bool) -> Result<ParakeetTranscript> {
        let mode = if timestamps {
            TimestampMode::Words
        } else {
            // The decoder still needs a mode to produce a coherent text result;
            // dropping the returned timing objects keeps the CLI output shape
            // consistent with Qwen's no-timestamp path.
            TimestampMode::Words
        };
        let result = self
            .model
            .transcribe_samples(samples.to_vec(), 16_000, 1, Some(mode))
            .context("Parakeet v3 transcription failed")?;
        let words = result
            .tokens
            .into_iter()
            .filter(|token| !token.text.trim().is_empty())
            .map(|token| ParakeetWord {
                text: token.text,
                start: token.start,
                end: token.end,
            })
            .collect();
        Ok(ParakeetTranscript {
            text: result.text,
            words,
        })
    }
}
