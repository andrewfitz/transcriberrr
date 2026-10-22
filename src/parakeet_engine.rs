//! Parakeet Unified English and legacy TDT backends.
//!
//! This module deliberately keeps model loading separate from the CLI.  Parakeet's
//! model files are downloaded into the user's cache at runtime (rather than being
//! embedded in the executable) and are then opened by `parakeet-rs`.  The crate's
//! `metal` feature selects Burn's native WGPU/Metal backend on Apple Silicon.

#![allow(dead_code)]

use anyhow::{Context, Result};
use std::path::Path;

use parakeet_rs::{
    ExecutionConfig, ExecutionProvider, ParakeetTDT, ParakeetUnified, TimestampMode, Transcriber,
};

/// Latest English Parakeet Unified ONNX export. The directory must contain
/// `encoder.onnx`, `decoder_joint.onnx`, and `tokenizer.model`.
pub const DEFAULT_MODEL_ID: &str = crate::model_cache::PARAKEET_UNIFIED_EN;

/// Longest chunk the Parakeet Unified encoder accepts, with margin below its ~400 s limit.
const MAX_CHUNK_SECONDS: f32 = 300.0;

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
    model: ParakeetBackend,
}

enum ParakeetBackend {
    Unified(ParakeetUnified),
    Tdt(ParakeetTDT),
}

impl ParakeetEngine {
    /// Load the Unified export by default, while retaining compatibility with
    /// the older TDT v3 directory layout when an explicit legacy path is used.
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
        let model = if path.join("tokenizer.model").is_file() && path.join("encoder.onnx").is_file()
        {
            ParakeetBackend::Unified(
                ParakeetUnified::from_pretrained(path, Some(config)).with_context(|| {
                    format!("loading Parakeet Unified model from {}", path.display())
                })?,
            )
        } else {
            ParakeetBackend::Tdt(
                ParakeetTDT::from_pretrained(path, Some(config)).with_context(|| {
                    format!("loading legacy Parakeet TDT model from {}", path.display())
                })?,
            )
        };
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
        let result = match &mut self.model {
            ParakeetBackend::Unified(model) => {
                model.transcribe_samples(samples.to_vec(), 16_000, 1, Some(mode))
            }
            ParakeetBackend::Tdt(model) => {
                model.transcribe_samples(samples.to_vec(), 16_000, 1, Some(mode))
            }
        }
        .context("Parakeet transcription failed")?;
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

    /// Transcribe bounded chunks so callers can report truthful completion as each model call
    /// finishes. The model is reused for every chunk; timestamps are offset back into source time.
    pub fn transcribe_with_progress(
        &mut self,
        samples: &[f32],
        timestamps: bool,
        chunk_seconds: f32,
        report: &mut dyn FnMut(f32, &str) -> Result<()>,
    ) -> Result<ParakeetTranscript> {
        // The Unified encoder's positional embedding covers ~5000 frames (~400 s); longer
        // chunks panic inside Burn and fail the file.
        let chunk_seconds = chunk_seconds.min(MAX_CHUNK_SECONDS);
        let chunk_samples = (chunk_seconds * 16_000.0).round().max(1.0) as usize;
        // Cut in pauses (as for Qwen) so words are not split across independent decodes.
        let chunks = crate::chunk_ranges(samples, chunk_samples);
        let total = chunks.len().max(1);
        let mut text_parts = Vec::with_capacity(chunks.len());
        let mut words = Vec::new();
        for (index, range) in chunks.iter().enumerate() {
            let mut result = self.transcribe(&samples[range.clone()], timestamps)?;
            let offset = range.start as f32 / 16_000.0;
            for word in &mut result.words {
                word.start += offset;
                word.end += offset;
            }
            if !result.text.trim().is_empty() {
                text_parts.push(result.text);
            }
            words.extend(result.words);
            let progress = 0.2 + 0.52 * (index + 1) as f32 / total as f32;
            report(
                progress,
                &format!("Transcribing chunk {}/{}", index + 1, total),
            )?;
        }
        Ok(ParakeetTranscript {
            text: text_parts.join(" "),
            words,
        })
    }
}
