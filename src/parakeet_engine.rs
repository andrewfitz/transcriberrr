//! Parakeet Unified English and multilingual TDT backends.
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

/// The Unified decoder stamps a word with the encoder frame that emitted its first
/// token, and gives it one or two 80 ms frames of duration. Against hand-labelled word
/// bounds the stamp trails the spoken onset by a steady 0.27-0.30 s (synthetic interview,
/// AMI headsets, and a held-out AMI set: mean +301, +268, +279 ms; sd 78-138 ms).
const UNIFIED_EMISSION_LATENCY: f32 = 0.28;
/// Speaking-rate estimate for a word whose end no following onset bounds.
const SECONDS_PER_CHARACTER: f32 = 0.075;
const SHORTEST_WORD: f32 = 0.14;
const LONGEST_WORD: f32 = 0.8;
/// A word before a pause is drawn out. Hand-labelled phrase-final words ended 0.13-0.29 s
/// after the in-phrase estimate, and a cut that clips a word is worse than one that keeps
/// a little of the pause.
const PHRASE_FINAL_WORD: f32 = 0.3;
const PHRASE_FINAL_SECONDS_PER_CHARACTER: f32 = 0.04;
const LONGEST_PHRASE_FINAL_WORD: f32 = 0.9;

/// Move Unified word onsets back by the decoder's emission latency and give each word a
/// spoken duration. A word runs to the next onset in continuous speech; before a pause it
/// keeps a length estimated from its spelling. Editors cut on these bounds, so a late onset
/// leaves the head of a deleted word in the program.
pub(crate) fn refine_unified_timing(words: &mut [ParakeetWord], duration: f32) {
    let mut floor = 0.0f32;
    for word in words.iter_mut() {
        if !word.start.is_finite() {
            continue;
        }
        // Onsets stay in order and inside the chunk.
        word.start = (word.start - UNIFIED_EMISSION_LATENCY).max(floor);
        floor = word.start;
    }
    estimate_word_ends(words, duration);
}

/// Give each word a spoken duration from onsets that are already where the words were
/// spoken. The listener calls this with onsets it corrected for its own streaming latency.
pub(crate) fn estimate_word_ends(words: &mut [ParakeetWord], duration: f32) {
    let mut next: Option<f32> = None;
    for word in words.iter_mut().rev() {
        if !word.start.is_finite() {
            continue;
        }
        if duration.is_finite() {
            word.start = word.start.clamp(0.0, duration.max(0.0));
        }
        let letters = word.text.chars().filter(|c| c.is_alphanumeric()).count();
        if letters == 0 {
            // Punctuation is attached to the preceding word by the caller.
            word.end = word.start;
            continue;
        }
        let start = word.start;
        let letters = letters as f32;
        let estimate = (letters * SECONDS_PER_CHARACTER).clamp(SHORTEST_WORD, LONGEST_WORD);
        let end = match next {
            Some(next) if next - start <= estimate * 1.6 => next.max(start),
            _ => {
                let drawn_out = (PHRASE_FINAL_WORD + letters * PHRASE_FINAL_SECONDS_PER_CHARACTER)
                    .clamp(estimate, LONGEST_PHRASE_FINAL_WORD);
                // The pause may be shorter than the drawn-out word.
                next.map_or(start + drawn_out, |next| next.min(start + drawn_out))
            }
        };
        word.end = if duration.is_finite() {
            end.min(duration.max(0.0))
        } else {
            end
        };
        next = Some(start);
    }
}

pub struct ParakeetEngine {
    model: ParakeetBackend,
}

enum ParakeetBackend {
    Unified(ParakeetUnified),
    Tdt(ParakeetTDT),
}

impl ParakeetEngine {
    /// Load a Unified folder (`tokenizer.model` beside `encoder.onnx`), or else a TDT v3 folder.
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
                    format!("loading Parakeet TDT model from {}", path.display())
                })?,
            )
        };
        Ok(Self { model })
    }

    /// Transcribe 16 kHz mono samples.  Parakeet's word grouping is performed
    /// by `parakeet-rs`; no secondary aligner model is necessary.
    fn transcribe(&mut self, samples: &[f32]) -> Result<ParakeetTranscript> {
        // Keep word grouping for sentence segmentation even when the output omits timestamps.
        let mode = Some(TimestampMode::Words);
        let result = match &mut self.model {
            ParakeetBackend::Unified(model) => {
                model.transcribe_samples(samples.to_vec(), 16_000, 1, mode)
            }
            ParakeetBackend::Tdt(model) => {
                model.transcribe_samples(samples.to_vec(), 16_000, 1, mode)
            }
        }
        .context("Parakeet transcription failed")?;
        let unified = matches!(self.model, ParakeetBackend::Unified(_));
        let mut words: Vec<_> = result
            .tokens
            .into_iter()
            .filter(|token| !token.text.trim().is_empty())
            .map(|token| ParakeetWord {
                text: token.text,
                start: token.start,
                end: token.end,
            })
            .collect();
        if unified {
            refine_unified_timing(&mut words, samples.len() as f32 / 16_000.0);
        }
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
            crate::check_cancelled()?;
            let progress = 0.2 + 0.52 * (index + 1) as f32 / total as f32;
            // Exact silence: Zoom pads a participant's track before they join and after they
            // leave, and bleed muting zeroes long stretches of a listener's track. Qwen's
            // voice detector skips such chunks; Parakeet would decode 120 s of nothing.
            if samples[range.clone()].iter().all(|&sample| sample == 0.0) {
                report(
                    progress,
                    &format!("Skipping silent chunk {}/{}", index + 1, total),
                )?;
                continue;
            }
            let mut result = self.transcribe(&samples[range.clone()])?;
            let offset = range.start as f32 / 16_000.0;
            for word in &mut result.words {
                word.start += offset;
                word.end += offset;
            }
            if !result.text.trim().is_empty() {
                text_parts.push(result.text);
            }
            words.extend(result.words);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f32) -> ParakeetWord {
        ParakeetWord {
            text: text.to_owned(),
            start,
            end: start + 0.08,
        }
    }

    #[test]
    fn unified_words_start_at_the_spoken_onset_and_run_to_the_next_word() {
        // Emission stamps for "Could you" then a pause before "slower?", spoken from 1.00 s.
        let mut words = vec![
            word("Could", 1.28),
            word(" you", 1.52),
            word(" slower", 2.60),
            word("?", 3.10),
        ];
        refine_unified_timing(&mut words, 4.0);
        assert!((words[0].start - 1.00).abs() < 1e-4);
        // Continuous speech: the next onset bounds the word.
        assert!((words[0].end - words[1].start).abs() < 1e-4);
        // Before a pause the word keeps a phrase-final estimate, not the next onset.
        assert!((words[1].end - (1.24 + 0.3 + 3.0 * 0.04)).abs() < 1e-4);
        assert!(words[1].end < words[2].start);
        // Six letters before the end of speech; punctuation does not bound it.
        assert!((words[2].end - (2.32 + 0.3 + 6.0 * 0.04)).abs() < 1e-4);
        assert_eq!(words[3].start, words[3].end);
    }

    #[test]
    fn unified_timing_stays_ordered_inside_the_chunk() {
        let mut words = vec![
            word("Hi", 0.08),
            word(" there", 0.16),
            word(" everyone", 9.9),
            word("!", 10.4),
            word("Again", 10.6),
        ];
        refine_unified_timing(&mut words, 10.0);
        assert_eq!(words[0].start, 0.0);
        assert!(words[1].start >= words[0].start);
        assert!(words.iter().all(|word| word.end >= word.start));
        assert_eq!(words[0].end, words[1].start);
        assert!(words.iter().all(|word| word.end <= 10.0));
        assert_eq!(words[4].start, 10.0);
    }
}
