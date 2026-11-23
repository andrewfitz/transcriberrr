//! Buffered streaming for Parakeet Unified with windows of varying length.
//!
//! `parakeet_rs::ParakeetUnified` fixes its hop when a decoder is created. A live track
//! that has fallen behind needs one window over all the audio waiting, so this module
//! drives the crate's public encoder and decoder itself. With a constant hop it
//! reproduces `ParakeetUnified::transcribe_chunk` and `flush` exactly;
//! `examples/parakeet_parity.rs` compares the two on recorded audio. Run it after
//! upgrading parakeet-rs.

use std::f32::consts::PI;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Context, Result, anyhow, ensure};
use ndarray::{Array2, Array3, s};
use parakeet_rs::{
    ExecutionConfig, FeatureCache, ParakeetUnifiedModel, PreprocessorConfig, SentencePieceVocab,
    TimedToken, TranscriptionResult, UnifiedModelConfig, UnifiedStreamingConfig,
};

const SAMPLE_RATE: usize = 16_000;
/// Samples between feature frames (10 ms).
const FEATURE_HOP: usize = 160;
const FFT_SIZE: usize = 512;
const WINDOW_LENGTH: usize = 400;
static HANN: LazyLock<[f32; WINDOW_LENGTH]> = LazyLock::new(|| {
    std::array::from_fn(|index| {
        0.5 - 0.5 * ((2.0 * PI * index as f32) / (WINDOW_LENGTH as f32 - 1.0)).cos()
    })
});
const MEL_BINS: usize = 128;
const PREEMPHASIS: f32 = 0.97;
/// The encoder emits one frame per eight feature frames.
const SUBSAMPLING: usize = 8;
/// Samples per encoder frame. Every window is a whole number of these.
pub(super) const FRAME_SAMPLES: usize = SUBSAMPLING * FEATURE_HOP;
const LSTM_LAYERS: usize = 2;
const LSTM_WIDTH: usize = 640;
const MAX_SYMBOLS_PER_FRAME: usize = 10;

/// The loaded networks, vocabulary and feature tables, shared by every stream.
pub(super) struct UnifiedModel {
    networks: Mutex<ParakeetUnifiedModel>,
    vocab: SentencePieceVocab,
    features: FeatureCache,
    blank: usize,
}

impl UnifiedModel {
    pub(super) fn load(path: &Path, execution: ExecutionConfig) -> Result<Arc<Self>> {
        let vocab = SentencePieceVocab::from_file(path.join("tokenizer.model"))?;
        let blank = vocab.size();
        let networks = ParakeetUnifiedModel::from_pretrained(
            path,
            execution,
            UnifiedModelConfig {
                vocab_size: blank + 1,
                blank_id: blank,
                decoder_lstm_dim: LSTM_WIDTH,
                decoder_lstm_layers: LSTM_LAYERS,
                subsampling_factor: SUBSAMPLING,
            },
        )?;
        let features = FeatureCache::from_config(&PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".into(),
            feature_size: MEL_BINS,
            hop_length: FEATURE_HOP,
            n_fft: FFT_SIZE,
            padding_side: "right".into(),
            padding_value: 0.0,
            preemphasis: PREEMPHASIS,
            processor_class: "ParakeetProcessor".into(),
            return_attention_mask: true,
            sampling_rate: SAMPLE_RATE,
            win_length: WINDOW_LENGTH,
        });
        Ok(Arc::new(Self {
            networks: Mutex::new(networks),
            vocab,
            features,
            blank,
        }))
    }
}

/// One stream's audio buffer, decoder state and tokens.
pub(super) struct StreamDecoder {
    model: Arc<UnifiedModel>,
    left_context: usize,
    right_context: usize,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    audio: Vec<f32>,
    /// Stream position of `audio[0]`.
    audio_start: usize,
    /// Stream position where the next window's own audio begins.
    next: usize,
    /// Token and the encoder frame, counted from the start of the stream, that emitted it.
    tokens: Vec<(usize, usize)>,
    /// For each window in order, the stream frame its hop began at and the encoder
    /// frames of left context it had.
    windows: Vec<(usize, usize)>,
}

/// The samples of one encoder window, relative to the start of the stream.
#[derive(Debug, PartialEq, Eq)]
struct Window {
    start: usize,
    end: usize,
    /// Encoder frames of left context, which are not decoded.
    context_frames: usize,
    /// Encoder frames decoded.
    frames: usize,
    /// Samples the stream advances by.
    advance: usize,
}

/// Plans the window that decodes the next `hop` samples. A window that is not the last
/// waits for its right context, so the caller ensures `hop + right_context` samples are
/// pending. The last takes what is there.
fn plan_window(
    next: usize,
    buffered_from: usize,
    received: usize,
    (left_context, hop, right_context): (usize, usize, usize),
    last: bool,
) -> Window {
    let pending = received - next;
    let (main, right) = if last {
        let main = pending.min(hop);
        (main, (pending - main).min(right_context))
    } else {
        (hop, right_context)
    };
    let left = (next - buffered_from).min(left_context);
    Window {
        start: next - left,
        end: next + main + right,
        context_frames: left / FRAME_SAMPLES,
        frames: main / FRAME_SAMPLES,
        advance: main,
    }
}

impl StreamDecoder {
    /// Uses the context lengths of `config`. The caller chooses each window's hop.
    pub(super) fn new(model: &Arc<UnifiedModel>, config: UnifiedStreamingConfig) -> Result<Self> {
        let config = config.validate()?;
        Ok(Self {
            model: Arc::clone(model),
            left_context: config.left_context_samples(),
            right_context: config.right_context_samples(),
            state_1: Array3::zeros((LSTM_LAYERS, 1, LSTM_WIDTH)),
            state_2: Array3::zeros((LSTM_LAYERS, 1, LSTM_WIDTH)),
            last_token: model.blank as i32,
            audio: Vec::new(),
            audio_start: 0,
            next: 0,
            tokens: Vec::new(),
            windows: Vec::new(),
        })
    }

    pub(super) fn extend(&mut self, samples: &[f32]) {
        self.audio.extend_from_slice(samples);
    }

    /// Received samples that no window has decoded.
    pub(super) fn pending_samples(&self) -> usize {
        self.audio_start + self.audio.len() - self.next
    }

    pub(super) fn right_context_samples(&self) -> usize {
        self.right_context
    }

    /// Runs one window that decodes the next `hop` samples, a whole number of encoder
    /// frames, and reports whether it produced a token. Unless `last`, `hop` plus the
    /// right context must be pending. The last window of a stream decodes what remains.
    pub(super) fn decode_window(&mut self, hop: usize, last: bool) -> Result<bool> {
        ensure!(
            hop > 0 && hop.is_multiple_of(FRAME_SAMPLES),
            "a Parakeet hop is a whole number of encoder frames"
        );
        ensure!(
            last || self.pending_samples() >= hop + self.right_context,
            "a Parakeet window needs its right context"
        );
        let window = plan_window(
            self.next,
            self.audio_start,
            self.audio_start + self.audio.len(),
            (self.left_context, hop, self.right_context),
            last,
        );
        if window.frames == 0 {
            return Ok(false);
        }
        let samples = &self.audio[window.start - self.audio_start..window.end - self.audio_start];
        let features = log_mel(samples, &self.model.features)?;
        let mut networks = self
            .model
            .networks
            .lock()
            .map_err(|_| anyhow!("a Parakeet stream panicked while holding the model"))?;
        // The English model takes no language prompt.
        let (encoded, encoded_frames) = networks.run_encoder(&features, None)?;
        let available = (encoded_frames as usize).min(encoded.shape()[2]);
        let first = window.context_frames.min(available);
        let end = (first + window.frames).min(available);
        let width = encoded.shape()[1];
        let offset = self.next / FRAME_SAMPLES;
        self.windows.push((offset, first));
        let before = self.tokens.len();
        for frame in first..end {
            let encoding = encoded
                .slice(s![0, .., frame])
                .to_owned()
                .into_shape_with_order((1, width, 1))
                .context("shaping a Parakeet encoder frame")?;
            for _ in 0..MAX_SYMBOLS_PER_FRAME {
                let (logits, state_1, state_2) = networks.run_decoder(
                    &encoding,
                    self.last_token,
                    &self.state_1,
                    &self.state_2,
                )?;
                let token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(self.model.blank, |(token, _)| token);
                if token == self.model.blank {
                    break;
                }
                self.tokens.push((token, offset + frame - first));
                self.last_token = token as i32;
                self.state_1 = state_1;
                self.state_2 = state_2;
            }
        }
        drop(networks);
        self.next += window.advance;
        // Keep only the left context of the next window.
        let keep_from = self.next.saturating_sub(self.left_context);
        if keep_from > self.audio_start {
            let drop = (keep_from - self.audio_start).min(self.audio.len());
            self.audio.drain(..drop);
            self.audio_start = keep_from;
        }
        Ok(self.tokens[before..]
            .iter()
            .any(|&(token, _)| token < self.model.blank))
    }

    /// How many encoder frames the window that decoded stream frame `frame` held before
    /// it: the window's left context and the part of its hop up to the frame.
    pub(super) fn frames_heard_before(&self, frame: usize) -> usize {
        let window = self
            .windows
            .partition_point(|&(first, _)| first <= frame)
            .saturating_sub(1);
        self.windows
            .get(window)
            .map_or(0, |&(first, context)| frame.saturating_sub(first) + context)
    }

    /// The text so far and its words, timed by the encoder frame that emitted each.
    pub(super) fn transcript(&self) -> TranscriptionResult {
        let spoken = || {
            self.tokens
                .iter()
                .filter(|&&(token, _)| token < self.model.blank)
        };
        let seconds = |frame: usize| (frame * FRAME_SAMPLES) as f32 / SAMPLE_RATE as f32;
        let timed: Vec<_> = spoken()
            .map(|&(token, frame)| TimedToken {
                text: self.model.vocab.decode_single(token),
                start: seconds(frame),
                end: seconds(frame + 1),
            })
            .collect();
        let tokens: Vec<_> = spoken().map(|&(token, _)| token).collect();
        TranscriptionResult {
            text: self.model.vocab.decode(&tokens),
            tokens: group_words(&timed),
        }
    }
}

/// Log-mel features as parakeet-rs computes them: one row per 10 ms, normalized per mel
/// bin over the window.
fn log_mel(audio: &[f32], cache: &FeatureCache) -> Result<Array2<f32>> {
    let frames = audio.len() / FEATURE_HOP;
    let bins = FFT_SIZE / 2 + 1;
    // Pre-emphasis, then a centered short-time transform with zero padding at both ends.
    let mut padded = Vec::with_capacity(audio.len() + FFT_SIZE);
    padded.resize(FFT_SIZE / 2, 0.0f32);
    padded.extend(audio.first());
    padded.extend(audio.windows(2).map(|pair| pair[1] - PREEMPHASIS * pair[0]));
    padded.resize(padded.len() + FFT_SIZE / 2, 0.0);
    // The analysis window is shorter than the transform and sits in its middle.
    let inset = (FFT_SIZE - WINDOW_LENGTH) / 2;
    let mut power = Array2::<f32>::zeros((bins, frames));
    let mut input = vec![0.0f32; FFT_SIZE];
    let mut output = cache.fft_plan.make_output_vec();
    let mut scratch = cache.fft_plan.make_scratch_vec();
    for frame in 0..frames {
        let start = frame * FEATURE_HOP + inset;
        input.fill(0.0);
        for (index, weight) in HANN.iter().enumerate() {
            input[inset + index] = padded[start + index] * weight;
        }
        cache
            .fft_plan
            .process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|error| anyhow!("Parakeet feature transform failed: {error}"))?;
        for (bin, value) in output.iter().take(bins).enumerate() {
            power[[bin, frame]] = value.norm_sqr();
        }
    }
    let guard = 2.0f32.powi(-24);
    let mut mel = cache.mel_basis.dot(&power).reversed_axes();
    mel.mapv_inplace(|energy| (energy + guard).ln());
    if frames <= 1 {
        return Ok(mel);
    }
    for mut bin in mel.columns_mut() {
        let mean = bin.iter().sum::<f32>() / frames as f32;
        let variance =
            bin.iter().map(|&value| (value - mean).powi(2)).sum::<f32>() / (frames as f32 - 1.0);
        let deviation = variance.sqrt() + 1e-5;
        for value in bin.iter_mut() {
            *value = (*value - mean) / deviation;
        }
    }
    Ok(mel)
}

/// Groups tokens into words as parakeet-rs does: a token with leading whitespace or pure
/// punctuation starts a word, while a contraction or hyphenated suffix joins the previous.
fn group_words(tokens: &[TimedToken]) -> Vec<TimedToken> {
    let Some(last) = tokens.last() else {
        return Vec::new();
    };
    let mut words = Vec::new();
    let mut text = String::new();
    let mut start = 0.0;
    for (index, token) in tokens.iter().enumerate() {
        let previous_end = || {
            index
                .checked_sub(1)
                .map_or(token.end, |index| tokens[index].end)
        };
        // A token of only whitespace separates words and adds no text.
        if token.text.trim().is_empty() {
            if !text.is_empty() {
                words.push(TimedToken {
                    text: std::mem::take(&mut text),
                    start,
                    end: previous_end(),
                });
            }
            continue;
        }
        let piece = token
            .text
            .trim_start_matches('\u{2581}')
            .trim_start_matches(' ');
        let punctuation = token
            .text
            .chars()
            .all(|character| character.is_ascii_punctuation());
        let joins_previous = piece.starts_with('\'') || piece.starts_with('-');
        let starts_word =
            (token.text.starts_with('\u{2581}') || token.text.starts_with(' ') || punctuation)
                && !joins_previous
                || index == 0;
        if starts_word && !text.is_empty() {
            words.push(TimedToken {
                text: std::mem::take(&mut text),
                start,
                end: previous_end(),
            });
        }
        if text.is_empty() {
            start = token.start;
        }
        text.push_str(piece);
    }
    if !text.is_empty() {
        words.push(TimedToken {
            text,
            start,
            end: last.end,
        });
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEFT: usize = 8 * FRAME_SAMPLES;
    const HOP: usize = 12 * FRAME_SAMPLES;
    const RIGHT: usize = 2 * FRAME_SAMPLES;

    #[test]
    fn windows_follow_the_buffered_streaming_layout() {
        let contexts = (LEFT, HOP, RIGHT);
        // The first window has no left context and waits for its right context.
        assert_eq!(
            plan_window(0, 0, HOP + RIGHT, contexts, false),
            Window {
                start: 0,
                end: HOP + RIGHT,
                context_frames: 0,
                frames: 12,
                advance: HOP,
            }
        );
        // Later windows reach back over the retained left context.
        assert_eq!(
            plan_window(HOP, HOP - LEFT, 2 * HOP + RIGHT, contexts, false),
            Window {
                start: HOP - LEFT,
                end: 2 * HOP + RIGHT,
                context_frames: 8,
                frames: 12,
                advance: HOP,
            }
        );
        // A longer hop over a backlog keeps the same contexts around it.
        let backlog = (LEFT, 3 * HOP, RIGHT);
        assert_eq!(
            plan_window(HOP, HOP - LEFT, 4 * HOP + RIGHT, backlog, false),
            Window {
                start: HOP - LEFT,
                end: 4 * HOP + RIGHT,
                context_frames: 8,
                frames: 36,
                advance: 3 * HOP,
            }
        );
        // The last window decodes what remains, without right context.
        assert_eq!(
            plan_window(HOP, HOP - LEFT, HOP + 5 * FRAME_SAMPLES, contexts, true),
            Window {
                start: HOP - LEFT,
                end: HOP + 5 * FRAME_SAMPLES,
                context_frames: 8,
                frames: 5,
                advance: 5 * FRAME_SAMPLES,
            }
        );
        // One frame past a full hop: the hop takes it as right context and leaves it
        // for a window of its own.
        let received = 2 * HOP + FRAME_SAMPLES;
        let full = plan_window(HOP, HOP - LEFT, received, contexts, true);
        assert_eq!((full.end, full.frames, full.advance), (received, 12, HOP));
        let rest = plan_window(2 * HOP, 2 * HOP - LEFT, received, contexts, true);
        assert_eq!((rest.start, rest.frames), (2 * HOP - LEFT, 1));
    }

    #[test]
    fn features_are_normalized_per_mel_bin() {
        let cache = FeatureCache::from_config(&PreprocessorConfig {
            feature_extractor_type: String::new(),
            feature_size: MEL_BINS,
            hop_length: FEATURE_HOP,
            n_fft: FFT_SIZE,
            padding_side: String::new(),
            padding_value: 0.0,
            preemphasis: PREEMPHASIS,
            processor_class: String::new(),
            return_attention_mask: true,
            sampling_rate: SAMPLE_RATE,
            win_length: WINDOW_LENGTH,
        });
        // A rising tone, so every mel bin varies over the window.
        let audio: Vec<f32> = (0..SAMPLE_RATE)
            .map(|index| {
                let seconds = index as f32 / SAMPLE_RATE as f32;
                (2.0 * PI * (300.0 + 2_000.0 * seconds) * seconds).sin() * 0.3
            })
            .collect();
        let features = log_mel(&audio, &cache).unwrap();
        assert_eq!(features.dim(), (100, MEL_BINS));
        // The tone crosses mel bins 20 to 70 (about 0.5 to 2 kHz) during the window.
        for bin in 20..70 {
            let bin = features.column(bin);
            let mean = bin.iter().sum::<f32>() / 100.0;
            let variance = bin.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / 99.0;
            assert!(mean.abs() < 1e-3, "{mean}");
            assert!((variance.sqrt() - 1.0).abs() < 1e-2, "{variance}");
        }
        // It rises, so a low bin peaks before a high one.
        let peak_frame = |bin: usize| {
            (0..100)
                .max_by(|&a, &b| features[[a, bin]].total_cmp(&features[[b, bin]]))
                .unwrap()
        };
        assert!(peak_frame(20) < peak_frame(60));
    }

    #[test]
    fn words_group_as_parakeet_rs_groups_them() {
        let token = |text: &str, frame: usize| TimedToken {
            text: text.into(),
            start: frame as f32 * 0.08,
            end: (frame + 1) as f32 * 0.08,
        };
        let words = group_words(&[
            token(" Don", 0),
            token("'t", 1),
            token(" stop", 3),
            token(",", 4),
            token(" twenty", 6),
            token("-two", 7),
            token(" ", 8),
            token("times", 9),
            token(".", 11),
        ]);
        let texts: Vec<_> = words.iter().map(|word| word.text.as_str()).collect();
        assert_eq!(texts, ["Don't", "stop", ",", "twenty-two", "times", "."]);
        // A word runs from its first token to the end of its last.
        assert_eq!((words[0].start, words[0].end), (0.0, 2.0 * 0.08));
        assert_eq!((words[3].start, words[3].end), (6.0 * 0.08, 8.0 * 0.08));
        assert_eq!(words[5].end, 12.0 * 0.08);
        assert!(group_words(&[]).is_empty());
    }
}
