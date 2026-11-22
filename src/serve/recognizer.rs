//! Synchronous recognition interfaces. The inference worker owns the model and its
//! borrowing streams on one thread. Dropping a stream cancels it without flushing.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use qwen3_asr::{
    AsrStream, AsrTranscription, AudioInput, Batch, LoadOptions, Qwen3Asr, StreamOptions,
    TranscribeOptions, forced_aligner::Qwen3ForcedAligner,
};
use tokio::sync::watch;

use crate::{
    DeviceKind, EngineKind, QWEN_DEFAULT_CHUNK_SECONDS, QWEN_TOKENS_PER_SECOND, chunk_ranges,
    load_model,
    model_cache::ModelCache,
    transcript::{Word, normalize_word_timing, preserve_word_text},
};

pub(super) const SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Debug, Default)]
pub(super) struct RecognitionOptions {
    pub prompt: Option<String>,
    pub languages: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Hypothesis {
    /// Entire current text, including the prefix returned by previous pushes.
    pub text: String,
    /// Whether the text can only grow by appending characters.
    pub stable: bool,
    pub words: Vec<TimedWord>,
    /// ISO language code when the recognizer detected one unambiguous language.
    pub language: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct TimedWord {
    /// Exact fragment of the transcript, with its preceding whitespace if present.
    pub text: String,
    pub start_sample: u64,
    pub end_sample: u64,
}

pub(super) trait Recognizer {
    fn start(&self, options: &RecognitionOptions) -> Result<Box<dyn RecognitionStream + '_>>;

    /// Permanent loss of a backend that can fail independently of the worker thread.
    fn backend_failure(&self) -> Option<watch::Receiver<Option<String>>> {
        None
    }

    /// Retained audio that may seed a new native timed item after rollover.
    fn rollover_preroll_samples(&self) -> u64 {
        0
    }

    fn start_committed(
        &self,
        options: &RecognitionOptions,
    ) -> Result<Box<dyn RecognitionStream + '_>> {
        Ok(Box::new(CommittedStream::new(self.start(options)?)))
    }
}

pub(super) trait RecognitionStream {
    fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>>;
    fn finish(self: Box<Self>) -> Result<Hypothesis>;
}

pub(super) struct CommittedStream<'a> {
    stream: Box<dyn RecognitionStream + 'a>,
    samples: Vec<f32>,
}

impl<'a> CommittedStream<'a> {
    pub(super) fn new(stream: Box<dyn RecognitionStream + 'a>) -> Self {
        Self {
            stream,
            samples: Vec::new(),
        }
    }
}

impl RecognitionStream for CommittedStream<'_> {
    fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>> {
        self.samples.extend_from_slice(samples);
        Ok(None)
    }

    fn finish(mut self: Box<Self>) -> Result<Hypothesis> {
        self.stream.push(&self.samples)?;
        self.stream.finish()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Capabilities {
    pub stable_deltas: bool,
    pub revisable_drafts: bool,
    pub word_timing: bool,
    pub word_timing_granularity: Option<&'static str>,
    pub draft_word_timing: bool,
    pub prompt: bool,
    pub languages: &'static [&'static str],
}

impl Capabilities {
    pub(super) fn supports_language(&self, language: &str) -> bool {
        self.languages.contains(&language)
    }
}

const ENGLISH: &[&str] = &["en"];
const QWEN_LANGUAGES: &[&str] = &[
    "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi", "ja",
    "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "fil", "fa", "el", "ro", "hu", "mk",
];

pub(super) fn capabilities(engine: EngineKind) -> Capabilities {
    capabilities_with_aligner(engine, false)
}

pub(super) fn capabilities_with_aligner(engine: EngineKind, aligner: bool) -> Capabilities {
    match engine {
        EngineKind::AppleNative => Capabilities {
            stable_deltas: false,
            revisable_drafts: true,
            word_timing: true,
            word_timing_granularity: Some("word"),
            draft_word_timing: false,
            prompt: false,
            languages: ENGLISH,
        },
        EngineKind::Qwen3 => Capabilities {
            stable_deltas: false,
            revisable_drafts: true,
            word_timing: aligner,
            word_timing_granularity: aligner.then_some("word"),
            draft_word_timing: false,
            prompt: true,
            languages: QWEN_LANGUAGES,
        },
        EngineKind::ParakeetV3 => Capabilities {
            stable_deltas: true,
            revisable_drafts: false,
            word_timing: true,
            word_timing_granularity: Some("word"),
            draft_word_timing: true,
            prompt: false,
            languages: ENGLISH,
        },
    }
}

pub(super) fn load_with_aligner(
    engine: EngineKind,
    model: &str,
    device: DeviceKind,
    offline: bool,
    model_cache: Option<PathBuf>,
    forced_aligner: Option<&str>,
) -> Result<Box<dyn Recognizer>> {
    if forced_aligner.is_some() && !matches!(engine, EngineKind::Qwen3) {
        bail!("--forced-aligner requires the qwen3 engine");
    }
    let cache = ModelCache::new(model_cache, offline);
    match engine {
        EngineKind::AppleNative => {
            if model != crate::default_asr(engine) {
                bail!("apple-native uses the system speech model and takes no model override");
            }
            Ok(Box::new(super::apple::AppleRecognizer::load(offline)?))
        }
        EngineKind::Qwen3 => {
            let (device, dtype) = match device {
                DeviceKind::Metal => (Device::new_metal(0)?, DType::BF16),
                DeviceKind::Cpu | DeviceKind::Coreml => (Device::Cpu, DType::F32),
            };
            let options = LoadOptions {
                dtype,
                ..LoadOptions::default()
            };
            let model = load_model(&cache, model, |path| {
                let path = path
                    .to_str()
                    .context("Qwen model path is not valid UTF-8")?;
                let model = Qwen3Asr::from_pretrained(path, &device, &options)
                    .context("loading Qwen3-ASR")?;
                model.require_ready()?;
                Ok(model)
            })?;
            let aligner = forced_aligner
                .map(|id| {
                    load_model(&cache, id, |path| {
                        let path = path
                            .to_str()
                            .context("forced aligner path is not valid UTF-8")?;
                        Qwen3ForcedAligner::from_pretrained(path, &device, &options)
                            .context("loading Qwen3 forced aligner")
                    })
                })
                .transpose()?;
            Ok(Box::new(QwenRecognizer { model, aligner }))
        }
        #[cfg(feature = "parakeet")]
        EngineKind::ParakeetV3 => {
            use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetUnifiedHandle};
            let provider = match device {
                DeviceKind::Metal => ExecutionProvider::BurnWgpu,
                DeviceKind::Cpu | DeviceKind::Coreml => ExecutionProvider::BurnCpu,
            };
            let recognizer = load_model(&cache, model, |path| {
                let config = ExecutionConfig::new().with_execution_provider(provider);
                ParakeetUnifiedHandle::from_pretrained(path, Some(config))
                    .context("loading Parakeet Unified for live transcription")
            })?;
            let mut probe = parakeet::decoder(&recognizer)
                .context("initializing the Parakeet Unified startup probe")?;
            let samples = super::parakeet_profile::warmup_samples(probe.streaming_config());
            probe
                .transcribe_chunk(&vec![0.0; samples])
                .context("warming the Parakeet Unified streaming model")?;
            Ok(Box::new(recognizer))
        }
        #[cfg(not(feature = "parakeet"))]
        EngineKind::ParakeetV3 => {
            bail!(
                "Parakeet support is not included in this build; rebuild with --features parakeet"
            )
        }
    }
}

fn qwen_language(options: &RecognitionOptions) -> Result<Option<String>> {
    use qwen3_asr::inference::utils::SUPPORTED_LANGUAGES;
    let capabilities = capabilities(EngineKind::Qwen3);
    for code in &options.languages {
        if !capabilities.supports_language(code) {
            bail!("Qwen does not support language {code}");
        }
    }
    let Some(first) = options.languages.first() else {
        return Ok(None);
    };
    if options.languages.iter().any(|code| code != first) {
        // Qwen can force one language or detect it, including on mixed-language audio.
        return Ok(None);
    }
    Ok(QWEN_LANGUAGES
        .iter()
        .zip(SUPPORTED_LANGUAGES)
        .find_map(|(&code, &name)| (code == first).then(|| name.to_owned())))
}

struct QwenRecognizer {
    model: Qwen3Asr,
    aligner: Option<Qwen3ForcedAligner>,
}

impl Recognizer for QwenRecognizer {
    fn start(&self, options: &RecognitionOptions) -> Result<Box<dyn RecognitionStream + '_>> {
        let stream = self.model.start_stream(StreamOptions {
            context: options.prompt.clone().unwrap_or_default(),
            language: qwen_language(options)?,
            chunk_size_sec: 1.0,
            audio_window_sec: Some(8.0),
            text_window_tokens: Some(96),
            ..StreamOptions::default()
        })?;
        Ok(Box::new(QwenStream {
            stream,
            aligner: self.aligner.as_ref(),
            audio: Vec::new(),
        }))
    }

    fn start_committed(
        &self,
        options: &RecognitionOptions,
    ) -> Result<Box<dyn RecognitionStream + '_>> {
        let language = qwen_language(options)?;
        let detect_language = language.is_none();
        let options = TranscribeOptions {
            context: Batch::one(options.prompt.clone().unwrap_or_default()),
            language: Batch::one(language),
            max_new_tokens: (QWEN_TOKENS_PER_SECOND * QWEN_DEFAULT_CHUNK_SECONDS).ceil() as usize,
            max_batch_size: 1,
            ..TranscribeOptions::default()
        };
        Ok(Box::new(QwenCommitted {
            model: &self.model,
            options,
            detect_language,
            samples: Vec::new(),
        }))
    }
}

struct QwenCommitted<'a> {
    model: &'a Qwen3Asr,
    options: TranscribeOptions,
    detect_language: bool,
    samples: Vec<f32>,
}

impl RecognitionStream for QwenCommitted<'_> {
    fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>> {
        self.samples.extend_from_slice(samples);
        Ok(None)
    }

    fn finish(self: Box<Self>) -> Result<Hypothesis> {
        // Choose quiet cut points as the batch path does and join texts explicitly.
        // Qwen's internal splitting concatenates results without spacing.
        let chunk_samples = (QWEN_DEFAULT_CHUNK_SECONDS * SAMPLE_RATE as f32) as usize;
        let mut chunks = Vec::new();
        for range in chunk_ranges(&self.samples, chunk_samples) {
            let result = self.model.transcribe(
                vec![AudioInput::Waveform {
                    samples: &self.samples[range],
                    sample_rate: SAMPLE_RATE,
                }],
                self.options.clone(),
            )?;
            chunks.push(
                result
                    .into_iter()
                    .next()
                    .context("Qwen returned no transcription")?,
            );
        }
        Ok(qwen_committed(chunks, self.detect_language))
    }
}

fn qwen_committed(chunks: Vec<AsrTranscription>, detect_language: bool) -> Hypothesis {
    use qwen3_asr::inference::utils::SUPPORTED_LANGUAGES;

    let spoken: Vec<_> = chunks
        .into_iter()
        .filter(|chunk| !chunk.text.trim().is_empty())
        .collect();
    let language = spoken
        .first()
        .filter(|first| {
            detect_language && spoken.iter().all(|chunk| chunk.language == first.language)
        })
        .and_then(|first| {
            SUPPORTED_LANGUAGES
                .iter()
                .zip(QWEN_LANGUAGES)
                .find_map(|(&language, &code)| {
                    language
                        .eq_ignore_ascii_case(&first.language)
                        .then(|| code.to_owned())
                })
        });
    Hypothesis {
        text: spoken
            .into_iter()
            .map(|chunk| chunk.text)
            .collect::<Vec<_>>()
            .join(" "),
        stable: true,
        words: Vec::new(),
        language,
    }
}

struct QwenStream<'a> {
    stream: AsrStream<'a>,
    aligner: Option<&'a Qwen3ForcedAligner>,
    audio: Vec<f32>,
}

impl RecognitionStream for QwenStream<'_> {
    fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>> {
        if self.aligner.is_some() {
            self.audio.extend_from_slice(samples);
        }
        let result = self.stream.push_audio_chunk(&AudioInput::Waveform {
            samples,
            sample_rate: SAMPLE_RATE,
        })?;
        Ok(result.map(|result| Hypothesis {
            text: result.text,
            ..Hypothesis::default()
        }))
    }

    fn finish(self: Box<Self>) -> Result<Hypothesis> {
        Ok(qwen_final(self.stream.finish()?, self.aligner, &self.audio))
    }
}

fn qwen_final(
    result: AsrTranscription,
    aligner: Option<&Qwen3ForcedAligner>,
    samples: &[f32],
) -> Hypothesis {
    // An unsupported detected language or an unplaceable alignment should leave
    // the authoritative transcript available with no word timestamps.
    let words = aligner
        .and_then(|aligner| align_qwen_words(aligner, &result, samples).ok())
        .unwrap_or_default();
    Hypothesis {
        text: result.text,
        stable: true,
        words,
        language: None,
    }
}

fn align_qwen_words(
    aligner: &Qwen3ForcedAligner,
    result: &AsrTranscription,
    samples: &[f32],
) -> Result<Vec<TimedWord>> {
    if result.text.trim().is_empty() || result.language.is_empty() || samples.is_empty() {
        return Ok(Vec::new());
    }
    let aligned = aligner.align(
        &[AudioInput::Waveform {
            samples,
            sample_rate: SAMPLE_RATE,
        }],
        std::slice::from_ref(&result.text),
        std::slice::from_ref(&result.language),
    )?;
    let mut words: Vec<_> = aligned
        .into_iter()
        .flat_map(|result| result.items)
        .map(|item| Word {
            text: item.text,
            start: item.start_time as f32,
            end: item.end_time as f32,
            speaker: None,
        })
        .collect();
    preserve_word_text(&result.text, &mut words)?;
    normalize_word_timing(&mut words, samples.len() as f32 / SAMPLE_RATE as f32);
    Ok(words
        .into_iter()
        .map(|word| TimedWord {
            text: word.text,
            start_sample: (word.start * SAMPLE_RATE as f32).round() as u64,
            end_sample: (word.end * SAMPLE_RATE as f32).round() as u64,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_accepts_supported_language_lists() {
        let options = |codes: &[&str]| RecognitionOptions {
            languages: codes.iter().map(|code| (*code).to_owned()).collect(),
            ..RecognitionOptions::default()
        };
        assert_eq!(
            qwen_language(&options(&["yue", "yue"])).unwrap().as_deref(),
            Some("Cantonese"),
        );
        assert_eq!(
            qwen_language(&options(&["mk"])).unwrap().as_deref(),
            Some("Macedonian"),
        );
        assert_eq!(qwen_language(&options(&["en", "es"])).unwrap(), None);
        assert!(qwen_language(&options(&["en", "es", "zzz"])).is_err());
    }

    #[test]
    fn qwen_reports_only_detected_supported_unambiguous_language() {
        let result = |names: &[&str], detect_language| {
            let chunks = names
                .iter()
                .map(|&language| AsrTranscription {
                    text: "speech".to_owned(),
                    language: language.to_owned(),
                    timestamps: None,
                })
                .collect();
            qwen_committed(chunks, detect_language)
        };
        assert_eq!(
            result(&["English", "English"], true).language.as_deref(),
            Some("en")
        );
        assert_eq!(
            result(&["Cantonese"], true).language.as_deref(),
            Some("yue")
        );
        assert_eq!(
            result(&["Macedonian"], true).language.as_deref(),
            Some("mk")
        );
        assert_eq!(result(&["English"], false).language, None);
        assert_eq!(result(&["English", "Spanish"], true).language, None);
        assert_eq!(result(&["English", ""], true).language, None);
        assert_eq!(result(&["English,Spanish"], true).language, None);
        assert_eq!(result(&["Unknown"], true).language, None);
        assert_eq!(result(&[], true).language, None);
    }
}

#[cfg(feature = "parakeet")]
mod parakeet {
    use super::super::parakeet_profile::{STREAMING_CONFIG, final_padding_samples};
    use super::*;
    use parakeet_rs::{ParakeetUnified, ParakeetUnifiedHandle, TimestampMode, TranscriptionResult};

    pub(super) fn decoder(model: &ParakeetUnifiedHandle) -> Result<ParakeetUnified> {
        Ok(ParakeetUnified::from_shared_with_streaming_config(
            model,
            STREAMING_CONFIG,
        )?)
    }

    struct ParakeetStream {
        decoder: ParakeetUnified,
        received_samples: usize,
        drafts: bool,
    }

    impl ParakeetStream {
        fn open(model: &ParakeetUnifiedHandle, drafts: bool) -> Result<Box<Self>> {
            Ok(Box::new(Self {
                decoder: decoder(model)?,
                received_samples: 0,
                drafts,
            }))
        }
    }

    impl Recognizer for ParakeetUnifiedHandle {
        fn start(&self, _options: &RecognitionOptions) -> Result<Box<dyn RecognitionStream + '_>> {
            Ok(ParakeetStream::open(self, true)?)
        }

        fn start_committed(
            &self,
            _options: &RecognitionOptions,
        ) -> Result<Box<dyn RecognitionStream + '_>> {
            // The worker already retained this turn through commit. Feed its spans directly:
            // Parakeet drains one hop from its PCM buffer per window, so giving it the entire
            // turn at once would shift all remaining samples on every inference.
            Ok(ParakeetStream::open(self, false)?)
        }

        fn rollover_preroll_samples(&self) -> u64 {
            STREAMING_CONFIG.chunk_samples() as u64
        }
    }

    impl RecognitionStream for ParakeetStream {
        fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>> {
            self.received_samples += samples.len();
            let emitted = self.decoder.transcribe_chunk(samples)?;
            if !self.drafts || emitted.is_empty() {
                return Ok(None);
            }
            Ok(Some(hypothesis(
                self.decoder.get_timed_transcript(TimestampMode::Words),
                false,
            )))
        }

        fn finish(mut self: Box<Self>) -> Result<Hypothesis> {
            // The decoder's flush processes only complete encoder frames. Keep the last
            // partial frame by padding here, after the item has recorded its real duration.
            let padding =
                final_padding_samples(self.received_samples, self.decoder.streaming_config());
            if padding > 0 {
                self.decoder.transcribe_chunk(&vec![0.0; padding])?;
            }
            self.decoder.flush()?;
            Ok(hypothesis(
                self.decoder.get_timed_transcript(TimestampMode::Words),
                true,
            ))
        }
    }

    fn hypothesis(result: TranscriptionResult, final_result: bool) -> Hypothesis {
        let mut remainder = result.text.as_str();
        let mut words = Vec::with_capacity(result.tokens.len());
        for word in result.tokens {
            let whitespace = remainder.len() - remainder.trim_start().len();
            let Some(next) = remainder[whitespace..].strip_prefix(&word.text) else {
                // A model/version that groups words differently cannot safely supply
                // exact transcript fragments for timings. Keep its authoritative text.
                words.clear();
                break;
            };
            let fragment = &remainder[..remainder.len() - next.len()];
            remainder = next;
            words.push(TimedWord {
                text: fragment.to_owned(),
                start_sample: (word.start * SAMPLE_RATE as f32).round() as u64,
                end_sample: (word.end * SAMPLE_RATE as f32).round() as u64,
            });
        }
        if !remainder.trim().is_empty() {
            words.clear();
        } else if final_result {
            if let Some(last) = words.last_mut() {
                last.text.push_str(remainder);
            }
        } else {
            // The next decoder step may extend the last word, even though its
            // previously emitted characters are stable.
            words.pop();
        }
        Hypothesis {
            text: result.text,
            stable: true,
            words,
            language: None,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use parakeet_rs::TimedToken;

        #[test]
        fn final_padding_keeps_partial_encoder_frames() {
            let padding = |samples| final_padding_samples(samples, STREAMING_CONFIG);
            assert_eq!(padding(0), 0);
            assert_eq!(padding(1), 1279);
            assert_eq!(padding(1279), 1);
            assert_eq!(padding(1280), 0);

            let chunk = STREAMING_CONFIG.chunk_samples();
            assert_eq!(padding(chunk + 640), 640);
        }

        #[test]
        fn fragments_preserve_exact_spacing_and_punctuation() {
            let result = TranscriptionResult {
                text: "Well,  well.".into(),
                tokens: ["Well", ",", "well", "."]
                    .into_iter()
                    .map(|text| TimedToken {
                        text: text.into(),
                        start: 0.25,
                        end: 0.5,
                    })
                    .collect(),
            };
            let draft = hypothesis(result.clone(), false);
            let finalized = hypothesis(result, true);
            let fragments: Vec<_> = finalized
                .words
                .iter()
                .map(|word| word.text.as_str())
                .collect();
            assert_eq!(fragments, ["Well", ",", "  well", "."]);
            assert_eq!(fragments.concat(), finalized.text);
            assert_eq!(draft.words.len(), 3);
            assert_eq!(draft.text, finalized.text);
            assert_eq!(finalized.words[0].start_sample, 4000);
            assert_eq!(finalized.words[0].end_sample, 8000);
        }
    }
}
