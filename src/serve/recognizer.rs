//! Synchronous recognition interfaces. The inference worker owns the model and its
//! borrowing streams on one thread. Dropping a stream cancels it without flushing.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use qwen3_asr::{
    AsrStream, AsrTranscription, AudioInput, Batch, LoadOptions, Qwen3Asr, StreamOptions,
    TranscribeOptions, forced_aligner::Qwen3ForcedAligner,
};
use tokio::sync::{Notify, watch};

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
    /// On the recognizer's own clock. Rollover cuts and bleed decisions use it.
    pub start_sample: u64,
    pub end_sample: u64,
    /// When the word was spoken, from a recognizer whose clock runs behind the speech.
    /// Clients and speaker labels get these times in place of the recognizer's.
    pub spoken: Option<(u64, u64)>,
}

pub(super) trait Recognizer {
    fn start(&self, options: &RecognitionOptions) -> Result<Box<dyn RecognitionStream + '_>>;

    /// Whether a native stream decodes audio that arrives together in fewer, longer
    /// windows. The worker then collects a track's queued audio before decoding.
    fn coalesces_backlog(&self) -> bool {
        false
    }

    /// Native tracks may use a decoder profile of their own.
    fn start_native(
        &self,
        options: &RecognitionOptions,
    ) -> Result<Box<dyn RecognitionStream + '_>> {
        self.start(options)
    }

    /// Permanent loss of a backend that can fail independently of the worker thread.
    fn backend_failure(&self) -> Option<watch::Receiver<Option<String>>> {
        None
    }

    /// Retained audio that may seed a new native timed item after rollover.
    fn rollover_preroll_samples(&self) -> u64 {
        0
    }

    /// Signaled whenever a final requested by `RecognitionStream::request_final` may
    /// have resolved. A recognizer with only synchronous finals returns `None`.
    fn final_notify(&self) -> Option<Arc<Notify>> {
        None
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

    /// Feeds consecutive spans that are all available now. A recognizer that sizes its
    /// inference window by the audio waiting decodes them together.
    fn push_spans(&mut self, spans: &[&[f32]]) -> Result<Option<Hypothesis>> {
        let mut latest = None;
        for span in spans {
            if let Some(hypothesis) = self.push(span)? {
                latest = Some(hypothesis);
            }
        }
        Ok(latest)
    }
    /// Blocks until the final is available. Valid after `request_final` as well.
    fn finish(self: Box<Self>) -> Result<Hypothesis>;

    /// Asks for the final without waiting for it. `Some(deadline)` means the result
    /// arrives through `try_final`; `None` means the caller must use `finish`.
    fn request_final(&mut self) -> Result<Option<Instant>> {
        Ok(None)
    }

    /// The requested final once it has arrived, or a failure once `now` reaches the
    /// deadline. Never blocks.
    fn try_final(&mut self, _now: Instant) -> Option<Result<Hypothesis>> {
        None
    }
}

pub(super) struct CommittedStream<'a> {
    stream: Box<dyn RecognitionStream + 'a>,
    samples: Vec<f32>,
    flushed: bool,
}

impl<'a> CommittedStream<'a> {
    pub(super) fn new(stream: Box<dyn RecognitionStream + 'a>) -> Self {
        Self {
            stream,
            samples: Vec::new(),
            flushed: false,
        }
    }

    fn flush(&mut self) -> Result<()> {
        if !self.flushed {
            self.flushed = true;
            self.stream.push(&std::mem::take(&mut self.samples))?;
        }
        Ok(())
    }
}

impl RecognitionStream for CommittedStream<'_> {
    fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>> {
        self.samples.extend_from_slice(samples);
        Ok(None)
    }

    fn request_final(&mut self) -> Result<Option<Instant>> {
        self.flush()?;
        self.stream.request_final()
    }

    fn try_final(&mut self, now: Instant) -> Option<Result<Hypothesis>> {
        self.stream.try_final(now)
    }

    fn finish(mut self: Box<Self>) -> Result<Hypothesis> {
        self.flush()?;
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
            use parakeet_rs::{ExecutionConfig, ExecutionProvider};
            let provider = match device {
                DeviceKind::Metal => ExecutionProvider::BurnWgpu,
                DeviceKind::Cpu | DeviceKind::Coreml => ExecutionProvider::BurnCpu,
            };
            let recognizer = load_model(&cache, model, |path| {
                let config = ExecutionConfig::new().with_execution_provider(provider);
                parakeet::ParakeetRecognizer::load(path, config)
                    .context("loading Parakeet Unified for live transcription")
            })?;
            recognizer
                .warmup()
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
            spoken: None,
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
    use super::super::parakeet_profile::{
        STREAMING_CONFIG, final_padding_samples, single_window, warmup_window_frames,
    };
    use super::super::parakeet_stream::{FRAME_SAMPLES, StreamDecoder, UnifiedModel};
    use super::super::trace;
    use super::*;
    use crate::parakeet_engine::{ParakeetWord, estimate_word_ends};
    use parakeet_rs::{ExecutionConfig, TranscriptionResult, UnifiedStreamingConfig};

    const STANDARD: UnifiedStreamingConfig = UnifiedStreamingConfig {
        chunk_secs: 1.44,
        ..STREAMING_CONFIG
    };
    /// A native stream that has fallen behind decodes up to this many hops in one window.
    /// An encoder call costs about the same for 1 s as for 4 s of audio, so one long
    /// window catches up where a window per hop would fall further behind.
    const NATIVE_MAX_HOPS: usize = 3;
    const FRAME_SECONDS: f32 = FRAME_SAMPLES as f32 / SAMPLE_RATE as f32;
    /// A streaming window emits a word's first token some time after the word began,
    /// and later the less audio the window held before it. With only the 0.64 s left
    /// context before it, the token trails the onset by `LATENCY_AFTER_LEFT_CONTEXT`.
    /// Each further encoder frame heard takes off `LATENCY_PER_FRAME_HEARD`, down to
    /// `SHORTEST_LATENCY`, which a 0.96 s hop reaches at its end and a window over a
    /// backlog keeps. Measured on read speech against forced alignment, through the
    /// socket and with windows of one to three hops.
    const LEFT_CONTEXT_FRAMES: usize = 8;
    const LATENCY_AFTER_LEFT_CONTEXT: f32 = 0.41;
    const LATENCY_PER_FRAME_HEARD: f32 = 0.01;
    const SHORTEST_LATENCY: f32 = 0.3;
    /// A stream's first window has no left context. It emits nothing for about 0.7 s
    /// and then its first words together, `FIRST_WINDOW_LATENCY` after they began. A
    /// token it emits after `FIRST_WINDOW_SETTLING_FRAME` trails by
    /// `FIRST_WINDOW_LATENCY_PER_FRAME` less for each frame past it.
    const FIRST_WINDOW_LATENCY: f32 = 0.5;
    const FIRST_WINDOW_SETTLING_FRAME: usize = 9;
    const FIRST_WINDOW_LATENCY_PER_FRAME: f32 = 0.07;

    /// Seconds by which the token emitted at stream frame `frame` trails the onset of
    /// its word, when its window held `heard_before` encoder frames before it.
    fn emission_latency(frame: usize, heard_before: usize) -> f32 {
        // Only a window that starts the stream holds exactly the stream so far.
        if heard_before == frame {
            let settled = frame.saturating_sub(FIRST_WINDOW_SETTLING_FRAME) as f32;
            return (FIRST_WINDOW_LATENCY - settled * FIRST_WINDOW_LATENCY_PER_FRAME)
                .max(SHORTEST_LATENCY);
        }
        let further = heard_before.saturating_sub(LEFT_CONTEXT_FRAMES) as f32;
        (LATENCY_AFTER_LEFT_CONTEXT - further * LATENCY_PER_FRAME_HEARD).max(SHORTEST_LATENCY)
    }

    pub(super) struct ParakeetRecognizer {
        model: Arc<UnifiedModel>,
    }

    impl ParakeetRecognizer {
        pub(super) fn load(path: &std::path::Path, execution: ExecutionConfig) -> Result<Self> {
            Ok(Self {
                model: UnifiedModel::load(path, execution)?,
            })
        }

        /// Runs the encoder once on every window length a stream can produce, before the
        /// listener reports ready. The GPU backend prepares its kernels for a length the
        /// first time it sees one, which takes up to a second (more on a busy host) and
        /// would otherwise stall the first stream that needs it. A full window is one hop
        /// or a backlog of several; the last window of an utterance covers whatever
        /// remains, so it can be any shorter length.
        pub(super) fn warmup(&self) -> Result<()> {
            let mut profiles = vec![STANDARD];
            profiles.extend((1..=NATIVE_MAX_HOPS).map(|hops| UnifiedStreamingConfig {
                chunk_secs: hops as f32 * STREAMING_CONFIG.chunk_secs,
                ..STREAMING_CONFIG
            }));
            for frames in 1..=warmup_window_frames(&profiles) {
                warm(&self.model, frames)?;
            }
            Ok(())
        }
    }

    fn warm(model: &Arc<UnifiedModel>, encoder_frames: usize) -> Result<()> {
        let mut decoder = StreamDecoder::new(model, single_window(encoder_frames))?;
        let samples = encoder_frames * FRAME_SAMPLES;
        decoder.extend(&vec![0.0; samples]);
        decoder.decode_window(samples, false)?;
        Ok(())
    }

    struct ParakeetStream {
        decoder: StreamDecoder,
        received_samples: usize,
        drafts: bool,
        /// The profile's hop: the shortest window a push decodes, and the grid on which
        /// every window starts.
        hop_samples: usize,
        /// The longest window. Equal to the hop on a stream with fixed windows.
        max_hop_samples: usize,
    }

    impl ParakeetStream {
        fn open(
            recognizer: &ParakeetRecognizer,
            config: UnifiedStreamingConfig,
            drafts: bool,
            max_hops: usize,
        ) -> Result<Box<Self>> {
            let hop_samples = config.chunk_samples();
            let decoder = StreamDecoder::new(&recognizer.model, config)?;
            Ok(Box::new(Self {
                decoder,
                received_samples: 0,
                drafts,
                hop_samples,
                max_hop_samples: max_hops * hop_samples,
            }))
        }

        /// Decodes every buffered window that has its right context. A
        /// window covers the whole hops waiting, up to the longest, so a stream that has
        /// fallen behind catches up in one encoder call. Reports whether text was added.
        fn decode_ready(&mut self) -> Result<bool> {
            let mut emitted = false;
            loop {
                let waiting = self
                    .decoder
                    .pending_samples()
                    .saturating_sub(self.decoder.right_context_samples());
                let hops = waiting / self.hop_samples * self.hop_samples;
                if hops == 0 {
                    return Ok(emitted);
                }
                emitted |= self.window("push", hops.min(self.max_hop_samples), false)?;
            }
        }

        fn window(&mut self, call: &str, hop: usize, last: bool) -> Result<bool> {
            let started = trace::enabled().then(Instant::now);
            let emitted = self.decoder.decode_window(hop, last)?;
            if let Some(started) = started {
                trace::emit(
                    "decode",
                    serde_json::json!({
                        "call": call,
                        "ms": started.elapsed().as_secs_f64() * 1e3,
                        "received_samples": self.received_samples,
                        "hop_secs": hop as f32 / SAMPLE_RATE as f32,
                        "drafts": self.drafts,
                        "emitted": emitted,
                    }),
                );
            }
            Ok(emitted)
        }
    }

    impl Recognizer for ParakeetRecognizer {
        fn start(&self, _options: &RecognitionOptions) -> Result<Box<dyn RecognitionStream + '_>> {
            // Standard turns keep the fixed 1.44 s hop their fixtures were accepted with.
            Ok(ParakeetStream::open(self, STANDARD, true, 1)?)
        }

        fn start_native(
            &self,
            _options: &RecognitionOptions,
        ) -> Result<Box<dyn RecognitionStream + '_>> {
            Ok(ParakeetStream::open(
                self,
                STREAMING_CONFIG,
                true,
                NATIVE_MAX_HOPS,
            )?)
        }

        fn start_committed(
            &self,
            _options: &RecognitionOptions,
        ) -> Result<Box<dyn RecognitionStream + '_>> {
            // The worker already retained this turn through commit and feeds its spans
            // directly, so the stream never holds more than a window of PCM.
            Ok(ParakeetStream::open(self, STANDARD, false, 1)?)
        }

        fn coalesces_backlog(&self) -> bool {
            true
        }

        fn rollover_preroll_samples(&self) -> u64 {
            // The new decoder needs the tested 1.44 s of preceding speech at a forced cut.
            STANDARD.chunk_samples() as u64
        }
    }

    impl RecognitionStream for ParakeetStream {
        fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>> {
            self.push_spans(&[samples])
        }

        fn push_spans(&mut self, spans: &[&[f32]]) -> Result<Option<Hypothesis>> {
            // One buffer lets the decoder see everything waiting before it sizes a window.
            for span in spans {
                self.decoder.extend(span);
                self.received_samples += span.len();
            }
            let emitted = self.decode_ready()?;
            Ok((self.drafts && emitted).then(|| self.hypothesis(false)))
        }

        fn finish(mut self: Box<Self>) -> Result<Hypothesis> {
            // Windows are whole encoder frames. Keep the last partial frame by padding
            // here, after the item has recorded its real duration.
            let padding = final_padding_samples(self.received_samples, STREAMING_CONFIG);
            self.decoder.extend(&[0.0; FRAME_SAMPLES][..padding]);
            self.decode_ready()?;
            // Nothing follows what is left, so it is decoded without right context: in
            // one window on a stream with variable windows, else hop by hop.
            while self.decoder.pending_samples() >= FRAME_SAMPLES {
                let rest = self.decoder.pending_samples() / FRAME_SAMPLES * FRAME_SAMPLES;
                let hop = if self.max_hop_samples > self.hop_samples {
                    rest.min(self.max_hop_samples)
                } else {
                    self.hop_samples
                };
                self.window("finish", hop, true)?;
            }
            Ok(self.hypothesis(true))
        }
    }

    /// The decoder's words on its own clock, each with the time it was spoken.
    /// `heard_before` gives the encoder frames a frame's window held before it, and
    /// `received_samples` is the audio the stream has been given.
    fn hypothesis(
        result: TranscriptionResult,
        heard_before: impl Fn(usize) -> usize,
        received_samples: usize,
        final_result: bool,
    ) -> Hypothesis {
        let mut remainder = result.text.as_str();
        let mut words = Vec::with_capacity(result.tokens.len());
        let mut spoken = Vec::with_capacity(result.tokens.len());
        let mut floor = 0.0f32;
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
                spoken: None,
            });
            let frame = (word.start / FRAME_SECONDS).round() as usize;
            // Onsets stay in order and inside the stream.
            let onset = (word.start - emission_latency(frame, heard_before(frame))).max(floor);
            floor = onset;
            spoken.push(ParakeetWord {
                text: word.text,
                start: onset,
                end: onset,
            });
        }
        place_spoken(&mut words, spoken, received_samples);
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

    /// Records when each word was spoken, from its corrected onset. A word runs to the
    /// next onset in continuous speech and keeps an estimated length before a pause, as
    /// in a recording decoded whole. Words the decoder emitted in one frame share an
    /// onset: they divide the time to the next onset by their length in letters.
    /// Punctuation closes the word before it.
    fn place_spoken(words: &mut [TimedWord], mut spoken: Vec<ParakeetWord>, received: usize) {
        if words.len() != spoken.len() {
            return;
        }
        estimate_word_ends(&mut spoken, received as f32 / SAMPLE_RATE as f32);
        let letters = |word: &ParakeetWord| {
            word.text
                .chars()
                .filter(|letter| letter.is_alphanumeric())
                .count()
        };
        let voiced: Vec<usize> = (0..spoken.len())
            .filter(|&index| letters(&spoken[index]) > 0)
            .collect();
        let mut first = 0;
        while first < voiced.len() {
            let onset = spoken[voiced[first]].start;
            let run = voiced[first..]
                .iter()
                .take_while(|&&index| spoken[index].start == onset)
                .count();
            let group = &voiced[first..first + run];
            // The run ends where its last word does, and no later than the next onset.
            let mut end = spoken[group[run - 1]].end.max(onset);
            if let Some(&next) = voiced.get(first + run) {
                end = end.min(spoken[next].start);
            }
            let total: usize = group.iter().map(|&index| letters(&spoken[index])).sum();
            let mut before = 0;
            for &index in group {
                let share = |count: usize| onset + (end - onset) * count as f32 / total as f32;
                spoken[index].start = share(before);
                before += letters(&spoken[index]);
                spoken[index].end = share(before);
            }
            first += run;
        }
        let samples = |seconds: f32| (seconds * SAMPLE_RATE as f32).round() as u64;
        let mut previous_end = None;
        for (word, timed) in spoken.iter().zip(words) {
            timed.spoken = Some(if letters(word) > 0 {
                let bound = (samples(word.start), samples(word.end));
                previous_end = Some(bound.1);
                bound
            } else {
                let at = previous_end.unwrap_or_else(|| samples(word.start));
                (at, at)
            });
        }
    }

    impl ParakeetStream {
        fn hypothesis(&self, final_result: bool) -> Hypothesis {
            hypothesis(
                self.decoder.transcript(),
                |frame| self.decoder.frames_heard_before(frame),
                self.received_samples,
                final_result,
            )
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
            let draft = hypothesis(result.clone(), |_| 20, 16_000, false);
            let finalized = hypothesis(result, |_| 20, 16_000, true);
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

        /// A token of `text` emitted at encoder frame `frame`.
        fn token(text: &str, frame: usize) -> TimedToken {
            TimedToken {
                text: text.into(),
                start: frame as f32 * FRAME_SECONDS,
                end: (frame + 1) as f32 * FRAME_SECONDS,
            }
        }

        /// Windows of one 12-frame hop: the first with no left context, the rest with 8.
        fn single_hops(frame: usize) -> usize {
            if frame < 12 { frame } else { frame % 12 + 8 }
        }

        fn close(samples: u64, seconds: f32) -> bool {
            samples.abs_diff((seconds * SAMPLE_RATE as f32).round() as u64) <= 2
        }

        #[test]
        fn latency_is_longest_in_a_first_window_and_shrinks_with_audio_heard() {
            assert_eq!(emission_latency(9, 9), FIRST_WINDOW_LATENCY);
            assert_eq!(emission_latency(12, 8), LATENCY_AFTER_LEFT_CONTEXT);
            assert!(emission_latency(9, 9) > emission_latency(12, 8));
            // The first window settles toward the latency of the windows after it.
            assert!(emission_latency(11, 11) < emission_latency(12, 8));
            assert!(emission_latency(11, 11) > SHORTEST_LATENCY);
            assert!(emission_latency(23, 19) < emission_latency(13, 9));
            // A window over a backlog hears more, down to the latency of a whole decode.
            assert_eq!(emission_latency(47, 43), SHORTEST_LATENCY);
            assert_eq!(emission_latency(200, 43), SHORTEST_LATENCY);
        }

        #[test]
        fn words_carry_the_time_they_were_spoken_beside_the_decoder_clock() {
            let result = TranscriptionResult {
                text: "Hello there. Friend".into(),
                tokens: vec![
                    token("Hello", 9),
                    token("there", 14),
                    token(".", 15),
                    token("Friend", 40),
                ],
            };
            let onset =
                |frame| frame as f32 * FRAME_SECONDS - emission_latency(frame, single_hops(frame));
            let finalized = hypothesis(result.clone(), single_hops, 4 * 16_000, true);
            let spoken: Vec<_> = finalized
                .words
                .iter()
                .map(|word| word.spoken.unwrap())
                .collect();
            // The decoder's clock is untouched.
            assert_eq!(finalized.words[0].start_sample, 9 * 1_280);
            assert_eq!(finalized.words[3].end_sample, 41 * 1_280);
            // Continuous speech: a word runs from its onset to the next.
            assert!(close(spoken[0].0, onset(9)) && close(spoken[0].1, onset(14)));
            assert!(close(spoken[1].0, onset(14)));
            // Before a pause it keeps an estimated length instead.
            assert!(spoken[1].1 > spoken[1].0 && spoken[1].1 < spoken[3].0 - 16_000);
            // Punctuation closes the word before it.
            assert_eq!(spoken[2], (spoken[1].1, spoken[1].1));
            assert!(close(spoken[3].0, onset(40)));
            assert!(spoken[3].1 > spoken[3].0 && spoken[3].1 <= 4 * 16_000);

            // A draft withholds its last word, whose onset still bounds the one before.
            let draft = hypothesis(result, single_hops, 4 * 16_000, false);
            let drafted: Vec<_> = draft
                .words
                .iter()
                .map(|word| word.spoken.unwrap())
                .collect();
            assert_eq!(drafted, spoken[..3]);
        }

        #[test]
        fn spoken_times_stay_ordered_inside_the_stream() {
            let result = TranscriptionResult {
                text: "Oh, no wait".into(),
                tokens: vec![
                    token("Oh", 2),
                    token(",", 2),
                    token("no", 3),
                    token("wait", 20),
                ],
            };
            let finalized = hypothesis(result, single_hops, 21 * 1_280, true);
            let spoken: Vec<_> = finalized
                .words
                .iter()
                .map(|word| word.spoken.unwrap())
                .collect();
            // Both onsets fall before the stream began and are held at its start, where
            // the two words share the time to the next onset.
            assert_eq!(spoken[0].0, 0);
            assert!(spoken[0].1 > 0);
            assert_eq!(spoken[1], (spoken[0].1, spoken[0].1));
            assert_eq!(spoken[2].0, spoken[0].1);
            assert_eq!(spoken[2].1 - spoken[2].0, spoken[0].1);
            assert!(spoken[2].1 <= spoken[3].0);
            // The last word cannot outlast the audio received.
            assert_eq!(spoken[3].1, 21 * 1_280);
        }
    }
}
