//! Manual check that the listener's Parakeet stream (`src/serve/parakeet_stream.rs`) decodes
//! exactly as `parakeet_rs::ParakeetUnified` does at a fixed hop. Pass the Unified model
//! directory and one or more raw 24 kHz mono signed little-endian PCM16 recordings. Prints
//! one JSON line per recording and hop, and fails if any transcript or word timing differs.
//! Each line also reports how many words change when backlogged windows span several hops.
//! Run it after upgrading parakeet-rs.

#[cfg(feature = "parakeet")]
#[allow(dead_code)]
#[path = "../src/serve/parakeet_profile.rs"]
mod parakeet_profile;
#[cfg(feature = "parakeet")]
#[allow(dead_code)]
#[path = "../src/serve/parakeet_stream.rs"]
mod parakeet_stream;

#[cfg(feature = "parakeet")]
mod parity {
    use super::parakeet_profile::{STREAMING_CONFIG, final_padding_samples};
    use super::parakeet_stream::{FRAME_SAMPLES, StreamDecoder, UnifiedModel};
    use anyhow::{Context, Result, ensure};
    use parakeet_rs::{
        ExecutionConfig, ExecutionProvider, ParakeetUnified, ParakeetUnifiedHandle, TimestampMode,
        TranscriptionResult, UnifiedStreamingConfig,
    };
    use rubato::{FftFixedInOut, Resampler};
    use std::{fs, path::Path, sync::Arc};

    const PACKET: usize = 2_400;
    /// The listener feeds the decoder 32 ms spans. Any size gives the same windows.
    const SPAN: usize = 512;

    fn recording(path: &Path) -> Result<Vec<f32>> {
        let bytes = fs::read(path)?;
        ensure!(
            !bytes.starts_with(b"RIFF") || bytes.get(8..12) != Some(b"WAVE"),
            "expected raw PCM, received a WAV container: {}",
            path.display()
        );
        ensure!(!bytes.is_empty(), "PCM recording is empty");
        ensure!(bytes.len() % 2 == 0, "PCM recording has a partial sample");
        let pcm: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|sample| f32::from(i16::from_le_bytes(*sample)) / 32768.0)
            .collect();
        resample_recording(&pcm)
    }

    fn resample_recording(pcm: &[f32]) -> Result<Vec<f32>> {
        let mut resampler = FftFixedInOut::<f32>::new(24_000, 16_000, PACKET, 1)?;
        let expected = pcm.len() * 2 / 3;
        let delay = resampler.output_delay();
        let mut speech = Vec::with_capacity(expected + delay);
        let (packets, tail) = pcm.as_chunks::<PACKET>();
        for packet in packets {
            speech.extend_from_slice(&resampler.process(&[packet.as_slice()], None)?[0]);
        }
        if !tail.is_empty() {
            speech.extend_from_slice(&resampler.process_partial(Some(&[tail]), None)?[0]);
        }
        while speech.len() < expected + delay {
            speech.extend_from_slice(&resampler.process_partial::<&[f32]>(None, None)?[0]);
        }
        speech.copy_within(delay..delay + expected, 0);
        speech.truncate(expected);
        Ok(speech)
    }

    #[test]
    fn recording_preserves_audio_after_the_last_full_packet() {
        let mut pcm = vec![0.0; PACKET];
        pcm.resize(PACKET + 600, 0.5);
        let speech = resample_recording(&pcm).unwrap();
        assert_eq!(speech.len(), (PACKET + 600) * 2 / 3);
        assert!(speech[PACKET * 2 / 3..].iter().any(|sample| *sample > 0.4));
    }

    fn reference(
        handle: &ParakeetUnifiedHandle,
        config: UnifiedStreamingConfig,
        speech: &[f32],
    ) -> Result<TranscriptionResult> {
        let mut decoder = ParakeetUnified::from_shared_with_streaming_config(handle, config)?;
        for span in speech.chunks(SPAN) {
            decoder.transcribe_chunk(span)?;
        }
        let padding = final_padding_samples(speech.len(), config);
        decoder.transcribe_chunk(&vec![0.0; padding])?;
        decoder.flush()?;
        Ok(decoder.get_timed_transcript(TimestampMode::Words))
    }

    /// Decodes as the listener does. `burst` spans arrive before each decode, as when the
    /// worker has fallen behind, and one window then covers up to `max_hops` hops.
    fn listener(
        model: &Arc<UnifiedModel>,
        config: UnifiedStreamingConfig,
        speech: &[f32],
        burst: usize,
        max_hops: usize,
    ) -> Result<TranscriptionResult> {
        let hop = config.chunk_samples();
        let mut decoder = StreamDecoder::new(model, config)?;
        let ready = |decoder: &mut StreamDecoder| -> Result<()> {
            loop {
                let waiting = decoder
                    .pending_samples()
                    .saturating_sub(decoder.right_context_samples());
                let hops = waiting / hop * hop;
                if hops == 0 {
                    return Ok(());
                }
                decoder.decode_window(hops.min(max_hops * hop), false)?;
            }
        };
        for spans in speech.chunks(SPAN * burst) {
            decoder.extend(spans);
            ready(&mut decoder)?;
        }
        decoder.extend(&vec![0.0; final_padding_samples(speech.len(), config)]);
        ready(&mut decoder)?;
        while decoder.pending_samples() >= FRAME_SAMPLES {
            let rest = decoder.pending_samples() / FRAME_SAMPLES * FRAME_SAMPLES;
            let last = if max_hops > 1 {
                rest.min(max_hops * hop)
            } else {
                hop
            };
            decoder.decode_window(last, true)?;
        }
        Ok(decoder.transcript())
    }

    fn identical(expected: &TranscriptionResult, actual: &TranscriptionResult) -> bool {
        expected.text == actual.text
            && expected.tokens.len() == actual.tokens.len()
            && expected
                .tokens
                .iter()
                .zip(&actual.tokens)
                .all(|(a, b)| a.text == b.text && a.start == b.start && a.end == b.end)
    }

    /// Lower-case words without punctuation, for scoring against a transcript.
    fn normalize(text: &str) -> String {
        text.to_lowercase()
            .split(|letter: char| !letter.is_alphanumeric() && letter != '\'')
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Word-level edit distance between two transcripts.
    fn edits(expected: &str, actual: &str) -> usize {
        let expected: Vec<_> = expected.split_whitespace().collect();
        let actual: Vec<_> = actual.split_whitespace().collect();
        let mut cost: Vec<usize> = (0..=actual.len()).collect();
        for (row, expected_word) in expected.iter().enumerate() {
            let mut diagonal = cost[0];
            cost[0] = row + 1;
            for (column, actual_word) in actual.iter().enumerate() {
                let previous = cost[column + 1];
                cost[column + 1] = (cost[column] + 1)
                    .min(previous + 1)
                    .min(diagonal + usize::from(expected_word != actual_word));
                diagonal = previous;
            }
        }
        cost[actual.len()]
    }

    pub fn run() -> Result<()> {
        let mut args = std::env::args_os().skip(1);
        let directory = args.next().context("pass a Unified model path")?;
        let recordings: Vec<_> = args.collect();
        ensure!(!recordings.is_empty(), "pass at least one PCM recording");
        let execution =
            || ExecutionConfig::new().with_execution_provider(ExecutionProvider::BurnWgpu);
        let handle = ParakeetUnifiedHandle::from_pretrained(&directory, Some(execution()))?;
        let model = UnifiedModel::load(Path::new(&directory), execution())?;
        let mut differing = 0;
        for path in &recordings {
            let speech = recording(Path::new(path))?;
            for chunk_secs in [0.96, 1.44, 1.92] {
                let config = UnifiedStreamingConfig {
                    chunk_secs,
                    ..STREAMING_CONFIG
                };
                let expected = reference(&handle, config, &speech)?;
                let fixed = listener(&model, config, &speech, 1, 1)?;
                let same = identical(&expected, &fixed);
                differing += usize::from(!same);
                // About 2.9 s of spans per decode: every window spans three hops.
                let backlogged = listener(&model, config, &speech, 90, 3)?;
                // With a transcript beside the recording, score both against it.
                let scored = fs::read_to_string(Path::new(path).with_extension("txt"))
                    .ok()
                    .map(|transcript| {
                        let transcript = normalize(&transcript);
                        serde_json::json!({
                            "words": transcript.split_whitespace().count(),
                            "edits": edits(&transcript, &normalize(&fixed.text)),
                            "edits_with_three_hop_windows":
                                edits(&transcript, &normalize(&backlogged.text)),
                        })
                    });
                println!(
                    "{}",
                    serde_json::json!({
                        "recording": Path::new(path).file_name().map(|name| name.to_string_lossy()),
                        "seconds": speech.len() as f32 / 16_000.0,
                        "hop_seconds": chunk_secs,
                        "words": expected.tokens.len(),
                        "identical": same,
                        "word_edits_with_three_hop_windows": edits(&expected.text, &backlogged.text),
                        "against_transcript": scored,
                    })
                );
            }
        }
        ensure!(
            differing == 0,
            "{differing} fixed-hop decodes differ from parakeet-rs"
        );
        Ok(())
    }
}

#[cfg(feature = "parakeet")]
fn main() -> anyhow::Result<()> {
    parity::run()
}

#[cfg(not(feature = "parakeet"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("Build this check with --features parakeet");
}
