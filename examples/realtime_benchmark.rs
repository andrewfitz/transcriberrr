//! Manual real-time Parakeet benchmark. Pass the model directory, a repeat count,
//! and one or more raw 24 kHz mono signed little-endian PCM16 fixture paths with
//! adjacent .txt references.

#[cfg(feature = "parakeet")]
#[allow(dead_code)]
#[path = "../src/serve/parakeet_profile.rs"]
mod parakeet_profile;

#[cfg(feature = "parakeet")]
mod benchmark {
    use super::parakeet_profile::{
        STREAMING_CONFIG, final_padding_samples, single_window, warmup_window_frames,
    };
    use anyhow::{Context, Result, ensure};
    use parakeet_rs::{
        ExecutionConfig, ExecutionProvider, ParakeetUnified, ParakeetUnifiedHandle, TimestampMode,
    };
    use rubato::{FftFixedInOut, Resampler};
    use serde_json::{Value, json};
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::{Arc, Barrier, mpsc},
        thread,
        time::{Duration, Instant},
    };

    const RATE: usize = 24_000;
    const PACKET: usize = RATE / 10;
    const MODEL_RATE: usize = 16_000;

    struct Packet {
        pcm: Option<Vec<f32>>,
        due: Instant,
        sent: Instant,
    }

    fn milliseconds(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1000.0
    }

    fn normalize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|letter: char| !letter.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_owned)
            .collect()
    }

    fn errors(reference: &str, hypothesis: &str) -> (usize, usize) {
        let expected = normalize(reference);
        let actual = normalize(hypothesis);
        let mut cost: Vec<usize> = (0..=actual.len()).collect();
        for (i, reference_word) in expected.iter().enumerate() {
            let mut diagonal = cost[0];
            cost[0] = i + 1;
            for (j, actual_word) in actual.iter().enumerate() {
                let previous = cost[j + 1];
                cost[j + 1] = (cost[j] + 1)
                    .min(previous + 1)
                    .min(diagonal + usize::from(reference_word != actual_word));
                diagonal = previous;
            }
        }
        (cost[actual.len()], expected.len())
    }

    fn fixture(path: &Path) -> Result<(Vec<f32>, usize)> {
        let bytes = fs::read(path)?;
        ensure!(
            !bytes.starts_with(b"RIFF") || bytes.get(8..12) != Some(b"WAVE"),
            "expected raw PCM, received a WAV container: {}",
            path.display()
        );
        ensure!(bytes.len() % 2 == 0, "PCM fixture has a partial sample");
        let pcm: Vec<_> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|sample| f32::from(i16::from_le_bytes(*sample)) / 32768.0)
            .collect();
        let onset = pcm
            .iter()
            .position(|sample| sample.abs() > 250.0 / 32768.0)
            .context("fixture has no onset above 250 PCM units")?;
        Ok((pcm, onset))
    }

    fn capture(
        sender: mpsc::SyncSender<Packet>,
        pcm: Vec<f32>,
        repeat: usize,
        start: Instant,
    ) -> Result<()> {
        let mut due = start;
        let mut capture_deadline = start;
        for _ in 0..repeat {
            for audio in pcm.chunks(PACKET) {
                let duration = Duration::from_secs_f64(audio.len() as f64 / RATE as f64);
                let jitter = Duration::from_millis(1).min(duration / 10);
                due += duration;
                capture_deadline = capture_deadline.max(Instant::now() - jitter) + duration;
                thread::sleep(capture_deadline.saturating_duration_since(Instant::now()));
                let sent = Instant::now();
                capture_deadline = capture_deadline.max(sent - jitter);
                sender
                    .try_send(Packet {
                        pcm: Some(audio.to_vec()),
                        due,
                        sent,
                    })
                    .context("benchmark capture queue filled")?;
            }
            sender
                .try_send(Packet {
                    pcm: None,
                    due,
                    sent: Instant::now(),
                })
                .context("benchmark capture queue filled at commit")?;
        }
        Ok(())
    }

    fn run_track(
        track: usize,
        receiver: mpsc::Receiver<Packet>,
        mut decoder: ParakeetUnified,
        onset: usize,
        reference: String,
        results: mpsc::Sender<Value>,
    ) -> Result<()> {
        let mut resampler = FftFixedInOut::<f32>::new(RATE, MODEL_RATE, PACKET, 1)?;
        let config = decoder.streaming_config();
        let delay_frames = resampler.output_delay();
        let resampler_delay = milliseconds(Duration::from_secs_f64(
            delay_frames as f64 / MODEL_RATE as f64,
        ));
        let mut delay_to_drop = delay_frames;
        let mut captured_frames = 0;
        let mut resampled_frames = 0;
        let mut turn = 0;
        let mut onset_sent = None;
        let mut first_text = None;
        let mut inference = Vec::new();
        let mut backlog = Vec::new();
        let mut capture_lateness = Vec::new();
        for packet in receiver {
            let received = Instant::now();
            capture_lateness.push(milliseconds(
                packet.sent.saturating_duration_since(packet.due),
            ));
            backlog.push(milliseconds(received.duration_since(packet.sent)));
            if let Some(pcm) = packet.pcm {
                if captured_frames <= onset && onset < captured_frames + pcm.len() {
                    onset_sent = Some(packet.sent);
                }
                captured_frames += pcm.len();
                let speech = if pcm.len() == PACKET {
                    resampler.process(&[pcm], None)?
                } else {
                    resampler.process_partial(Some(&[pcm]), None)?
                };
                let skip = delay_to_drop.min(speech[0].len());
                delay_to_drop -= skip;
                let remaining = captured_frames * MODEL_RATE / RATE - resampled_frames;
                let usable = remaining.min(speech[0].len() - skip);
                resampled_frames += usable;
                if usable == 0 {
                    continue;
                }
                let started = Instant::now();
                let delta = decoder.transcribe_chunk(&speech[0][skip..skip + usable])?;
                inference.push(milliseconds(started.elapsed()));
                if let Some(sent) = onset_sent
                    && first_text.is_none()
                    && !delta.trim().is_empty()
                {
                    let now = Instant::now();
                    first_text = Some(now);
                    eprintln!(
                        "track {track} turn {turn} first text {:.1} ms after onset packet",
                        milliseconds(now.duration_since(sent)),
                    );
                }
            } else {
                let remaining = captured_frames * MODEL_RATE / RATE - resampled_frames;
                let started = Instant::now();
                if remaining > 0 {
                    let speech = resampler.process_partial::<&[f32]>(None, None)?;
                    decoder
                        .transcribe_chunk(&speech[0][delay_to_drop..delay_to_drop + remaining])?;
                }
                let padding = final_padding_samples(resampled_frames + remaining, config);
                if padding > 0 {
                    decoder.transcribe_chunk(&vec![0.0; padding])?;
                }
                decoder.flush()?;
                let transcript = decoder.get_timed_transcript(TimestampMode::Words);
                let finalized = Instant::now();
                let finalize_ms = milliseconds(finalized.duration_since(started));
                if first_text.is_none() && !transcript.text.trim().is_empty() {
                    first_text = Some(finalized);
                }
                let words: Vec<_> = transcript
                    .tokens
                    .iter()
                    .map(|word| json!([&word.text, word.start, word.end]))
                    .collect();
                let (word_errors, reference_words) = errors(&reference, &transcript.text);
                let first_text_ms = first_text
                    .zip(onset_sent)
                    .map(|(text, sent)| milliseconds(text.duration_since(sent)));
                results.send(json!({
                    "track": track, "turn": turn, "first_text_ms": first_text_ms,
                    "first_text_before_commit": first_text.is_some_and(|time| time < packet.sent),
                    "commit_to_final_ms": milliseconds(finalized.duration_since(packet.sent)),
                    "resampler_delay_ms": resampler_delay, "onset_frame": onset,
                    "left_context_secs": config.left_context_secs, "chunk_secs": config.chunk_secs,
                    "right_context_secs": config.right_context_secs, "finalize_ms": finalize_ms,
                    "inference_ms": inference, "queue_ms": backlog,
                    "capture_lateness_ms": capture_lateness, "word_errors": word_errors,
                    "reference_words": reference_words, "text": transcript.text, "words": words,
                }))?;
                decoder.reset();
                resampler.reset();
                captured_frames = 0;
                resampled_frames = 0;
                delay_to_drop = delay_frames;
                turn += 1;
                onset_sent = None;
                first_text = None;
                inference = Vec::new();
                backlog = Vec::new();
                capture_lateness = Vec::new();
            }
        }
        Ok(())
    }

    pub fn run() -> Result<()> {
        let mut args = std::env::args_os().skip(1);
        let model = args.next().context("pass a Unified model path")?;
        let repeat: usize = args
            .next()
            .context("pass a positive repeat count")?
            .to_string_lossy()
            .parse()?;
        let fixtures: Vec<_> = args.map(PathBuf::from).collect();
        ensure!(
            repeat > 0 && !fixtures.is_empty(),
            "pass at least one fixture and one repeat"
        );
        let execution = ExecutionConfig::new().with_execution_provider(ExecutionProvider::BurnWgpu);
        let mut config = STREAMING_CONFIG;
        if let Ok(left) = std::env::var("TRANSCRIBERRR_BENCH_LEFT_SECONDS") {
            config.left_context_secs = left.parse()?;
        }
        if let Ok(right) = std::env::var("TRANSCRIBERRR_BENCH_RIGHT_SECONDS") {
            config.right_context_secs = right.parse()?;
        }
        if let Ok(chunk) = std::env::var("TRANSCRIBERRR_BENCH_CHUNK_SECONDS") {
            config.chunk_secs = chunk.parse()?;
        }
        config.validate()?;
        let shared = ParakeetUnifiedHandle::from_pretrained(model, Some(execution))?;
        // As the listener does: one call on every window length this profile can produce.
        for frames in 1..=warmup_window_frames(&[config]) {
            let window = single_window(frames);
            let mut warm = ParakeetUnified::from_shared_with_streaming_config(&shared, window)?;
            warm.transcribe_chunk(&vec![0.0; window.chunk_samples()])?;
        }
        let tracks = fixtures
            .iter()
            .map(|path| {
                let (pcm, onset) = fixture(path)?;
                let reference = fs::read_to_string(path.with_extension("txt"))?;
                let decoder = ParakeetUnified::from_shared_with_streaming_config(&shared, config)?;
                Ok((pcm, onset, reference, decoder))
            })
            .collect::<Result<Vec<_>>>()?;
        let barrier = Arc::new(Barrier::new(tracks.len() + 1));
        let (results, lines) = mpsc::channel();
        thread::scope(|scope| -> Result<()> {
            let mut children = Vec::new();
            let start = Instant::now() + Duration::from_secs(1);
            for (track, (pcm, onset, reference, decoder)) in tracks.into_iter().enumerate() {
                let (sender, receiver) = mpsc::sync_channel(64);
                let results = results.clone();
                let barrier = Arc::clone(&barrier);
                children.push(scope.spawn(move || {
                    barrier.wait();
                    capture(sender, pcm, repeat, start)
                }));
                children.push(
                    scope.spawn(move || {
                        run_track(track, receiver, decoder, onset, reference, results)
                    }),
                );
            }
            drop(results);
            barrier.wait();
            for line in lines {
                println!("{line}");
            }
            for child in children {
                child
                    .join()
                    .map_err(|_| anyhow::anyhow!("benchmark worker panicked"))??;
            }
            Ok(())
        })
    }
}

#[cfg(feature = "parakeet")]
fn main() -> anyhow::Result<()> {
    benchmark::run()
}

#[cfg(not(feature = "parakeet"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("Build this benchmark with --features parakeet");
}
