//! Voice activity detection for the Qwen3 path.
//!
//! Qwen3-ASR invents text ("Okay.", "I'm not sure.") for audio with no speech, and 30 s chunks
//! make any long pause a chunk of its own. Level heuristics cannot tell low-frequency hum or
//! slowly swelling room tone from speech, so chunks are judged by Silero VAD instead (pure Rust,
//! weights compiled in). Parakeet and Apple SpeechTranscriber emit nothing for such audio.

use anyhow::{Result, anyhow};
use rustvani_vad::{BYTES_PER_WINDOW, SAMPLES_PER_WINDOW, SileroVad};

/// Silero's own default decision threshold. Measured on speech kept vs. noise skipped, scores
/// were 1.00 for every speech clip and at most 0.06 for every noise-only clip.
const SPEECH_PROBABILITY: f32 = 0.5;
/// Consecutive 32 ms windows over the threshold (~100 ms). A lone "No." gave 13.
const SHORTEST_SPEECH_WINDOWS: usize = 3;
/// Silero misses quiet speech (peaks near -60 dBFS scored 0.68 at best), so each chunk is
/// judged at a fixed peak level. Only the detector sees the gain; the model gets the original.
const JUDGED_PEAK: f32 = 0.7;

pub(crate) struct VoiceDetector(SileroVad);

impl VoiceDetector {
    pub(crate) fn new() -> Result<Self> {
        SileroVad::new()
            .map(Self)
            .map_err(|error| anyhow!("loading the voice activity model: {error}"))
    }

    /// Whether a chunk contains speech.
    pub(crate) fn has_speech(&self, samples: &[f32]) -> bool {
        self.0.reset_state();
        // Judge above ~150 Hz (one-pole high-pass). Loud mains hum or rumble otherwise sets
        // the peak, and speech 24 dB under a 60 Hz hum scored as no speech while Qwen3
        // transcribed it word for word.
        let mut previous_input = 0.0f32;
        let mut previous_output = 0.0f32;
        let samples = samples
            .iter()
            .map(|&sample| {
                previous_output = sample - previous_input + 0.943 * previous_output;
                previous_input = sample;
                previous_output
            })
            .collect::<Vec<_>>();
        let peak = samples
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        if peak == 0.0 || !peak.is_finite() {
            return false;
        }
        let gain = JUDGED_PEAK / peak;
        let mut bytes = Vec::with_capacity(BYTES_PER_WINDOW);
        let mut run = 0;
        for window in samples.chunks(SAMPLES_PER_WINDOW) {
            bytes.clear();
            for index in 0..SAMPLES_PER_WINDOW {
                let sample = window.get(index).copied().unwrap_or(0.0) * gain;
                let sample = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            // `infer` only fails on a wrong window length; keep the chunk if it ever does.
            let probability = self.0.infer(&bytes).unwrap_or(1.0);
            run = if probability >= SPEECH_PROBABILITY {
                run + 1
            } else {
                0
            };
            if run >= SHORTEST_SPEECH_WINDOWS {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TARGET_SAMPLE_RATE;

    const RATE: usize = TARGET_SAMPLE_RATE as usize;

    /// Deterministic pseudo-random noise in [-1, 1].
    fn noise(seed: u32, samples: usize) -> Vec<f32> {
        let mut state = seed;
        (0..samples)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// Spoken English from macOS `say`, resampled to 16 kHz, or None where `say` is missing.
    fn spoken(text: &str) -> Option<Vec<f32>> {
        let aiff = crate::audio_convert::TempPath::new("aiff");
        let status = std::process::Command::new("/usr/bin/say")
            .arg("-o")
            .arg(aiff.path())
            .arg(text)
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        Some(crate::decode_audio(aiff.path()).unwrap().samples)
    }

    #[test]
    fn noise_hum_clicks_and_silence_are_not_speech() {
        let detector = VoiceDetector::new().unwrap();
        assert!(!detector.has_speech(&vec![0.0; RATE * 30]));
        // Steady broadband room tone.
        let room = noise(1, RATE * 30)
            .iter()
            .map(|n| n * 0.003)
            .collect::<Vec<_>>();
        assert!(!detector.has_speech(&room));
        // Low-frequency (brown-like) hum, as from a car or HVAC.
        let mut state = 0.0f32;
        let hum = noise(11, RATE * 30)
            .into_iter()
            .map(|n| {
                state = 0.995 * state + n * 0.002;
                state
            })
            .collect::<Vec<_>>();
        assert!(!detector.has_speech(&hum));
        // Room tone swelling every five seconds (sox `tremolo 0.2 30`: gain 0.7-1.0).
        let swell = noise(12, RATE * 30)
            .into_iter()
            .enumerate()
            .map(|(index, n)| {
                let time = index as f32 / RATE as f32;
                n * 0.003 * (1.0 - 0.15 * (1.0 - (std::f32::consts::TAU * 0.2 * time).sin()))
            })
            .collect::<Vec<_>>();
        assert!(!detector.has_speech(&swell));
        // An isolated click between muted stretches.
        let mut click = vec![0.0; RATE * 12];
        click.extend(noise(9, RATE / 20).iter().map(|n| n * 0.3));
        click.extend(std::iter::repeat_n(0.0, RATE * 17));
        assert!(!detector.has_speech(&click));
    }

    #[test]
    fn quiet_noisy_and_short_speech_is_kept() {
        let Some(speech) = spoken("The delivery truck arrived early this morning.") else {
            eprintln!("skipping: /usr/bin/say unavailable");
            return;
        };
        let detector = VoiceDetector::new().unwrap();
        let peak = speech.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
        // Clean 24-bit speech peaking at -60 dBFS, padded with digital silence to a chunk.
        let mut quiet = vec![0.0; RATE * 10];
        quiet.extend(speech.iter().map(|s| s / peak * 0.001));
        quiet.extend(std::iter::repeat_n(0.0, RATE * 15));
        assert!(detector.has_speech(&quiet));
        // Speech in white noise at roughly 0 dB SNR.
        let power =
            |samples: &[f32]| samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        let floor = noise(3, speech.len());
        let scale = (power(&speech) / power(&floor)).sqrt();
        let noisy = speech
            .iter()
            .zip(&floor)
            .map(|(s, n)| (s + n * scale) * 0.5)
            .collect::<Vec<_>>();
        assert!(detector.has_speech(&noisy));
        // Speech 24 dB (RMS) under a loud 60 Hz mains hum; peaks 13 dB under it.
        let hummed = speech
            .iter()
            .enumerate()
            .map(|(index, s)| {
                let time = index as f32 / RATE as f32;
                s / peak * 0.16 + 0.7 * (std::f32::consts::TAU * 60.0 * time).sin()
            })
            .collect::<Vec<_>>();
        assert!(detector.has_speech(&hummed));
        // A lone short reply between long muted stretches.
        let reply = spoken("No.").unwrap();
        let mut chunk = vec![0.0; RATE * 12];
        chunk.extend(reply);
        chunk.extend(std::iter::repeat_n(0.0, RATE * 17));
        assert!(detector.has_speech(&chunk));
        // Quiet speech beside a full-scale click in the same chunk.
        let mut clicked = noise(5, RATE / 50)
            .iter()
            .map(|n| n * 0.99)
            .collect::<Vec<_>>();
        clicked.extend(std::iter::repeat_n(0.0, RATE * 2));
        clicked.extend(speech.iter().map(|s| s / peak * 0.003));
        assert!(detector.has_speech(&clicked));
    }
}
