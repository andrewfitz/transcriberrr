//! Mute other speakers' bleed in the per-speaker (iso) tracks of one session.
//!
//! Each mic also picks up the other speakers, quieter, and the recognizer transcribes that
//! too, so a merge would show every sentence twice. When another track carries the same moment
//! much louder, the sound is that speaker's: it is muted here, before transcription, to exact
//! silence, which the voice detector then skips (it normalizes each chunk to a fixed peak, so
//! merely attenuated bleed would still be judged speech).

use crate::TARGET_SAMPLE_RATE;

/// 20 ms analysis frames.
const FRAME_SAMPLES: usize = (TARGET_SAMPLE_RATE / 50) as usize;
const FRAMES_PER_SECOND: f32 = 50.0;
/// A track leads when it is this much louder than every other: that speaker talks alone, and
/// the other tracks carry only their bleed. Also the muting threshold for a pair of tracks
/// whose bleed could not be measured. A lav or headset mic picks up a speaker a metre or two
/// away 15-35 dB down.
const MARGIN_DB: f32 = 10.0;
/// A frame is bleed when the leading track is louder by at least the measured bleed less
/// this much (frame-to-frame spread of the level difference). Someone talking over another
/// raises their own mic far above the bleed level, so they stay.
const TOLERANCE_DB: f32 = 6.0;
/// Never mute a frame less than this much quieter than the other track.
const LOWEST_THRESHOLD_DB: f32 = 8.0;
/// The louder track must also be this far above its own noise floor, i.e. actually speaking,
/// so a noisy track cannot mute a quiet speaker.
const ACTIVE_DB: f32 = 15.0;
/// Floors lower than this are digital silence (Zoom pads absent participants' tracks).
const LOWEST_FLOOR_DB: f32 = -70.0;
const SILENCE_DB: f32 = -120.0;
/// Muted frames this close to a track's own speech are kept, so quiet word onsets and tails
/// under another speaker survive (120 ms; longer lets the end of the other speaker's last word
/// through at a turn change).
const HOLD_FRAMES: usize = 6;
/// After another speaker's bleed ends, their room's reverberation lingers on this mic while
/// their own close mic has gone quiet, so level comparison cannot see it. Keep muting for up
/// to 500 ms while this track's level keeps falling; speech of its own rises.
const RELEASE_FRAMES: usize = 25;
const RELEASE_RISE_DB: f32 = 3.0;
/// Between another speaker's words their own mic dips while their reverberation still fills
/// this one, leaving holes in the bleed that a forced aligner then times words into. Holes up
/// to 400 ms are muted too when they get no louder than the bleed around them (half a second
/// either side); this speaker's own interjection is much louder than bleed on their own mic.
const HOLE_FRAMES: usize = 20;
const HOLE_CONTEXT_FRAMES: usize = 25;
/// Gain ramp at mute boundaries (10 ms), so cuts do not click.
const RAMP_SAMPLES: f32 = 160.0;
/// Measuring a pair's bleed takes a second of frames where one of them leads.
const MIN_LEAD_FRAMES: usize = 50;

/// Loudness per 20 ms frame, in dB, above ~150 Hz (the voice detector's high-pass), so mains
/// hum and rumble do not count as level.
pub(crate) fn frame_levels(samples: &[f32]) -> Vec<f32> {
    let mut previous_input = 0.0f32;
    let mut previous_output = 0.0f32;
    samples
        .chunks(FRAME_SAMPLES)
        .map(|frame| {
            let mut energy = 0.0f64;
            for &sample in frame {
                previous_output = sample - previous_input + 0.943 * previous_output;
                previous_input = sample;
                energy += f64::from(previous_output).powi(2);
            }
            let mean = energy / frame.len() as f64;
            if mean > 0.0 {
                ((10.0 * mean.log10()) as f32).max(SILENCE_DB)
            } else {
                SILENCE_DB
            }
        })
        .collect()
}

/// Which frames of one track are another speaker's bleed. The default mutes nothing.
#[derive(Debug, Clone, Default)]
pub(crate) struct Mask {
    muted: Vec<bool>,
}

impl Mask {
    pub(crate) fn muted_seconds(&self) -> f32 {
        self.muted.iter().filter(|&&muted| muted).count() as f32 / FRAMES_PER_SECOND
    }

    /// Silence muted frames exactly, with short ramps at the edges.
    pub(crate) fn apply(&self, samples: &mut [f32]) {
        let mut gain = 1.0f32;
        for (index, sample) in samples.iter_mut().enumerate() {
            let muted = self
                .muted
                .get(index / FRAME_SAMPLES)
                .copied()
                .unwrap_or(false);
            gain = if muted {
                (gain - 1.0 / RAMP_SAMPLES).max(0.0)
            } else {
                (gain + 1.0 / RAMP_SAMPLES).min(1.0)
            };
            *sample *= gain;
        }
    }

    /// Whether every frame of a span is muted, so a word timed there is not this speaker's.
    pub(crate) fn mutes(&self, start: f32, end: f32) -> bool {
        let first = (start.max(0.0) * FRAMES_PER_SECOND).floor() as usize;
        let last = ((end.max(start) * FRAMES_PER_SECOND).ceil() as usize).max(first + 1);
        last <= self.muted.len() && self.muted[first..last].iter().all(|&muted| muted)
    }
}

/// Muted stretches longer than this (2 s) are cut out before transcription: a listener's
/// track is mostly the others' turns, muted, and the recognizer costs the same per second of
/// silence as of speech.
const CUT_FRAMES: usize = 100;
/// Silence kept from either end of each cut (about 0.5 s in all): a natural pause, so words
/// either side stay apart and the mute ramps are kept whole.
const PAUSE_FRAMES: usize = 12;

/// Where the pieces of a cut-down track came from, to put its words back on its own timeline.
#[derive(Debug, Clone, Default)]
pub(crate) struct Timeline {
    /// (start in the cut-down track, start in the track), in samples, in order.
    pieces: Vec<(usize, usize)>,
}

impl Timeline {
    /// A time in the cut-down track, in seconds, on the track's own timeline.
    pub(crate) fn original(&self, seconds: f32) -> f32 {
        let rate = TARGET_SAMPLE_RATE as f32;
        let at = (seconds.max(0.0) * rate) as usize;
        let piece = self.pieces.partition_point(|&(start, _)| start <= at);
        match piece.checked_sub(1).map(|piece| self.pieces[piece]) {
            Some((compact, original)) => {
                (original as f32 + (seconds * rate - compact as f32)) / rate
            }
            None => seconds,
        }
    }
}

impl Mask {
    /// The track with every muted stretch longer than `CUT_FRAMES` cut down to a pause, and the
    /// timeline to map its words back. Muted audio is silence (see `apply`), so the recognizer
    /// hears the same speech with shorter pauses.
    pub(crate) fn cut(&self, samples: &[f32]) -> (Vec<f32>, Timeline) {
        let mut kept: Vec<std::ops::Range<usize>> = Vec::new();
        let mut keep = |range: std::ops::Range<usize>| match kept.last_mut() {
            Some(last) if last.end == range.start => last.end = range.end,
            _ if range.is_empty() => {}
            _ => kept.push(range),
        };
        let mut frame = 0;
        let mut from = 0;
        while frame < self.muted.len() {
            if !self.muted[frame] {
                frame += 1;
                continue;
            }
            let start = frame;
            while frame < self.muted.len() && self.muted[frame] {
                frame += 1;
            }
            if frame - start > CUT_FRAMES {
                // No pause is needed before the first word or after the last.
                let first = if start == 0 { 0 } else { start + PAUSE_FRAMES };
                let last = if frame == self.muted.len() {
                    usize::MAX / FRAME_SAMPLES
                } else {
                    frame - PAUSE_FRAMES
                };
                let cut = first * FRAME_SAMPLES..last * FRAME_SAMPLES;
                keep(from..cut.start.min(samples.len()));
                from = cut.end.min(samples.len());
            }
        }
        keep(from..samples.len());
        let mut timeline = Timeline::default();
        let mut audio = Vec::with_capacity(kept.iter().map(|range| range.len()).sum());
        for range in kept {
            timeline.pieces.push((audio.len(), range.start));
            audio.extend_from_slice(&samples[range]);
        }
        (audio, timeline)
    }
}

fn percentile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return SILENCE_DB;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted[((sorted.len() - 1) as f32 * fraction).round() as usize]
}

/// Bleed masks for the tracks of one session, from their `frame_levels`. `starts` places
/// each track on the session's timeline, in seconds: 0 for tracks recorded from the start, later
/// for the files of a restarted Zoom recording's next segment. Outside its span a track is
/// silent.
pub(crate) fn masks(levels: &[Vec<f32>], starts: &[f32]) -> Vec<Mask> {
    let tracks = levels.len();
    let offsets = starts
        .iter()
        .map(|&start| (start.max(0.0) * FRAMES_PER_SECOND).round() as usize)
        .collect::<Vec<_>>();
    let frames = levels
        .iter()
        .zip(&offsets)
        .map(|(track, offset)| offset + track.len())
        .max()
        .unwrap_or(0);
    // By session frame.
    let level = |track: usize, frame: usize| {
        frame
            .checked_sub(offsets[track])
            .and_then(|frame| levels[track].get(frame))
            .copied()
            .unwrap_or(SILENCE_DB)
    };
    // Digital silence is no mic at all: Zoom fills a participant's file with it before they
    // join, after they leave, and while they are muted, so it says nothing about the room.
    let has_audio = |track: usize, frame: usize| level(track, frame) > SILENCE_DB;
    let floors = levels
        .iter()
        .map(|track| {
            let heard = track
                .iter()
                .copied()
                .filter(|&level| level > SILENCE_DB)
                .collect::<Vec<_>>();
            percentile(&heard, 0.1).max(LOWEST_FLOOR_DB)
        })
        .collect::<Vec<_>>();
    let active = |track: usize, frame: usize| level(track, frame) > floors[track] + ACTIVE_DB;

    // How much quieter each speaker is on each other mic: the median level difference over
    // frames where that speaker leads. Measured from the bleed itself, this absorbs mic gains,
    // distances, and loud or soft talkers alike.
    let mut differences = vec![vec![Vec::new(); tracks]; tracks];
    for frame in 0..frames {
        for leader in 0..tracks {
            let value = level(leader, frame);
            if active(leader, frame)
                && (0..tracks)
                    .all(|other| other == leader || value >= level(other, frame) + MARGIN_DB)
            {
                for (track, differences) in differences[leader].iter_mut().enumerate() {
                    // Only where the track has audio: before or after it, it says nothing.
                    if track != leader && has_audio(track, frame) {
                        differences.push(value - level(track, frame));
                    }
                }
            }
        }
    }
    // threshold[speaker][track]: how much louder the speaker's track must be for a frame of
    // `track` to be that speaker's bleed.
    let threshold = differences
        .iter()
        .map(|pairs| {
            pairs
                .iter()
                .map(|values| {
                    if values.len() >= MIN_LEAD_FRAMES {
                        (percentile(values, 0.5) - TOLERANCE_DB).max(LOWEST_THRESHOLD_DB)
                    } else {
                        MARGIN_DB
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    (0..tracks)
        .map(|track| {
            let length = levels[track].len();
            // By this track's own frame.
            let offset = offsets[track];
            let mut muted = (0..length)
                .map(|frame| {
                    let own = level(track, offset + frame);
                    (0..tracks).any(|other| {
                        other != track
                            && active(other, offset + frame)
                            && level(other, offset + frame) - own >= threshold[other][track]
                    })
                })
                .collect::<Vec<_>>();
            let mut frame = 1;
            while frame < length {
                if muted[frame - 1] && !muted[frame] {
                    let start = level(track, offset + frame - 1);
                    let mut next = frame;
                    while next < length
                        && next < frame + RELEASE_FRAMES
                        && !muted[next]
                        && level(track, offset + next) <= start
                        && level(track, offset + next)
                            <= level(track, offset + next - 1) + RELEASE_RISE_DB
                    {
                        muted[next] = true;
                        next += 1;
                    }
                    // Past the release, so its end does not start another.
                    frame = next + 1;
                } else {
                    frame += 1;
                }
            }
            let mut frame = 0;
            while frame < length {
                if muted[frame] {
                    frame += 1;
                    continue;
                }
                let start = frame;
                while frame < length && !muted[frame] {
                    frame += 1;
                }
                if start == 0 || frame == length || frame - start > HOLE_FRAMES {
                    continue;
                }
                let loudest = |frames: &mut dyn Iterator<Item = usize>| {
                    frames
                        .map(|frame| level(track, offset + frame))
                        .fold(SILENCE_DB, f32::max)
                };
                let bleed = loudest(
                    &mut (start.saturating_sub(HOLE_CONTEXT_FRAMES)..start)
                        .chain(frame..(frame + HOLE_CONTEXT_FRAMES).min(length))
                        .filter(|&frame| muted[frame]),
                );
                if loudest(&mut (start..frame)) <= bleed {
                    muted[start..frame].fill(true);
                }
            }
            // Keep muted frames next to this track's own (kept, active) speech.
            let speaking = (0..length)
                .map(|frame| !muted[frame] && active(track, offset + frame))
                .collect::<Vec<_>>();
            for (frame, muted) in muted.iter_mut().enumerate() {
                let near = frame.saturating_sub(HOLD_FRAMES)..(frame + HOLD_FRAMES + 1).min(length);
                if *muted && speaking[near].iter().any(|&speaking| speaking) {
                    *muted = false;
                }
            }
            Mask { muted }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: usize = TARGET_SAMPLE_RATE as usize;

    #[test]
    fn long_muted_stretches_are_cut_and_words_put_back() {
        // 0-2 s speech, 2-10 s muted, 10-11 s speech, 11-12 s muted (kept: too short).
        let mut muted = vec![false; 600];
        muted[100..500].fill(true);
        muted[550..].fill(true);
        let mask = Mask { muted };
        let samples = (0..12 * RATE).map(|index| index as f32).collect::<Vec<_>>();
        let (audio, timeline) = mask.cut(&samples);
        // 8 s muted, cut to a pause.
        let pause = PAUSE_FRAMES * FRAME_SAMPLES;
        assert_eq!(audio.len(), RATE * 2 + 2 * pause + RATE * 2);
        assert_eq!(audio[2 * RATE + pause - 1], (2 * RATE + pause - 1) as f32);
        assert_eq!(audio[2 * RATE + pause], (10 * RATE - pause) as f32);
        assert_eq!(timeline.original(1.0), 1.0);
        let seconds = |samples: usize| samples as f32 / RATE as f32;
        assert!((timeline.original(2.0 + seconds(2 * pause) + 0.2) - 10.2).abs() < 1e-3);
        assert!((timeline.original(4.0 + seconds(2 * pause) - 0.1) - 11.9).abs() < 1e-3);
        // Nothing muted: nothing cut.
        let (same, timeline) = Mask::default().cut(&samples);
        assert_eq!(same.len(), samples.len());
        assert_eq!(timeline.original(7.0), 7.0);
        // A track muted from the start, and to the end: no pause there.
        let mut muted = vec![true; 600];
        muted[300..400].fill(false);
        let (audio, timeline) = Mask { muted }.cut(&samples);
        assert_eq!(audio.len(), pause + 2 * RATE + pause);
        assert!((timeline.original(seconds(pause) + 0.5) - 6.5).abs() < 1e-3);
    }

    /// Deterministic noise in [-1, 1].
    fn noise(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }

    /// A voice-like signal: a 180 Hz buzz with harmonics, modulated at a syllable rate.
    fn voice(seconds: f32, pitch: f32) -> Vec<f32> {
        (0..(seconds * RATE as f32) as usize)
            .map(|index| {
                let time = index as f32 / RATE as f32;
                let syllables = 0.6 + 0.4 * (std::f32::consts::TAU * 4.0 * time).sin();
                let buzz = (1..6)
                    .map(|harmonic| {
                        (std::f32::consts::TAU * pitch * harmonic as f32 * time).sin()
                            / harmonic as f32
                    })
                    .sum::<f32>();
                0.2 * syllables * buzz
            })
            .collect()
    }

    fn gain(db: f32) -> f32 {
        10f32.powf(db / 20.0)
    }

    /// One mic: background noise plus each source at a level in dB (None = not heard).
    fn mic(seconds: f32, noise_db: f32, sources: &[(&[f32], f32, f32)], seed: u32) -> Vec<f32> {
        let mut seed = seed;
        let mut samples = (0..(seconds * RATE as f32) as usize)
            .map(|_| gain(noise_db) * noise(&mut seed))
            .collect::<Vec<_>>();
        for &(source, start, db) in sources {
            let offset = (start * RATE as f32) as usize;
            for (index, value) in source.iter().enumerate() {
                if let Some(sample) = samples.get_mut(offset + index) {
                    *sample += gain(db) * value;
                }
            }
        }
        samples
    }

    fn muted_between(mask: &Mask, start: f32, end: f32) -> f32 {
        let frames = &mask.muted[(start * 50.0) as usize..(end * 50.0) as usize];
        frames.iter().filter(|&&muted| muted).count() as f32 / frames.len() as f32
    }

    /// Alice talks 1-4 s, Bob 5-8 s, both 9-11 s. `bob_gain_db` is his mic's gain;
    /// `bob_voice_db` how loudly he talks.
    fn session(bleed_db: f32, bob_gain_db: f32, bob_voice_db: f32) -> (Vec<f32>, Vec<f32>) {
        let alice = voice(3.0, 180.0);
        let bob = voice(3.0, 120.0);
        let both_a = voice(2.0, 200.0);
        let both_b = voice(2.0, 110.0);
        let alice_mic = mic(
            12.0,
            -60.0,
            &[
                (&alice, 1.0, 0.0),
                (&bob, 5.0, bleed_db + bob_voice_db),
                (&both_a, 9.0, 0.0),
                (&both_b, 9.0, bleed_db + bob_voice_db),
            ],
            1,
        );
        let bob_mic = mic(
            12.0,
            -60.0,
            &[
                (&alice, 1.0, bleed_db + bob_gain_db),
                (&bob, 5.0, bob_gain_db + bob_voice_db),
                (&both_a, 9.0, bleed_db + bob_gain_db),
                (&both_b, 9.0, bob_gain_db + bob_voice_db),
            ],
            2,
        );
        (alice_mic, bob_mic)
    }

    #[test]
    fn mutes_each_speaker_on_the_other_mic_but_not_overlapping_speech() {
        for (bleed, bob_gain, bob_voice) in [
            (-20.0, 0.0, 0.0),
            (-15.0, 0.0, 0.0),
            (-12.0, 0.0, 0.0),
            (-20.0, 8.0, 0.0),
            (-20.0, -8.0, 0.0),
            // A soft talker: his voice is quieter on both mics, his bleed on Alice's too.
            (-18.0, 0.0, -8.0),
            (-18.0, 0.0, 8.0),
        ] {
            let (alice, bob) = session(bleed, bob_gain, bob_voice);
            let masks = masks(&[frame_levels(&alice), frame_levels(&bob)], &[0.0; 2]);
            let case = format!("bleed {bleed} dB, Bob's gain {bob_gain} dB, voice {bob_voice} dB");
            assert!(
                muted_between(&masks[0], 1.2, 3.8) < 0.01,
                "{case}: Alice kept"
            );
            assert!(
                muted_between(&masks[0], 5.2, 7.8) > 0.99,
                "{case}: Bob's bleed"
            );
            assert!(
                muted_between(&masks[1], 1.2, 3.8) > 0.99,
                "{case}: Alice's bleed"
            );
            assert!(
                muted_between(&masks[1], 5.2, 7.8) < 0.01,
                "{case}: Bob kept"
            );
            assert!(
                muted_between(&masks[0], 9.2, 10.8) < 0.01,
                "{case}: overlap"
            );
            assert!(
                muted_between(&masks[1], 9.2, 10.8) < 0.01,
                "{case}: overlap"
            );
        }
    }

    #[test]
    fn a_listeners_track_of_only_bleed_is_muted() {
        let (alice, bob) = session(-20.0, 0.0, 0.0);
        let speech = voice(3.0, 180.0);
        let other = voice(3.0, 120.0);
        // The listener's mic hears both, a bit louder than their mics hear each other.
        let listener = mic(
            12.0,
            -60.0,
            &[(&speech, 1.0, -14.0), (&other, 5.0, -14.0)],
            3,
        );
        let masks = masks(
            &[
                frame_levels(&alice),
                frame_levels(&bob),
                frame_levels(&listener),
            ],
            &[0.0; 3],
        );
        assert!(muted_between(&masks[2], 1.2, 3.8) > 0.99);
        assert!(muted_between(&masks[2], 5.2, 7.8) > 0.99);
    }

    #[test]
    fn a_noisy_mic_does_not_mute_a_quiet_speaker() {
        let quiet = voice(3.0, 180.0);
        let speaker = mic(5.0, -70.0, &[(&quiet, 1.0, -32.0)], 4);
        let fan = mic(5.0, -22.0, &[], 5);
        let masks = masks(&[frame_levels(&speaker), frame_levels(&fan)], &[0.0; 2]);
        assert!(muted_between(&masks[0], 1.2, 3.8) < 0.01);
    }

    #[test]
    fn a_reconnected_speakers_files_are_compared_where_they_have_audio() {
        // Alice reconnected at 6 s. Like all of Zoom's files of a recording, both of hers
        // cover it, silent where she was not connected.
        let (alice, bob) = session(-18.0, 0.0, 0.0);
        let split = 6 * RATE;
        let mut first = alice.clone();
        first[split..].fill(0.0);
        let mut second = alice;
        second[..split].fill(0.0);
        let masks = masks(
            &[
                frame_levels(&first),
                frame_levels(&second),
                frame_levels(&bob),
            ],
            &[0.0; 3],
        );
        // Bob talks 5-8 s, both 9-11 s.
        assert!(muted_between(&masks[0], 5.2, 5.8) > 0.99);
        assert!(muted_between(&masks[1], 6.2, 7.8) > 0.99);
        assert!(muted_between(&masks[1], 9.2, 10.8) < 0.01);
        assert!(muted_between(&masks[2], 5.2, 7.8) < 0.01);
        assert!(muted_between(&masks[2], 1.2, 3.8) > 0.99);
    }

    #[test]
    fn digital_silence_does_not_skew_the_bleed_measure() {
        // A late joiner: Zoom's file is silent for the 60 s before Bob joined, while Alice
        // talked. Bob's mic then hears her 18 dB down.
        let (alice, bob) = session(-18.0, 0.0, 0.0);
        let early = voice(50.0, 180.0);
        let lead = 60 * RATE;
        let mut alice_mic = mic(60.0, -60.0, &[(&early, 5.0, 0.0)], 6);
        alice_mic.extend(alice);
        let mut bob_mic = vec![0.0; lead];
        bob_mic.extend(bob);
        let masks = masks(
            &[frame_levels(&alice_mic), frame_levels(&bob_mic)],
            &[0.0; 2],
        );
        // Alice talks at 61-64 s: her bleed on Bob's mic is muted.
        assert!(muted_between(&masks[1], 61.2, 63.8) > 0.99);
        assert!(muted_between(&masks[1], 65.2, 67.8) < 0.01);
    }

    #[test]
    fn applying_silences_muted_frames_exactly() {
        let (alice, bob) = session(-20.0, 0.0, 0.0);
        let masks = masks(&[frame_levels(&alice), frame_levels(&bob)], &[0.0; 2]);
        let mut samples = alice.clone();
        masks[0].apply(&mut samples);
        assert!(
            samples[(5.5 * RATE as f32) as usize..(7.5 * RATE as f32) as usize]
                .iter()
                .all(|&sample| sample == 0.0)
        );
        let kept = (2.0 * RATE as f32) as usize;
        assert_eq!(samples[kept], alice[kept]);
        assert!(masks[0].mutes(6.0, 6.4));
        assert!(!masks[0].mutes(2.0, 2.4));
        assert!(!masks[0].mutes(7.9, 9.5));
        assert!(masks[0].muted_seconds() > 2.5);
    }
}
