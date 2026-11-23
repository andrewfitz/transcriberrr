//! Recognize directional microphone spill from repeatable lexical and acoustic evidence.
use super::{ACTIVITY_WINDOW, Activity, Atom, MatchScore, TARGET_SAMPLE_RATE, check_cancelled};
use anyhow::Result;
use std::collections::BTreeMap;

// Calibrate over multiple recognized words, then require local acoustic evidence on
// any additional candidate. The similarity window covers several syllables.
const HALF_WINDOW: i64 = 20;
const MAX_LAG: i64 = 2;
const MIN_FRAMES: usize = 24;
const MIN_AGREEMENT: f64 = 0.75;
const MIN_ANCHORS: usize = 3;
const MIN_ANCHOR_SPAN: u64 = (TARGET_SAMPLE_RATE as u64) * 2 / 5;
const NEAR_ANCHOR: u64 = (TARGET_SAMPLE_RATE as u64) * 3 / 5;

// A wearer's own word reaches this share of the microphone's loud speech. A copy of a
// neighbour's word begins while the microphone still sits 16 dB or more below it,
// where it has been for the 0.3 s before.
const OWN_LEVEL: f32 = 0.3;
const FAINT_LEVEL: f32 = 0.15;
const ONSET: i64 = (TARGET_SAMPLE_RATE as i64) / 5;
const LEAD_IN: i64 = (TARGET_SAMPLE_RATE as i64) * 3 / 10;

/// A microphone can recognize its neighbour's word where no spill was calibrated, or
/// where its own wearer starts to talk before the word ends. The neighbour's
/// microphone recognized the same word at speaking level and this one was still quiet
/// as the word began, so the word is the neighbour's. A level by itself cannot tell a
/// copy from a quiet voice. Here both microphones also recognized the same word at the
/// same moment. Two wearers who say the same word together are each loud on their own
/// microphone and stay apart. A wearer who echoes the word below the faint level is
/// taken for a copy.
pub(super) fn mark_faint(
    atoms: &mut [Atom<'_>],
    activity: &[(&[Activity], i64)],
    references: &[Option<f32>],
    edges: &[(MatchScore, usize, usize)],
) -> bool {
    let quiet_onset = |atom: &Atom<'_>| {
        let (frames, start) = activity[atom.track];
        let reference = references[atom.track]?;
        let from = atom
            .start
            .checked_sub(start)?
            .saturating_sub(LEAD_IN)
            .max(0);
        let until = atom
            .end
            .min(atom.start.saturating_add(ONSET))
            .checked_sub(start)?;
        let first = usize::try_from(from).ok()? / ACTIVITY_WINDOW;
        let last = usize::try_from(until).ok()?.div_ceil(ACTIVITY_WINDOW);
        let frames = frames
            .get(first..last)
            .filter(|frames| !frames.is_empty())?;
        let power = frames
            .iter()
            .map(|frame| frame.level * frame.level)
            .sum::<f32>();
        Some((power / frames.len() as f32).sqrt() < FAINT_LEVEL * reference)
    };
    let own =
        |atom: &Atom<'_>| atom.isolated && atom.speaker.is_some() && atom.acoustic_source.is_none();
    let mut copies = Vec::new();
    for &(_, left, right) in edges {
        for (source, copy) in [(left, right), (right, left)] {
            let (source, word) = (&atoms[source], &atoms[copy]);
            if own(source)
                && own(word)
                && source.track != word.track
                && source.level.is_some_and(|level| level >= OWN_LEVEL)
                && quiet_onset(source) == Some(false)
                && quiet_onset(word) == Some(true)
            {
                copies.push((copy, source.track));
            }
        }
    }
    for &(copy, source) in &copies {
        atoms[copy].acoustic_source = Some(source);
        atoms[copy].speaker = None;
        atoms[copy].likely_bleed = true;
    }
    !copies.is_empty()
}

// A copy within 12 dB of its floor has little visible modulation. Require a
// further 20 dB of source-to-floor contrast to calibrate from these quiet frames.
fn quiet_copy(source_above_floor: f64, copy_above_floor: f64) -> bool {
    copy_above_floor > 0.0
        && copy_above_floor < 4.0
        && source_above_floor > 10.0 * copy_above_floor.max(1.0)
}

struct Envelope<'a> {
    frames: &'a [Activity],
    start: i64,
    floor: f64,
}

impl<'a> Envelope<'a> {
    fn new(frames: &'a [Activity], start: i64) -> Option<Self> {
        let mut values: Vec<f32> = frames
            .iter()
            .filter_map(|frame| (frame.level > 0.0).then_some(frame.level))
            .collect();
        if values.len() < MIN_FRAMES {
            return None;
        }
        let rank = (values.len() - 1) / 10;
        let floor = f64::from(*values.select_nth_unstable_by(rank, f32::total_cmp).1);
        Some(Self {
            frames,
            start,
            floor,
        })
    }

    fn rms_near(&self, sample: i64) -> Option<f64> {
        let step = ACTIVITY_WINDOW as i64;
        let mut values =
            (-5..=5).filter_map(|offset| self.level_at(sample.checked_add(offset * step)?));
        let first = values.next()?;
        let (sum, count) = values.fold((first * first, 1.0), |(sum, count), level| {
            (sum + level * level, count + 1.0)
        });
        Some((sum / count).sqrt())
    }

    fn level_at(&self, sample: i64) -> Option<f64> {
        let local = usize::try_from(sample.checked_sub(self.start)?).ok()?;
        let frame = self.frames.get(local / ACTIVITY_WINDOW)?;
        (local < frame.end_sample as usize).then_some(f64::from(frame.level))
    }
}

/// Test whether the same modulated signal is substantially louder on the source mic.
/// Level differences alone cannot distinguish copied speech from a separate quiet voice.
fn agreement(source: &Envelope<'_>, copy: &Envelope<'_>, center: i64) -> Option<(f64, f64)> {
    let mut best: Option<(f64, f64)> = None;
    let step = ACTIVITY_WINDOW as i64;
    for lag in -MAX_LAG..=MAX_LAG {
        let (mut count, mut x, mut y, mut xx, mut yy, mut xy) = (0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for frame in -HALF_WINDOW..=HALF_WINDOW {
            let Some(sample) = center.checked_add(frame * step) else {
                continue;
            };
            let Some(delayed) = sample.checked_add(lag * step) else {
                continue;
            };
            let (Some(a), Some(b)) = (source.level_at(sample), copy.level_at(delayed)) else {
                continue;
            };
            count += 1;
            x += a;
            y += b;
            xx += a * a;
            yy += b * b;
            xy += a * b;
        }
        if count < MIN_FRAMES || xx < yy * 10.0 || yy == 0.0 {
            continue;
        }
        let n = count as f64;
        let source_rms = (xx / n).sqrt();
        let copy_rms = (yy / n).sqrt();
        if source_rms < source.floor * 5.6
            || source_rms / source.floor < (copy_rms / copy.floor) * 2.0
        {
            continue;
        }
        let variance = (xx - x * x / n) * (yy - y * y / n);
        if variance <= 0.0 {
            continue;
        }
        let correlation = (xy - x * y / n) / variance.sqrt();
        if correlation >= MIN_AGREEMENT && best.is_none_or(|(old, _)| correlation > old) {
            best = Some((correlation, source_rms / copy_rms));
        }
    }
    best
}

pub(super) fn mark(
    atoms: &mut [Atom<'_>],
    activity: &[(&[Activity], i64)],
    edges: &[(MatchScore, usize, usize)],
) -> Result<bool> {
    let mut participating = vec![false; activity.len()];
    for atom in atoms.iter().filter(|atom| atom.isolated) {
        participating[atom.track] = true;
    }
    if participating.iter().filter(|&&active| active).count() < 2 {
        return Ok(false);
    }
    let envelopes: Vec<_> = activity
        .iter()
        .zip(participating)
        .map(|(&(frames, start), active)| active.then(|| Envelope::new(frames, start)).flatten())
        .collect();
    let center = |atom: &Atom<'_>| i64::midpoint(atom.start, atom.end);
    #[derive(Default)]
    struct Pair {
        lexical: Vec<(usize, i64)>,
        anchors: Vec<(i64, f64)>,
        gain: f64,
    }
    let mut shared_isolated_word = vec![false; atoms.len()];
    let mut pairs = BTreeMap::<(usize, usize), Pair>::new();
    for (index, &(_, a, b)) in edges.iter().enumerate() {
        if index % 512 == 0 {
            check_cancelled()?;
        }
        for (leader, copy) in [(a, b), (b, a)] {
            let (source, target) = (&atoms[leader], &atoms[copy]);
            if !source.isolated || !target.isolated {
                continue;
            }
            shared_isolated_word[copy] = true;
            let pair = pairs.entry((target.track, source.track)).or_default();
            pair.lexical.push((copy, center(target)));
        }
    }
    // A source headset need not decode every word heard on its neighbours. Measure
    // its transfer into each headset during other recognized speech on that source.
    for (index, atom) in atoms.iter().enumerate() {
        if index % 512 == 0 {
            check_cancelled()?;
        }
        let Some(source) = &envelopes[atom.track] else {
            continue;
        };
        let sample = center(atom);
        for (target_track, target) in envelopes.iter().enumerate() {
            let Some(target) = target else { continue };
            if target_track == atom.track || target.level_at(sample).is_none() {
                continue;
            }
            let gain = agreement(source, target, sample)
                .map(|(_, gain)| gain)
                .or_else(|| {
                    let source_rms = source.rms_near(sample)?;
                    let target_rms = target.rms_near(sample)?;
                    // A distant microphone can be near its noise floor, where its
                    // envelope is poorly correlated even during an isolated syllable.
                    quiet_copy(source_rms / source.floor, target_rms / target.floor)
                        .then(|| source_rms / target_rms)
                });
            if let Some(gain) = gain {
                pairs
                    .entry((target_track, atom.track))
                    .or_default()
                    .anchors
                    .push((sample, gain));
            }
        }
    }
    pairs.retain(|_, pair| {
        pair.anchors.sort_by_key(|anchor| anchor.0);
        pair.anchors.dedup_by_key(|anchor| anchor.0);
        if pair.anchors.len() < MIN_ANCHORS
            || pair.anchors[0]
                .0
                .abs_diff(pair.anchors[pair.anchors.len() - 1].0)
                < MIN_ANCHOR_SPAN
        {
            return false;
        }
        let mut gains: Vec<_> = pair.anchors.iter().map(|anchor| anchor.1).collect();
        let midpoint = gains.len() / 2;
        pair.gain = *gains.select_nth_unstable_by(midpoint, f64::total_cmp).1;
        pair.lexical.sort_unstable();
        true
    });
    if pairs.is_empty() {
        return Ok(false);
    }
    let mut changed = false;
    for (index, atom) in atoms.iter_mut().enumerate() {
        if index % 512 == 0 {
            check_cancelled()?;
        }
        let sample = center(atom);
        let best = pairs
            .range((atom.track, 0)..=(atom.track, usize::MAX))
            .filter_map(|(&(_, leader), pair)| {
                let near = pair.anchors.partition_point(|&(time, _)| time < sample);
                let nearest = near
                    .checked_sub(1)
                    .into_iter()
                    .chain(Some(near))
                    .filter_map(|index| pair.anchors.get(index))
                    .min_by_key(|&&(time, _)| time.abs_diff(sample))?;
                let position = pair.lexical.partition_point(|&(i, _)| i < index);
                let direct = pair.lexical.get(position).is_some_and(|&(i, _)| i == index);
                let between_words = !direct
                    && position > 0
                    && pair.lexical.get(position).is_some_and(|&(_, after)| {
                        sample.abs_diff(after) <= u64::from(TARGET_SAMPLE_RATE)
                            && sample.abs_diff(pair.lexical[position - 1].1)
                                <= u64::from(TARGET_SAMPLE_RATE)
                    });
                let source = envelopes[leader].as_ref()?;
                let target = envelopes[atom.track].as_ref()?;
                let source_rms = source.rms_near(sample)?;
                let target_rms = target.rms_near(sample)?;
                let source_above_floor = source_rms / source.floor;
                let target_above_floor = target_rms / target.floor;
                // Combine measured spill with the target noise floor. As the source
                // gets quieter the raw level ratio alone is biased towards unity.
                let predicted = (source_rms / pair.gain).hypot(target.floor);
                if target_rms == 0.0
                    || target_rms > predicted * 2.0
                    || source_above_floor < 3.16
                    || source_above_floor < target_above_floor * 1.41
                {
                    return None;
                }
                let local = agreement(source, target, sample);
                let near = nearest.0.abs_diff(sample) <= NEAR_ANCHOR;
                let quiet_and_corroborated = shared_isolated_word[index]
                    && quiet_copy(source_above_floor, target_above_floor);
                if !(direct || between_words || near && (local.is_some() || quiet_and_corroborated))
                {
                    return None;
                }
                // Prefer the channel that accounts for the observed modulation and
                // decoded the word. Simultaneous speakers can both be loud nearby.
                let strength = ((direct, between_words, local.is_some()), source_above_floor);
                Some((leader, strength))
            })
            .max_by(|(_, a), (_, b)| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        if let Some((leader, _)) = best {
            atom.acoustic_source = Some(leader);
            atom.speaker = None;
            atom.likely_bleed = true;
            changed = true;
        }
    }
    Ok(changed)
}
