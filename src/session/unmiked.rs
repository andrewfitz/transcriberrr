//! Speech from a participant without a microphone. A mixed recording hears it clearly
//! while every isolated microphone holds it at spill level.
use super::{ACTIVITY_WINDOW, Activity, Atom, MatchScore, SpeakerIdentity};
use std::collections::{BTreeMap, HashMap};

// An isolated word this strong, relative to its track's loud speech, is its wearer talking.
const OWN_LEVEL: f32 = 0.3;
// Words needed to learn how a wearer's own speech compares between two recordings.
const MIN_OWN_WORDS: usize = 4;
// A mixed word must be clearly audible before silence on the microphones means anything.
const AUDIBLE_LEVEL: f32 = 0.15;
// The wearer is not the speaker when the microphone is at spill level and at least
// 16 dB below the balance its own speech keeps with that mixed recording.
const SPILL_LEVEL: f32 = 0.25;
const SPILL_BALANCE: f32 = 0.15;
// A diarized voice that is mostly off microphone keeps its own label.
const MOSTLY_UNMIKED: usize = 7;
// Words of one phrase follow each other within half a second, the pause at a comma
// included. A recognizer that estimates word ends leaves gaps of that size.
const PHRASE_GAP: i64 = 8_000;
// A microphone that recognized the same word within a second may hold a copy of it.
const COPY_WINDOW: u64 = 16_000;
// A second voice well above the wearer's on a mixed recording leaves the wearer's
// microphone accounting for under half of that recording's level.
const UNACCOUNTED: f32 = 0.5;
// One word can sit off balance by chance. A phrase of three does not.
const PHRASE_WORDS: usize = 3;
// A microphone's word 16 dB or more below its wearer's loud speech is not its wearer's
// when it stands beside faint copies of someone else's phrase.
const FAINT_LEVEL: f32 = 0.15;
const LEAD_IN: i64 = 4_800;

/// Words in start-time order and the furthest end seen at each position. A long
/// word can overlap a query even when several shorter words after it have ended.
#[derive(Default)]
struct TimedWords {
    entries: Vec<(usize, i64, i64)>,
}

impl TimedWords {
    fn push(&mut self, index: usize, start: i64, end: i64) {
        let furthest = self
            .entries
            .last()
            .map_or(end, |&(_, _, last)| last.max(end));
        self.entries.push((index, start, furthest));
    }

    fn near(&self, start: i64, end: i64) -> impl Iterator<Item = usize> + '_ {
        let first = self
            .entries
            .partition_point(|&(_, _, furthest)| furthest < start);
        self.entries[first..]
            .iter()
            .take_while(move |&&(_, word_start, _)| word_start <= end)
            .map(|&(index, _, _)| index)
    }
}

/// Summed power and frame count of a span on the session clock.
fn span_power(
    activity: &[Activity],
    track_start: i64,
    start: i64,
    end: i64,
) -> Option<(f32, usize)> {
    let first = usize::try_from(start.checked_sub(track_start)?).ok()? / ACTIVITY_WINDOW;
    let last = usize::try_from(end.checked_sub(track_start)?)
        .ok()?
        .div_ceil(ACTIVITY_WINDOW);
    let frames = activity.get(first..last.max(first + 1))?;
    let power = frames.iter().map(|frame| frame.level * frame.level).sum();
    Some((power, frames.len()))
}

/// How loud a wearer's own word is on the microphone relative to each mixed recording
/// that also recognized it, by microphone and mixed recording.
pub(super) type Balances = BTreeMap<(usize, usize), f32>;

pub(super) fn balances(atoms: &[Atom<'_>], edges: &[(MatchScore, usize, usize)]) -> Balances {
    let mut balances = BTreeMap::<(usize, usize), Vec<f32>>::new();
    for &(_, left, right) in edges {
        for (own, mixed) in [(left, right), (right, left)] {
            let (own, mixed) = (&atoms[own], &atoms[mixed]);
            if !own.isolated || mixed.isolated || own.speaker.is_none() {
                continue;
            }
            if let (Some(level), Some(mixed_level)) = (own.level, mixed.level)
                && level >= OWN_LEVEL
                && mixed_level > 0.0
            {
                balances
                    .entry((own.track, mixed.track))
                    .or_default()
                    .push(level / mixed_level);
            }
        }
    }
    balances
        .into_iter()
        .filter_map(|(pair, mut values)| {
            if values.len() < MIN_OWN_WORDS {
                return None;
            }
            let middle = values.len() / 2;
            let (_, balance, _) = values.select_nth_unstable_by(middle, f32::total_cmp);
            Some((pair, *balance))
        })
        .collect()
}

/// The share of a mixed word's level that a wearer's own speech accounts for. It is 1
/// when the microphone is as loud against the mixed recording as that wearer's speech
/// usually is, and falls as another voice adds to the mixed recording.
pub(super) fn accounted(
    atom: &Atom<'_>,
    microphone: usize,
    balances: &Balances,
    activity: &[(&[Activity], i64)],
    references: &[Option<f32>],
) -> Option<f32> {
    let level = atom.level.filter(|&level| level > 0.0)?;
    let balance = balances.get(&(microphone, atom.track))?;
    let (frames, start) = activity[microphone];
    let (power, count) = span_power(frames, start, atom.start, atom.end)?;
    let heard = (power / count as f32).sqrt() / references[microphone]?;
    Some(heard / level / balance)
}

/// With two or more isolated microphones, a word that all of them hold at spill level
/// was spoken by someone else. Each microphone would otherwise claim its faint copy for
/// its wearer, and the conversation would repeat the word once per microphone. The
/// mixed recording's own observation supplies the wording, timing and speaker.
pub(super) fn mark(
    atoms: &mut [Atom<'_>],
    identities: &mut Vec<SpeakerIdentity>,
    activity: &[(&[Activity], i64)],
    references: &[Option<f32>],
    isolated: &[bool],
    edges: &[(MatchScore, usize, usize)],
    balances: &Balances,
) -> Option<usize> {
    let microphones: Vec<_> = (0..isolated.len())
        .filter(|&track| isolated[track] && references[track].is_some())
        .collect();
    if microphones.len() < 2 || balances.is_empty() {
        return None;
    }
    // Whether a microphone holds a mixed word at spill level. The spans of the word in
    // which its wearer says something else are left out.
    let quiet_outside = |atom: &Atom<'_>, microphone: usize, busy: &[(i64, i64)]| {
        let Some(level) = atom.level.filter(|&level| level >= AUDIBLE_LEVEL) else {
            return false;
        };
        let (frames, start) = activity[microphone];
        let (Some(&balance), Some(reference)) = (
            balances.get(&(microphone, atom.track)),
            references[microphone],
        ) else {
            return false;
        };
        // A level frame cannot show a pause shorter than itself. It would report the
        // wearer's words on either side.
        let frame = ACTIVITY_WINDOW as i64;
        let mut pauses = Vec::new();
        let mut from = atom.start;
        for &(busy_start, busy_end) in busy {
            if busy_start - from >= frame {
                pauses.push((from, busy_start));
            }
            from = from.max(busy_end);
        }
        if atom.end - from >= frame || busy.is_empty() {
            pauses.push((from, atom.end));
        }
        let (mut power, mut count) = (0.0, 0);
        for (from, until) in pauses {
            let Some((sum, frames)) = span_power(frames, start, from, until) else {
                return false;
            };
            power += sum;
            count += frames;
        }
        if count == 0 {
            return true;
        }
        let heard = (power / count as f32).sqrt() / reference;
        heard < SPILL_LEVEL && heard / level < balance * SPILL_BALANCE
    };
    let quiet = |atom: &Atom<'_>, microphone: usize| quiet_outside(atom, microphone, &[]);
    // A word that a wearer's microphone recognized at speaking level is that wearer's,
    // wherever a recognizer that could not place it on the mixed recording put it.
    let mut worn = vec![false; atoms.len()];
    for &(_, left, right) in edges {
        for (own, mixed) in [(left, right), (right, left)] {
            let own = &atoms[own];
            if own.isolated
                && own.speaker.is_some()
                && own.level.is_some_and(|level| level >= OWN_LEVEL)
            {
                worn[mixed] = true;
            }
        }
    }
    let mut unmiked: Vec<bool> = atoms
        .iter()
        .zip(&worn)
        .map(|(atom, &worn)| {
            !atom.isolated
                && !worn
                && microphones
                    .iter()
                    .all(|&microphone| quiet(atom, microphone))
        })
        .collect();
    let mut order = vec![Vec::new(); isolated.len()];
    for (index, atom) in atoms.iter().enumerate() {
        order[atom.track].push(index);
    }
    let mut timeline: Vec<_> = isolated.iter().map(|_| TimedWords::default()).collect();
    for (index, word) in atoms.iter().enumerate() {
        timeline[word.track].push(index, word.start, word.end);
    }
    for words in &mut order {
        words.sort_unstable_by_key(|&index| atoms[index].position);
    }
    // A wearer who talks through the word is loud on their microphone, yet that
    // microphone recognized their own words there and no copy of this one.
    let saying_something_else = |atom: &Atom<'_>, microphone: usize| {
        let center = i64::midpoint(atom.start, atom.end);
        let mut own_words = Vec::new();
        for own in timeline[microphone]
            .near(
                atom.start.min(center.saturating_sub(COPY_WINDOW as i64)),
                atom.end.max(center.saturating_add(COPY_WINDOW as i64)),
            )
            .map(|index| &atoms[index])
        {
            if own.spoken() == atom.spoken()
                && i64::midpoint(own.start, own.end).abs_diff(center) <= COPY_WINDOW
            {
                return false;
            }
            if own.speaker.is_some()
                && own.level.is_some_and(|level| level >= OWN_LEVEL)
                && own.start < atom.end
                && own.end > atom.start
            {
                own_words.push((own.start, own.end));
            }
        }
        // The rest of the word falls in that wearer's pauses.
        !own_words.is_empty() && quiet_outside(atom, microphone, &own_words)
    };
    // The voice keeps talking when a wearer resumes. Follow its phrase on the mixed
    // recording from a word in the wearers' pause through the words they talk over.
    for words in order
        .iter()
        .filter(|words| words.first().is_some_and(|&index| !atoms[index].isolated))
    {
        let ruled_out = |word: usize| {
            !worn[word]
                && microphones.iter().all(|&microphone| {
                    quiet(&atoms[word], microphone)
                        || saying_something_else(&atoms[word], microphone)
                })
        };
        let adjoins = |earlier: usize, later: usize| {
            atoms[later].start.saturating_sub(atoms[earlier].end) <= PHRASE_GAP
        };
        // A phrase spoken entirely over a wearer has no word in a pause. It still shows
        // in the balance: through several words in a row the mixed recording is at
        // least twice as loud as any wearer's microphone accounts for, while those
        // microphones recognized their wearers saying something else.
        let unaccounted = |word: usize| {
            microphones.iter().all(|&microphone| {
                accounted(&atoms[word], microphone, balances, activity, references)
                    .is_some_and(|share| share < UNACCOUNTED)
            }) && ruled_out(word)
        };
        let (mut run, mut phrases) = (Vec::new(), Vec::new());
        // A recognizer can also set a wearer's own words down early, under the other
        // voice. Such a phrase repeats what that microphone recognized close by.
        let repeated = |word: usize| {
            let center = i64::midpoint(atoms[word].start, atoms[word].end);
            microphones.iter().any(|&microphone| {
                timeline[microphone]
                    .near(
                        center.saturating_sub(2 * COPY_WINDOW as i64),
                        center.saturating_add(2 * COPY_WINDOW as i64),
                    )
                    .any(|own| {
                        atoms[own].spoken() == atoms[word].spoken()
                            && atoms[own].speaker.is_some()
                            && atoms[own].level.is_some_and(|level| level >= OWN_LEVEL)
                            && i64::midpoint(atoms[own].start, atoms[own].end).abs_diff(center)
                                <= 2 * COPY_WINDOW
                    })
            })
        };
        let phrase = |run: &[usize]| {
            run.len() >= PHRASE_WORDS
                && run.iter().filter(|&&word| repeated(word)).count() * 2 < run.len()
        };
        for &word in words {
            let apart = unaccounted(word);
            if !apart || run.last().is_some_and(|&last| !adjoins(last, word)) {
                if phrase(&run) {
                    phrases.append(&mut run);
                }
                run.clear();
            }
            if apart {
                run.push(word);
            }
        }
        if phrase(&run) {
            phrases.append(&mut run);
        }
        for word in phrases {
            unmiked[word] = true;
        }
        for pair in words.windows(2) {
            let (earlier, later) = (pair[0], pair[1]);
            if unmiked[earlier] && !unmiked[later] && adjoins(earlier, later) && ruled_out(later) {
                unmiked[later] = true;
            }
        }
        for pair in words.windows(2).rev() {
            let (earlier, later) = (pair[0], pair[1]);
            if unmiked[later] && !unmiked[earlier] && adjoins(earlier, later) && ruled_out(earlier)
            {
                unmiked[earlier] = true;
            }
        }
    }
    if !unmiked.contains(&true) {
        return None;
    }
    let mut copies = Vec::new();
    for &(_, left, right) in edges {
        for (copy, source) in [(left, right), (right, left)] {
            if unmiked[source]
                && atoms[copy].isolated
                && atoms[copy].acoustic_source.is_none()
                && atoms[copy].level.is_none_or(|level| level < SPILL_LEVEL)
            {
                atoms[copy].speaker = None;
                atoms[copy].likely_bleed = true;
                copies.push((copy, source));
            }
        }
    }
    // A microphone can also read that faint voice as a different word. A word it
    // recognized at spill level, mostly inside a word the mixed recording heard while
    // the microphone stayed quiet, is such a reading and not its wearer's.
    let mut away = TimedWords::default();
    for (index, word) in atoms
        .iter()
        .enumerate()
        .filter(|&(index, _)| unmiked[index])
    {
        away.push(index, word.start, word.end);
    }
    let mut faint = Vec::new();
    for (index, word) in atoms.iter().enumerate() {
        if !word.isolated
            || word.speaker.is_none()
            || word.acoustic_source.is_some()
            || !word.level.is_some_and(|level| level < SPILL_LEVEL)
        {
            continue;
        }
        let heard = away
            .near(word.start, word.end)
            .map(|other| (other, &atoms[other]))
            .take_while(|(_, other)| other.start < word.end)
            .filter(|(_, other)| quiet(other, word.track))
            .map(|(index, other)| {
                let shared = other.end.min(word.end) - other.start.max(word.start);
                (shared, std::cmp::Reverse(index), other.track)
            })
            .filter(|&(shared, ..)| {
                i128::from(shared) * 2 >= i128::from(word.end) - i128::from(word.start)
            })
            .max();
        if let Some((.., recording)) = heard {
            faint.push((index, recording));
        }
    }
    for (index, recording) in faint {
        atoms[index].speaker = None;
        atoms[index].likely_bleed = true;
        atoms[index].acoustic_source = Some(recording);
    }
    // A diarizer can give a participant without a microphone the label of someone who
    // wears one. The microphones refute that label for these words only.
    let mut heard = HashMap::<Option<usize>, (usize, usize)>::new();
    for (atom, &off) in atoms.iter().zip(&unmiked) {
        if !atom.isolated && atom.level.is_some_and(|level| level >= AUDIBLE_LEVEL) {
            let (all, away) = heard.entry(atom.speaker).or_default();
            *all += 1;
            *away += usize::from(off);
        }
    }
    // A label shared with a microphone wearer cannot tell two such participants apart,
    // and the same person can borrow a different wearer's label in each turn. These
    // words share one speaker; a voice the diarizer kept separate keeps its own label.
    // That label is off microphone as well, so the same participant heard under a kept
    // label on one recording and a borrowed one on another can be recognized as one.
    let mut shadow = None;
    for (atom, off) in atoms.iter_mut().zip(unmiked) {
        if !off {
            continue;
        }
        let kept = atom.speaker.filter(|_| {
            heard
                .get(&atom.speaker)
                .is_some_and(|&(all, away)| away * 10 >= all * MOSTLY_UNMIKED)
        });
        if let Some(label) = kept {
            identities[label].unmiked = true;
            continue;
        }
        let track = atom.track;
        atom.speaker = Some(*shadow.get_or_insert_with(|| {
            identities.push(SpeakerIdentity {
                track,
                hint: None,
                isolated: false,
                unmiked: true,
            });
            identities.len() - 1
        }));
    }
    // The mixed recordings can miss or misplace a word of that participant's phrase
    // which a microphone still caught: far below its wearer's speech and directly
    // beside its faint copies of the rest. The word is that participant's.
    let mut voice = vec![None; atoms.len()];
    for (copy, source) in copies {
        voice[copy] = atoms[source].speaker;
    }
    // The microphone is quiet through the word and the 0.3 s before it, so a wearer's
    // own word that a recognizer set down late, into silence, still counts as theirs.
    let caught = |atom: &Atom<'_>| {
        let (frames, start) = activity[atom.track];
        let from = atom.start.saturating_sub(LEAD_IN).max(start);
        atom.isolated
            && atom.speaker.is_some()
            && atom.acoustic_source.is_none()
            && references[atom.track]
                .zip(span_power(frames, start, from, atom.end))
                .is_some_and(|(reference, (power, count))| {
                    (power / count as f32).sqrt() < FAINT_LEVEL * reference
                })
    };
    let mut recovered = Vec::new();
    for words in order.iter().filter(|words| !words.is_empty()) {
        for forward in [true, false] {
            let mut pairs: Vec<_> = words.windows(2).map(|pair| (pair[0], pair[1])).collect();
            if !forward {
                pairs.reverse();
            }
            for (earlier, later) in pairs {
                let (from, to) = if forward {
                    (earlier, later)
                } else {
                    (later, earlier)
                };
                if voice[from].is_some()
                    && voice[to].is_none()
                    && caught(&atoms[to])
                    && atoms[later].start.saturating_sub(atoms[earlier].end) <= PHRASE_GAP
                {
                    voice[to] = voice[from];
                    recovered.push(to);
                }
            }
        }
    }
    for word in recovered {
        atoms[word].speaker = voice[word];
        atoms[word].likely_bleed = true;
    }
    shadow
}

/// Words under a borrowed label say nothing about which participant off microphone
/// spoke them. When the session has exactly one such participant that a diarizer kept
/// apart, they are that participant's. With none, or with several to choose from, they
/// stay a speaker of their own.
pub(super) fn unify(identities: &[SpeakerIdentity], roots: &mut [usize], shadow: usize) {
    let borrowed = roots[shadow];
    let mut kept = (0..identities.len())
        .filter(|&identity| identities[identity].unmiked && roots[identity] != borrowed)
        .map(|identity| roots[identity]);
    let Some(voice) = kept.next() else { return };
    if kept.all(|other| other == voice) {
        for root in roots.iter_mut().filter(|root| **root == borrowed) {
            *root = voice;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TimedWords;

    #[test]
    fn time_query_keeps_a_long_overlapping_word() {
        let mut words = TimedWords::default();
        words.push(0, 0, 100_000);
        words.push(1, 10_000, 12_000);
        words.push(2, 75_000, 77_000);
        words.push(3, 85_000, 87_000);
        let candidates: Vec<_> = words.near(76_000, 80_000).collect();
        assert!(candidates.contains(&0) && candidates.contains(&2));
        assert!(!candidates.contains(&3));
        assert_eq!(words.near(101_000, 110_000).count(), 0);
    }
}
