//! Associate competing readings without changing track-local words.
use super::{ACTIVITY_WINDOW, Activity, Atom, Manifest, Numbers, Role, check_cancelled};
use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// A word written in digits and a word that is part of how it is said are one reading.
pub(super) fn number_basis(numbers: &Numbers, a: &Atom<'_>, b: &Atom<'_>) -> Option<&'static str> {
    let within = |number: &Atom<'_>, word: &Atom<'_>| {
        !word.normalized.is_empty()
            && numbers
                .figure(&number.local.text)
                .is_some_and(|figure| figure.includes(&word.normalized))
    };
    (within(a, b) || within(b, a)).then_some("equivalent_spoken_number")
}

/// Stands for the pause after a phrase where a slot has no following word.
const PHRASE_END: usize = usize::MAX;

// A one-letter substitution in a longer word is a plausible competing reading.
// Unrelated words, including backchannels, keep their own conversation entries.
fn similar(a: &str, b: &str) -> bool {
    let (a, b): (Vec<_>, Vec<_>) = (a.chars().collect(), b.chars().collect());
    let (short, long) = if a.len() <= b.len() {
        (&a, &b)
    } else {
        (&b, &a)
    };
    if short.len() < 4 || long.len() - short.len() > 1 {
        return false;
    }
    if short.len() == long.len() {
        return short
            .iter()
            .zip(long.iter())
            .filter(|(a, b)| a != b)
            .count()
            == 1;
    }
    // One letter heard or missed, as in "hole" for "whole".
    let prefix = short
        .iter()
        .zip(long.iter())
        .take_while(|(a, b)| a == b)
        .count();
    short[prefix..] == long[prefix + 1..]
}

// Words of one phrase follow each other within a quarter of a second.
const PHRASE_GAP: i64 = 4_000;
// A recognizer can misplace a word under other speech by most of a second.
const ORDERED_REACH: u64 = 16_000;

/// A recognizer places a word poorly when other speech covers it, and a forced aligner
/// can squeeze such words to no length. Their order still holds. Directly before or
/// after a word that two recordings share, the same word from the same speaker is
/// one word. Order alone cannot tell a phrase from someone's echo of it, so both
/// observations need a known voice. Returns the observations joined this way.
pub(super) fn join_neighbors(
    atoms: &[Atom<'_>],
    clusters: &mut [Vec<usize>],
    membership: &mut [usize],
    voices: &[Option<usize>],
    track_count: usize,
) -> Result<Vec<bool>> {
    let mut tracks = vec![Vec::new(); track_count];
    for (index, atom) in atoms.iter().enumerate() {
        tracks[atom.track].push(index);
    }
    for track in &mut tracks {
        track.sort_unstable_by_key(|&index| atoms[index].position);
    }
    let same_word = |clusters: &[Vec<usize>], a: usize, b: usize| {
        clusters[a].iter().all(|&left| {
            clusters[b].iter().all(|&right| {
                let same_voice = voices[left]
                    .zip(voices[right])
                    .is_none_or(|(left, right)| left == right);
                let (left, right) = (&atoms[left], &atoms[right]);
                !left.excludes(right)
                    && same_voice
                    && i64::midpoint(left.start, left.end)
                        .abs_diff(i64::midpoint(right.start, right.end))
                        <= ORDERED_REACH
            })
        })
    };
    let mut joined = vec![false; atoms.len()];
    let mut shared: VecDeque<_> = (0..clusters.len())
        .filter(|&cluster| clusters[cluster].len() > 1)
        .collect();
    let mut visited = 0_usize;
    while let Some(anchor) = shared.pop_front() {
        visited += 1;
        if visited.is_multiple_of(1024) {
            check_cancelled()?;
        }
        for forward in [false, true] {
            let neighbors: Vec<_> = clusters[anchor]
                .iter()
                .filter_map(|&member| {
                    let atom = &atoms[member];
                    let position = if forward {
                        atom.position + 1
                    } else {
                        atom.position.checked_sub(1)?
                    };
                    let &neighbor = tracks[atom.track].get(position)?;
                    let gap = if forward {
                        atoms[neighbor].start - atom.end
                    } else {
                        atom.start - atoms[neighbor].end
                    };
                    (gap <= PHRASE_GAP).then_some(neighbor)
                })
                .collect();
            for (index, &left) in neighbors.iter().enumerate() {
                for &right in &neighbors[index + 1..] {
                    let (keep, replace) = (membership[left], membership[right]);
                    if keep == replace
                        || voices[left].is_none()
                        || voices[left] != voices[right]
                        || !same_word(clusters, keep, replace)
                    {
                        continue;
                    }
                    let moved = std::mem::take(&mut clusters[replace]);
                    for &atom in &moved {
                        membership[atom] = keep;
                    }
                    clusters[keep].extend(moved);
                    for &atom in &clusters[keep] {
                        joined[atom] = true;
                    }
                    shared.push_back(keep);
                }
            }
        }
    }
    Ok(joined)
}

pub(super) fn reconcile(
    atoms: &[Atom<'_>],
    clusters: &mut [Vec<usize>],
    roots: &[usize],
    manifest: &Manifest,
    activity: &[(&[Activity], i64)],
    far: &Far<'_>,
    numbers: &Numbers,
) -> Result<Vec<bool>> {
    let mut tracks = vec![Vec::new(); manifest.tracks.len()];
    let mut membership = vec![0; atoms.len()];
    for (c, cluster) in clusters.iter().enumerate() {
        for &i in cluster {
            tracks[atoms[i].track].push(i);
            membership[i] = c;
        }
    }
    for track in &mut tracks {
        track.sort_unstable_by_key(|&i| atoms[i].position);
    }
    align_shifted_prefixes(
        atoms,
        clusters,
        &mut membership,
        &tracks,
        roots,
        manifest,
        activity,
    )?;
    join_compounds(atoms, clusters, &mut membership, &tracks, roots, numbers)?;
    let owner = |i: usize| atoms[i].speaker.map(|id| roots[id]);
    let mut slots = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for track in &tracks {
        for triple in track.windows(3) {
            let [left, middle, right] = *triple else {
                unreachable!()
            };
            let before = membership[left];
            let after = membership[right];
            if clusters[before].len() < 2
                || clusters[after].len() < 2
                || clusters[membership[middle]].len()
                    >= clusters[before].len().min(clusters[after].len())
                || atoms[middle].start.abs_diff(atoms[left].end) > 4_000
                || atoms[right].start.abs_diff(atoms[middle].end) > 4_000
            {
                continue;
            }
            let Some(speaker) = owner(clusters[before][0]) else {
                continue;
            };
            if owner(clusters[after][0]) != Some(speaker) {
                continue;
            }
            let bounded_spill = atoms[middle].isolated
                && atoms[left].likely_bleed
                && atoms[right].likely_bleed
                && atoms[left].local.utterance_id == atoms[middle].local.utterance_id
                && atoms[right].local.utterance_id == atoms[middle].local.utterance_id
                && atoms[middle]
                    .level
                    .zip(atoms[left].level.zip(atoms[right].level))
                    .is_some_and(|(level, (left, right))| level <= 1.41 * left.max(right));
            if owner(middle) == Some(speaker) || bounded_spill {
                slots.entry((before, after)).or_default().push(middle);
            }
        }
        // The last word before a pause has no following word to bound it. Recognizers
        // disagree there most often, on names and on trailing consonants.
        for (position, pair) in track.windows(2).enumerate() {
            let [left, middle] = *pair else {
                unreachable!()
            };
            let before = membership[left];
            let continues = track
                .get(position + 2)
                .is_some_and(|&next| atoms[next].start.abs_diff(atoms[middle].end) <= 4_000);
            if continues
                || clusters[before].len() < 2
                || clusters[membership[middle]].len() >= clusters[before].len()
                || atoms[middle].start.abs_diff(atoms[left].end) > 4_000
            {
                continue;
            }
            let Some(speaker) = owner(clusters[before][0]) else {
                continue;
            };
            let trailing_spill = atoms[middle].isolated
                && atoms[middle].likely_bleed
                && atoms[left].likely_bleed
                && atoms[left].local.utterance_id == atoms[middle].local.utterance_id;
            if owner(middle) == Some(speaker) || trailing_spill {
                slots.entry((before, PHRASE_END)).or_default().push(middle);
            }
        }
    }
    for ((before, after), slot) in slots {
        check_cancelled()?;
        let groups: BTreeSet<_> = slot.iter().map(|&i| membership[i]).collect();
        if groups.len() < 2
            || groups.iter().any(|&group| clusters[group].is_empty())
            || clusters[before].is_empty()
            || (after != PHRASE_END && clusters[after].is_empty())
        {
            continue;
        }
        // The reading most recordings agree on stays. The speaker's own microphone
        // settles an even split between recordings.
        let own = |cluster: usize| {
            clusters[cluster]
                .iter()
                .any(|&i| atoms[i].isolated && !atoms[i].likely_bleed)
        };
        let most = groups.iter().map(|&group| clusters[group].len()).max();
        let mut leading: Vec<_> = groups
            .iter()
            .copied()
            .filter(|&group| Some(clusters[group].len()) == most)
            .collect();
        if leading.len() > 1 {
            leading.retain(|&group| own(group));
        }
        let [winner] = leading[..] else {
            continue;
        };
        // Require independent agreement on the canonical spelling and account for
        // every observation of every reading in this exact, one-word slot.
        let observed: usize = groups.iter().map(|&group| clusters[group].len()).sum();
        if observed != slot.len()
            || slot
                .iter()
                .filter(|&&i| atoms[i].isolated && !atoms[i].likely_bleed)
                .count()
                > 1
        {
            continue;
        }
        let start = slot.iter().map(|&i| atoms[i].start).max().unwrap();
        let end = slot.iter().map(|&i| atoms[i].end).min().unwrap();
        let shortest = slot
            .iter()
            .map(|&i| atoms[i].end - atoms[i].start)
            .min()
            .unwrap();
        if end <= start || i128::from(end - start) * 2 < i128::from(shortest) {
            continue;
        }
        for other in groups {
            let primary = &atoms[clusters[winner][0]].normalized;
            if other != winner && similar(primary, &atoms[clusters[other][0]].normalized) {
                let alternate = std::mem::take(&mut clusters[other]);
                clusters[winner].extend(alternate);
            }
        }
    }
    fold_spill_readings(atoms, clusters, &tracks);
    Ok(fold_far_readings(atoms, clusters, &tracks, far))
}

/// What the microphones show about the words of a recording that is not one of them.
pub(super) struct Far<'a> {
    /// Words that belong to a participant without a microphone.
    pub(super) apart: &'a [bool],
    /// The share of a word's level that a microphone's wearer accounts for.
    pub(super) accounted: &'a dyn Fn(usize, usize) -> Option<f32>,
}

// A wearer's speech accounts for a far recording's word when the microphone is no more
// than 2 dB below the balance it usually keeps against that recording. Another voice
// on the far recording can only lower that share.
const ACCOUNTED: f32 = 0.8;
// A far word must be clearly audible on its recording for that balance to mean anything.
const AUDIBLE_LEVEL: f32 = 0.15;
// A wearer's own word reaches this share of the microphone's loud speech.
const OWN_LEVEL: f32 = 0.3;

/// A far recording garbles what a speaker's own microphone hears cleanly: other words,
/// and often a different number of them. Its unmatched word is such a reading when it
/// stands beside a word both recordings share, inside the stretch the microphone
/// recognized at speaking level, and the wearer's speech accounts for the far
/// recording's level there. The microphone's wording stands and the far word joins
/// the word it overlaps most. A second voice raises the far recording above what the
/// microphone accounts for and stays a word. So does a word that two far recordings
/// agree on, a single word that stands for a single word of the microphone, and a
/// word where the microphone recognized nothing. Returns the words folded this way.
fn fold_far_readings(
    atoms: &[Atom<'_>],
    clusters: &mut [Vec<usize>],
    tracks: &[Vec<usize>],
    far: &Far<'_>,
) -> Vec<bool> {
    let mut membership = vec![0; atoms.len()];
    for (cluster, members) in clusters.iter().enumerate() {
        for &index in members {
            membership[index] = cluster;
        }
    }
    let own = |index: usize| {
        let atom = &atoms[index];
        atom.isolated && atom.speaker.is_some() && !atom.likely_bleed
    };
    // A far word that no other recording, not even a faint copy, agrees with.
    let stray: Vec<_> = clusters
        .iter()
        .map(|cluster| match cluster[..] {
            [word] if !atoms[word].isolated && !far.apart[word] => Some(word),
            _ => None,
        })
        .collect();
    let shared = |cluster: usize| clusters[cluster].iter().copied().find(|&index| own(index));
    if stray.iter().all(Option::is_none) {
        return vec![false; atoms.len()];
    }
    // Every word in an uninterrupted run has the same anchors.
    let mut runs = vec![(0, 0); atoms.len()];
    for track in tracks {
        let mut first = 0;
        while first < track.len() {
            let mut end = first;
            while end < track.len() && stray[membership[track[end]]].is_some() {
                end += 1;
            }
            for &word in &track[first..end] {
                runs[word] = (first, end - 1);
            }
            first = end + 1;
        }
    }
    let windows: Vec<_> = tracks
        .iter()
        .map(|track| {
            let mut by_start = track.clone();
            by_start.sort_unstable_by_key(|&index| atoms[index].start);
            let mut max_end = i64::MIN;
            by_start
                .into_iter()
                .map(|index| {
                    max_end = max_end.max(atoms[index].end);
                    (index, max_end)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let overlapping = |track: usize, atom: &Atom<'_>| {
        let words = &windows[track];
        let from = words.partition_point(|&(_, max_end)| max_end <= atom.start);
        let until = words[from..].partition_point(|&(index, _)| atoms[index].start < atom.end);
        &words[from..from + until]
    };
    let mut folds = Vec::new();
    for (cluster, word) in stray.iter().enumerate() {
        let Some(word) = *word else { continue };
        let atom = &atoms[word];
        let track = &tracks[atom.track];
        let (first, last) = runs[word];
        let before = first
            .checked_sub(1)
            .and_then(|position| shared(membership[track[position]]));
        let after = track
            .get(last + 1)
            .and_then(|&index| shared(membership[index]));
        // How many words the microphone has for the stretch between two shared words.
        let between = before.zip(after).and_then(|(before, after)| {
            (atoms[before].track == atoms[after].track).then(|| {
                atoms[after]
                    .position
                    .saturating_sub(atoms[before].position + 1)
            })
        });
        if between == Some(0) || (between == Some(1) && first == last) {
            continue;
        }
        let overlap = |index: usize| {
            atoms[index]
                .end
                .min(atom.end)
                .saturating_sub(atoms[index].start.max(atom.start))
                .max(0)
        };
        let target = [before, after]
            .into_iter()
            .flatten()
            .map(|anchor| atoms[anchor].track)
            .filter(|&microphone| {
                atom.level.is_some_and(|level| level >= AUDIBLE_LEVEL)
                    && (far.accounted)(word, microphone).is_some_and(|share| share >= ACCOUNTED)
            })
            .filter_map(|microphone| {
                let spoken = || {
                    overlapping(microphone, atom)
                        .iter()
                        .map(|&(index, _)| index)
                        .filter(|&index| {
                            own(index) && atoms[index].level.is_some_and(|level| level >= OWN_LEVEL)
                        })
                };
                let covered: i128 = spoken().map(overlap).map(i128::from).sum();
                let target = spoken().max_by_key(|&index| (overlap(index), index))?;
                (covered > 0 && covered * 2 >= i128::from(atom.end - atom.start))
                    .then_some((overlap(target), target))
            })
            .max();
        // Another far recording that heard nearly the same word there supports it.
        let supported = || {
            (0..tracks.len()).any(|other| {
                other != atom.track
                    && overlapping(other, atom)
                        .iter()
                        .map(|&(index, _)| &atoms[index])
                        .filter(|other| other.end > atom.start)
                        .any(|other| {
                            !other.isolated
                                && (other.normalized == atom.normalized
                                    || similar(&other.normalized, &atom.normalized))
                        })
            })
        };
        // The microphone's word already has its reading on this recording when another
        // word of the stretch is one letter from it. This one is then an extra word.
        let reads = |position: usize, target: usize| {
            let (far, near) = (
                &atoms[track[position]].normalized,
                &atoms[target].normalized,
            );
            far == near || similar(far, near)
        };
        let extra = |target: usize| {
            !reads(atom.position, target) && (first..=last).any(|position| reads(position, target))
        };
        if let Some((_, target)) = target
            && !extra(target)
            && !supported()
        {
            folds.push((cluster, membership[target]));
        }
    }
    let mut folded = vec![false; atoms.len()];
    for (cluster, target) in folds {
        let moved = std::mem::take(&mut clusters[cluster]);
        for &index in &moved {
            folded[index] = true;
        }
        clusters[target].extend(moved);
    }
    folded
}

/// A microphone hears its neighbour's speech faintly and can read it as other words.
/// Where the neighbour's own microphone recognized that speech, its wording stands and
/// the faint reading joins it as an alternative. A faint word the source microphone
/// has nothing for remains a word of its own. The same holds for a microphone's faint
/// reading of a participant who has none, against the mixed recording that heard them.
fn fold_spill_readings(atoms: &[Atom<'_>], clusters: &mut [Vec<usize>], tracks: &[Vec<usize>]) {
    let mut membership = vec![0; atoms.len()];
    for (cluster, members) in clusters.iter().enumerate() {
        for &index in members {
            membership[index] = cluster;
        }
    }
    for cluster in 0..clusters.len() {
        let Some(&first) = clusters[cluster].first() else {
            continue;
        };
        let (spill, Some(source)) = (&atoms[first], atoms[first].acoustic_source) else {
            continue;
        };
        if clusters[cluster]
            .iter()
            .any(|&index| atoms[index].acoustic_source != Some(source))
        {
            continue;
        }
        let shared = |word: &Atom<'_>| {
            word.end
                .min(spill.end)
                .saturating_sub(word.start.max(spill.start))
                .max(0)
        };
        let own_words = || {
            tracks[source]
                .iter()
                .map(|&index| (index, &atoms[index]))
                .filter(|(_, word)| word.speaker.is_some() && !word.likely_bleed)
        };
        let covered: i128 = own_words().map(|(_, word)| i128::from(shared(word))).sum();
        let Some((target, word)) = own_words().max_by_key(|&(index, word)| (shared(word), index))
        else {
            continue;
        };
        let target = membership[target];
        // A faint "sour" and "dough" under the source's "sourdough" are its two halves.
        // Both halves have to be there: a short word the source microphone missed can
        // sit inside a longer one that its recognizer stretched.
        let half = |offset: isize| {
            let position = spill.position.checked_add_signed(offset)?;
            let other = &atoms[*tracks[spill.track].get(position)?];
            (other.acoustic_source == Some(source)).then_some(other.normalized.as_str())
        };
        let part = half(1).is_some_and(|next| {
            word.normalized.strip_prefix(spill.normalized.as_str()) == Some(next)
        }) || half(-1).is_some_and(|previous| {
            word.normalized.strip_suffix(spill.normalized.as_str()) == Some(previous)
        });
        // The two readings occupy the same stretch of the source's speech. A recording
        // that already holds the source's word there heard this one as well.
        if covered == 0
            || covered * 2 < i128::from(spill.end - spill.start)
            || (i128::from(shared(word)) * 2 < i128::from(word.end - word.start) && !part)
            || clusters[target].iter().any(|&other| {
                atoms[other].normalized == word.normalized
                    && clusters[cluster]
                        .iter()
                        .any(|&index| atoms[index].track == atoms[other].track)
            })
        {
            continue;
        }
        let moved = std::mem::take(&mut clusters[cluster]);
        for &index in &moved {
            membership[index] = target;
        }
        clusters[target].extend(moved);
    }
}

/// One recognizer writes a compound as a single word where another writes its two
/// parts, or a number in digits where another says it in words, in any of the ways its
/// language has. After the same shared word and before the same word or the end of the
/// recording, from the same speaker, these are one reading. The spelling most recordings
/// agree on stays, and the speaker's own microphone settles an even split.
fn join_compounds(
    atoms: &[Atom<'_>],
    clusters: &mut [Vec<usize>],
    membership: &mut [usize],
    tracks: &[Vec<usize>],
    roots: &[usize],
    numbers: &Numbers,
) -> Result<()> {
    let at = |index: usize, offset: usize| {
        let atom = &atoms[index];
        tracks[atom.track].get(atom.position + offset).copied()
    };
    let ahead = |membership: &[usize], index: usize| {
        at(index, 1).map_or(PHRASE_END, |word| membership[word])
    };
    let mut readings = BTreeSet::new();
    for track in tracks {
        for pair in track.windows(2) {
            let [left, whole] = *pair else { unreachable!() };
            let figure = numbers.figure(&atoms[whole].local.text);
            for &other in &clusters[membership[left]] {
                if other == left {
                    continue;
                }
                let (mut spoken, mut parts) = (String::new(), vec![]);
                while let Some(part) = at(other, parts.len() + 1)
                    && !atoms[part].normalized.is_empty()
                {
                    parts.push(membership[part]);
                    spoken += &atoms[part].normalized;
                    if parts.len() < 2 {
                        continue;
                    }
                    let same = (parts.len() == 2 && spoken == atoms[whole].normalized)
                        || figure.is_some_and(|figure| figure.says(&spoken));
                    if same && ahead(membership, part) == ahead(membership, whole) {
                        let after = ahead(membership, whole);
                        readings.insert((membership[left], after, membership[whole], parts));
                        break;
                    }
                    // The words go on only while they can still say the number.
                    if !figure.is_some_and(|figure| figure.begins(&spoken)) {
                        break;
                    }
                }
            }
        }
    }
    for (before, after, whole, parts) in readings {
        check_cancelled()?;
        let previous = |index: usize| {
            let atom = &atoms[index];
            let position = atom.position.checked_sub(1)?;
            tracks[atom.track].get(position).copied()
        };
        // Every observation of a reading sits between the same neighbours.
        let between = |cluster: &[usize], behind: usize, next: usize| {
            !cluster.is_empty()
                && cluster.iter().all(|&index| {
                    previous(index).is_some_and(|word| membership[word] == behind)
                        && ahead(membership, index) == next
                })
        };
        let first = parts[0];
        let chain: Vec<_> = [before]
            .into_iter()
            .chain(parts.iter().copied())
            .chain([after])
            .collect();
        let distinct: BTreeSet<_> = chain.iter().copied().chain([whole]).collect();
        if distinct.len() != parts.len() + 3
            || parts
                .iter()
                .any(|&part| clusters[part].len() != clusters[first].len())
            || !between(&clusters[whole], before, after)
            || chain
                .windows(3)
                .any(|link| !between(&clusters[link[1]], link[0], link[2]))
        {
            continue;
        }
        let mut speakers = parts
            .iter()
            .copied()
            .chain([whole])
            .flat_map(|cluster| &clusters[cluster])
            .filter_map(|&index| atoms[index].speaker.map(|id| roots[id]));
        let speaker = speakers.next();
        if speakers.any(|other| Some(other) != speaker) {
            continue;
        }
        let own = |cluster: usize| {
            clusters[cluster]
                .iter()
                .any(|&index| atoms[index].isolated && !atoms[index].likely_bleed)
        };
        let keep_whole = match clusters[whole].len().cmp(&clusters[first].len()) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal if own(whole) != own(first) => own(whole),
            std::cmp::Ordering::Equal => continue,
        };
        let (keep, replaced) = if keep_whole {
            (whole, parts)
        } else {
            (first, vec![whole])
        };
        for cluster in replaced {
            let moved = std::mem::take(&mut clusters[cluster]);
            for &index in &moved {
                membership[index] = keep;
            }
            clusters[keep].extend(moved);
        }
    }
    Ok(())
}

// The earlier words of an interruption can be assigned to the preceding speaker
// on a mixed mic. Recover a contiguous two-word prefix only if the mics agree
// on the final word AND speaker, and their measured envelopes corroborate the
// prefix on the shared clock. Keep the reading with the consistent speaker.
fn align_shifted_prefixes(
    atoms: &[Atom<'_>],
    clusters: &mut [Vec<usize>],
    membership: &mut [usize],
    tracks: &[Vec<usize>],
    roots: &[usize],
    manifest: &Manifest,
    activity: &[(&[Activity], i64)],
) -> Result<()> {
    let owner = |i: usize| atoms[i].speaker.map(|id| roots[id]);
    let mut links = Vec::new();
    for (index, cluster) in clusters.iter().enumerate() {
        if index.is_multiple_of(1024) {
            check_cancelled()?;
        }
        if cluster.len() < 2 {
            continue;
        }
        for &anchor in cluster {
            let mic = atoms[anchor].track;
            let position = atoms[anchor].position;
            let Some(speaker) = owner(anchor) else {
                continue;
            };
            if manifest.tracks[mic].role != Role::MixedAudio || position < 2 {
                continue;
            }
            let stable = &tracks[mic][position - 2..=position];
            if !stable.iter().all(|&i| {
                atoms[i].local.utterance_id == atoms[anchor].local.utterance_id
                    && owner(i) == Some(speaker)
            }) {
                continue;
            }
            for &other in cluster {
                let peer = atoms[other].track;
                let position = atoms[other].position;
                if mic == peer
                    || position < 2
                    || manifest.tracks[peer].role != Role::MixedAudio
                    || owner(other) != Some(speaker)
                {
                    continue;
                }
                let shifted = &tracks[peer][position - 2..=position];
                let peer_prefix = owner(shifted[0]);
                if peer_prefix.is_none()
                    || owner(shifted[1]) != peer_prefix
                    || peer_prefix == Some(speaker)
                {
                    continue;
                }
                let words = [
                    atoms[stable[0]].spoken(),
                    atoms[stable[1]].spoken(),
                    atoms[anchor].spoken(),
                ];
                if words[0] == words[1]
                    || words[0] == words[2]
                    || words[1] == words[2]
                    || words.contains(&"")
                {
                    continue;
                }
                let same_prefix = (0..2).all(|n| {
                    let (a, b) = (&atoms[stable[n]], &atoms[shifted[n]]);
                    words[n] == b.spoken()
                        && i64::midpoint(a.start, a.end).abs_diff(i64::midpoint(b.start, b.end))
                            < 12_000
                        && a.end.abs_diff(atoms[stable[n + 1]].start) < 2_400
                        && b.end.abs_diff(atoms[shifted[n + 1]].start) < 2_400
                        && clusters[membership[stable[n]]].len() == 1
                        && clusters[membership[shifted[n]]].len() == 1
                });
                let start = atoms[stable[0]].start.min(atoms[shifted[0]].start);
                let end = atoms[anchor].start.min(atoms[other].start);
                if same_prefix && correlated(activity[mic], activity[peer], start, end) {
                    links.push([stable[0], shifted[0], stable[1], shifted[1]]);
                }
            }
        }
    }
    if links.is_empty() {
        return Ok(());
    }
    let mut uses = vec![0_u8; atoms.len()];
    for &i in links.iter().flatten() {
        uses[i] = uses[i].saturating_add(1);
    }
    for link in links {
        if link.iter().any(|&i| uses[i] != 1) {
            continue;
        }
        for [stable, shifted] in [[link[0], link[1]], [link[2], link[3]]] {
            let (a, b) = (membership[stable], membership[shifted]);
            clusters[b].clear();
            clusters[a].push(shifted);
            membership[shifted] = a;
        }
    }
    Ok(())
}

fn correlated(a: (&[Activity], i64), b: (&[Activity], i64), start: i64, end: i64) -> bool {
    if start >= end || start.abs_diff(end) > 24_000 {
        return false;
    }
    let level = |(frames, origin): (&[Activity], i64), sample: i64| {
        let local = usize::try_from(sample.checked_sub(origin)?).ok()?;
        let frame = frames.get(local / ACTIVITY_WINDOW)?;
        (local < frame.end_sample as usize).then_some(f64::from(frame.level))
    };
    let (mut n, mut x, mut y, mut xx, mut yy, mut xy) = (0., 0., 0., 0., 0., 0.);
    for sample in (start..end).step_by(ACTIVITY_WINDOW) {
        if let (Some(a), Some(b)) = (level(a, sample), level(b, sample)) {
            n += 1.;
            x += a;
            y += b;
            xx += a * a;
            yy += b * b;
            xy += a * b;
        }
    }
    if n < 16. {
        return false;
    }
    let (variance_a, variance_b) = (xx - x * x / n, yy - y * y / n);
    variance_a > 0.01 * xx
        && variance_b > 0.01 * yy
        && xy - x * y / n > 0.75 * (variance_a * variance_b).sqrt()
}
