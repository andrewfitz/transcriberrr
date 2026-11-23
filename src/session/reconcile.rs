//! Match track-local observations on a shared clock and reconcile local speaker IDs.
use super::numbers::Numbers;
use super::{
    ACTIVITY_WINDOW, Activity, ConversationWord, LocalWord, Manifest, Observation, Outcome,
    Precision, Role, SessionResult, Speaker, TrackResult, VERSION,
};
use crate::{TARGET_SAMPLE_RATE, check_cancelled, transcript};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};

#[path = "isolation.rs"]
mod isolation;
#[path = "readings.rs"]
mod readings;
#[path = "unmiked.rs"]
mod unmiked;

struct Atom<'a> {
    track: usize,
    position: usize,
    local: &'a LocalWord,
    start: i64,
    end: i64,
    normalized: String,
    /// The plain reading of a number, for a word written in digits and for a word that
    /// says a number the session also has in digits.
    alias: Option<String>,
    previous: Option<(String, i64)>,
    next: Option<(String, i64)>,
    speaker: Option<usize>,
    level: Option<f32>,
    likely_bleed: bool,
    isolated: bool,
    acoustic_source: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MatchScore {
    /// Penalize a placement disagreement before comparing overlap percentage.
    error: u64,
    inverse_overlap: i64,
    has_context: bool,
}

impl Atom<'_> {
    fn spoken(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.normalized)
    }

    /// Observations that cannot be one spoken word, however their timing compares.
    fn excludes(&self, other: &Atom<'_>) -> bool {
        self.track == other.track
            || self.normalized.is_empty()
            || self.spoken() != other.spoken()
            || self.acoustic_source.is_some_and(|source| {
                other.isolated && other.track != source && other.acoustic_source != Some(source)
            })
            || other.acoustic_source.is_some_and(|source| {
                self.isolated && self.track != source && self.acoustic_source != Some(source)
            })
    }

    /// Recordings that each match a third can disagree on a word's edges by more than a
    /// direct match allows, most often on a final word stretched into the pause after
    /// it. The match they share makes any common span enough.
    fn overlaps_same_word(&self, other: &Atom<'_>) -> bool {
        !self.excludes(other)
            && self.normalized == other.normalized
            && self.end.min(other.end) > self.start.max(other.start)
    }

    fn matches(&self, other: &Atom<'_>) -> Option<MatchScore> {
        if self.excludes(other) {
            return None;
        }
        let acoustic_match = (self.acoustic_source == Some(other.track)
            || other.acoustic_source == Some(self.track)
            || (self.acoustic_source.is_some() && self.acoustic_source == other.acoustic_source))
            && i64::midpoint(self.start, self.end).abs_diff(i64::midpoint(other.start, other.end))
                <= 6_400;
        let overlap = self
            .end
            .min(other.end)
            .saturating_sub(self.start.max(other.start));
        let duration = (self.end - self.start).min(other.end - other.start);
        let error = self
            .start
            .abs_diff(other.start)
            .saturating_add(self.end.abs_diff(other.end));
        // Short words can land on adjacent 80 ms ASR frames. Neighboring words provide
        // independent lexical evidence in this case. The signed placement still has to be close.
        let shared = |a: &Option<(String, i64)>, b: &Option<(String, i64)>| {
            a.as_ref()
                .zip(b.as_ref())
                .is_some_and(|((a, at), (b, bt))| {
                    a == b && a != self.spoken() && at.abs_diff(*bt) <= 6_400
                })
        };
        let has_context =
            shared(&self.previous, &other.previous) || shared(&self.next, &other.next);
        if self.normalized != other.normalized && !has_context {
            return None;
        }
        let overlap_matches = overlap > 0 && i128::from(overlap) * 3 >= i128::from(duration);
        let context_matches = has_context
            && i64::midpoint(self.start, self.end).abs_diff(i64::midpoint(other.start, other.end))
                <= 6_400;
        (acoustic_match || (error < 12_000 && (overlap_matches || context_matches))).then_some(
            MatchScore {
                error,
                inverse_overlap: overlap.saturating_neg(),
                has_context,
            },
        )
    }
}

pub(super) fn word_text(text: &str) -> String {
    text.chars()
        .filter(|&character| transcript::is_word_char(character))
        .flat_map(char::to_lowercase)
        .collect()
}

// A speaker's turn continues across pauses of up to a second.
const TURN_GAP: i64 = 16_000;

struct SpeakerIdentity {
    track: usize,
    hint: Option<String>,
    isolated: bool,
    /// Speech the microphones show belongs to none of their wearers.
    unmiked: bool,
}

pub(super) fn reconcile<'a>(
    manifest: &'a Manifest,
    outcomes: &'a [Outcome],
) -> Result<SessionResult<'a>> {
    let mut atoms = Vec::new();
    let mut identities = Vec::new();
    let mut track_results = Vec::new();
    let mut references = Vec::new();
    let numbers = Numbers::of(
        manifest.language.as_deref(),
        outcomes
            .iter()
            .filter_map(|outcome| outcome.artifact.as_ref())
            .flat_map(|artifact| &artifact.words)
            .map(|word| word.text.as_str()),
    );
    for (track_index, (track, outcome)) in manifest.tracks.iter().zip(outcomes).enumerate() {
        let artifact = outcome.artifact.as_ref();
        let words = artifact.map_or(&[][..], |artifact| &artifact.words);
        let activity = artifact.map_or(&[][..], |artifact| &artifact.activity);
        let reference = reference_level(activity);
        references.push(reference);
        let mut local_speakers = BTreeMap::new();
        for (index, word) in words.iter().enumerate() {
            let context = |neighbor: &LocalWord| {
                let center = i64::midpoint(neighbor.start_sample, neighbor.end_sample);
                let word_center = i64::midpoint(word.start_sample, word.end_sample);
                let text = numbers
                    .alias(&neighbor.text)
                    .unwrap_or_else(|| word_text(&neighbor.text));
                (!text.is_empty() && center.abs_diff(word_center) <= 16_000)
                    .then_some((text, track.start_frame + center))
            };
            let key = if track.role == Role::IsolatedMic {
                Some("s0")
            } else {
                word.speaker_id.as_deref()
            };
            let speaker = key.map(|key| {
                *local_speakers.entry(key).or_insert_with(|| {
                    let id = identities.len();
                    identities.push(SpeakerIdentity {
                        track: track_index,
                        hint: None,
                        isolated: track.role == Role::IsolatedMic,
                        unmiked: false,
                    });
                    id
                })
            });
            atoms.push(Atom {
                track: track_index,
                position: index,
                local: word,
                start: track.start_frame + word.start_sample,
                end: track.start_frame + word.end_sample,
                normalized: word_text(&word.text),
                alias: numbers.alias(&word.text),
                previous: index
                    .checked_sub(1)
                    .and_then(|index| context(&words[index])),
                next: words.get(index + 1).and_then(context),
                speaker,
                level: reference.and_then(|reference| word_level(activity, word, reference)),
                likely_bleed: false,
                isolated: track.role == Role::IsolatedMic,
                acoustic_source: None,
            });
        }
        let speaker_hint = track
            .speaker
            .as_ref()
            .filter(|name| !name.trim().is_empty() && local_speakers.len() == 1);
        if let Some((&speaker, name)) = local_speakers.values().next().zip(speaker_hint) {
            identities[speaker].hint = Some(name.clone());
        }
        track_results.push(TrackResult {
            id: &track.track_id,
            name: &track.name,
            source_id: &track.source_id,
            stream_index: track.binding.stream_index,
            channel: track.channel,
            origin: track.binding.origin,
            session_start: track.binding.session_start,
            stream_evidence: &track.stream_evidence,
            coverage: outcome.coverage,
            role: track.role,
            error: outcome.error.as_deref(),
            warning: artifact.and_then(|artifact| artifact.warning.as_deref()),
            activity,
            words,
            speakers: local_speakers
                .keys()
                .map(|&id| Speaker {
                    id: id.to_owned(),
                    name: speaker_hint.cloned().unwrap_or_else(|| {
                        format!("Speaker {}", id.strip_prefix('s').unwrap_or(id))
                    }),
                })
                .collect(),
        });
    }
    atoms.sort_by_key(|atom| (atom.start, atom.track, atom.end));
    let mut matches = word_matches(&atoms)?;
    let activity: Vec<_> = manifest
        .tracks
        .iter()
        .zip(&track_results)
        .map(|(track, result)| (result.activity, track.start_frame))
        .collect();
    let calibrated = isolation::mark(&mut atoms, &activity, &matches.confirmed)?;
    let faint = isolation::mark_faint(&mut atoms, &activity, &references, &matches.confirmed);
    if calibrated || faint {
        matches = word_matches(&atoms)?;
    }
    let isolated: Vec<_> = manifest
        .tracks
        .iter()
        .map(|track| track.role == Role::IsolatedMic)
        .collect();
    let balances = unmiked::balances(&atoms, &matches.confirmed);
    let shadow = unmiked::mark(
        &mut atoms,
        &mut identities,
        &activity,
        &references,
        &isolated,
        &matches.confirmed,
        &balances,
    );
    // A mixed recording's diarizer can switch labels in the middle of a headset turn.
    // With multiple isolated identities, qualify spill using the other headsets directly.
    if identities
        .iter()
        .filter(|identity| identity.isolated)
        .count()
        < 2
    {
        discount_bleed(&mut atoms, &identities, &matches.confirmed);
    }
    let mut roots = reconcile_speakers(&atoms, &identities, &matches.confirmed)?;
    if let Some(shadow) = shadow {
        unmiked::unify(&identities, &mut roots, shadow);
    }
    let mut clusters: Vec<Vec<usize>> = (0..atoms.len()).map(|index| vec![index]).collect();
    let mut membership: Vec<usize> = (0..atoms.len()).collect();
    let mut alternatives = HashMap::<usize, Vec<(usize, &'static str)>>::new();
    for (left, right) in matches.uncertain {
        alternatives
            .entry(left)
            .or_default()
            .push((right, "ambiguous_lexical_and_timing"));
    }
    for (_, left_word, right_word) in matches.confirmed {
        let left = membership[left_word];
        let right = membership[right_word];
        if left == right {
            continue;
        }
        let incompatible = |a: usize, b: usize| {
            let (a, b) = (&atoms[a], &atoms[b]);
            let score = a.matches(b);
            if let (Some(left), Some(right)) = (a.speaker, b.speaker)
                && roots[left] != roots[right]
            {
                // Local diarizers can split the same physical turn or carry the
                // preceding label into an interjection. Confirmed lexical context can
                // link a mixed recording to its source while retaining both labels.
                return !score.is_some_and(|score| score.has_context)
                    || (identities[left].isolated && identities[right].isolated);
            }
            score.is_none() && !a.overlaps_same_word(b)
        };
        if clusters[left]
            .iter()
            .any(|&a| clusters[right].iter().any(|&b| incompatible(a, b)))
        {
            if let (Some(a), Some(b)) = (atoms[left_word].speaker, atoms[right_word].speaker)
                && roots[a] != roots[b]
                && identities[a].isolated != identities[b].isolated
            {
                let basis = "lexical_timing_with_conflicting_speaker";
                alternatives
                    .entry(left_word)
                    .or_default()
                    .push((right_word, basis));
                alternatives
                    .entry(right_word)
                    .or_default()
                    .push((left_word, basis));
            }
            continue;
        }
        let moved = std::mem::take(&mut clusters[right]);
        for &index in &moved {
            membership[index] = left;
        }
        clusters[left].extend(moved);
    }

    let owners: HashMap<_, _> = identities
        .iter()
        .enumerate()
        .filter(|(_, identity)| identity.isolated)
        .map(|(id, identity)| (identity.track, id))
        .collect();
    let output_speaker = |atom: &Atom<'_>| {
        atom.speaker.or_else(|| {
            atom.acoustic_source
                .and_then(|source| owners.get(&source).copied())
        })
    };
    let voices: Vec<_> = atoms
        .iter()
        .map(|atom| output_speaker(atom).map(|speaker| roots[speaker]))
        .collect();
    let ordered = readings::join_neighbors(
        &atoms,
        &mut clusters,
        &mut membership,
        &voices,
        manifest.tracks.len(),
    )?;
    let mut clusters: Vec<_> = clusters
        .into_iter()
        .filter(|cluster| !cluster.is_empty())
        .collect();
    let has_conflicting_labels = |cluster: &[usize]| {
        let mut labels = cluster
            .iter()
            .filter_map(|&index| atoms[index].speaker.map(|id| roots[id]));
        labels
            .next()
            .is_some_and(|label| labels.any(|other| label != other))
    };
    let utterance = |atom: &Atom<'a>| (atom.track, atom.local.utterance_id.as_str());
    // Conflicts are adjudicated over the recognized span. Picking a different microphone
    // for each short word can reorder words when those microphones disagree on boundaries.
    let mut turn_levels = HashMap::<_, (f32, f32)>::new();
    for cluster in clusters
        .iter()
        .filter(|cluster| has_conflicting_labels(cluster))
    {
        for &index in cluster {
            let atom = &atoms[index];
            if let Some(level) = atom.level {
                let (sum, duration) = turn_levels.entry(utterance(atom)).or_default();
                let weight = (atom.end - atom.start).min(8_000) as f32;
                *sum += level * weight;
                *duration += weight;
            }
        }
    }
    let turn_level = |index: usize| {
        turn_levels
            .get(&utterance(&atoms[index]))
            .map_or(0.0, |(sum, duration)| sum / duration)
    };
    let preference = |index: usize| {
        let atom = &atoms[index];
        (
            atom.likely_bleed,
            manifest.tracks[atom.track].role != Role::IsolatedMic,
            atom.speaker.is_none(),
            atom.alias.is_some(),
        )
    };
    for cluster in &mut clusters {
        let conflicted = has_conflicting_labels(cluster);
        // Isolated speech supplies the canonical wording and timing whenever possible.
        cluster.sort_by(|&a, &b| {
            preference(a)
                .cmp(&preference(b))
                .then_with(|| {
                    if conflicted {
                        turn_level(b).total_cmp(&turn_level(a))
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .then(a.cmp(&b))
        });
    }
    // Recordings with an equal claim to a word each time it a little differently, so
    // taking each word from whichever heard it first can reorder a phrase and split
    // it between their utterances. Without the speaker's own microphone, the recording
    // that heard the most of a turn supplies all of it. Between equals, the nearer
    // recording is the one that hears more of the words first.
    let rivals = |cluster: &[usize]| {
        cluster
            .iter()
            .take_while(|&&index| preference(index) == preference(cluster[0]))
            .count()
    };
    let speaker_of = |cluster: &[usize]| {
        cluster
            .iter()
            .find_map(|&index| output_speaker(&atoms[index]))
            .map(|speaker| roots[speaker])
    };
    let mut open: Vec<_> = (0..clusters.len())
        .filter(|&cluster| {
            let primary = &atoms[clusters[cluster][0]];
            (!primary.isolated || primary.likely_bleed)
                && !has_conflicting_labels(&clusters[cluster])
        })
        .collect();
    open.sort_by_key(|&cluster| (atoms[clusters[cluster][0]].start, clusters[cluster][0]));
    let mut turn = 0;
    while turn < open.len() {
        let mut end = turn + 1;
        while end < open.len() {
            let (previous, next) = (&clusters[open[end - 1]], &clusters[open[end]]);
            if speaker_of(previous) != speaker_of(next)
                || atoms[next[0]].start.saturating_sub(atoms[previous[0]].end) > TURN_GAP
            {
                break;
            }
            end += 1;
        }
        let mut heard = BTreeMap::<usize, (usize, usize)>::new();
        for &cluster in &open[turn..end] {
            let cluster = &clusters[cluster];
            for (place, &index) in cluster[..rivals(cluster)].iter().enumerate() {
                let (words, first) = heard.entry(atoms[index].track).or_default();
                *words += 1;
                *first += usize::from(place == 0);
            }
        }
        for &cluster in &open[turn..end] {
            let count = rivals(&clusters[cluster]);
            clusters[cluster][..count].sort_by(|&a, &b| {
                heard[&atoms[b].track]
                    .cmp(&heard[&atoms[a].track])
                    .then(a.cmp(&b))
            });
        }
        turn = end;
    }
    let apart: Vec<_> = atoms
        .iter()
        .map(|atom| {
            atom.speaker
                .is_some_and(|speaker| identities[roots[speaker]].unmiked)
        })
        .collect();
    let accounted = |word: usize, microphone: usize| {
        unmiked::accounted(&atoms[word], microphone, &balances, &activity, &references)
    };
    let far = readings::Far {
        apart: &apart,
        accounted: &accounted,
    };
    let far = readings::reconcile(
        &atoms,
        &mut clusters,
        &roots,
        manifest,
        &activity,
        &far,
        &numbers,
    )?;
    clusters.retain(|cluster| !cluster.is_empty());
    clusters.sort_by_key(|cluster| (atoms[cluster[0]].start, cluster[0]));
    // A host keeps its speaker renames by ID. Named speakers are numbered as they first
    // appear. The others follow in an order fixed by the recordings, so a voice that
    // comes or goes when a recording is moved leaves the named speakers' IDs alone.
    let mut heard = Vec::new();
    for &index in clusters.iter().flatten() {
        if let Some(speaker) = output_speaker(&atoms[index])
            && !heard.contains(&roots[speaker])
        {
            heard.push(roots[speaker]);
        }
    }
    heard.sort_by_key(|&root| {
        let identity = &identities[root];
        match (&identity.hint, identity.unmiked) {
            (Some(_), _) => (0, 0),
            (None, true) => (1, root),
            (None, false) => (2, root),
        }
    });
    let mut speakers = Vec::new();
    let mut speaker_ids = HashMap::new();
    for root in heard {
        let number = speakers.len() + 1;
        let id = format!("s{number}");
        let name = identities[root]
            .hint
            .clone()
            .unwrap_or_else(|| format!("Speaker {number}"));
        speakers.push(Speaker {
            id: id.clone(),
            name,
        });
        speaker_ids.insert(root, id);
    }
    let mut words = Vec::with_capacity(clusters.len());
    // An utterance is one speaker's stretch of one recording's utterance. It keeps its
    // ID when another speaker's words fall between its own.
    let mut utterance_numbers = HashMap::new();
    for cluster in clusters {
        let primary = &atoms[cluster[0]];
        // A wearer's own word on their microphone names the speaker. Next comes a
        // participant the microphones placed off all of them, then the wearer whose
        // spill a microphone holds, which another voice at the same moment can mimic.
        // A diarizer's label on a mixed recording confuses similar voices and is last.
        let speaker = cluster
            .iter()
            .filter_map(|&index| {
                let atom = &atoms[index];
                let speaker = roots[output_speaker(atom)?];
                let rank = match (identities[speaker].unmiked, atom.isolated) {
                    (false, true) if atom.speaker.is_some() => 0,
                    (true, _) => 1,
                    (false, true) => 2,
                    (false, false) => 3,
                };
                Some((rank, speaker))
            })
            .min_by_key(|&(rank, _)| rank)
            .map(|(_, speaker)| speaker);
        let speaker_id = speaker.and_then(|speaker| speaker_ids.get(&speaker).cloned());
        let utterance = (primary.track, primary.local.utterance_id.as_str(), speaker);
        let next_number = utterance_numbers.len() + 1;
        let utterance_number = *utterance_numbers.entry(utterance).or_insert(next_number);
        let observe = |index: usize, evidence, basis| {
            let atom = &atoms[index];
            Observation {
                track_id: &manifest.tracks[atom.track].track_id,
                word_id: &atom.local.id,
                text: &atom.local.text,
                start_sample: atom.local.start_sample,
                end_sample: atom.local.end_sample,
                evidence,
                basis,
                speaker_id: output_speaker(atom)
                    .and_then(|speaker| speaker_ids.get(&roots[speaker]).cloned()),
            }
        };
        let mut observations: Vec<_> = cluster
            .iter()
            .map(|&index| {
                if far[index] {
                    let basis = "far_reading_within_own_microphone_phrase";
                    return observe(index, "inferred", Some(basis));
                }
                if atoms[index].normalized != primary.normalized {
                    let ordinal = |atom: &Atom<'_>| {
                        (numbers.figure(&atom.local.text)).is_some_and(|figure| figure.ordinal)
                    };
                    let basis = if atoms[index].spoken() != primary.spoken() {
                        readings::number_basis(&numbers, &atoms[index], primary)
                    } else if ordinal(&atoms[index]) || ordinal(primary) {
                        Some("equivalent_spoken_ordinal")
                    } else {
                        Some("equivalent_spoken_number")
                    };
                    let basis = if let Some(basis) = basis {
                        basis
                    } else if atoms[index].acoustic_source == Some(primary.track) {
                        "alternative_reading_of_spill"
                    } else {
                        "alternative_reading_between_matched_words"
                    };
                    return observe(index, "inferred", Some(basis));
                }
                if index != cluster[0]
                    && primary.matches(&atoms[index]).is_none()
                    && (ordered[index] || !primary.overlaps_same_word(&atoms[index]))
                {
                    let basis = if ordered[index] {
                        "ordered_phrase_beside_matched_word"
                    } else {
                        "ordered_phrase_and_correlated_audio"
                    };
                    return observe(index, "recognized", Some(basis));
                }
                observe(
                    index,
                    "recognized",
                    match (index == cluster[0], atoms[index].acoustic_source.is_some()) {
                        (true, true) => Some("calibrated_acoustic_spill"),
                        (true, false) => None,
                        (false, true) => Some("lexical_timing_and_acoustic_spill"),
                        (false, false) => Some("lexical_and_timing"),
                    },
                )
            })
            .collect();
        let mut used_tracks: HashSet<_> = cluster.iter().map(|&index| atoms[index].track).collect();
        for &index in &cluster {
            for &(alternate, basis) in alternatives.get(&index).into_iter().flatten() {
                let atom = &atoms[alternate];
                if matches!((atoms[index].speaker, atom.speaker), (Some(a), Some(b))
                    if roots[a] != roots[b] && identities[a].isolated && identities[b].isolated)
                {
                    continue;
                }
                if used_tracks.insert(atom.track) {
                    observations.push(observe(alternate, "inferred", Some(basis)));
                }
            }
        }
        words.push(ConversationWord {
            id: format!("w{}", words.len() + 1),
            utterance_id: format!("u{utterance_number}"),
            text: &primary.local.text,
            start_sample: primary.start,
            end_sample: primary.end,
            speaker_id,
            precision: Precision::Word,
            observations,
        });
    }
    Ok(SessionResult {
        version: VERSION,
        clock: "session",
        sample_rate: TARGET_SAMPLE_RATE,
        job_id: &manifest.job_id,
        fingerprint: &manifest.fingerprint,
        tracks: track_results,
        speakers,
        words,
    })
}

/// Mutual best, unambiguous match for each pair of tracks. Matching is one-to-one within
/// each track, and close competing candidates are retained as separate conversation words.
struct WordMatches {
    confirmed: Vec<(MatchScore, usize, usize)>,
    uncertain: Vec<(usize, usize)>,
}

fn word_matches(atoms: &[Atom<'_>]) -> Result<WordMatches> {
    struct Candidate {
        score: MatchScore,
        index: usize,
        alternate_error: u64,
    }
    let mut best = HashMap::<(usize, usize), Candidate>::new();
    for (left, atom) in atoms.iter().enumerate() {
        if left % 1024 == 0 {
            check_cancelled()?;
        }
        for (distance, other) in atoms[left + 1..].iter().enumerate() {
            if atom.start.abs_diff(other.start) >= 12_000 {
                break;
            }
            let right = left + 1 + distance;
            if let Some(score) = atom.matches(other) {
                for (index, other_index) in [(left, right), (right, left)] {
                    best.entry((index, atoms[other_index].track))
                        .and_modify(|previous| {
                            if (score.error, score.inverse_overlap)
                                < (previous.score.error, previous.score.inverse_overlap)
                            {
                                previous.alternate_error = previous.score.error;
                                previous.score = score;
                                previous.index = other_index;
                            } else {
                                previous.alternate_error =
                                    previous.alternate_error.min(score.error);
                            }
                        })
                        .or_insert(Candidate {
                            score,
                            index: other_index,
                            alternate_error: u64::MAX,
                        });
                }
            }
        }
    }
    let clear = |candidate: &Candidate| {
        candidate
            .alternate_error
            .saturating_sub(candidate.score.error)
            >= 640
    };
    let mut confirmed = Vec::new();
    let mut uncertain = Vec::new();
    for (&(left, _), candidate) in &best {
        let right = candidate.index;
        let reciprocal = best
            .get(&(right, atoms[left].track))
            .is_some_and(|other| other.index == left && clear(other));
        if clear(candidate) && reciprocal {
            if left < right {
                confirmed.push((candidate.score, left, right));
            }
        } else {
            uncertain.push((left, right));
        }
    }
    confirmed.sort_unstable();
    uncertain.sort_unstable();
    Ok(WordMatches {
        confirmed,
        uncertain,
    })
}

fn reference_level(activity: &[Activity]) -> Option<f32> {
    let mut active: Vec<_> = activity
        .iter()
        .filter_map(|frame| (frame.level > 0.001).then_some(frame.level))
        .collect();
    if active.is_empty() {
        return None;
    }
    let index = (active.len() - 1) * 9 / 10;
    let (_, level, _) = active.select_nth_unstable_by(index, f32::total_cmp);
    Some(*level)
}

fn word_level(activity: &[Activity], word: &LocalWord, reference: f32) -> Option<f32> {
    let first = usize::try_from(word.start_sample).ok()? / ACTIVITY_WINDOW;
    let end = usize::try_from(word.end_sample)
        .ok()?
        .div_ceil(ACTIVITY_WINDOW);
    let frames = activity.get(first..end)?;
    let rms = frames
        .iter()
        .map(|frame| frame.level * frame.level)
        .sum::<f32>()
        / frames.len() as f32;
    Some(rms.sqrt() / reference)
}

/// Relative gain qualifies a speaker label only after both tracks actually recognized the
/// same aligned words. A consistently faint pair can be spill onto an isolated mic when
/// that mic also has stronger lexical evidence for a different local speaker.
fn discount_bleed(
    atoms: &mut [Atom<'_>],
    identities: &[SpeakerIdentity],
    edges: &[(MatchScore, usize, usize)],
) {
    let mut levels = HashMap::<(usize, usize), Vec<f32>>::new();
    let mut pairs = Vec::new();
    for &(_, left, right) in edges {
        let (Some(a), Some(b)) = (atoms[left].speaker, atoms[right].speaker) else {
            continue;
        };
        let (isolated, mixed, isolated_word) =
            match (identities[a].isolated, identities[b].isolated) {
                (true, false) => (a, b, left),
                (false, true) => (b, a, right),
                _ => continue,
            };
        let levels_for_pair = levels.entry((isolated, mixed)).or_default();
        let (Some(isolated_level), Some(mixed_level)) = (
            atoms[isolated_word].level,
            atoms[if isolated_word == left { right } else { left }].level,
        ) else {
            continue;
        };
        if mixed_level < 0.15 {
            continue;
        }
        levels_for_pair.push(isolated_level / mixed_level);
        pairs.push((isolated_word, (isolated, mixed)));
    }
    let medians: HashMap<_, _> = levels
        .into_iter()
        .filter_map(|(pair, mut levels)| {
            if levels.len() < 2 {
                return None;
            }
            let index = levels.len() / 2;
            let (_, level, _) = levels.select_nth_unstable_by(index, f32::total_cmp);
            Some((pair, *level))
        })
        .collect();
    let mut suspect = HashSet::new();
    let mut supported = HashSet::new();
    for (index, (isolated, mixed)) in pairs {
        let Some(&median) = medians.get(&(isolated, mixed)) else {
            continue;
        };
        let has_stronger_speaker =
            medians
                .iter()
                .any(|(&(other_mic, other_speaker), &other_level)| {
                    isolated == other_mic
                        && identities[mixed].track == identities[other_speaker].track
                        && mixed != other_speaker
                        && other_level > 0.8
                        && median * 1.65 < other_level
                });
        let is_strong = medians
            .iter()
            .any(|(&(other_mic, other_speaker), &other_level)| {
                isolated == other_mic
                    && identities[mixed].track == identities[other_speaker].track
                    && mixed != other_speaker
                    && median > 0.8
                    && other_level * 1.65 < median
            });
        if has_stronger_speaker {
            suspect.insert(index);
        } else if is_strong {
            supported.insert(index);
        }
    }
    for index in suspect.difference(&supported).copied() {
        atoms[index].speaker = None;
        atoms[index].likely_bleed = true;
    }
}

/// Combine speaker labels only when multiple aligned word observations agree. Competing
/// mappings and conflicting isolated speaker hints remain distinct.
fn reconcile_speakers(
    atoms: &[Atom<'_>],
    identities: &[SpeakerIdentity],
    edges: &[(MatchScore, usize, usize)],
) -> Result<Vec<usize>> {
    let mut votes = HashMap::<(usize, usize), usize>::new();
    for (_, left, right) in edges {
        if let (Some(left), Some(right)) = (atoms[*left].speaker, atoms[*right].speaker) {
            *votes.entry((left.min(right), left.max(right))).or_default() += 1;
        }
    }
    let mut candidates: Vec<_> = votes
        .iter()
        .filter_map(|(&(left, right), &count)| {
            (count >= 2).then_some((std::cmp::Reverse(count), left, right))
        })
        .collect();
    for (left, identity) in identities.iter().enumerate() {
        if !identity.isolated || identity.hint.is_none() {
            continue;
        }
        for (right, other) in identities.iter().enumerate().skip(left + 1) {
            if other.isolated && identity.hint == other.hint {
                candidates.push((std::cmp::Reverse(usize::MAX), left, right));
            }
        }
    }
    candidates.sort_unstable();
    let mut roots: Vec<_> = (0..identities.len()).collect();
    for (std::cmp::Reverse(count), left, right) in candidates {
        check_cancelled()?;
        // A recording that heard only part of such a turn must not hand it back to a wearer.
        if identities[left].unmiked != identities[right].unmiked {
            continue;
        }
        let has_rival = |id: usize, target: usize| {
            identities.iter().enumerate().any(|(candidate, other)| {
                other.track == identities[target].track
                    && candidate != target
                    && votes
                        .get(&(id.min(candidate), id.max(candidate)))
                        .copied()
                        .unwrap_or(0)
                        >= count
            })
        };
        if has_rival(left, right) || has_rival(right, left) {
            continue;
        }
        let (left_root, right_root) = (roots[left], roots[right]);
        if left_root == right_root {
            continue;
        }
        let overlaps = identities
            .iter()
            .enumerate()
            .filter(|(id, _)| roots[*id] == left_root)
            .any(|(_, left)| {
                identities
                    .iter()
                    .enumerate()
                    .filter(|(id, _)| roots[*id] == right_root)
                    .any(|(_, right)| {
                        left.track == right.track
                            || (left.isolated
                                && right.isolated
                                && left
                                    .hint
                                    .as_ref()
                                    .zip(right.hint.as_ref())
                                    .is_none_or(|(left, right)| left != right))
                    })
            });
        if overlaps {
            continue;
        }
        let preferred = |id: usize| (!identities[id].isolated, identities[id].hint.is_none(), id);
        let (keep, replace) = if preferred(left_root) <= preferred(right_root) {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        for root in &mut roots {
            if *root == replace {
                *root = keep;
            }
        }
    }
    Ok(roots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Binding, Coverage, Time, Track, TrackArtifact};
    use serde_json::json;
    use std::path::PathBuf;

    fn track(id: &str, start_frame: i64, role: Role) -> Track {
        Track {
            track_id: id.to_owned(),
            source_id: "source".to_owned(),
            stream_evidence: Some(json!({"sample_digest":"fixture"})),
            binding: Binding {
                stream_index: Some(0),
                origin: Time {
                    numerator: 0,
                    denominator: 1,
                },
                session_start: Time {
                    numerator: start_frame,
                    denominator: 16_000,
                },
            },
            channel: None,
            name: "Interview audio".to_owned(),
            role,
            speaker: None,
            recognize: true,
            start_frame,
            audio: None,
            result: None,
            activity: None,
            recognition_key: None,
            unavailable: None,
            shared_cache: None,
            reuse: false,
        }
    }

    fn outcome(words: &[(&str, i64, i64, Option<&str>)], coverage: Coverage) -> Outcome {
        let words = words
            .iter()
            .enumerate()
            .map(|(index, &(text, start, end, speaker))| LocalWord {
                id: format!("w{}", index + 1),
                utterance_id: "u1".to_owned(),
                text: text.to_owned(),
                start_sample: start,
                end_sample: end,
                speaker_id: speaker.map(str::to_owned),
                precision: Precision::Word,
            })
            .collect();
        Outcome {
            coverage,
            artifact: Some(TrackArtifact {
                version: VERSION,
                sample_rate: TARGET_SAMPLE_RATE,
                recognition_key: None,
                stream_evidence: None,
                duration_samples: 16000,
                activity: vec![Activity {
                    start_sample: 0,
                    end_sample: 320,
                    level: 0.1,
                }],
                words,
                warning: None,
            }),
            error: None,
        }
    }

    fn manifest(tracks: Vec<Track>) -> Manifest {
        Manifest {
            schema_version: VERSION,
            job_id: "job".to_owned(),
            fingerprint: "fingerprint".to_owned(),
            engine: None,
            model: None,
            language: None,
            expected_speakers: None,
            diarization: Some(false),
            output: PathBuf::new(),
            tracks,
        }
    }

    #[test]
    fn reconciles_signed_clocks_speakers_and_repetition_without_using_activity_as_speech() {
        let mut lav = track("lav", -16_000, Role::IsolatedMic);
        lav.speaker = Some("Host".to_owned());
        let camera = track("camera", -15_500, Role::MixedAudio);
        let manifest = manifest(vec![lav, camera, track("cough", 0, Role::Unknown)]);
        let outcomes = [
            outcome(
                &[(" Here", 600, 2600, None), (" here.", 2800, 4600, None)],
                Coverage::Recognized,
            ),
            outcome(
                &[
                    ("Here", 100, 2100, Some("s0")),
                    (" Yes!", 1000, 1900, Some("s1")),
                    (" here", 2300, 4100, Some("s0")),
                    (" Off air", 6000, 8000, Some("s1")),
                ],
                Coverage::Recognized,
            ),
            outcome(&[], Coverage::Skipped),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 4);
        let first = &result.words[0];
        assert_eq!(first.start_sample, -15_400);
        assert_eq!(first.observations.len(), 2);
        assert_eq!(first.observations[1].start_sample, 100);
        assert_eq!(result.speakers[0].name, "Host");
        assert_eq!(first.speaker_id, result.words[2].speaker_id);
        assert_eq!(result.words[1].speaker_id, result.words[3].speaker_id);
        assert_ne!(first.speaker_id, result.words[1].speaker_id);
        assert_eq!(result.words[3].text, " Off air");
        assert_eq!(result.tracks[2].activity.len(), 1);
        assert!(
            result
                .words
                .iter()
                .flat_map(|word| &word.observations)
                .all(|observation| observation.track_id != "cough")
        );
    }

    #[test]
    fn ambiguous_word_match_retains_all_decoded_words_and_marks_inferred_links() {
        let manifest = manifest(vec![
            track("lav", 0, Role::IsolatedMic),
            track("cam", 0, Role::MixedAudio),
        ]);
        let outcomes = [
            outcome(
                &[("Yes", 0, 1000, None), (" yes", 1000, 2000, None)],
                Coverage::Recognized,
            ),
            outcome(&[("yes", 500, 1500, None)], Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 3);
        assert_eq!(
            result
                .words
                .iter()
                .flat_map(|word| &word.observations)
                .filter(|observation| observation.evidence == "recognized")
                .count(),
            3
        );
        assert!(
            result
                .words
                .iter()
                .flat_map(|word| &word.observations)
                .any(|observation| observation.evidence == "inferred"
                    && observation.basis == Some("ambiguous_lexical_and_timing"))
        );
    }

    #[test]
    fn simultaneous_same_word_from_distinct_isolated_speakers_stays_distinct() {
        let mut host = track("host-lav", 0, Role::IsolatedMic);
        host.speaker = Some("Host".to_owned());
        let mut guest = track("guest-lav", 0, Role::IsolatedMic);
        guest.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![host, guest]);
        let outcomes = [
            outcome(
                &[(" Yes", 0, 1200, None), (" yes!", 1400, 2000, None)],
                Coverage::Recognized,
            ),
            outcome(&[("Yes!", 0, 1200, None)], Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 3);
        assert_eq!(result.words[0].start_sample, result.words[1].start_sample);
        assert_ne!(result.words[0].speaker_id, result.words[1].speaker_id);
        assert_eq!(result.words[0].speaker_id, result.words[2].speaker_id);
        assert!(result.words.iter().all(|word| word.observations.len() == 1));
    }

    #[test]
    fn calibrated_spill_merges_copies_and_keeps_a_louder_concurrent_voice_distinct() {
        let mut near = track("near", -3200, Role::IsolatedMic);
        near.speaker = Some("Host".to_owned());
        let mut far = track("far", -3200, Role::IsolatedMic);
        far.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![near, far]);
        let words = [
            ("One", 8_000, 10_400, None),
            (" two", 17_600, 20_800, None),
            (" three", 27_200, 30_400, None),
            (" four", 36_800, 40_000, None),
            (" yes", 48_000, 51_200, None),
        ];
        let mut outcomes = [
            outcome(&words, Coverage::Recognized),
            outcome(&words, Coverage::Recognized),
        ];
        for (index, outcome) in outcomes.iter_mut().enumerate() {
            let artifact = outcome.artifact.as_mut().unwrap();
            artifact.duration_samples = 64_000;
            artifact.activity = (0..200)
                .map(|frame| {
                    let active = (15..145).contains(&frame) || (147..165).contains(&frame);
                    let modulation = if active {
                        0.15 + ((frame * 37) % 31) as f32 * 0.013
                    } else {
                        0.002
                    };
                    let level = if index == 0 {
                        modulation
                    } else if (147..165).contains(&frame) {
                        0.16 + ((frame * 19) % 11) as f32 * 0.013
                    } else {
                        modulation * 0.08 + 0.001
                    };
                    Activity {
                        start_sample: frame * 320,
                        end_sample: (frame + 1) * 320,
                        level,
                    }
                })
                .collect();
        }
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 6);
        assert_eq!(result.tracks[1].words.len(), words.len());
        for word in &result.words[..4] {
            assert_eq!(word.observations.len(), 2);
            assert_eq!(word.observations[1].evidence, "recognized");
            assert_eq!(
                word.observations[1].basis,
                Some("lexical_timing_and_acoustic_spill")
            );
        }
        assert_eq!(result.words[4].start_sample, result.words[5].start_sample);
        assert_ne!(result.words[4].speaker_id, result.words[5].speaker_id);
    }

    #[test]
    fn a_participant_without_a_microphone_is_heard_once_as_a_separate_speaker() {
        let mut host = track("host-lav", 0, Role::IsolatedMic);
        host.speaker = Some("Host".to_owned());
        let mut guest = track("guest-lav", 0, Role::IsolatedMic);
        guest.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![host, guest, track("camera", 0, Role::MixedAudio)]);
        let spoken = [
            "Welcome",
            " back",
            " everyone",
            " today",
            " we",
            " restore",
            " cameras",
            " thanks",
            " for",
            " having",
            " me",
            " here",
            " this",
            " morning",
            " that",
            " was",
            " great",
            " take",
        ];
        // Seven Host words, seven Guest words, then four from a producer behind the camera.
        let owner = |word: usize| match word {
            0..7 => 0,
            7..14 => 1,
            _ => 2,
        };
        // The camera's diarizer hears the producer as the Host.
        let labels = [Some("s0"), Some("s1"), Some("s0")];
        // Level of each voice on the Host lav, the Guest lav, and the camera.
        let gains = [[0.2, 0.014, 0.1], [0.012, 0.2, 0.1], [0.005, 0.007, 0.18]];
        let mut outcomes = [0, 1, 2].map(|microphone| {
            let words: Vec<_> = spoken
                .iter()
                .enumerate()
                .map(|(word, &text)| {
                    let start = 3_200 + word as i64 * 6_400;
                    let label = if microphone == 2 {
                        labels[owner(word)]
                    } else {
                        None
                    };
                    (text, start, start + 3_200, label)
                })
                .collect();
            let mut outcome = outcome(&words, Coverage::Recognized);
            let artifact = outcome.artifact.as_mut().unwrap();
            artifact.duration_samples = 128_000;
            artifact.activity = (0..400)
                .map(|frame: i64| {
                    let word = (frame - 10).div_euclid(20);
                    let speaking = frame >= 10
                        && (frame - 10).rem_euclid(20) < 10
                        && (word as usize) < spoken.len();
                    let level = if speaking {
                        let syllable = 0.6 + ((frame * 37) % 31) as f32 * 0.025;
                        gains[owner(word as usize)][microphone] * syllable + 0.0006
                    } else {
                        0.0006
                    };
                    Activity {
                        start_sample: frame * 320,
                        end_sample: (frame + 1) * 320,
                        level,
                    }
                })
                .collect();
            outcome
        });
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        let expected: Vec<_> = spoken.iter().map(|word| word.trim()).collect();
        assert_eq!(texts, expected);
        let name = |word: &ConversationWord<'_>| {
            let id = word.speaker_id.as_ref().unwrap();
            result
                .speakers
                .iter()
                .find(|speaker| &speaker.id == id)
                .unwrap()
                .name
                .clone()
        };
        assert!(result.words[..7].iter().all(|word| name(word) == "Host"));
        assert!(result.words[7..14].iter().all(|word| name(word) == "Guest"));
        for word in &result.words[14..] {
            assert_eq!(name(word), "Speaker 3");
            assert_eq!(word.observations.len(), 3);
            assert_eq!(word.observations[0].track_id, "camera");
            assert!(
                word.observations[1..]
                    .iter()
                    .all(|observation| observation.evidence == "recognized"
                        && observation.speaker_id.is_none())
            );
        }
        assert_eq!(result.speakers.len(), 3);

        // The Host turns to speak quietly: softer everywhere, in the same balance.
        for outcome in &mut outcomes {
            for frame in &mut outcome.artifact.as_mut().unwrap().activity[10..150] {
                if frame.level > 0.001 {
                    frame.level *= 0.2;
                }
            }
        }
        let quiet = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(quiet.words.len(), spoken.len());
        assert!(quiet.words[..7].iter().all(|word| {
            quiet.speakers.iter().any(|speaker| {
                Some(&speaker.id) == word.speaker_id.as_ref() && speaker.name == "Host"
            })
        }));
    }

    /// Recordings of voices that speak one word per 6 400-sample slot. Each voice has a
    /// level on every recording, and `hears` selects the words a recording recognized.
    fn slotted(
        spoken: &[(&str, i64, usize)],
        gains: &[[f32; 3]],
        labels: [Option<&str>; 3],
        hears: impl Fn(usize, i64, usize) -> bool,
    ) -> [Outcome; 3] {
        [0, 1, 2].map(|recording| {
            let words: Vec<_> = spoken
                .iter()
                .filter(|&&(_, slot, voice)| hears(recording, slot, voice))
                .map(|&(text, slot, voice)| {
                    let start = 3_200 + slot * 6_400;
                    let label = if recording == 2 { labels[voice] } else { None };
                    (text, start, start + 3_200, label)
                })
                .collect();
            let mut outcome = outcome(&words, Coverage::Recognized);
            let artifact = outcome.artifact.as_mut().unwrap();
            artifact.duration_samples = 160_000;
            artifact.activity = (0..500)
                .map(|frame: i64| {
                    let slot = (frame - 10).div_euclid(20);
                    let power: f32 = spoken
                        .iter()
                        .filter(|word| word.1 == slot && (frame - 10).rem_euclid(20) < 10)
                        .map(|word| gains[word.2][recording].powi(2))
                        .sum();
                    let syllable = 0.6 + ((frame * 37) % 31) as f32 * 0.025;
                    Activity {
                        start_sample: frame * 320,
                        end_sample: (frame + 1) * 320,
                        level: power.sqrt() * syllable + 0.0006,
                    }
                })
                .collect();
            outcome
        })
    }

    #[test]
    fn an_interjection_off_microphone_keeps_its_speaker_while_a_wearer_talks() {
        let mut host = track("host-lav", 0, Role::IsolatedMic);
        host.speaker = Some("Host".to_owned());
        let mut guest = track("guest-lav", 0, Role::IsolatedMic);
        guest.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![host, guest, track("camera", 0, Role::MixedAudio)]);
        // The Host, then the Guest. A producer behind the camera starts in the Guest's
        // pause and keeps talking over the Guest's next two words.
        let spoken = [
            ("Welcome", 0, 0),
            (" back", 1, 0),
            (" everyone", 2, 0),
            (" today", 3, 0),
            (" we", 4, 0),
            (" restore", 5, 0),
            (" cameras", 6, 0),
            (" thanks", 7, 1),
            (" for", 8, 1),
            (" having", 9, 1),
            (" me", 10, 1),
            (" sorry", 11, 2),
            (" one", 12, 2),
            (" moment", 13, 2),
            (" this", 12, 1),
            (" morning", 13, 1),
            (" it", 14, 1),
            (" was", 15, 1),
            (" great", 16, 1),
        ];
        // Level of each voice on the Host lav, the Guest lav, and the camera.
        let gains = [[0.2, 0.014, 0.1], [0.012, 0.2, 0.1], [0.005, 0.007, 0.18]];
        // The camera's diarizer hears the producer as the Guest.
        let labels = [Some("s0"), Some("s1"), Some("s1")];
        // The camera follows the producer over the Guest. The lavs miss the producer.
        let hears = |recording: usize, slot: i64, voice: usize| match recording {
            2 => !(voice == 1 && (12..14).contains(&slot)),
            _ => voice != 2,
        };
        let outcomes = slotted(&spoken, &gains, labels, hears);
        let result = reconcile(&manifest, &outcomes).unwrap();
        let name = |result: &SessionResult<'_>, text: &str| {
            let word = result.words.iter().find(|word| word.text == text).unwrap();
            let id = word.speaker_id.as_ref().unwrap();
            let speaker = result.speakers.iter().find(|speaker| &speaker.id == id);
            speaker.unwrap().name.clone()
        };
        assert_eq!(result.words.len(), spoken.len());
        for word in [" sorry", " one", " moment"] {
            assert_eq!(name(&result, word), "Speaker 3", "{word}");
        }
        for word in [" me", " this", " morning", " it"] {
            assert_eq!(name(&result, word), "Guest", "{word}");
        }
        assert_eq!(result.speakers.len(), 3);
        drop(result);

        // The camera's aligner drops the Guest's last word before the pause into it.
        // The Guest's own microphone recognized that word, so it stays the Guest's.
        let mut misplaced = slotted(&spoken, &gains, labels, hears);
        let word = &mut misplaced[2].artifact.as_mut().unwrap().words[10];
        assert_eq!(word.text, " me");
        (word.start_sample, word.end_sample) = (74_000, 74_001);
        let result = reconcile(&manifest, &misplaced).unwrap();
        let guest = result
            .speakers
            .iter()
            .find(|speaker| speaker.name == "Guest");
        let word = result.words.iter().find(|word| word.text == " me").unwrap();
        assert_eq!(word.observations.len(), 3);
        for observation in &word.observations {
            assert!(
                observation.speaker_id.is_none()
                    || observation.speaker_id == guest.map(|g| g.id.clone())
            );
        }
        drop(result);

        // The camera starts the producer's second word a hundredth of a second before
        // the Guest's next one, while the Guest's microphone still rings from a sound
        // it recognized nothing in. A gap that short is no pause to measure.
        let mut early = slotted(&spoken, &gains, labels, hears);
        early[1].artifact.as_mut().unwrap().activity[249].level = 0.15;
        let one = &mut early[2].artifact.as_mut().unwrap().words[12];
        assert_eq!(one.text, " one");
        one.start_sample -= 160;
        let result = reconcile(&manifest, &early).unwrap();
        for word in [" sorry", " one", " moment"] {
            assert_eq!(name(&result, word), "Speaker 3", "{word}");
        }
        drop(result);

        // The Host's microphone reads the producer's faint first word as another one.
        // It stayed quiet through that word, so the reading is not the Host's.
        let mut misread = slotted(&spoken, &gains, labels, hears);
        let words = &mut misread[0].artifact.as_mut().unwrap().words;
        let mut story = words[11].clone();
        assert_eq!(words[10].text, " me");
        (story.id, story.text) = ("w99".to_owned(), " story".to_owned());
        (story.start_sample, story.end_sample) = (3_200 + 11 * 6_400, 6_400 + 11 * 6_400);
        words.insert(11, story);
        let result = reconcile(&manifest, &misread).unwrap();
        assert_eq!(result.words.len(), spoken.len());
        let sorry = result.words.iter().find(|word| word.text == " sorry");
        let reading = sorry.unwrap().observations.last().unwrap();
        assert_eq!(
            (
                reading.track_id,
                reading.text,
                reading.evidence,
                reading.basis
            ),
            (
                "host-lav",
                " story",
                "inferred",
                Some("alternative_reading_of_spill")
            )
        );
        assert_eq!(result.speakers.len(), 3);
        drop(result);

        // With no word in the Guest's pause, nothing places the voice off microphone.
        let outcomes = slotted(&spoken, &gains, labels, |recording, slot, voice| {
            hears(recording, slot, voice) && slot != 11
        });
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(name(&result, " one"), "Guest");
        assert_eq!(result.speakers.len(), 2);
    }

    /// Two lavs and a camera: the Host speaks four words, then the Guest ten.
    fn interview() -> (Manifest, Vec<(&'static str, i64, usize)>) {
        let mut host = track("host-lav", 0, Role::IsolatedMic);
        host.speaker = Some("Host".to_owned());
        let mut guest = track("guest-lav", 0, Role::IsolatedMic);
        guest.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![host, guest, track("camera", 0, Role::MixedAudio)]);
        let host = ["Welcome", " back", " everyone", " today"];
        let guest = [
            " it", " started", " with", " three", " pots", " and", " broken", " cans", " last",
            " spring",
        ];
        let spoken = host
            .iter()
            .map(|&word| (word, 0))
            .chain(guest.iter().map(|&word| (word, 1)))
            .enumerate()
            .map(|(slot, (word, voice))| (word, slot as i64, voice))
            .collect();
        (manifest, spoken)
    }

    fn speaker_name(result: &SessionResult<'_>, text: &str) -> String {
        let word = result.words.iter().find(|word| word.text == text).unwrap();
        let id = word.speaker_id.as_ref().unwrap();
        let speaker = result.speakers.iter().find(|speaker| &speaker.id == id);
        speaker.unwrap().name.clone()
    }

    #[test]
    fn a_far_recordings_garbled_stretch_joins_the_wearers_own_words() {
        let (manifest, spoken) = interview();
        let gains = [[0.2, 0.014, 0.1], [0.012, 0.2, 0.1], [0.005, 0.007, 0.3]];
        let labels = [Some("s0"), Some("s1"), Some("s1")];
        let everyone = |_: usize, _: i64, voice: usize| voice != 2;
        let texts = |result: &SessionResult<'_>| -> Vec<String> {
            let words = result.words.iter();
            words.map(|word| word.text.trim().to_owned()).collect()
        };
        let said: Vec<_> = spoken.iter().map(|word| word.0.trim().to_owned()).collect();

        // The camera hears "three pots" as two other words.
        let mut garbled = slotted(&spoken, &gains, labels, everyone);
        let camera = &mut garbled[2].artifact.as_mut().unwrap().words;
        camera[7].text = " free".to_owned();
        camera[8].text = " lots".to_owned();
        let result = reconcile(&manifest, &garbled).unwrap();
        assert_eq!(texts(&result), said);
        let far = result.words[7].observations.last().unwrap();
        assert_eq!(
            (far.track_id, far.text, far.evidence, far.basis),
            (
                "camera",
                " free",
                "inferred",
                Some("far_reading_within_own_microphone_phrase")
            )
        );
        drop(result);

        // One word against one word keeps the rule for a single alternate reading.
        garbled[2].artifact.as_mut().unwrap().words[8].text = " pots".to_owned();
        assert_eq!(
            reconcile(&manifest, &garbled).unwrap().words.len(),
            said.len() + 1
        );

        // Two garbled words on a camera that is all but silent there show nothing
        // about whose speech they are, and stay.
        let mut silent = slotted(&spoken, &gains, labels, everyone);
        let camera = silent[2].artifact.as_mut().unwrap();
        camera.words[7].text = " free".to_owned();
        camera.words[8].text = " lots".to_owned();
        for frame in &mut camera.activity[150..190] {
            frame.level = 0.0006;
        }
        assert_eq!(
            reconcile(&manifest, &silent).unwrap().words.len(),
            said.len() + 2
        );

        // A word the camera hears where the Guest's microphone has none stays, even
        // though that microphone's recognizer ran the word before across it.
        let mut between = slotted(&spoken, &gains, labels, everyone);
        let camera = &mut between[2].artifact.as_mut().unwrap().words;
        let mut heard = camera[7].clone();
        (heard.id, heard.text) = ("w99".to_owned(), " really".to_owned());
        heard.start_sample += 1_700;
        camera[7].end_sample -= 1_700;
        camera.insert(8, heard);
        let result = reconcile(&manifest, &between).unwrap();
        assert_eq!(result.words.len(), said.len() + 1);
        assert!(result.words.iter().any(|word| word.text == " really"));
        drop(result);

        // It splits "broken" into an extra word and a near copy. The extra word stays.
        let mut split = slotted(&spoken, &gains, labels, everyone);
        let camera = &mut split[2].artifact.as_mut().unwrap().words;
        let mut extra = camera[10].clone();
        (extra.id, extra.text) = ("w99".to_owned(), " a".to_owned());
        extra.end_sample = extra.start_sample + 1_200;
        camera[10].text = " brokens".to_owned();
        camera[10].start_sample += 1_400;
        camera.insert(10, extra);
        let result = reconcile(&manifest, &split).unwrap();
        assert_eq!(result.words.len(), said.len() + 1);
        assert!(result.words.iter().any(|word| word.text == " a"));
        assert!(result.words.iter().all(|word| word.text != " brokens"));
        drop(result);

        // Someone off microphone says two words over the Guest. The camera is louder
        // than the Guest's microphone accounts for, and the words stay.
        let mut over = spoken.clone();
        over.extend([(" hold", 7, 2), (" on", 8, 2)]);
        over.sort_by_key(|&(_, slot, voice)| (slot, voice));
        let outcomes = slotted(
            &over,
            &gains,
            labels,
            |recording, slot, voice| match recording {
                2 => !(voice == 1 && (7..9).contains(&slot)),
                _ => voice != 2,
            },
        );
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), said.len() + 2);
        assert!(result.words.iter().any(|word| word.text == " hold"));
        drop(result);

        // A second far recording that hears nearly the same word supports it.
        let mut wide = track("wide", 0, Role::MixedAudio);
        wide.source_id = "wide".to_owned();
        let mut tracks = manifest.tracks;
        tracks.push(wide);
        let manifest = super::tests::manifest(tracks);
        let mut outcomes = Vec::from(garbled);
        let mut second = outcomes[2].artifact.clone().unwrap();
        outcomes[2].artifact.as_mut().unwrap().words[8].text = " lots".to_owned();
        second.words[7].text = " freed".to_owned();
        second.words[8].text = " slots".to_owned();
        outcomes.push(Outcome {
            coverage: Coverage::Recognized,
            artifact: Some(second),
            error: None,
        });
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), said.len() + 4);
    }

    #[test]
    fn a_phrase_spoken_entirely_over_a_wearer_is_placed_by_the_level_balance() {
        let (manifest, mut spoken) = interview();
        // A producer behind the camera talks over three of the Guest's words. The
        // camera follows the producer there and gives the words the Guest's label.
        spoken.extend([(" hold", 6, 2), (" on", 7, 2), (" please", 8, 2)]);
        spoken.sort_by_key(|&(_, slot, voice)| (slot, voice));
        let gains = [[0.2, 0.014, 0.1], [0.012, 0.2, 0.1], [0.005, 0.007, 0.3]];
        let labels = [Some("s0"), Some("s1"), Some("s1")];
        // The Host's microphone recognizes the producer's words too, faintly.
        let hears = |recording: usize, slot: i64, voice: usize| match recording {
            2 => !(voice == 1 && (6..9).contains(&slot)),
            1 => voice != 2,
            _ => true,
        };
        let mut outcomes = slotted(&spoken, &gains, labels, hears);
        // The recognizer ends the first word early, a comma's pause before the next.
        let hold = &mut outcomes[2].artifact.as_mut().unwrap().words[6];
        assert_eq!(hold.text, " hold");
        hold.end_sample -= 1_600;
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), spoken.len());
        for word in [" hold", " on", " please"] {
            assert_eq!(speaker_name(&result, word), "Speaker 3", "{word}");
        }
        for word in [" with", " three", " pots", " and"] {
            assert_eq!(speaker_name(&result, word), "Guest", "{word}");
        }
        assert_eq!(result.speakers.len(), 3);
        drop(result);

        // Words the camera's recognizer set down early repeat what the Guest's own
        // microphone recognizes a moment later. They are not another voice.
        let camera = &mut outcomes[2].artifact.as_mut().unwrap().words;
        for (word, text) in camera[6..9].iter_mut().zip([" broken", " cans", " last"]) {
            word.text = text.to_owned();
        }
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.speakers.len(), 2);
    }

    #[test]
    fn words_under_a_borrowed_label_join_the_one_voice_a_diarizer_kept_apart() {
        let (manifest, mut spoken) = interview();
        // A producer talks over three of the Guest's words and again after the Guest.
        spoken.extend([(" hold", 6, 2), (" on", 7, 2), (" please", 8, 2)]);
        spoken.extend([
            (" okay", 14, 2),
            (" it", 15, 2),
            (" stopped", 16, 2),
            (" now", 17, 2),
        ]);
        spoken.sort_by_key(|&(_, slot, voice)| (slot, voice));
        let gains = [[0.2, 0.014, 0.1], [0.012, 0.2, 0.1], [0.005, 0.007, 0.3]];
        let hears = |recording: usize, slot: i64, voice: usize| match recording {
            2 => !(voice == 1 && (6..9).contains(&slot)),
            _ => voice != 2,
        };
        // The camera's diarizer keeps the producer apart in the pause and hears the
        // Guest in the words spoken over the Guest.
        let labels = [Some("s0"), Some("s1"), Some("s2")];
        let mut outcomes = slotted(&spoken, &gains, labels, hears);
        let camera = &mut outcomes[2].artifact.as_mut().unwrap().words;
        for word in &mut camera[6..9] {
            word.speaker_id = Some("s1".to_owned());
        }
        let result = reconcile(&manifest, &outcomes).unwrap();
        let names: Vec<_> = result
            .speakers
            .iter()
            .map(|speaker| &speaker.name)
            .collect();
        assert_eq!(names, ["Host", "Guest", "Speaker 3"]);
        for word in [" hold", " on", " please", " okay", " stopped", " now"] {
            assert_eq!(speaker_name(&result, word), "Speaker 3", "{word}");
        }
        drop(result);

        // With two voices kept apart, nothing says which one spoke over the Guest.
        let camera = &mut outcomes[2].artifact.as_mut().unwrap().words;
        for word in camera.iter_mut().rev().take(2) {
            word.speaker_id = Some("s3".to_owned());
        }
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.speakers.len(), 5);
        assert_ne!(
            speaker_name(&result, " hold"),
            speaker_name(&result, " okay")
        );
        assert_ne!(
            speaker_name(&result, " hold"),
            speaker_name(&result, " now")
        );
    }

    #[test]
    fn a_word_placed_off_microphone_keeps_that_speaker_beside_a_diarizers_label() {
        let (manifest, mut spoken) = interview();
        spoken.extend([(" four", 6, 2), (" five", 7, 2), (" six", 8, 2)]);
        spoken.sort_by_key(|&(_, slot, voice)| (slot, voice));
        let gains = [[0.2, 0.014, 0.1], [0.012, 0.2, 0.1], [0.005, 0.007, 0.3]];
        let labels = [Some("s0"), Some("s1"), Some("s1")];
        let hears = |recording: usize, slot: i64, voice: usize| match recording {
            2 => !(voice == 1 && (6..9).contains(&slot)),
            _ => voice != 2,
        };
        let mut outcomes = Vec::from(slotted(&spoken, &gains, labels, hears));
        // A second camera hears the producer's first word no louder than the Guest. The
        // phrase is too short there to place, and its words keep the Guest's label.
        let mut wide = track("wide", 0, Role::MixedAudio);
        wide.source_id = "wide".to_owned();
        let mut tracks = manifest.tracks;
        tracks.push(wide);
        let manifest = super::tests::manifest(tracks);
        let mut second = outcomes[2].artifact.clone().unwrap();
        for (frame, level) in second.activity.iter_mut().enumerate() {
            if (frame as i64 - 10).div_euclid(20) == 6 {
                level.level *= 0.3;
            }
        }
        outcomes.push(Outcome {
            coverage: Coverage::Recognized,
            artifact: Some(second),
            error: None,
        });
        // The first camera writes digits, so the second supplies the wording.
        let camera = &mut outcomes[2].artifact.as_mut().unwrap().words;
        for (word, digit) in camera[6..9].iter_mut().zip([" 4", " 5", " 6"]) {
            word.text = digit.to_owned();
        }
        let result = reconcile(&manifest, &outcomes).unwrap();
        for word in [" four", " five", " six"] {
            assert_eq!(speaker_name(&result, word), "Speaker 3", "{word}");
            let word = result.words.iter().find(|item| item.text == word).unwrap();
            assert_eq!(word.observations[0].track_id, "wide");
            assert_eq!(word.observations.len(), 2);
        }
        assert_eq!(result.speakers.len(), 3);
    }

    #[test]
    fn a_participant_off_microphone_is_one_speaker_on_every_recording() {
        let (manifest, mut spoken) = interview();
        // The Host says a word under the Guest's, and later the producer speaks.
        spoken.extend([
            (" right", 12, 0),
            (" could", 14, 2),
            (" you", 15, 2),
            (" both", 16, 2),
            (" lean", 17, 2),
        ]);
        spoken.sort_by_key(|&(_, slot, voice)| (slot, voice));
        let gains = [[0.2, 0.014, 0.1], [0.012, 0.2, 0.1], [0.005, 0.007, 0.3]];
        // The camera's diarizer keeps the producer apart. The Host's microphone picks
        // up the producer faintly, and the camera misses the first word.
        let labels = [Some("s0"), Some("s1"), Some("s2")];
        let hears = |recording: usize, slot: i64, voice: usize| match recording {
            0 => voice != 1,
            1 => voice == 1,
            _ => slot != 14 && !(slot == 12 && voice == 0),
        };
        let mut outcomes = Vec::from(slotted(&spoken, &gains, labels, hears));
        // The Host's recognizer sets that word down late, where the microphone is quiet
        // again. The microphone was loud just before it, so the word stays the Host's.
        let late = &mut outcomes[0].artifact.as_mut().unwrap().words[4];
        assert_eq!(late.text, " right");
        (late.start_sample, late.end_sample) = (late.start_sample + 4_000, late.end_sample + 4_000);
        // A second camera's diarizer hears the producer as the Guest.
        let mut wide = track("wide", 0, Role::MixedAudio);
        wide.source_id = "wide".to_owned();
        let mut tracks = manifest.tracks;
        tracks.push(wide);
        let manifest = super::tests::manifest(tracks);
        let mut second = outcomes[2].artifact.clone().unwrap();
        for word in &mut second.words {
            if word.speaker_id.as_deref() == Some("s2") {
                word.speaker_id = Some("s1".to_owned());
            }
        }
        outcomes.push(Outcome {
            coverage: Coverage::Recognized,
            artifact: Some(second),
            error: None,
        });
        let result = reconcile(&manifest, &outcomes).unwrap();
        let names: Vec<_> = result
            .speakers
            .iter()
            .map(|speaker| &speaker.name)
            .collect();
        assert_eq!(names, ["Host", "Guest", "Speaker 3"]);
        for word in [" could", " you", " both", " lean"] {
            assert_eq!(speaker_name(&result, word), "Speaker 3", "{word}");
        }
        assert_eq!(speaker_name(&result, " right"), "Host");
        // Only the Host's microphone caught the first word, far below the Host's level.
        let could = result.words.iter().find(|word| word.text == " could");
        assert_eq!(could.unwrap().observations.len(), 1);
        assert_eq!(could.unwrap().observations[0].track_id, "host-lav");
    }

    #[test]
    fn a_microphones_faint_copy_names_the_speaker_over_a_diarizers_label() {
        let (manifest, spoken) = interview();
        let gains = [[0.2, 0.014, 0.1], [0.012, 0.2, 0.1], [0.0, 0.0, 0.0]];
        // The Guest's own microphone misses one word. The camera has it under the
        // Host's label, and the Host's microphone holds it as the Guest's spill.
        let labels = [Some("s0"), Some("s1"), None];
        let hears = |recording: usize, slot: i64, _: usize| !(recording == 1 && slot == 8);
        let mut outcomes = slotted(&spoken, &gains, labels, hears);
        let pots = &mut outcomes[2].artifact.as_mut().unwrap().words[8];
        assert_eq!(pots.text, " pots");
        pots.speaker_id = Some("s0".to_owned());
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), spoken.len());
        assert_eq!(speaker_name(&result, " pots"), "Guest");
    }

    #[test]
    fn a_neighbours_word_heard_before_the_wearer_talks_is_the_neighbours() {
        let mut host = track("host-lav", 0, Role::IsolatedMic);
        host.speaker = Some("Host".to_owned());
        let mut guest = track("guest-lav", 0, Role::IsolatedMic);
        guest.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![host, guest, track("camera", 0, Role::MixedAudio)]);
        // The Host cuts in straight after the Guest's third word.
        let spoken = [
            ("Welcome", 0, 0),
            (" back", 1, 0),
            (" everyone", 2, 0),
            (" today", 3, 0),
            (" the", 4, 1),
            (" bread", 5, 1),
            (" takes", 6, 1),
            (" time", 7, 1),
            (" wow", 7, 0),
            (" nice", 8, 0),
        ];
        let gains = [[0.2, 0.014, 0.1], [0.012, 0.2, 0.1], [0.0, 0.0, 0.0]];
        let labels = [Some("s0"), Some("s1"), None];
        // Each lav hears its wearer. The Host's also picks up that third word, and its
        // recognizer runs the word on into the Host's own speech.
        let hears = |recording: usize, slot: i64, voice: usize| {
            recording == 2 || recording == voice || (recording == 0 && slot == 6)
        };
        let mut outcomes = slotted(&spoken, &gains, labels, hears);
        let spill = &mut outcomes[0].artifact.as_mut().unwrap().words[4];
        assert_eq!(spill.text, " takes");
        spill.end_sample += 6_400;
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), spoken.len());
        let takes = result.words.iter().find(|word| word.text == " takes");
        let guest = result
            .speakers
            .iter()
            .find(|speaker| speaker.name == "Guest");
        assert_eq!(
            takes.unwrap().speaker_id,
            guest.map(|guest| guest.id.clone())
        );
        assert_eq!(takes.unwrap().observations.len(), 3);
        drop(result);

        // A Host who says the word too is loud on the Host's microphone from its start.
        let mut both = spoken.to_vec();
        both.push((" takes", 6, 0));
        let mut outcomes = slotted(&both, &gains, labels, |recording, slot, voice| {
            recording == voice || (recording == 2 && !(slot == 6 && voice == 0))
        });
        outcomes[0].artifact.as_mut().unwrap().words.swap(4, 6);
        outcomes[0].artifact.as_mut().unwrap().words.swap(5, 6);
        let result = reconcile(&manifest, &outcomes).unwrap();
        let said: Vec<_> = result
            .words
            .iter()
            .filter(|word| word.text == " takes")
            .map(|word| word.speaker_id.clone())
            .collect();
        assert_eq!(said.len(), 2);
        assert_ne!(said[0], said[1]);
    }

    #[test]
    fn a_phrase_final_reading_follows_the_speakers_own_microphone() {
        let mut lav = track("lav", 0, Role::IsolatedMic);
        lav.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![lav, track("camera", 0, Role::MixedAudio)]);
        let outcomes = [
            outcome(
                &[
                    ("new", 0, 3_000, None),
                    (" silk", 3_200, 6_400, None),
                    (" from", 6_600, 9_000, None),
                    (" Kyoto.", 9_200, 14_000, None),
                    (" Thanks", 40_000, 44_000, None),
                ],
                Coverage::Recognized,
            ),
            outcome(
                &[
                    ("new", 40, 3_000, Some("s0")),
                    (" silk", 3_240, 6_400, Some("s0")),
                    (" from", 6_640, 9_000, Some("s0")),
                    (" Gyoto.", 9_240, 14_000, Some("s0")),
                    (" Thanks", 40_040, 44_000, Some("s0")),
                ],
                Coverage::Recognized,
            ),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["new", "silk", "from", "Kyoto.", "Thanks"]);
        let name = &result.words[3];
        assert_eq!(name.observations.len(), 2);
        assert_eq!(name.observations[1].text, " Gyoto.");
        assert_eq!(name.observations[1].evidence, "inferred");
        assert_eq!(
            name.observations[1].basis,
            Some("alternative_reading_between_matched_words")
        );

        // An unrelated word in the same place is someone else's and stays.
        let mut different = outcomes;
        different[1].artifact.as_mut().unwrap().words[3].text = " Really?".to_owned();
        let result = reconcile(&manifest, &different).unwrap();
        assert_eq!(result.words.len(), 6);
    }

    #[test]
    fn a_stretched_final_word_joins_the_recordings_it_overlaps() {
        let mut lav = track("lav", 0, Role::IsolatedMic);
        lav.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![
            lav,
            track("close", 0, Role::MixedAudio),
            track("wide", 0, Role::MixedAudio),
        ]);
        // The lav's recognizer runs the last word 0.8 s into the pause. That is within
        // reach of the wide camera's edges and beyond the close camera's.
        let ends = [25_000, 9_000, 14_000];
        let outcomes = [0, 1, 2].map(|recording| {
            let label = (recording > 0).then_some("s0");
            outcome(
                &[
                    ("stuck", 0, 2_800, label),
                    (" for", 3_000, 5_000, label),
                    (" decades.", 5_200, ends[recording], label),
                    (" Next", 60_000, 63_000, label),
                ],
                Coverage::Recognized,
            )
        });
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["stuck", "for", "decades.", "Next"]);
        assert!(result.words.iter().all(|word| word.observations.len() == 3));
        assert_eq!(result.words[2].observations[0].track_id, "lav");
        assert_eq!(result.speakers.len(), 1);
    }

    #[test]
    fn a_misplaced_word_joins_its_copy_beside_a_shared_word() {
        let mut lav = track("lav", 0, Role::IsolatedMic);
        lav.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![lav, track("camera", 0, Role::MixedAudio)]);
        // The lav's recognizer starts "because" in the pause before it, 0.47 s early.
        let outcomes = [
            outcome(
                &[
                    ("because", 9_760, 17_440, None),
                    (" that", 17_440, 22_240, None),
                    (" is", 22_240, 25_120, None),
                    (" where", 25_120, 27_040, None),
                ],
                Coverage::Recognized,
            ),
            outcome(
                &[
                    ("Because", 17_280, 22_080, Some("s0")),
                    (" that", 22_080, 24_000, Some("s0")),
                    (" is", 24_000, 25_920, Some("s0")),
                    (" where", 25_920, 27_840, Some("s0")),
                ],
                Coverage::Recognized,
            ),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["because", "that", "is", "where"]);
        let copy = &result.words[0].observations[1];
        assert_eq!(
            (copy.text, copy.evidence, copy.basis),
            (
                "Because",
                "recognized",
                Some("ordered_phrase_beside_matched_word")
            )
        );
        drop(result);

        // A recording with no speaker labels cannot show whose word it heard.
        let mut unlabeled = outcomes;
        for word in &mut unlabeled[1].artifact.as_mut().unwrap().words {
            word.speaker_id = None;
        }
        assert_eq!(reconcile(&manifest, &unlabeled).unwrap().words.len(), 5);
    }

    #[test]
    fn words_an_aligner_could_not_place_join_their_copies_in_order() {
        let manifest = manifest(vec![
            track("wide", 0, Role::MixedAudio),
            track("close", 0, Role::MixedAudio),
        ]);
        // The close camera's aligner leaves three words of the phrase with no length.
        let outcomes = [
            outcome(
                &[
                    ("Sorry,", 1_000, 4_000, Some("s0")),
                    (" one", 4_000, 7_000, Some("s0")),
                    (" moment,", 7_000, 12_000, Some("s0")),
                    (" please.", 12_000, 16_000, Some("s0")),
                    (" Thanks", 40_000, 44_000, Some("s0")),
                ],
                Coverage::Recognized,
            ),
            outcome(
                &[
                    ("Sorry.", 11_000, 15_000, Some("s0")),
                    (" One", 15_000, 15_001, Some("s0")),
                    (" moment,", 15_000, 15_001, Some("s0")),
                    (" please.", 15_000, 15_001, Some("s0")),
                    (" Thanks", 40_000, 44_000, Some("s0")),
                ],
                Coverage::Recognized,
            ),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["Sorry,", "one", "moment,", "please.", "Thanks"]);
        assert!(result.words.iter().all(|word| word.observations.len() == 2));
        drop(result);

        // Another speaker's echo of the phrase stays apart from it.
        let mut echo = outcomes;
        for word in &mut echo[1].artifact.as_mut().unwrap().words[..2] {
            word.speaker_id = Some("s1".to_owned());
        }
        assert_eq!(reconcile(&manifest, &echo).unwrap().words.len(), 7);

        // So do words that no recording gives a speaker.
        for outcome in &mut echo {
            for word in &mut outcome.artifact.as_mut().unwrap().words {
                word.speaker_id = None;
            }
        }
        assert_eq!(reconcile(&manifest, &echo).unwrap().words.len(), 7);
    }

    #[test]
    fn a_compound_and_its_two_parts_are_one_reading() {
        let mut lav = track("lav", 0, Role::IsolatedMic);
        lav.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![
            lav,
            track("close", 0, Role::MixedAudio),
            track("wide", 0, Role::MixedAudio),
        ]);
        let whole = [
            ("brought", 0, 2_800, None),
            (" a", 3_000, 4_000, None),
            (" rangefinder", 4_200, 12_000, None),
            (" that", 12_200, 14_000, None),
            (" was", 14_200, 16_000, None),
        ];
        let parts = [
            ("brought", 0, 2_800, Some("s0")),
            (" a", 3_000, 4_000, Some("s0")),
            (" range", 4_200, 7_000, Some("s0")),
            (" finder", 7_200, 12_000, Some("s0")),
            (" that", 12_200, 14_000, Some("s0")),
            (" was", 14_200, 16_000, Some("s0")),
        ];
        // One camera against the speaker's own microphone.
        let outcomes = [
            outcome(&whole, Coverage::Recognized),
            outcome(&parts, Coverage::Recognized),
            outcome(&[], Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["brought", "a", "rangefinder", "that", "was"]);
        let readings: Vec<_> = result.words[2]
            .observations
            .iter()
            .map(|observation| (observation.track_id, observation.text, observation.evidence))
            .collect();
        assert_eq!(
            readings,
            [
                ("lav", " rangefinder", "recognized"),
                ("close", " range", "inferred"),
                ("close", " finder", "inferred"),
            ]
        );
        drop(result);

        // Two cameras outvote the microphone.
        let outcomes = [
            outcome(&whole, Coverage::Recognized),
            outcome(&parts, Coverage::Recognized),
            outcome(&parts, Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["brought", "a", "range", "finder", "that", "was"]);
        assert!(result.words[2].observations.iter().any(|observation| {
            observation.text == " rangefinder" && observation.evidence == "inferred"
        }));
        drop(result);

        // Another speaker's words in the same place stay.
        let mut other = outcomes;
        for word in &mut other[1].artifact.as_mut().unwrap().words[2..4] {
            word.speaker_id = Some("s1".to_owned());
        }
        other[2] = outcome(&[], Coverage::Recognized);
        assert_eq!(reconcile(&manifest, &other).unwrap().words.len(), 7);
    }

    #[test]
    fn three_readings_of_one_word_keep_only_the_unrelated_one_apart() {
        let mut lav = track("lav", 0, Role::IsolatedMic);
        lav.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![
            lav,
            track("close", 0, Role::MixedAudio),
            track("wide", 0, Role::MixedAudio),
        ]);
        let outcomes = [" Salvador", " salvado", " sourdough"].map(|reading| {
            let label = (reading != " Salvador").then_some("s0");
            outcome(
                &[
                    ("The", 0, 1_000, label),
                    (reading, 1_200, 8_000, label),
                    (" takes", 8_200, 11_000, label),
                    (" time", 11_200, 14_000, label),
                ],
                Coverage::Recognized,
            )
        });
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["The", "Salvador", "sourdough", "takes", "time"]);
        let near = &result.words[1].observations[1];
        assert_eq!((near.text, near.evidence), (" salvado", "inferred"));
        assert_eq!(result.words[2].observations.len(), 1);
    }

    #[test]
    fn a_number_in_digits_and_its_spelling_are_one_reading() {
        let mut lav = track("lav", 0, Role::IsolatedMic);
        lav.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![
            lav,
            track("close", 0, Role::MixedAudio),
            track("wide", 0, Role::MixedAudio),
        ]);
        let spelled = [
            ("Roughly", 0, 2_800, None),
            (" two", 3_000, 4_000, None),
            (" hundred", 4_200, 7_000, None),
            (" loaves", 7_200, 9_000, None),
            (" at", 9_200, 10_000, None),
            (" four", 10_200, 12_000, None),
            (" or", 12_200, 13_000, None),
            (" three", 13_200, 14_000, None),
            (" hundred", 14_200, 16_000, None),
            (" and", 16_200, 17_000, None),
            (" fifty.", 17_200, 19_000, None),
        ];
        let digits = [
            ("Roughly", 0, 2_800, Some("s0")),
            (" 200", 3_000, 7_000, Some("s0")),
            (" loaves", 7_200, 9_000, Some("s0")),
            (" at", 9_200, 10_000, Some("s0")),
            (" 4:00", 10_200, 12_000, Some("s0")),
            (" or", 12_200, 13_000, Some("s0")),
            (" 350.", 13_200, 19_000, Some("s0")),
        ];
        let outcomes = [
            outcome(&spelled, Coverage::Recognized),
            outcome(&digits, Coverage::Recognized),
            outcome(&[], Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        let expected: Vec<_> = spelled.iter().map(|word| word.0.trim()).collect();
        assert_eq!(texts, expected);
        let copy = |word: usize| {
            let copy = &result.words[word].observations[1];
            (copy.text, copy.evidence, copy.basis)
        };
        let number = Some("equivalent_spoken_number");
        assert_eq!(copy(1), (" 200", "inferred", number));
        assert_eq!(copy(5), (" 4:00", "inferred", number));
        assert_eq!(copy(7), (" 350.", "inferred", number));
        drop(result);

        // Two cameras outvote the microphone on the spelling. The hour is one word
        // either way and keeps the microphone's.
        let outcomes = [
            outcome(&spelled, Coverage::Recognized),
            outcome(&digits, Coverage::Recognized),
            outcome(&digits, Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(
            texts,
            ["Roughly", "200", "loaves", "at", "four", "or", "350."]
        );
        drop(result);

        // A different number in the same place is a different word.
        let mut other = digits;
        other[1].0 = " 300";
        let outcomes = [
            outcome(&spelled, Coverage::Recognized),
            outcome(&other, Coverage::Recognized),
            outcome(&[], Coverage::Recognized),
        ];
        assert_eq!(reconcile(&manifest, &outcomes).unwrap().words.len(), 12);
    }

    /// A microphone whose recognizer wrote each number in words and a camera whose
    /// recognizer wrote it in digits. In `phrase`, "250=two+hundred+fifty" is the camera's
    /// word and the microphone's words for it. Returns the reconciled words, and the
    /// camera's words that were joined to another spelling with the reason given.
    fn said_and_written(language: &str, phrase: &str) -> (Vec<String>, Vec<(String, String)>) {
        fn spans<'a>(
            words: &'a [(String, i64, i64)],
            speaker: Option<&'a str>,
        ) -> Vec<(&'a str, i64, i64, Option<&'a str>)> {
            (words.iter())
                .map(|(text, start, end)| (text.as_str(), *start, *end, speaker))
                .collect()
        }
        let mut lav = track("lav", 0, Role::IsolatedMic);
        lav.speaker = Some("Guest".to_owned());
        let mut manifest = manifest(vec![
            lav,
            track("close", 0, Role::MixedAudio),
            track("wide", 0, Role::MixedAudio),
        ]);
        manifest.language = Some(language.to_owned());
        let (mut said, mut written) = (Vec::new(), Vec::new());
        let mut clock = 0;
        for item in phrase.split(' ') {
            let (figure, words) = item.split_once('=').unwrap_or((item, item));
            let start = clock;
            for word in words.split('+') {
                said.push((format!(" {word}"), clock, clock + 1_800));
                clock += 2_000;
            }
            written.push((format!(" {figure}"), start, clock - 200));
        }
        let outcomes = [
            outcome(&spans(&said, None), Coverage::Recognized),
            outcome(&spans(&written, Some("s0")), Coverage::Recognized),
            outcome(&[], Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts = (result.words.iter())
            .map(|word| word.text.trim().to_owned())
            .collect();
        let joined = (result.words.iter())
            .flat_map(|word| &word.observations)
            .filter(|copy| copy.track_id == "close")
            .filter_map(|copy| Some((copy.text.trim().to_owned(), copy.basis?.to_owned())))
            .filter(|(_, basis)| basis.starts_with("equivalent_spoken"))
            .collect();
        (texts, joined)
    }

    #[test]
    fn a_figure_meets_the_words_that_say_it_in_the_language_of_the_session() {
        let number = "equivalent_spoken_number";
        let ordinal = "equivalent_spoken_ordinal";
        for (language, phrase, expected) in [
            (
                "en",
                "It cost $200=two+hundred+dollars or 50%=fifty+percent more at \
                 4:05=four+oh+five in 1905=nineteen+oh+five on the 21st=twenty-first for \
                 12,500=twelve+thousand+five+hundred people",
                vec![
                    ("$200", number),
                    ("50%", number),
                    ("4:05", number),
                    ("1905", number),
                    ("21st", ordinal),
                    ("12,500", number),
                ],
            ),
            (
                "de",
                "Wir haben 250=zweihundertfünfzig Stunden am 3.=dritten Mai und \
                 1.250=ein+tausend+zwei+hundert+fünfzig Euro gebraucht",
                vec![("250", number), ("3.", number), ("1.250", number)],
            ),
            (
                "es",
                "Tenemos 1=una casa con 1.250=mil+doscientas+cincuenta personas en la \
                 3.ª=tercera planta",
                vec![("1", number), ("1.250", number), ("3.ª", ordinal)],
            ),
            (
                "fr",
                "Il a 80=quatre-vingts ans le 1er=premier jour avec 200€=deux+cents+euros \
                 en poche",
                vec![("80", number), ("1er", ordinal), ("200€", number)],
            ),
            (
                "ru",
                "Около 250=двухсот+пятидесяти человек и 21=двадцать+один день",
                vec![("250", number), ("21", number)],
            ),
        ] {
            let (texts, joined) = said_and_written(language, phrase);
            let said: Vec<_> = phrase
                .split(' ')
                .flat_map(|item| {
                    item.split_once('=')
                        .map_or(item, |(_, words)| words)
                        .split('+')
                })
                .collect();
            assert_eq!(texts, said, "{language}");
            let joined: Vec<_> = (joined.iter())
                .map(|(text, basis)| (text.as_str(), basis.as_str()))
                .collect();
            assert_eq!(joined, expected, "{language}");
        }
    }

    #[test]
    fn a_figure_is_not_joined_to_another_number_or_read_in_another_language() {
        // Another number in the same place is another word.
        let (texts, joined) = said_and_written("de", "Wir haben 250=dreihundert Stunden gebraucht");
        assert_eq!(texts.len(), 6);
        assert!(joined.is_empty());
        // German words do not say a figure in a session declared as English.
        let (texts, _) =
            said_and_written("en", "Wir haben 250=zwei+hundert+fünfzig Stunden gebraucht");
        assert_eq!(texts.len(), 8);
        let (texts, _) =
            said_and_written("de", "Wir haben 250=zwei+hundert+fünfzig Stunden gebraucht");
        assert_eq!(texts.len(), 7);
    }

    #[test]
    fn one_recording_carries_a_turn_that_no_microphone_heard() {
        let manifest = manifest(vec![
            track("close", 0, Role::MixedAudio),
            track("wide", 0, Role::MixedAudio),
        ]);
        // The close camera misses the first word and starts two of the others earlier.
        let outcomes = [
            outcome(
                &[
                    ("One", 3_500, 6_800, Some("s0")),
                    (" moment,", 7_200, 12_000, Some("s0")),
                    (" please.", 11_800, 16_000, Some("s0")),
                ],
                Coverage::Recognized,
            ),
            outcome(
                &[
                    ("Sorry,", 1_000, 4_000, Some("s0")),
                    (" one", 4_000, 7_000, Some("s0")),
                    (" moment,", 7_000, 12_000, Some("s0")),
                    (" please.", 12_000, 16_000, Some("s0")),
                ],
                Coverage::Recognized,
            ),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["Sorry,", "one", "moment,", "please."]);
        for word in &result.words {
            assert_eq!(word.observations[0].track_id, "wide");
            assert_eq!(word.utterance_id, "u1");
        }
        assert_eq!(result.words[1].start_sample, 4_000);
    }

    #[test]
    fn an_utterance_keeps_its_id_while_another_speaker_talks_over_it() {
        let mut host = track("host-lav", 0, Role::IsolatedMic);
        host.speaker = Some("Host".to_owned());
        let mut guest = track("guest-lav", 0, Role::IsolatedMic);
        guest.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![host, guest]);
        let outcomes = [
            outcome(
                &[("Welcome", 0, 3_000, None), (" back", 6_000, 9_000, None)],
                Coverage::Recognized,
            ),
            outcome(&[("Yes", 3_200, 5_800, None)], Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        let utterances: Vec<_> = result
            .words
            .iter()
            .map(|word| (word.text.trim(), word.utterance_id.as_str()))
            .collect();
        assert_eq!(
            utterances,
            [("Welcome", "u1"), ("Yes", "u2"), ("back", "u1")]
        );
    }

    #[test]
    fn a_spill_misreading_joins_the_word_the_source_microphone_recognized() {
        let mut near = track("near", -3200, Role::IsolatedMic);
        near.speaker = Some("Host".to_owned());
        let mut far = track("far", -3200, Role::IsolatedMic);
        far.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![near, far]);
        let words = [
            ("One", 8_000, 10_400, None),
            (" two", 17_600, 20_800, None),
            (" three", 27_200, 30_400, None),
            (" four", 36_800, 40_000, None),
        ];
        // The Guest's microphone reads the Host's faint third word as something else.
        let mut misread = words;
        misread[2].0 = " sea";
        type Words<'a> = [(&'a str, i64, i64, Option<&'a str>)];
        let recorded = |near: &Words<'_>, far: &Words<'_>| {
            let mut outcomes = [
                outcome(near, Coverage::Recognized),
                outcome(far, Coverage::Recognized),
            ];
            for (index, outcome) in outcomes.iter_mut().enumerate() {
                let artifact = outcome.artifact.as_mut().unwrap();
                artifact.duration_samples = 64_000;
                artifact.activity = (0..200)
                    .map(|frame| {
                        let source = if (15..145).contains(&frame) {
                            0.15 + ((frame * 37) % 31) as f32 * 0.013
                        } else {
                            0.002
                        };
                        Activity {
                            start_sample: frame * 320,
                            end_sample: (frame + 1) * 320,
                            level: if index == 0 {
                                source
                            } else {
                                source * 0.08 + 0.001
                            },
                        }
                    })
                    .collect();
            }
            outcomes
        };
        // It can also hear the two halves of a compound, with or without the next word.
        let compound = [
            ("One", 8_000, 10_400, None),
            (" two", 17_600, 20_800, None),
            (" sunrise", 27_200, 30_400, None),
            (" four", 36_800, 40_000, None),
            (" five", 41_600, 44_000, None),
        ];
        let halves = [
            ("One", 8_000, 10_400, None),
            (" two", 17_600, 20_800, None),
            (" sun", 27_200, 28_600, None),
            (" rise", 29_000, 30_400, None),
            (" five", 41_600, 44_000, None),
        ];
        let outcomes = recorded(&compound, &halves);
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["One", "two", "sunrise", "four", "five"]);
        let readings: Vec<_> = result.words[2].observations[1..]
            .iter()
            .map(|reading| (reading.track_id, reading.text, reading.basis))
            .collect();
        let basis = Some("alternative_reading_of_spill");
        assert_eq!(readings, [("far", " sun", basis), ("far", " rise", basis)]);
        drop(result);

        // A short word the Host's microphone missed is no half of the longer word its
        // recognizer stretched across it.
        let mut stretched = compound;
        stretched[2].0 = " then";
        let recovered = [
            ("One", 8_000, 10_400, None),
            (" two", 17_600, 20_800, None),
            (" the", 29_000, 30_400, None),
            (" five", 41_600, 44_000, None),
        ];
        let outcomes = recorded(&stretched, &recovered);
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["One", "two", "then", "the", "four", "five"]);
        drop(result);

        let outcomes = recorded(&words, &misread);
        let result = reconcile(&manifest, &outcomes).unwrap();
        let texts: Vec<_> = result.words.iter().map(|word| word.text.trim()).collect();
        assert_eq!(texts, ["One", "two", "three", "four"]);
        let reading = &result.words[2].observations[1];
        assert_eq!(
            (
                reading.track_id,
                reading.text,
                reading.evidence,
                reading.basis
            ),
            (
                "far",
                " sea",
                "inferred",
                Some("alternative_reading_of_spill")
            )
        );
        assert_eq!(result.speakers.len(), 1);
    }

    #[test]
    fn a_named_speaker_keeps_its_id_wherever_an_unnamed_voice_falls() {
        let mut lav = track("lav", 0, Role::IsolatedMic);
        lav.speaker = Some("Guest".to_owned());
        let manifest = manifest(vec![track("camera", 0, Role::MixedAudio), lav]);
        // Someone off microphone speaks before the Guest, or after.
        for stranger in [0, 40_000] {
            let outcomes = [
                outcome(
                    &[
                        ("Thanks", 20_000, 23_000, Some("s1")),
                        (" for", 23_200, 25_000, Some("s1")),
                        (" coming", 25_200, 28_000, Some("s1")),
                        (" Hello", stranger, stranger + 3_000, Some("s0")),
                        (" there", stranger + 3_200, stranger + 6_000, Some("s0")),
                    ],
                    Coverage::Recognized,
                ),
                outcome(
                    &[
                        ("Thanks", 20_000, 23_000, None),
                        (" for", 23_200, 25_000, None),
                        (" coming", 25_200, 28_000, None),
                    ],
                    Coverage::Recognized,
                ),
            ];
            let result = reconcile(&manifest, &outcomes).unwrap();
            let speakers: Vec<_> = result
                .speakers
                .iter()
                .map(|speaker| (speaker.id.as_str(), speaker.name.as_str()))
                .collect();
            assert_eq!(speakers, [("s1", "Guest"), ("s2", "Speaker 2")]);
            let hello = result.words.iter().find(|word| word.text == " Hello");
            assert_eq!(hello.unwrap().speaker_id.as_deref(), Some("s2"));
        }
    }

    #[test]
    fn two_quiet_copies_can_recover_a_word_the_source_headset_missed() {
        let tracks = ["Host", "Guest", "Producer"].map(|name| {
            let mut mic = track(name, -3200, Role::IsolatedMic);
            mic.speaker = Some(name.to_owned());
            mic
        });
        let manifest = manifest(tracks.into());
        let anchors = [
            ("one", 8000, 10400, None),
            ("two", 17600, 20800, None),
            ("four", 36800, 40000, None),
        ];
        let copy = [("three", 27200, 30400, None)];
        let mut outcomes = [
            outcome(&anchors, Coverage::Recognized),
            outcome(&copy, Coverage::Recognized),
            outcome(&copy, Coverage::Recognized),
        ];
        for (mic, outcome) in outcomes.iter_mut().enumerate() {
            let artifact = outcome.artifact.as_mut().unwrap();
            artifact.duration_samples = 64000;
            artifact.activity = (0..200)
                .map(|frame| {
                    let source = if (15..145).contains(&frame) {
                        0.15 + ((frame * 37) % 31) as f32 * 0.013
                    } else {
                        0.002
                    };
                    let level = match mic {
                        0 => source,
                        1 => source * 0.06 + 0.002,
                        _ => source * 0.04 + 0.002,
                    };
                    Activity {
                        start_sample: frame * 320,
                        end_sample: (frame + 1) * 320,
                        level,
                    }
                })
                .collect();
        }
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 4);
        let recovered = &result.words[2];
        assert_eq!(recovered.text, "three");
        assert_eq!(recovered.speaker_id, result.words[0].speaker_id);
        assert_eq!(result.speakers[0].name, "Host");
        assert_eq!(recovered.observations.len(), 2);
        assert!(
            recovered
                .observations
                .iter()
                .all(|observation| observation.evidence == "recognized"
                    && observation.text == "three"
                    && observation.track_id != "Host")
        );
        assert_eq!(result.tracks[0].words.len(), 3);
    }

    #[test]
    fn a_mixed_diarizer_switch_inside_a_confirmed_phrase_keeps_the_headset_owner() {
        let mut mic = track("headset", 0, Role::IsolatedMic);
        mic.speaker = Some("Host".to_owned());
        let manifest = manifest(vec![mic, track("room", 0, Role::MixedAudio)]);
        let outcomes = [
            outcome(
                &[
                    ("one", 100, 1000, None),
                    ("two", 1200, 2200, None),
                    ("three", 2400, 3400, None),
                ],
                Coverage::Recognized,
            ),
            outcome(
                &[
                    ("one", 100, 1000, Some("s0")),
                    ("two", 1200, 2200, Some("s0")),
                    ("three", 2400, 3400, Some("s1")),
                ],
                Coverage::Recognized,
            ),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 3);
        assert!(
            result
                .words
                .iter()
                .all(|word| word.speaker_id == result.words[0].speaker_id
                    && word.observations.len() == 2)
        );
        let final_word = &result.words[2];
        assert_ne!(
            final_word.observations[0].speaker_id,
            final_word.observations[1].speaker_id
        );
    }

    #[test]
    fn unhinted_isolated_mics_preserve_simultaneous_matching_phrases() {
        let manifest = manifest(vec![
            track("one", 0, Role::IsolatedMic),
            track("two", 0, Role::IsolatedMic),
        ]);
        let outcomes = [
            outcome(
                &[("Thank", 0, 1200, None), (" you", 1400, 2200, None)],
                Coverage::Recognized,
            ),
            outcome(
                &[("Thank", 0, 1200, None), (" you", 1400, 2200, None)],
                Coverage::Recognized,
            ),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 4);
        assert_eq!(result.speakers.len(), 2);
        assert!(result.words.iter().all(|word| word.observations.len() == 1));
    }

    #[test]
    fn lone_mixed_word_with_a_competing_speaker_has_an_inferred_link() {
        let manifest = manifest(vec![
            track("lav", 0, Role::IsolatedMic),
            track("room", 0, Role::MixedAudio),
        ]);
        let outcomes = [
            outcome(&[("Yes", 0, 1200, None)], Coverage::Recognized),
            outcome(&[("Yes!", 0, 1200, Some("s0"))], Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 2);
        for word in &result.words {
            assert_eq!(word.observations.len(), 2);
            assert_eq!(word.observations[0].evidence, "recognized");
            assert_eq!(word.observations[1].evidence, "inferred");
            assert_eq!(
                word.observations[1].basis,
                Some("lexical_timing_with_conflicting_speaker")
            );
        }
    }

    #[test]
    fn phrase_context_links_adjacent_estimates_without_repeating_the_word() {
        let manifest = manifest(vec![
            track("left", 0, Role::Unknown),
            track("right", 0, Role::Unknown),
        ]);
        let outcomes = [
            outcome(
                &[
                    ("Some", 0, 2000, None),
                    (" words", 2200, 2450, None),
                    (" here", 4900, 7500, None),
                ],
                Coverage::Recognized,
            ),
            outcome(
                &[
                    ("Some", 0, 2000, None),
                    (" words", 4000, 4300, None),
                    (" here", 4900, 7500, None),
                ],
                Coverage::Recognized,
            ),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 3);
        assert_eq!(result.words[1].text, " words");
        assert_eq!(result.words[1].observations.len(), 2);
        assert!(
            result.words[1]
                .observations
                .iter()
                .all(|item| item.evidence == "recognized")
        );
    }

    #[test]
    fn alternate_spelling_requires_matching_flanks_lane_and_independent_word_votes() {
        let manifest = manifest(
            ["one", "two", "three"]
                .map(|id| track(id, 0, Role::MixedAudio))
                .into(),
        );
        let words = [
            ("very", 100, 1000, Some("s0")),
            ("quick", 1000, 3000, Some("s0")),
            ("fox", 3000, 4000, Some("s0")),
        ];
        let mut dissent = words;
        dissent[1].0 = "quack";
        let mut outcomes = [
            outcome(&words, Coverage::Recognized),
            outcome(&words, Coverage::Recognized),
            outcome(&dissent, Coverage::Recognized),
        ];
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 3);
        let word = &result.words[1];
        assert_eq!(word.text, "quick");
        let alternate = word
            .observations
            .iter()
            .find(|observation| observation.track_id == "three")
            .unwrap();
        assert_eq!(
            (
                alternate.text,
                alternate.word_id,
                alternate.start_sample,
                alternate.end_sample
            ),
            ("quack", "w2", 1000, 3000)
        );
        assert_eq!(
            (alternate.evidence, alternate.basis),
            (
                "inferred",
                Some("alternative_reading_between_matched_words")
            )
        );
        assert_eq!(result.tracks[2].words[1].text, "quack");
        drop(result);
        // A distinct speaker interrupting with the same near spelling keeps a word.
        outcomes[2].artifact.as_mut().unwrap().words[1].speaker_id = Some("s1".to_owned());
        assert_eq!(reconcile(&manifest, &outcomes).unwrap().words.len(), 4);
        // Repetition on a single track has two positions between the shared anchors.
        let repeated = [
            dissent[0],
            ("quack", 1000, 1800, Some("s0")),
            ("quack", 1800, 3000, Some("s0")),
            dissent[2],
        ];
        outcomes[2] = outcome(&repeated, Coverage::Recognized);
        assert_eq!(reconcile(&manifest, &outcomes).unwrap().words.len(), 5);
    }

    #[test]
    fn shifted_interruption_needs_shared_audio_and_retains_the_actual_ordinal_readings() {
        let manifest = manifest(vec![
            track("stable", 0, Role::MixedAudio),
            track("late", 0, Role::MixedAudio),
        ]);
        let mut outcomes = [
            outcome(
                &[
                    ("Try", 26560, 31360, Some("s0")),
                    ("the", 31360, 36160, Some("s0")),
                    ("3rd.", 36160, 42880, Some("s0")),
                    ("time", 43600, 47000, Some("s0")),
                    ("again", 48000, 52000, Some("s0")),
                ],
                Coverage::Recognized,
            ),
            outcome(
                &[
                    ("Try", 36160, 39040, Some("s0")),
                    ("the", 39040, 40960, Some("s0")),
                    ("third.", 40960, 45760, Some("s1")),
                    ("time", 45800, 47000, Some("s1")),
                    ("again", 48000, 52000, Some("s1")),
                ],
                Coverage::Recognized,
            ),
        ];
        for mic in &mut outcomes {
            let artifact = mic.artifact.as_mut().unwrap();
            artifact.duration_samples = 64000;
            artifact.activity = (0..200)
                .map(|frame| Activity {
                    start_sample: frame * 320,
                    end_sample: (frame + 1) * 320,
                    level: 0.1 + ((frame * 37) % 31) as f32 * 0.01,
                })
                .collect();
        }
        let result = reconcile(&manifest, &outcomes).unwrap();
        assert_eq!(result.words.len(), 5);
        for word in &result.words[..2] {
            assert_eq!(word.observations[0].track_id, "stable");
            assert_eq!(
                word.observations[1].basis,
                Some("ordered_phrase_and_correlated_audio")
            );
            assert_eq!(word.speaker_id, result.words[2].speaker_id);
        }
        let ordinal = &result.words[2];
        assert_eq!(ordinal.text, "third.");
        assert!(ordinal.observations.iter().any(|o| o.evidence == "inferred"
            && o.text == "3rd."
            && o.basis == Some("equivalent_spoken_ordinal")));
        drop(result);
        for frame in &mut outcomes[1].artifact.as_mut().unwrap().activity {
            frame.level = 0.1;
        }
        assert_eq!(reconcile(&manifest, &outcomes).unwrap().words.len(), 7);
    }

    #[test]
    fn signed_clock_rounds_exactly_and_reports_overflow() {
        for (numerator, denominator, expected) in
            [(1, 32_000, 1), (-1, 32_000, -1), (-2, 48_000, -1)]
        {
            assert_eq!(
                Time {
                    numerator,
                    denominator
                }
                .samples()
                .unwrap(),
                expected
            );
        }
        assert!(
            Time {
                numerator: i64::MIN,
                denominator: 1
            }
            .samples()
            .is_err()
        );
        assert!(
            Time {
                numerator: 0,
                denominator: 0
            }
            .samples()
            .is_err()
        );
    }
}
