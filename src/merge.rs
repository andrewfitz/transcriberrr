//! Merge transcripts of one session's per-speaker ("iso") tracks into a single transcript.
//!
//! Each track already holds one speaker, so no diarization is involved: every word of a track
//! gets that track's speaker, words from all tracks are ordered by start time, and segments
//! are rebuilt exactly as transcription builds them, giving the shape of a diarized transcript.
use crate::transcript::{
    DiarizationMetadata, MergedTrack, TranscriptDocument, TranscriptMetadata, Word, build_segments,
};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

/// Tracks of one session share a timeline, so their lengths should agree to within a few
/// seconds; a larger spread suggests a track from another recording.
const DURATION_TOLERANCE_SECONDS: f32 = 5.0;
/// File name words that describe the recording rather than the speaker.
const NOISE_WORDS: &[&str] = &[
    "audio",
    "iso",
    "isolated",
    "mic",
    "track",
    "trk",
    "raw",
    "rec",
    "recording",
    "take",
    "channel",
    "ch",
    "mono",
    "stereo",
    "final",
    "master",
    "riverside",
    "zencastr",
    "squadcast",
    "zoom",
    "craig",
];

/// What a track's file name says about its speaker.
#[derive(Debug, PartialEq)]
pub(crate) enum TrackName {
    /// A Zoom per-participant recording ("Record a separate audio file for each participant").
    /// Zoom numbers a meeting's files in the order they start, across all participants:
    /// `part` is that number, so a participant who reconnects gets a second file with a later
    /// one. `segment` counts the times the recording was stopped and started again, and
    /// `meeting` (the last 9 digits) is shared by all of them. Every file of a segment covers
    /// the whole segment, silent where its participant was absent.
    Zoom {
        speaker: String,
        part: u32,
        segment: u32,
        meeting: String,
    },
    /// Zoom's recording of everyone, which belongs to no single speaker.
    ZoomMixed {
        segment: u32,
        meeting: Option<String>,
    },
    /// Any other file name, with a best-effort speaker name.
    Other(Option<String>),
}

/// Parse a track's file stem. Zoom formats (from wisdom's track discovery):
/// - `audioAndrewDaniel21067067067`: CamelCase name, then the file number (1–2 digits), the
///   segment digit and the 9-digit meeting ID. Older files lack the segment digit.
/// - `audio_jordan_k.41864297531`: the same with a lowercase, underscored name.
/// - `audio_only_13572468_Dana_Reyes`: 2018–2019 format, meeting ID first.
/// - `audio1067067067` (segment digit and meeting ID), `audio_only`: the mixed recording.
pub(crate) fn parse_track_name(stem: &str) -> TrackName {
    // `audio_only01` and the like: renamed copies.
    if stem
        .strip_prefix("audio_only")
        .is_some_and(|rest| rest.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return TrackName::ZoomMixed {
            segment: 0,
            meeting: None,
        };
    }
    if let Some(rest) = stem.strip_prefix("audio_only_")
        && let Some((meeting, name)) = rest.split_once('_')
        && !meeting.is_empty()
        && meeting.bytes().all(|byte| byte.is_ascii_digit())
        && !name.is_empty()
    {
        return TrackName::Zoom {
            speaker: name.replace('_', " "),
            part: 0,
            segment: 0,
            meeting: meeting.to_owned(),
        };
    }
    if let Some(rest) = stem.strip_prefix("audio") {
        let name = rest.trim_end_matches(|c: char| c.is_ascii_digit());
        let digits = &rest[name.len()..];
        if name.is_empty() {
            let segment = match digits.len() {
                9 => Some(0),
                10 => Some(u32::from(digits.as_bytes()[0] - b'0')),
                _ => None,
            };
            if let Some(segment) = segment {
                return TrackName::ZoomMixed {
                    segment,
                    meeting: Some(digits[digits.len() - 9..].to_owned()),
                };
            }
        }
        // The part number takes what the segment digit and meeting ID leave.
        let (part, segment) = match digits.len() {
            10 => (&digits[..1], "0"),
            11 => (&digits[..1], &digits[1..2]),
            12 => (&digits[..2], &digits[2..3]),
            _ => ("", ""),
        };
        let speaker = match name.strip_prefix('_') {
            Some(name) if is_name(name, "_.'’-") => {
                let name = name.replace('_', " ");
                let name = name.trim_end_matches('.').trim();
                if name.chars().any(char::is_uppercase) {
                    split_camel_case(name)
                } else {
                    title_case(name)
                }
            }
            None if is_name(name, "'’-") => split_camel_case(name),
            _ => String::new(),
        };
        if let (Ok(part), Ok(segment)) = (part.parse(), segment.parse())
            && !speaker.is_empty()
        {
            return TrackName::Zoom {
                speaker,
                part,
                segment,
                meeting: digits[digits.len() - 9..].to_owned(),
            };
        }
    }
    TrackName::Other(generic_name(stem))
}

/// Whether a file name's speaker part reads as a name: a letter, then letters, combining
/// accents (names copied from HFS+ or a NAS are decomposed: "e" + U+0301) or `punctuation`.
fn is_name(name: &str, punctuation: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(char::is_alphabetic)
        && chars.all(|c| c.is_alphabetic() || is_mark(c) || punctuation.contains(c))
}

fn is_mark(c: char) -> bool {
    use unicode_general_category::{GeneralCategory, get_general_category};
    matches!(
        get_general_category(c),
        GeneralCategory::NonspacingMark | GeneralCategory::SpacingMark
    )
}

/// Parse a file's name as `parse_track_name` does its stem.
pub(crate) fn parse_path(path: &Path) -> TrackName {
    parse_track_name(
        &path
            .file_stem()
            .map(|stem| stem.to_string_lossy())
            .unwrap_or_default(),
    )
}

/// Parse the file names of tracks that may belong together, as `parse_path` does, reading
/// their 12-digit tails in context (see `resolve_name_digits`).
pub(crate) fn parse_paths<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Vec<TrackName> {
    let mut names = paths.into_iter().map(parse_path).collect::<Vec<_>>();
    resolve_name_digits(&mut names);
    names
}

/// A 12-digit tail is either a 2-digit file number or a name ending in a digit ("Guest2",
/// file 1). Zoom gives each file of a meeting its own number from 1, so the tail is a file
/// number when reading it as a name digit would leave file 0 or a number another file has,
/// or when the speaker has files numbered below 10. Otherwise its first digit belongs to the
/// name, which keeps "Guest1" and "Guest2" apart.
pub(crate) fn resolve_name_digits(names: &mut [TrackName]) {
    let mut speakers = std::collections::HashSet::new();
    let mut numbers = std::collections::HashSet::new();
    for name in names.iter() {
        if let TrackName::Zoom {
            speaker,
            part,
            meeting,
            ..
        } = name
            && *part < 10
        {
            speakers.insert((speaker.clone(), meeting.clone()));
            numbers.insert((*part, meeting.clone()));
        }
    }
    for name in names {
        if let TrackName::Zoom {
            speaker,
            part,
            meeting,
            ..
        } = name
            && *part >= 10
            && *part % 10 != 0
            && !numbers.contains(&(*part % 10, meeting.clone()))
            && !speakers.contains(&(speaker.clone(), meeting.clone()))
        {
            speaker.extend(char::from_digit(*part / 10, 10));
            *part %= 10;
        }
    }
}

/// Whether `id`, as typed by a user, names `meeting`: the meeting ID itself, or the 10-digit
/// ID in the file names (the segment digit, then the meeting ID).
pub(crate) fn names_meeting(id: &str, meeting: &str) -> bool {
    id == meeting
        || id
            .strip_suffix(meeting)
            .is_some_and(|digit| digit.len() == 1 && digit.bytes().all(|b| b.is_ascii_digit()))
}

/// The speaker names a session's track file names suggest, as `merge_tracks` would name them.
pub(crate) fn speaker_names(paths: &[&Path]) -> Vec<String> {
    parse_paths(paths.iter().copied())
        .into_iter()
        .zip(paths)
        .map(|(name, path)| match name {
            TrackName::Zoom { speaker, .. } => speaker,
            TrackName::Other(Some(name)) => name,
            TrackName::ZoomMixed { .. } | TrackName::Other(None) => path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default(),
        })
        .collect()
}

/// Where each track starts on the session's timeline, in seconds, given each file's length.
/// Zoom's files of one segment all start with it, a reconnected participant's later file
/// silent until they rejoined. A stopped recording leaves no gap in the files, so each
/// segment starts where the previous one's longest file ended. Other tracks start at 0.
pub(crate) fn track_starts(names: &[TrackName], durations: &[f32]) -> Vec<f32> {
    let segment_of = |name: &TrackName| match name {
        TrackName::Zoom { segment, .. } => Some(*segment),
        _ => None,
    };
    let mut segments = names.iter().filter_map(segment_of).collect::<Vec<_>>();
    segments.sort_unstable();
    segments.dedup();
    let mut starts = HashMap::new();
    let mut at = 0.0f32;
    for segment in segments {
        starts.insert(segment, at);
        at += names
            .iter()
            .zip(durations)
            .filter(|(name, _)| segment_of(name) == Some(segment))
            .map(|(_, &duration)| duration)
            .fold(0.0, f32::max);
    }
    names
        .iter()
        .map(|name| segment_of(name).map_or(0.0, |segment| starts[&segment]))
        .collect()
}

/// "AndrewDaniel" -> "Andrew Daniel".
fn split_camel_case(name: &str) -> String {
    let mut result = String::with_capacity(name.len() + 4);
    let mut previous_lowercase = false;
    for c in name.chars() {
        if c.is_uppercase() && previous_lowercase {
            result.push(' ');
        }
        // An accent belongs to the letter before it ("Rene\u{301}Dupont").
        if !is_mark(c) {
            previous_lowercase = c.is_lowercase();
        }
        result.push(c);
    }
    result
}

/// "jordan k" -> "Jordan K".
fn title_case(text: &str) -> String {
    text.split(' ')
        .map(|word| {
            let mut chars = word.chars();
            chars
                .next()
                .map(|first| {
                    first
                        .to_uppercase()
                        .chain(chars.flat_map(char::to_lowercase))
                })
                .into_iter()
                .flatten()
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `1-alice` -> "Alice", `bob_iso_mic2` -> "Bob". Drops numbers and recording words;
/// capitalizes all-lowercase words but keeps deliberate casing ("McDonald").
fn generic_name(stem: &str) -> Option<String> {
    let words = stem
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|word| {
            let letters = word.trim_end_matches(|c: char| c.is_ascii_digit());
            !letters.is_empty() && !NOISE_WORDS.contains(&letters.to_lowercase().as_str())
        })
        .map(|word| {
            if word.chars().any(char::is_uppercase) {
                word.to_owned()
            } else {
                title_case(word)
            }
        })
        .collect::<Vec<_>>();
    (!words.is_empty()).then(|| words.join(" "))
}

/// The speaker `merge_tracks` gives a meeting's mixed recording.
pub(crate) const OTHER_SPEAKERS: &str = "Other speakers";
/// Track words closer than this form one stretch of speech when deciding which of the mixed
/// recording's words a track already has.
const COVER_GAP_SECONDS: f32 = 0.5;

/// One transcript to merge.
pub(crate) struct TrackInput {
    pub(crate) doc: TranscriptDocument,
    /// Where the transcript was read from, for messages and as a name fallback.
    pub(crate) path: String,
    /// Speaker name given on the command line; guessed from the file name otherwise.
    pub(crate) name: Option<String>,
    /// The meeting's mixed recording, for the speakers without a track (wisdom's "subtraction"
    /// for a meeting where only one participant has a track): only its words no track
    /// covers are kept, as `OTHER_SPEAKERS`.
    pub(crate) mixed: bool,
}

impl TrackInput {
    /// Parse the recorded audio's file name, falling back to the transcript's.
    fn track_name(&self) -> (String, TrackName) {
        let stem = Path::new(&self.doc.source)
            .file_stem()
            .or_else(|| Path::new(&self.path).file_stem())
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        let name = parse_track_name(&stem);
        (stem, name)
    }

    fn is_merged(&self) -> bool {
        !self.doc.speaker_names.is_empty() || !self.doc.metadata.merged_from.is_empty()
    }
}

/// Pick the transcripts of one session from a folder's transcripts (sorted by path). The
/// session is the Zoom meeting `session` names if given, else that of the `anchor` transcript,
/// else the folder's only Zoom meeting. Without Zoom tracks, every transcript is a track. A
/// meeting with one participant's tracks takes in its mixed recording for everyone else.
/// Returns the tracks and notes about the transcripts left out.
pub(crate) fn select_session(
    candidates: Vec<TrackInput>,
    anchor: Option<&str>,
    session: Option<&str>,
) -> Result<(Vec<TrackInput>, Vec<String>)> {
    let mut notes = Vec::new();
    let mut zoom: Vec<(String, String, TrackInput)> = Vec::new();
    let mut mixed: Vec<(Option<String>, TrackInput)> = Vec::new();
    let mut others = Vec::new();
    let mut anchor_meeting = None;
    let mut names = candidates
        .iter()
        .map(|input| input.track_name().1)
        .collect::<Vec<_>>();
    resolve_name_digits(&mut names);
    for (input, name) in candidates.into_iter().zip(names) {
        if anchor == Some(input.path.as_str()) {
            match &name {
                TrackName::Zoom { meeting, .. }
                | TrackName::ZoomMixed {
                    meeting: Some(meeting),
                    ..
                } => anchor_meeting = Some(meeting.clone()),
                // The old `audio_only` names no meeting: the folder's only one.
                TrackName::ZoomMixed { meeting: None, .. } => {}
                TrackName::Other(_) => bail!(
                    "{} is not a Zoom participant track, so its session's other tracks cannot \
                     be found; list the transcripts to merge",
                    input.path
                ),
            }
        }
        if input.is_merged() {
            notes.push(format!(
                "skipping {}: already a merged transcript",
                input.path
            ));
            continue;
        }
        match name {
            TrackName::ZoomMixed { meeting, .. } => mixed.push((meeting, input)),
            TrackName::Zoom {
                speaker, meeting, ..
            } => zoom.push((meeting, speaker, input)),
            TrackName::Other(_) => others.push((input.path.clone(), input)),
        }
    }
    let mut speakers = std::collections::HashSet::new();
    let mut selected = Vec::new();
    let mut wanted_meeting = None;
    if zoom.is_empty() {
        if let Some(session) = session {
            bail!("no Zoom participant tracks for session {session}");
        }
        for (path, input) in others {
            speakers.insert(path);
            selected.push(input);
        }
    } else {
        let mut meetings: Vec<(String, Vec<String>)> = Vec::new();
        for (meeting, speaker, _) in &zoom {
            match meetings.iter_mut().find(|(id, _)| id == meeting) {
                Some((_, speakers)) if speakers.contains(speaker) => {}
                Some((_, speakers)) => speakers.push(speaker.clone()),
                None => meetings.push((meeting.clone(), vec![speaker.clone()])),
            }
        }
        let listing = || {
            meetings
                .iter()
                .map(|(id, speakers)| format!("{id} ({})", speakers.join(", ")))
                .collect::<Vec<_>>()
                .join("; ")
        };
        let wanted = match (session, anchor_meeting) {
            (Some(session), _) => {
                match meetings.iter().find(|(id, _)| names_meeting(session, id)) {
                    Some((id, _)) => id.clone(),
                    None => bail!("no tracks for Zoom session {session}; found {}", listing()),
                }
            }
            (None, Some(meeting)) => meeting,
            (None, None) if meetings.len() == 1 => meetings[0].0.clone(),
            (None, None) => bail!(
                "found {} Zoom sessions: {}; pick one with --session ID",
                meetings.len(),
                listing()
            ),
        };
        for (meeting, speaker, input) in zoom {
            if meeting == wanted {
                speakers.insert(speaker);
                selected.push(input);
            } else {
                notes.push(format!("skipping {}: Zoom session {meeting}", input.path));
            }
        }
        for (path, _) in others {
            notes.push(format!("skipping {path}: not a Zoom participant track"));
        }
        wanted_meeting = Some(wanted);
    }
    // One speaker's tracks alone are no conversation; the mixed recording has the rest.
    for (meeting, mut input) in mixed {
        let matches = wanted_meeting.is_some()
            && (meeting.is_none() || meeting == wanted_meeting)
            && speakers.len() == 1;
        if matches && !selected.iter().any(|input| input.mixed) {
            input.mixed = true;
            input.name.get_or_insert_with(|| OTHER_SPEAKERS.to_owned());
            selected.push(input);
        } else if matches {
            // Its transcript starts at 0 wherever its segment falls; transcribing the
            // meeting folder joins the segments first.
            notes.push(format!(
                "skipping {}: a later part of Zoom's mixed recording; transcribe the meeting \
                 folder to join its parts",
                input.path
            ));
        } else {
            notes.push(format!("skipping {}: Zoom's mixed recording", input.path));
        }
    }
    if selected.len() < 2 {
        bail!(
            "found {} track transcript(s) to merge; need at least 2",
            selected.len()
        );
    }
    Ok((selected, notes))
}

struct Speaker {
    name: String,
    /// Zoom (segment, part) and input index of each of the speaker's files.
    parts: Vec<((u32, u32), usize)>,
}

/// Merge per-speaker tracks of one session. Returns the merged transcript and warnings about
/// inputs that look inconsistent but may be intended.
pub(crate) fn merge_tracks(inputs: Vec<TrackInput>) -> Result<(TranscriptDocument, Vec<String>)> {
    if inputs.is_empty() {
        bail!("nothing to merge");
    }
    let mut warnings = Vec::new();
    let mut speakers: Vec<Speaker> = Vec::new();
    let mut zoom_speakers = HashMap::new();
    let mut meetings = Vec::new();
    let (stems, mut names): (Vec<_>, Vec<_>) = inputs.iter().map(TrackInput::track_name).unzip();
    resolve_name_digits(&mut names);
    for (index, input) in inputs.iter().enumerate() {
        input
            .doc
            .validate()
            .with_context(|| format!("invalid transcript {}", input.path))?;
        if input.is_merged() {
            bail!("{} is already a merged transcript", input.path);
        }
        if input
            .doc
            .segments
            .iter()
            .any(|segment| segment.words.is_empty())
        {
            bail!(
                "{} has no word timings; merging needs word-level transcripts (the \
                 transcription default, --timestamps words)",
                input.path
            );
        }
        if input
            .doc
            .segments
            .iter()
            .any(|segment| segment.speaker.is_some())
        {
            warnings.push(format!(
                "{} was diarized; its speaker labels are replaced by the track's speaker",
                input.path
            ));
        }
        let given = input.name.clone();
        match &names[index] {
            _ if input.mixed => speakers.push(Speaker {
                name: given.unwrap_or_else(|| OTHER_SPEAKERS.to_owned()),
                parts: vec![((0, 0), index)],
            }),
            TrackName::ZoomMixed { .. } => bail!(
                "{} is Zoom's mixed recording of everyone, not a speaker track",
                input.path
            ),
            TrackName::Zoom {
                speaker,
                part,
                segment,
                meeting,
            } => {
                if !meetings.contains(meeting) {
                    meetings.push(meeting.clone());
                }
                // Reconnecting gives a participant several files: the same speaker.
                let key = (speaker.clone(), meeting.clone());
                let place = (*segment, *part);
                if let Some(&id) = zoom_speakers.get(&key) {
                    let existing: &mut Speaker = &mut speakers[id];
                    if let Some(&(_, other)) = existing.parts.iter().find(|(p, _)| *p == place) {
                        bail!(
                            "{} and {} are the same Zoom track (part {part} of {})",
                            inputs[other].path,
                            input.path,
                            existing.name
                        );
                    }
                    existing.parts.push((place, index));
                    if let Some(name) = given {
                        existing.name = name;
                    }
                } else {
                    zoom_speakers.insert(key, speakers.len());
                    speakers.push(Speaker {
                        name: given.unwrap_or_else(|| speaker.clone()),
                        parts: vec![(place, index)],
                    });
                }
            }
            TrackName::Other(guess) => speakers.push(Speaker {
                name: given
                    .or_else(|| guess.clone())
                    .unwrap_or_else(|| stems[index].clone()),
                parts: vec![((0, 0), index)],
            }),
        }
        if input.mixed {
            names[index] = TrackName::Other(None);
        }
    }
    if meetings.len() > 1 {
        warnings.push(format!(
            "tracks come from {} Zoom sessions ({}); merging them on one timeline",
            meetings.len(),
            meetings.join(", ")
        ));
    }
    let segments = names
        .iter()
        .filter_map(|name| match name {
            TrackName::Zoom { segment, .. } => Some(*segment),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    if segments.len() > 1 {
        warnings.push(format!(
            "the recording was stopped and started again: joining its {} parts end to end",
            segments.len()
        ));
    }
    // Two people guessed to the same name stay distinct speakers.
    let mut seen: HashMap<String, usize> = HashMap::new();
    for speaker in &mut speakers {
        let count = seen.entry(speaker.name.clone()).or_default();
        *count += 1;
        if *count > 1 {
            speaker.name = format!("{} ({count})", speaker.name);
        }
    }

    let durations = inputs
        .iter()
        .map(|input| input.doc.duration_seconds)
        .collect::<Vec<_>>();
    let starts = track_starts(&names, &durations);
    let mut words = Vec::new();
    let mut mixed_words = Vec::new();
    let mut merged_from = Vec::new();
    let mut lengths = Vec::new();
    for (id, speaker) in speakers.iter_mut().enumerate() {
        speaker.parts.sort_unstable();
        let mut length = 0.0f32;
        for (part, &(_, index)) in speaker.parts.iter().enumerate() {
            let input = &inputs[index];
            let offset = starts[index];
            merged_from.push(MergedTrack {
                source: input.doc.source.clone(),
                speaker: id,
                offset_seconds: offset,
            });
            let target = if input.mixed {
                &mut mixed_words
            } else {
                &mut words
            };
            let first = target.len();
            for word in input.doc.segments.iter().flat_map(|segment| &segment.words) {
                target.push(Word {
                    text: word.text.clone(),
                    start: word.start + offset,
                    end: word.end + offset,
                    speaker: Some(id),
                });
            }
            // Like a speaker change: a later file's first word may have no space before it.
            if part > 0
                && let Some(word) = target.get_mut(first)
                && !word.text.starts_with(char::is_whitespace)
            {
                word.text.insert(0, ' ');
            }
            length = length.max(offset + input.doc.duration_seconds);
        }
        lengths.push(length);
    }
    let duration_seconds = lengths.iter().copied().fold(0.0, f32::max);
    let shortest = lengths.iter().copied().fold(f32::INFINITY, f32::min);
    if duration_seconds - shortest > DURATION_TOLERANCE_SECONDS {
        warnings.push(format!(
            "track lengths differ by {:.1} s; tracks of one session should share a timeline",
            duration_seconds - shortest
        ));
    }

    // Stable: equal starts keep speaker order, and each track keeps its own word order.
    words.sort_by(|a, b| a.start.total_cmp(&b.start));
    if !mixed_words.is_empty() {
        let covered = speech_spans(&words);
        mixed_words.retain(|word| !overlaps(&covered, word.start, word.end));
        words.append(&mut mixed_words);
        words.sort_by(|a, b| a.start.total_cmp(&b.start));
    }
    separate_speakers(&mut words);
    // Like Sortformer's count: speakers who said something.
    let speaker_count = (0..speakers.len())
        .filter(|&id| words.iter().any(|word| word.speaker == Some(id)))
        .count();

    let docs = inputs.iter().map(|input| &input.doc).collect::<Vec<_>>();
    let distinct = |field: fn(&TranscriptDocument) -> &str| {
        let mut values: Vec<&str> = Vec::new();
        for doc in &docs {
            if !values.contains(&field(doc)) {
                values.push(field(doc));
            }
        }
        values.join(", ")
    };
    let model = distinct(|doc| &doc.model);
    let doc = TranscriptDocument {
        source: distinct(|doc| &doc.source),
        model: model.clone(),
        device: distinct(|doc| &doc.device),
        language: distinct(|doc| &doc.language),
        duration_seconds,
        segments: build_segments(words),
        speaker_names: speakers.into_iter().map(|speaker| speaker.name).collect(),
        metadata: TranscriptMetadata {
            transcribed_with: "transcriberrr".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            // All UTC RFC 3339 at second precision, so string order is time order.
            source_created_at: docs
                .iter()
                .filter_map(|doc| doc.metadata.source_created_at.clone())
                .min(),
            source_modified_at: docs
                .iter()
                .filter_map(|doc| doc.metadata.source_modified_at.clone())
                .max(),
            transcribed_at: docs
                .iter()
                .map(|doc| doc.metadata.transcribed_at.clone())
                .max()
                .unwrap_or_default(),
            model,
            // The speakers come from the tracks, not Sortformer; `merged_from` says so.
            diarization: Some(DiarizationMetadata {
                enabled: true,
                speakers: None,
                detected_speakers: Some(speaker_count),
            }),
            merged_from,
        },
    };
    doc.validate()?;
    Ok((doc, warnings))
}

/// Stretches of speech in time-ordered words, with pauses under `COVER_GAP_SECONDS` bridged.
fn speech_spans(words: &[Word]) -> Vec<(f32, f32)> {
    let mut spans: Vec<(f32, f32)> = Vec::new();
    for word in words {
        match spans.last_mut() {
            Some(last) if word.start <= last.1 + COVER_GAP_SECONDS => last.1 = last.1.max(word.end),
            _ => spans.push((word.start, word.end)),
        }
    }
    spans
}

fn overlaps(spans: &[(f32, f32)], start: f32, end: f32) -> bool {
    let index = spans.partition_point(|span| span.1 <= start);
    spans.get(index).is_some_and(|span| span.0 < end)
}

/// Keep speakers apart in the text. A word carries its own leading whitespace, and a track's
/// first word has none, so a speaker change would otherwise run two words together
/// ("there.Fine"). Diarized transcripts are one stream separated by spaces; so is this.
fn separate_speakers(words: &mut [Word]) {
    for index in 1..words.len() {
        let (before, after) = words.split_at_mut(index);
        let (previous, word) = (&before[index - 1], &mut after[0]);
        if word.speaker != previous.speaker
            && !word.text.starts_with(char::is_whitespace)
            && !previous.text.ends_with(char::is_whitespace)
        {
            word.text.insert(0, ' ');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{RenderOptions, render_output};
    use crate::{OutputFormat, TimestampGranularity};

    fn track(source: &str, duration: f32, words: &[(&str, f32)]) -> TrackInput {
        let words = words
            .iter()
            .map(|&(text, start)| Word {
                text: text.to_owned(),
                start,
                end: start + 0.4,
                speaker: None,
            })
            .collect();
        TrackInput {
            doc: TranscriptDocument {
                source: source.to_owned(),
                model: "m".to_owned(),
                device: "cpu".to_owned(),
                language: "English".to_owned(),
                duration_seconds: duration,
                segments: build_segments(words),
                speaker_names: Vec::new(),
                metadata: TranscriptMetadata {
                    transcribed_with: "transcriberrr".to_owned(),
                    version: "0".to_owned(),
                    source_created_at: None,
                    source_modified_at: None,
                    transcribed_at: "2026-10-22T00:00:00Z".to_owned(),
                    model: "m".to_owned(),
                    diarization: None,
                    merged_from: Vec::new(),
                },
            },
            path: format!("{source}.json"),
            name: None,
            mixed: false,
        }
    }

    fn render(doc: &TranscriptDocument, speaker_turns: bool) -> String {
        render_output(
            doc,
            RenderOptions {
                format: OutputFormat::Txt,
                timestamps: TimestampGranularity::None,
                speaker_turns,
            },
        )
        .unwrap()
    }

    fn turns(doc: &TranscriptDocument) -> Vec<(Option<usize>, String)> {
        doc.segments
            .iter()
            .map(|segment| (segment.speaker, segment.content()))
            .collect()
    }

    fn paths(inputs: &[TrackInput]) -> Vec<&str> {
        inputs.iter().map(|input| input.path.as_str()).collect()
    }

    #[test]
    fn parses_zoom_participant_file_names() {
        let zoom = |speaker: &str, part, segment, meeting: &str| TrackName::Zoom {
            speaker: speaker.to_owned(),
            part,
            segment,
            meeting: meeting.to_owned(),
        };
        assert_eq!(
            parse_track_name("audioAndrewDaniel21067067067"),
            zoom("Andrew Daniel", 2, 1, "067067067")
        );
        assert_eq!(
            parse_track_name("audioHana62062062062"),
            zoom("Hana", 6, 2, "062062062")
        );
        // A twelfth reconnect, and a file from before the segment digit.
        assert_eq!(
            parse_track_name("audioHana122062062062"),
            zoom("Hana", 12, 2, "062062062")
        );
        assert_eq!(
            parse_track_name("audioHana3878787878"),
            zoom("Hana", 3, 0, "878787878")
        );
        assert_eq!(
            parse_track_name("audio_jordan_k.41864297531"),
            zoom("Jordan K", 4, 1, "864297531")
        );
        assert_eq!(
            parse_track_name("audio_AndrewDaniel21067067067"),
            zoom("Andrew Daniel", 2, 1, "067067067")
        );
        assert_eq!(
            parse_track_name("audio_only_13572468_Dana_Reyes"),
            zoom("Dana Reyes", 0, 0, "13572468")
        );
        let mixed = |segment, meeting: Option<&str>| TrackName::ZoomMixed {
            segment,
            meeting: meeting.map(str::to_owned),
        };
        assert_eq!(
            parse_track_name("audio2062062062"),
            mixed(2, Some("062062062"))
        );
        assert_eq!(
            parse_track_name("audio123456789"),
            mixed(0, Some("123456789"))
        );
        assert_eq!(parse_track_name("audio_only"), mixed(0, None));
        assert_eq!(parse_track_name("audio_only01"), mixed(0, None));
        assert!(names_meeting("2062062062", "062062062"));
        assert!(names_meeting("062062062", "062062062"));
        assert!(!names_meeting("12062062062", "062062062"));
    }

    #[test]
    fn places_reconnects_and_restarted_recordings_on_one_timeline() {
        // As Zoom numbered a meeting recorded in two segments, in which Bob reconnected in
        // the first and Alice joined twice in the second; every file of a segment is as long
        // as the segment.
        let names = [
            "audioAlice11067067067",
            "audioBob21067067067",
            "audioCarol31067067067",
            "audioBob41067067067",
            "audioAlice52067067067",
            "audioCarol62067067067",
            "audioBob72067067067",
            "audioAlice82067067067",
            "notes",
        ]
        .map(parse_track_name);
        let (first, second) = (5346.24, 4159.15);
        let durations = [
            first, first, first, first, second, second, second, second, 3.0,
        ];
        let starts = track_starts(&names, &durations);
        assert_eq!(
            starts,
            [0.0, 0.0, 0.0, 0.0, first, first, first, first, 0.0]
        );
    }

    #[test]
    fn guesses_names_from_other_file_names() {
        let name = |stem| match parse_track_name(stem) {
            TrackName::Other(name) => name,
            other => panic!("{stem} parsed as {other:?}"),
        };
        assert_eq!(name("1-alice").as_deref(), Some("Alice"));
        assert_eq!(name("bob_iso_mic2").as_deref(), Some("Bob"));
        assert_eq!(name("Track 3 - McDonald").as_deref(), Some("McDonald"));
        assert_eq!(name("audio1234").as_deref(), None);
    }

    #[test]
    fn builds_diarized_segments_from_interleaved_words() {
        let alice = track(
            "/r/audioAliceSmith11067067067.m4a",
            10.0,
            &[
                ("Hello", 0.0),
                (" there.", 0.5),
                (" Shall", 3.0),
                (" we?", 3.5),
            ],
        );
        let bob = track(
            "/r/audioBob11067067067.m4a",
            10.0,
            &[("Yeah.", 1.2), (" Sure.", 5.0)],
        );
        let (doc, warnings) = merge_tracks(vec![alice, bob]).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(doc.speaker_names, ["Alice Smith", "Bob"]);
        assert_eq!(
            turns(&doc),
            [
                (Some(0), "Hello there.".to_owned()),
                (Some(1), " Yeah.".to_owned()),
                (Some(0), " Shall we?".to_owned()),
                (Some(1), " Sure.".to_owned()),
            ]
        );
        assert!(doc.segments.iter().all(|segment| {
            segment
                .words
                .iter()
                .all(|word| word.speaker == segment.speaker)
        }));
        assert_eq!(render(&doc, false), "Hello there. Yeah. Shall we? Sure.\n");
        assert_eq!(
            render(&doc, true),
            "[Alice Smith] Hello there.\n[Bob] Yeah.\n[Alice Smith] Shall we?\n[Bob] Sure.\n"
        );
        let diarization = doc.metadata.diarization.as_ref().unwrap();
        assert_eq!(diarization.detected_speakers, Some(2));
        assert_eq!(doc.metadata.merged_from.len(), 2);
    }

    #[test]
    fn overlapping_speech_interleaves_by_word_like_diarization() {
        let alice = track(
            "alice.wav",
            5.0,
            &[("I", 0.0), (" think", 0.4), (" so.", 1.2)],
        );
        let bob = track("bob.wav", 5.0, &[("Yeah.", 0.8)]);
        let (doc, _) = merge_tracks(vec![alice, bob]).unwrap();
        assert_eq!(
            turns(&doc),
            [
                (Some(0), "I think".to_owned()),
                (Some(1), " Yeah.".to_owned()),
                (Some(0), " so.".to_owned()),
            ]
        );
    }

    #[test]
    fn equal_starts_keep_argument_order() {
        let a = track("a.wav", 2.0, &[("One.", 1.0)]);
        let b = track("b.wav", 2.0, &[("Two.", 1.0)]);
        let (doc, _) = merge_tracks(vec![a, b]).unwrap();
        assert_eq!(doc.segments[0].speaker, Some(0));
        assert_eq!(doc.speaker_names, ["A", "B"]);
    }

    #[test]
    fn joins_a_reconnected_speakers_files_on_one_timeline() {
        let late = track("audioAlice31067067067.m4a", 10.0, &[("Back.", 7.0)]);
        let early = track("audioAlice11067067067.m4a", 10.0, &[("Hi.", 1.0)]);
        let bob = track("audioBob21067067067.m4a", 10.0, &[("Hey.", 2.0)]);
        let (doc, warnings) = merge_tracks(vec![late, bob, early]).unwrap();
        assert_eq!(doc.speaker_names, ["Alice", "Bob"]);
        assert!(warnings.is_empty(), "{warnings:?}");
        let alice = doc
            .segments
            .iter()
            .filter(|segment| segment.speaker == Some(0))
            .map(|segment| (segment.start, segment.content()))
            .collect::<Vec<_>>();
        assert_eq!(alice, [(1.0, "Hi.".to_owned()), (7.0, " Back.".to_owned())]);
        assert_eq!(doc.duration_seconds, 10.0);
    }

    #[test]
    fn rejects_mixed_recordings_duplicates_merges_and_sentence_only_tracks() {
        let bob = || track("audioBob11067067067.m4a", 1.0, &[("Hey.", 0.0)]);
        let mixed = track("audio1067067067.m4a", 1.0, &[("All.", 0.0)]);
        assert!(merge_tracks(vec![mixed, bob()]).is_err());
        assert!(merge_tracks(vec![bob(), bob()]).is_err());
        let (merged, _) = merge_tracks(vec![bob(), track("x.wav", 1.0, &[("Hi.", 0.0)])]).unwrap();
        let merged = TrackInput {
            doc: merged,
            path: "talk.json".to_owned(),
            name: None,
            mixed: false,
        };
        let error = merge_tracks(vec![bob(), merged]).err().unwrap();
        assert!(error.to_string().contains("already a merged"));
        let mut sentences = track("y.wav", 1.0, &[("Hi.", 0.0)]);
        sentences.doc.collapse_words();
        let error = merge_tracks(vec![bob(), sentences]).err().unwrap();
        assert!(error.to_string().contains("no word timings"));
    }

    #[test]
    fn replaces_diarized_labels_and_keeps_same_named_people_apart() {
        let mut a = track("/x/mic1.wav", 3.0, &[("Hi.", 0.0)]);
        for segment in &mut a.doc.segments {
            segment.speaker = Some(3);
            for word in &mut segment.words {
                word.speaker = Some(3);
            }
        }
        let b = track("/y/mic1.wav", 30.0, &[("Yo.", 1.0)]);
        let (doc, warnings) = merge_tracks(vec![a, b]).unwrap();
        assert_eq!(doc.segments[0].speaker, Some(0));
        assert_eq!(doc.speaker_names, ["mic1", "mic1 (2)"]);
        assert!(warnings.iter().any(|w| w.contains("diarized")));
        assert!(warnings.iter().any(|w| w.contains("lengths differ")));
    }

    #[test]
    fn joins_a_restarted_recordings_segments() {
        let alice1 = track("audioAlice11067067067.m4a", 8.0, &[("Hi.", 1.0)]);
        let bob1 = track("audioBob21067067067.m4a", 8.0, &[("Hey.", 2.0)]);
        let alice2 = track("audioAlice32067067067.m4a", 5.0, &[("Back.", 1.0)]);
        let (doc, warnings) = merge_tracks(vec![alice2, bob1, alice1]).unwrap();
        assert_eq!(doc.speaker_names, ["Alice", "Bob"]);
        assert!(warnings.iter().any(|w| w.contains("stopped and started")));
        assert_eq!(
            turns(&doc),
            [
                (Some(0), "Hi.".to_owned()),
                (Some(1), " Hey.".to_owned()),
                (Some(0), " Back.".to_owned())
            ]
        );
        assert_eq!(doc.segments[2].start, 9.0);
        assert_eq!(doc.duration_seconds, 13.0);
    }

    #[test]
    fn a_mixed_recording_fills_in_speakers_without_a_track() {
        let alice = track(
            "audioAlice11067067067.m4a",
            10.0,
            &[("So", 1.0), (" tell", 1.4), (" me.", 1.8), (" Great.", 7.0)],
        );
        let mut mixed = track(
            "audio1067067067.m4a",
            10.0,
            &[
                ("So", 1.05),
                (" tell", 1.45),
                (" me.", 1.85),
                (" I", 3.0),
                (" moved.", 3.4),
                (" Great.", 7.05),
            ],
        );
        mixed.mixed = true;
        let (doc, _) = merge_tracks(vec![alice, mixed]).unwrap();
        assert_eq!(doc.speaker_names, ["Alice", OTHER_SPEAKERS]);
        assert_eq!(
            turns(&doc),
            [
                (Some(0), "So tell me.".to_owned()),
                (Some(1), " I moved.".to_owned()),
                (Some(0), " Great.".to_owned())
            ]
        );

        // `merge` finds the mixed recording for a meeting with one participant's track.
        let candidates = vec![
            track("audio1067067067.m4a", 10.0, &[("I", 3.0)]),
            track("audioAlice11067067067.m4a", 10.0, &[("So", 1.0)]),
        ];
        let (tracks, _) = select_session(candidates, None, None).unwrap();
        assert_eq!(
            tracks.iter().map(|track| track.mixed).collect::<Vec<_>>(),
            [false, true]
        );
        assert_eq!(tracks[1].name.as_deref(), Some(OTHER_SPEAKERS));
    }

    #[test]
    fn a_speakers_next_part_starts_a_new_word() {
        let early = track(
            "audioAlice11067067067.m4a",
            6.0,
            &[("Hi", 1.0), (" there.", 1.4)],
        );
        let late = track("audioAlice31067067067.m4a", 6.0, &[("Back.", 3.0)]);
        let bob = track("audioBob21067067067.m4a", 6.0, &[("Hey.", 5.0)]);
        let (doc, _) = merge_tracks(vec![early, late, bob]).unwrap();
        let text = doc
            .segments
            .iter()
            .map(|segment| segment.content())
            .collect::<String>();
        assert_eq!(text, "Hi there. Back. Hey.");
    }

    #[test]
    fn names_ending_in_a_digit_stay_apart() {
        let one = track("audioGuest111067067067.m4a", 4.0, &[("Hi.", 1.0)]);
        let two = track("audioGuest221067067067.m4a", 4.0, &[("Hey.", 2.0)]);
        let (doc, warnings) = merge_tracks(vec![one, two]).unwrap();
        assert_eq!(doc.speaker_names, ["Guest1", "Guest2"]);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            doc.metadata
                .merged_from
                .iter()
                .map(|track| track.offset_seconds)
                .collect::<Vec<_>>(),
            [0.0, 0.0]
        );
        // Real file numbers of 10 or more: no file 0, file 1 is taken, or Hana has files
        // below 10.
        let mut names = [
            "audioAlice11062062062",
            "audioBob102062062062",
            "audioCarol112062062062",
            "audioHana92062062062",
            "audioHana122062062062",
        ]
        .map(parse_track_name);
        resolve_name_digits(&mut names);
        let read = names
            .iter()
            .map(|name| match name {
                TrackName::Zoom { speaker, part, .. } => (speaker.as_str(), *part),
                other => panic!("{other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            read,
            [
                ("Alice", 1),
                ("Bob", 10),
                ("Carol", 11),
                ("Hana", 9),
                ("Hana", 12)
            ]
        );
        // Decomposed accents (from HFS+ or a NAS) and punctuation are part of a name.
        let name = |stem| match parse_track_name(stem) {
            TrackName::Zoom { speaker, .. } => speaker,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            name("audioRene\u{301}Dupont11067067067"),
            "Rene\u{301} Dupont"
        );
        assert_eq!(
            name("audioMaryO'Brien-Smith11067067067"),
            "Mary O'Brien-Smith"
        );
    }

    #[test]
    fn an_old_mixed_recording_finds_its_meeting() {
        let candidates = vec![
            track("audio_only.m4a", 10.0, &[("All.", 0.0)]),
            track("audio_only_13572468_Hana.m4a", 10.0, &[("Hi.", 1.0)]),
        ];
        let (tracks, _) = select_session(candidates, Some("audio_only.m4a.json"), None).unwrap();
        assert_eq!(
            tracks.iter().map(|track| track.mixed).collect::<Vec<_>>(),
            [false, true]
        );
    }

    #[test]
    fn given_names_win() {
        let mut a = track("audioAndrewDaniel21067067067.m4a", 2.0, &[("Hi.", 0.0)]);
        a.name = Some("Andrew".to_owned());
        let b = track("b.wav", 2.0, &[("Hello.", 1.0)]);
        let (doc, _) = merge_tracks(vec![a, b]).unwrap();
        assert_eq!(doc.speaker_names, ["Andrew", "B"]);
    }

    fn folder() -> Vec<TrackInput> {
        vec![
            track("audio1067067067.m4a", 1.0, &[("All.", 0.0)]),
            track("audioAndrewDaniel21067067067.m4a", 1.0, &[("Hi.", 0.0)]),
            track("audioHana31067067067.m4a", 1.0, &[("Hey.", 0.0)]),
            track("audioHana12062062062.m4a", 1.0, &[("Yo.", 0.0)]),
            track("audioSam12062062062.m4a", 1.0, &[("Sup.", 0.0)]),
            track("notes.wav", 1.0, &[("Memo.", 0.0)]),
        ]
    }

    #[test]
    fn a_folder_with_two_sessions_needs_a_choice() {
        let error = select_session(folder(), None, None)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("067067067 (Andrew Daniel, Hana)"), "{error}");
        assert!(error.contains("062062062 (Hana, Sam)"), "{error}");
        let (tracks, notes) = select_session(folder(), None, Some("2062062062")).unwrap();
        assert_eq!(
            paths(&tracks),
            [
                "audioHana12062062062.m4a.json",
                "audioSam12062062062.m4a.json"
            ]
        );
        assert_eq!(notes.len(), 4, "{notes:?}");
        assert!(select_session(folder(), None, Some("999")).is_err());
    }

    #[test]
    fn one_track_finds_its_session_siblings_despite_a_different_part_digit() {
        let (tracks, _) =
            select_session(folder(), Some("audioHana31067067067.m4a.json"), None).unwrap();
        assert_eq!(
            paths(&tracks),
            [
                "audioAndrewDaniel21067067067.m4a.json",
                "audioHana31067067067.m4a.json"
            ]
        );
        // The mixed recording's transcript names the session too, but is not a track.
        let (tracks, notes) =
            select_session(folder(), Some("audio1067067067.m4a.json"), None).unwrap();
        assert_eq!(tracks.len(), 2);
        assert!(notes.iter().any(|note| note.contains("mixed recording")));
        assert!(select_session(folder(), Some("notes.wav.json"), None).is_err());
    }

    #[test]
    fn a_folder_without_zoom_tracks_merges_all_but_earlier_merges() {
        let a = track("a.wav", 1.0, &[("Hi.", 0.0)]);
        let b = track("b.wav", 1.0, &[("Yo.", 0.0)]);
        let (merged, _) = merge_tracks(vec![a, b]).unwrap();
        let candidates = vec![
            track("a.wav", 1.0, &[("Hi.", 0.0)]),
            track("b.wav", 1.0, &[("Yo.", 0.0)]),
            TrackInput {
                doc: merged,
                path: "talk.json".to_owned(),
                name: None,
                mixed: false,
            },
        ];
        let (tracks, notes) = select_session(candidates, None, None).unwrap();
        assert_eq!(paths(&tracks), ["a.wav.json", "b.wav.json"]);
        assert!(notes[0].contains("already a merged"));
        let single = vec![track("a.wav", 1.0, &[("Hi.", 0.0)])];
        assert!(select_session(single, None, None).is_err());
    }
}
