//! Read sample-timed native transcripts and project their canonical words for conversion.
use super::{Activity, Coverage, Speaker, Time, VERSION};
use crate::{TARGET_SAMPLE_RATE, transcript};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Reader for the native interchange result, also used by the lossless JSON converter.
#[derive(Deserialize)]
pub(crate) struct NativeResult {
    version: u8,
    clock: NativeClock,
    sample_rate: u32,
    tracks: Vec<NativeTrack>,
    speakers: Vec<Speaker>,
    words: Vec<NativeWord>,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum NativeClock {
    Presentation,
    Session,
}

#[derive(Deserialize)]
struct NativeTrack {
    id: String,
    name: String,
    origin: Time,
    session_start: Time,
    coverage: Coverage,
    #[serde(default)]
    speakers: Vec<Speaker>,
    #[serde(default)]
    words: Vec<NativeWord>,
    #[serde(default)]
    activity: Vec<Activity>,
}

#[derive(Deserialize)]
struct NativeWord {
    id: String,
    text: String,
    start_sample: i64,
    end_sample: i64,
    speaker_id: Option<String>,
    utterance_id: Option<String>,
    precision: NativePrecision,
    confidence: Option<f32>,
    #[serde(default)]
    observations: Vec<NativeObservation>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NativePrecision {
    Word,
    Cue,
}

#[derive(Deserialize)]
struct NativeObservation {
    track_id: String,
    text: Option<String>,
    start_sample: i64,
    end_sample: i64,
    speaker_id: Option<String>,
    confidence: Option<f32>,
    evidence: NativeEvidence,
    basis: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NativeEvidence {
    Recognized,
    Inferred,
}

impl NativeResult {
    pub(crate) fn parse(value: &Value) -> Result<Self> {
        let result = Self::deserialize(value).context("reading native session transcript")?;
        result.validate()?;
        Ok(result)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == VERSION && self.sample_rate == TARGET_SAMPLE_RATE,
            "unsupported native session version or sample rate"
        );
        ensure!(
            self.clock != NativeClock::Presentation || self.tracks.len() == 1,
            "a presentation-clock transcript needs exactly one track"
        );
        fn ids<'a>(items: impl Iterator<Item = &'a str>) -> Result<HashSet<&'a str>> {
            let mut ids = HashSet::new();
            for id in items {
                ensure!(
                    !id.is_empty() && ids.insert(id),
                    "duplicate or empty native session ID"
                );
            }
            Ok(ids)
        }
        fn bounds(start: i64, end: i64, signed: bool) -> bool {
            start <= end && (signed || start >= 0)
        }
        fn confidence(value: Option<f32>) -> bool {
            value.is_none_or(|confidence| (0.0..=1.0).contains(&confidence))
        }
        let speaker_ids = ids(self.speakers.iter().map(|speaker| speaker.id.as_str()))?;
        ids(self.tracks.iter().map(|track| track.id.as_str()))?;
        let coverage: HashMap<_, _> = self
            .tracks
            .iter()
            .map(|track| (track.id.as_str(), track.coverage))
            .collect();
        for track in &self.tracks {
            track.origin.samples()?;
            track.session_start.samples()?;
            ensure!(!track.name.is_empty(), "native session track needs a name");
            let speakers = ids(track.speakers.iter().map(|speaker| speaker.id.as_str()))?;
            ids(track.words.iter().map(|word| word.id.as_str()))?;
            for word in &track.words {
                ensure!(
                    track.coverage == Coverage::Recognized
                        && !word.text.is_empty()
                        && bounds(word.start_sample, word.end_sample, false)
                        && confidence(word.confidence)
                        && word.utterance_id.as_deref() != Some("")
                        && word.observations.is_empty()
                        && word
                            .speaker_id
                            .as_deref()
                            .is_none_or(|id| speakers.contains(id)),
                    "invalid native track-local word"
                );
            }
            for level in &track.activity {
                ensure!(
                    bounds(level.start_sample, level.end_sample, false)
                        && level.start_sample < level.end_sample
                        && (0.0..=1.0).contains(&level.level),
                    "invalid native track activity"
                );
            }
        }
        ids(self.words.iter().map(|word| word.id.as_str()))?;
        for word in &self.words {
            ensure!(
                !word.text.is_empty()
                    && bounds(
                        word.start_sample,
                        word.end_sample,
                        self.clock == NativeClock::Session
                    )
                    && confidence(word.confidence)
                    && word.utterance_id.as_deref() != Some("")
                    && word
                        .speaker_id
                        .as_deref()
                        .is_none_or(|id| speaker_ids.contains(id)),
                "invalid native conversation word"
            );
            for item in &word.observations {
                ensure!(
                    bounds(item.start_sample, item.end_sample, false)
                        && confidence(item.confidence)
                        && item
                            .speaker_id
                            .as_deref()
                            .is_none_or(|id| speaker_ids.contains(id))
                        && item.text.as_ref().is_none_or(|text| !text.is_empty()),
                    "invalid native track observation"
                );
                let coverage = coverage
                    .get(item.track_id.as_str())
                    .context("unknown observation track ID")?;
                match item.evidence {
                    NativeEvidence::Recognized => ensure!(
                        *coverage == Coverage::Recognized,
                        "recognized word on track without recognition"
                    ),
                    NativeEvidence::Inferred => ensure!(
                        item.basis
                            .as_ref()
                            .is_some_and(|basis| !basis.trim().is_empty()),
                        "inferred observation needs a basis"
                    ),
                }
            }
        }
        Ok(())
    }

    /// Legacy text and subtitle renderers take nonnegative time. Shift each cue by the same
    /// amount when a session has speech before zero, preserving spacing and duration.
    pub(crate) fn projected_document(
        self,
        source: &Path,
    ) -> (transcript::TranscriptDocument, (i128, i128)) {
        let origin = if self.clock == NativeClock::Presentation {
            self.tracks[0].origin
        } else {
            Time {
                numerator: 0,
                denominator: 1,
            }
        };
        // Keep the clock rational through the origin addition. Legacy text and subtitle
        // formatters quantize at formatting time.
        let rate = i128::from(self.sample_rate);
        let scale = i128::from(origin.denominator);
        let numerator =
            |sample: i64| i128::from(origin.numerator) * rate + i128::from(sample) * scale;
        let denominator = scale * rate;
        let offset = self
            .words
            .iter()
            .map(|word| numerator(word.start_sample))
            .min()
            .unwrap_or(0)
            .min(0);
        let seconds =
            |sample: i64| ((numerator(sample) - offset) as f64 / denominator as f64) as f32;
        let speakers: HashMap<_, _> = self
            .speakers
            .iter()
            .enumerate()
            .map(|(index, speaker)| (speaker.id.as_str(), index))
            .collect();
        let mut items = self.words;
        items.sort_by_key(|item| item.start_sample);
        let duration_seconds = items
            .iter()
            .map(|item| seconds(item.end_sample))
            .fold(0.0, f32::max);
        let mut segments = Vec::new();
        let mut words = Vec::new();
        for item in items {
            let start = seconds(item.start_sample);
            let end = seconds(item.end_sample);
            let speaker = item
                .speaker_id
                .and_then(|id| speakers.get(id.as_str()).copied());
            match item.precision {
                NativePrecision::Word => words.push(transcript::Word {
                    text: item.text,
                    start,
                    end,
                    speaker,
                }),
                NativePrecision::Cue => {
                    segments.extend(transcript::build_segments(std::mem::take(&mut words)));
                    segments.push(transcript::Segment {
                        text: Some(item.text),
                        start,
                        end,
                        speaker,
                        words: Vec::new(),
                    });
                }
            }
        }
        segments.extend(transcript::build_segments(words));
        let document = transcript::TranscriptDocument {
            source: source.display().to_string(),
            model: "session".to_owned(),
            device: "local".to_owned(),
            language: "English".to_owned(),
            duration_seconds,
            segments,
            speaker_names: self
                .speakers
                .into_iter()
                .map(|speaker| speaker.name)
                .collect(),
            metadata: transcript::TranscriptMetadata {
                transcribed_with: "transcriberrr".to_owned(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
                source_created_at: None,
                source_modified_at: None,
                transcribed_at: String::new(),
                model: "session".to_owned(),
                diarization: None,
                merged_from: Vec::new(),
            },
        };
        (document, (offset, denominator))
    }
}
