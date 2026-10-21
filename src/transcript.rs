//! Compact transcript schema and deterministic text/export operations.
use crate::{OutputFormat, TimestampGranularity};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Word {
    /// Exact transcript fragment, including preceding whitespace and punctuation.
    pub(crate) text: String,
    pub(crate) start: f32,
    pub(crate) end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) speaker: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Segment {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<String>,
    pub(crate) start: f32,
    pub(crate) end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) speaker: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) words: Vec<Word>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TranscriptDocument {
    pub(crate) source: String,
    pub(crate) model: String,
    pub(crate) device: String,
    pub(crate) language: String,
    pub(crate) duration_seconds: f32,
    pub(crate) segments: Vec<Segment>,
    pub(crate) metadata: TranscriptMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TranscriptMetadata {
    #[serde(rename = "transcribed-with")]
    pub(crate) transcribed_with: String,
    pub(crate) version: String,
    pub(crate) source_created_at: Option<String>,
    pub(crate) source_modified_at: Option<String>,
    pub(crate) transcribed_at: String,
    pub(crate) model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) diarization: Option<DiarizationMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DiarizationMetadata {
    pub(crate) enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) speakers: Option<usize>,
    /// Number of distinct speaker IDs returned by Sortformer. `speakers` retains the
    /// user-requested maximum for backwards compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detected_speakers: Option<usize>,
}

impl Segment {
    pub(crate) fn content(&self) -> String {
        self.text
            .clone()
            .unwrap_or_else(|| self.words.iter().map(|word| word.text.as_str()).collect())
    }
}

impl TranscriptDocument {
    pub(crate) fn content(&self) -> String {
        self.segments.iter().map(Segment::content).collect()
    }

    pub(crate) fn collapse_words(&mut self) {
        for segment in &mut self.segments {
            segment.text = Some(segment.content());
            segment.words.clear();
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        fn timing(start: f32, end: f32) -> Result<()> {
            if !start.is_finite() || !end.is_finite() || start < 0.0 || end < start {
                bail!("invalid transcript timing: {start} --> {end}");
            }
            Ok(())
        }
        timing(0.0, self.duration_seconds)?;
        for segment in &self.segments {
            timing(segment.start, segment.end)?;
            if segment.text.is_some() == !segment.words.is_empty() {
                bail!("each segment must contain either text or words, exclusively");
            }
            let mut last_start = segment.start;
            for word in &segment.words {
                timing(word.start, word.end)?;
                if word.start < last_start || word.end > segment.end || word.text.is_empty() {
                    bail!("invalid word order, segment bounds, or empty word text");
                }
                if word.speaker != segment.speaker {
                    bail!("word speaker must match its segment speaker");
                }
                last_start = word.start;
            }
        }
        Ok(())
    }
}

pub(crate) fn build_segments(words: Vec<Word>) -> Vec<Segment> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    for word in words {
        let split = current.last().is_some_and(|last: &Word| {
            word.start - last.end > 1.0
                || last
                    .text
                    .trim_end_matches(['”', '’', '"', '\'', ')', ']', '」', '』'])
                    .ends_with(['.', '?', '!', '。', '？', '！'])
                || word.speaker != last.speaker
        });
        if split && !current.is_empty() {
            result.push(segment_from_words(std::mem::take(&mut current)));
        }
        current.push(word);
    }
    if !current.is_empty() {
        result.push(segment_from_words(current));
    }
    result
}

fn segment_from_words(words: Vec<Word>) -> Segment {
    Segment {
        text: None,
        start: words.first().map(|word| word.start).unwrap_or_default(),
        end: words.iter().map(|word| word.end).fold(0.0, f32::max),
        speaker: words.first().and_then(|word| word.speaker),
        words,
    }
}

/// Restore exact source fragments to aligner tokens, which may omit punctuation.
/// Match every letter/number in order; never silently discard or invent spoken text.
pub(crate) fn preserve_word_text(text: &str, words: &mut [Word]) -> Result<()> {
    if words.is_empty() {
        return Ok(());
    }
    let mut cursor = 0;
    let mut spans = Vec::with_capacity(words.len());
    for word in words.iter() {
        let token = word.text.trim();
        let letters: Vec<char> = token.chars().filter(|c| c.is_alphanumeric()).collect();
        let mut start = None;
        if letters.is_empty() {
            if token.is_empty() {
                bail!("empty aligned word");
            }
            let offset = text[cursor..]
                .find(token)
                .context("aligned punctuation does not match transcript")?;
            if text[cursor..cursor + offset]
                .chars()
                .any(char::is_alphanumeric)
            {
                bail!("aligned punctuation skips transcript words");
            }
            start = Some(cursor + offset);
            cursor += offset + token.len();
        } else {
            for expected in letters {
                let (offset, actual) = text[cursor..]
                    .char_indices()
                    .find(|(_, c)| c.is_alphanumeric())
                    .context("aligned words extend beyond transcript text")?;
                // Aligners may case-fold non-ASCII letters too ("École" vs "école").
                if actual != expected && !actual.to_lowercase().eq(expected.to_lowercase()) {
                    bail!(
                        "aligned word {:?} does not match transcript near {:?}",
                        token,
                        text[cursor..].chars().take(40).collect::<String>()
                    );
                }
                start.get_or_insert(cursor + offset);
                cursor += offset + actual.len_utf8();
            }
        }
        spans.push((start.unwrap_or(cursor), cursor));
    }
    if text[cursor..].chars().any(char::is_alphanumeric) {
        bail!("transcript contains words without timestamps");
    }
    let mut previous_end = 0;
    for index in 0..words.len() {
        let end = if index + 1 == words.len() {
            text.len()
        } else {
            // Attach closing punctuation to this word, leaving whitespace and opening
            // quotes/brackets as the exact prefix of the next word.
            let end = spans[index].1;
            let gap = &text[end..spans[index + 1].0];
            end + gap
                .char_indices()
                .find(|(_, c)| {
                    c.is_whitespace() || matches!(c, '“' | '‘' | '(' | '[' | '{' | '「' | '『')
                })
                .map(|(offset, _)| offset)
                .unwrap_or(gap.len())
        };
        words[index].text = text[previous_end..end].to_owned();
        previous_end = end;
    }
    Ok(())
}

/// Make aligner/recognizer timings satisfy `validate` instead of discarding a finished
/// transcription at write time: times are clamped to the audio, starts never move backwards,
/// and every word ends at or after its start. Padded final chunks can otherwise report words
/// past the end of the audio.
pub(crate) fn normalize_word_timing(words: &mut [Word], duration: f32) {
    let duration = if duration.is_finite() {
        duration.max(0.0)
    } else {
        0.0
    };
    let clamp = |value: f32, floor: f32| {
        if value.is_finite() {
            value.clamp(floor, duration.max(floor))
        } else {
            floor
        }
    };
    let mut previous_start = 0.0f32;
    for word in words {
        word.start = clamp(word.start, previous_start);
        word.end = clamp(word.end, word.start);
        previous_start = word.start;
    }
}

/// Write via a temporary sibling and rename so a crash or cancellation never leaves a
/// truncated transcript at the final path (which `--resume` would then refuse to replace).
/// Without `overwrite`, the file is published with a hard link, which fails if the path exists,
/// so two processes cannot both pass an earlier existence check and replace each other's output.
/// How a transcript is rendered to a file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderOptions {
    pub(crate) format: OutputFormat,
    pub(crate) timestamps: TimestampGranularity,
    /// Plain text only: one `[Speaker N]` paragraph per speaker turn instead of the exact
    /// transcript text. Timestamped formats always carry speaker labels.
    pub(crate) speaker_turns: bool,
}

pub(crate) fn write_output(
    doc: &TranscriptDocument,
    path: &Path,
    options: RenderOptions,
    overwrite: bool,
) -> Result<()> {
    let content = render_output(doc, options)?;
    if path.file_name().is_none() {
        bail!("output path has no file name: {}", path.display());
    }
    // Fixed-length name: deriving it from a near-255-byte file name would exceed NAME_MAX.
    // One process writes one output at a time, so the PID keeps it unique.
    let temporary = path.with_file_name(format!(".transcriberrr-{}.tmp", std::process::id()));
    let result = std::fs::write(&temporary, content).and_then(|()| {
        if overwrite {
            std::fs::rename(&temporary, path)
        } else {
            // exFAT and some network volumes have no hard links; fall back to check-then-rename.
            std::fs::hard_link(&temporary, path).or_else(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    Err(error)
                } else if path.exists() {
                    Err(std::io::ErrorKind::AlreadyExists.into())
                } else {
                    std::fs::rename(&temporary, path)
                }
            })
        }
    });
    let _ = std::fs::remove_file(&temporary);
    match result {
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            bail!("output exists (use --overwrite): {}", path.display())
        }
        result => result.with_context(|| format!("writing {}", path.display())),
    }
}

pub(crate) fn render_output(doc: &TranscriptDocument, options: RenderOptions) -> Result<String> {
    let RenderOptions {
        format,
        timestamps,
        speaker_turns,
    } = options;
    doc.validate()?;
    if matches!(format, OutputFormat::Json) {
        return Ok(serde_json::to_string_pretty(doc)? + "\n");
    }
    let plain_text =
        matches!(format, OutputFormat::Txt) && timestamps == TimestampGranularity::None;
    if plain_text && speaker_turns && doc.segments.iter().any(|segment| segment.speaker.is_some()) {
        return Ok(speaker_turn_text(doc));
    }
    if plain_text {
        let text = doc.content();
        return Ok(if text.ends_with('\n') {
            text
        } else {
            text + "\n"
        });
    }
    let vtt = matches!(format, OutputFormat::Vtt);
    let mut output = if vtt {
        "WEBVTT\n\n".to_owned()
    } else {
        String::new()
    };
    let mut index = 0;
    for segment in &doc.segments {
        let cues = if timestamps == TimestampGranularity::Words && !segment.words.is_empty() {
            segment
                .words
                .iter()
                .map(|word| (word.start, word.end, word.text.clone(), word.speaker))
                .collect::<Vec<_>>()
        } else {
            vec![(
                segment.start,
                segment.end,
                segment.content(),
                segment.speaker,
            )]
        };
        for (start, end, text, speaker) in cues {
            index += 1;
            let speaker = speaker
                .map(|id| format!("[Speaker {id}] "))
                .unwrap_or_default();
            match format {
                OutputFormat::Txt => output.push_str(&format!(
                    "[{} --> {}] {}{}\n",
                    timestamp(start, true),
                    timestamp(end, true),
                    speaker,
                    text.trim()
                )),
                OutputFormat::Srt | OutputFormat::Vtt => {
                    if !vtt {
                        output.push_str(&format!("{index}\n"));
                    }
                    output.push_str(&format!(
                        "{} --> {}\n{}{}\n\n",
                        timestamp(start, vtt),
                        timestamp(end, vtt),
                        speaker,
                        // Blank lines delimit cues in SRT/WebVTT. Keep exact whitespace
                        // in JSON/plain text, and use a single line for subtitle bodies.
                        text.split_whitespace().collect::<Vec<_>>().join(" ")
                    ));
                }
                OutputFormat::Json => unreachable!(),
            }
        }
    }
    Ok(output)
}

/// Plain text with one paragraph per speaker turn, labeled like the timestamped formats.
/// Consecutive segments from the same speaker join into one paragraph.
fn speaker_turn_text(doc: &TranscriptDocument) -> String {
    let mut output = String::new();
    let mut previous: Option<Option<usize>> = None;
    for segment in &doc.segments {
        let text = segment.content();
        if previous == Some(segment.speaker) {
            output.push_str(&text);
            continue;
        }
        if previous.is_some() {
            output.truncate(output.trim_end().len());
            output.push('\n');
        }
        if let Some(id) = segment.speaker {
            output.push_str(&format!("[Speaker {id}] "));
        }
        output.push_str(text.trim_start());
        previous = Some(segment.speaker);
    }
    output.truncate(output.trim_end().len());
    output + "\n"
}

fn timestamp(seconds: f32, vtt: bool) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1000;
    let millis = total_ms % 1000;
    let separator = if vtt { '.' } else { ',' };
    format!("{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn words(tokens: &[&str]) -> Vec<Word> {
        tokens
            .iter()
            .enumerate()
            .map(|(index, text)| Word {
                text: (*text).to_owned(),
                start: index as f32,
                end: index as f32 + 0.5,
                speaker: Some(index / 2),
            })
            .collect()
    }

    #[test]
    fn restores_punctuation_and_exact_spacing_from_recognized_text() {
        let mut words = words(&["Hello", "world", "Dont", "panic"]);
        preserve_word_text("  “Hello,  world!”\nDon't panic…  ", &mut words).unwrap();
        assert_eq!(
            words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<String>(),
            "  “Hello,  world!”\nDon't panic…  "
        );
        let segments = build_segments(words);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].content(), "  “Hello,  world!”");
        assert_eq!(segments[1].content(), "\nDon't panic…  ");
        assert_eq!(segments[1].speaker, Some(1));
    }

    #[test]
    fn restores_unspaced_unicode_and_separate_punctuation_tokens() {
        let mut words = words(&["你好", "。", "再见"]);
        preserve_word_text("你好。再见！", &mut words).unwrap();
        assert_eq!(
            words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<String>(),
            "你好。再见！"
        );
        assert_eq!(words[1].text, "。");
    }

    #[test]
    fn refuses_unmatched_text_instead_of_losing_words() {
        for (text, tokens) in [
            ("Hello missing world.", vec!["Hello", "world"]),
            ("Hello world extra.", vec!["Hello", "world"]),
            ("Hello.", vec!["Hello", "extra"]),
        ] {
            assert!(preserve_word_text(text, &mut words(&tokens)).is_err());
        }
    }

    #[test]
    fn matches_non_ascii_letters_case_insensitively() {
        let mut words = words(&["école", "ÉTÉ"]);
        preserve_word_text("École été.", &mut words).unwrap();
        assert_eq!(words[0].text, "École");
        assert_eq!(words[1].text, " été.");
    }

    #[test]
    fn normalizes_out_of_order_and_overlong_word_timing() {
        let mut words = words(&["a", "b", "c", "d"]);
        words[2].start = 0.5; // starts before word 1 (1.0)
        words[2].end = 0.4; // ends before it starts
        words[3].start = f32::NAN;
        words[3].end = 99.0; // padded tail chunk beyond the audio
        normalize_word_timing(&mut words, 3.2);
        let mut last = 0.0;
        for word in &words {
            assert!(word.start >= last && word.end >= word.start && word.end <= 3.2);
            last = word.start;
        }
        assert_eq!(words[3].end, 3.2);
        let doc_words = words
            .into_iter()
            .map(|word| Word {
                speaker: None,
                ..word
            })
            .collect();
        let segment = segment_from_words(doc_words);
        let doc = TranscriptDocument {
            source: "a.wav".to_owned(),
            model: "m".to_owned(),
            device: "cpu".to_owned(),
            language: "English".to_owned(),
            duration_seconds: 3.2,
            segments: vec![segment],
            metadata: TranscriptMetadata {
                transcribed_with: "transcriberrr".to_owned(),
                version: "0".to_owned(),
                source_created_at: None,
                source_modified_at: None,
                transcribed_at: "2026-01-01T00:00:00Z".to_owned(),
                model: "m".to_owned(),
                diarization: None,
            },
        };
        doc.validate().expect("normalized timing must validate");
        let directory =
            std::env::temp_dir().join(format!("transcriberrr-write-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("out.txt");
        std::fs::write(&path, "stale").unwrap();
        write_output(&doc, &path, PLAIN, true).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "abcd\n");
        let error = write_output(&doc, &path, PLAIN, false)
            .expect_err("must not replace an existing output without overwrite");
        assert!(error.to_string().contains("output exists"));
        assert_eq!(
            std::fs::read_dir(&directory).unwrap().count(),
            1,
            "no temp files left"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    const PLAIN: RenderOptions = RenderOptions {
        format: OutputFormat::Txt,
        timestamps: TimestampGranularity::None,
        speaker_turns: false,
    };

    #[test]
    fn plain_text_speaker_turns_are_opt_in() {
        let segment = |text: &str, start: f32, speaker: Option<usize>| Segment {
            text: Some(text.to_owned()),
            start,
            end: start + 1.0,
            speaker,
            words: Vec::new(),
        };
        let mut doc = TranscriptDocument {
            source: "a.wav".to_owned(),
            model: "m".to_owned(),
            device: "cpu".to_owned(),
            language: "English".to_owned(),
            duration_seconds: 4.0,
            segments: vec![
                segment("Hello there.", 0.0, Some(0)),
                segment(" How are you?", 1.0, Some(0)),
                segment("\n Fine, thanks.", 2.0, Some(1)),
                segment(" Good.", 3.0, Some(0)),
            ],
            metadata: TranscriptMetadata {
                transcribed_with: "transcriberrr".to_owned(),
                version: "0".to_owned(),
                source_created_at: None,
                source_modified_at: None,
                transcribed_at: "2026-01-01T00:00:00Z".to_owned(),
                model: "m".to_owned(),
                diarization: None,
            },
        };
        let turns = RenderOptions {
            speaker_turns: true,
            ..PLAIN
        };
        assert_eq!(
            render_output(&doc, turns).unwrap(),
            "[Speaker 0] Hello there. How are you?\n[Speaker 1] Fine, thanks.\n[Speaker 0] Good.\n"
        );
        // Off by default: the exact transcript text.
        assert_eq!(
            render_output(&doc, PLAIN).unwrap(),
            "Hello there. How are you?\n Fine, thanks. Good.\n"
        );
        // Without diarization the option changes nothing.
        for segment in &mut doc.segments {
            segment.speaker = None;
        }
        assert_eq!(
            render_output(&doc, turns).unwrap(),
            render_output(&doc, PLAIN).unwrap()
        );
    }

    #[test]
    fn timestamps_round_across_hour_boundary() {
        assert_eq!(timestamp(3599.9998, false), "01:00:00,000");
    }
}
