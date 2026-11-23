//! Compact transcript schema and deterministic text/export operations.
use crate::{OutputFormat, TimestampGranularity};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    /// Display name for each speaker ID, when known (merged speaker tracks). Unnamed IDs are
    /// labeled `Speaker N`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) speaker_names: Vec<String>,
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
    /// For a merge of per-speaker tracks: each track and where it was placed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) merged_from: Vec<MergedTrack>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MergedTrack {
    pub(crate) source: String,
    pub(crate) speaker: usize,
    /// Seconds added to the track's timings (where the track's segment of a restarted Zoom
    /// recording starts).
    pub(crate) offset_seconds: f32,
    /// Length of this source part on the same clock, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) duration_seconds: Option<f32>,
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
    pub(crate) fn speaker_label(&self, id: usize) -> String {
        match self.speaker_names.get(id) {
            Some(name) if !name.is_empty() => name.clone(),
            _ => format!("Speaker {id}"),
        }
    }

    pub(crate) fn collapse_words(&mut self) {
        for segment in &mut self.segments {
            if segment.text.is_none() {
                segment.text = Some(segment.content());
            }
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
        for track in &self.metadata.merged_from {
            timing(0.0, track.offset_seconds)?;
            if let Some(duration) = track.duration_seconds {
                timing(0.0, duration)?;
            }
        }
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
            let ending = last
                .text
                .trim_end_matches(['”', '’', '"', '\'', ')', ']', '」', '』']);
            word.start - last.end > 1.0
                // Full-width terminators end a sentence outright. ASCII ones only when a space
                // follows: recognizers split "3.45pm" and "$5.50" into "3." + "45pm", which
                // otherwise cut segments (and subtitle cues) mid-number.
                || ending.ends_with(['。', '？', '！'])
                || (ending.ends_with(['.', '?', '!'])
                    && word.text.starts_with(char::is_whitespace))
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

/// The Qwen aligner's notion of a word character: Unicode letters and numbers only (its
/// `is_kept_char`, minus the apostrophe, which this matcher skips on both sides).
/// `char::is_alphanumeric` also counts combining vowel signs (Devanagari, Thai), which the
/// aligner drops, so matching failed whole files of such speech after transcription.
pub(crate) fn is_word_char(c: char) -> bool {
    use unicode_general_category::{GeneralCategory, get_general_category};
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
            | GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
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
        let mut letters = token.chars().filter(|&c| is_word_char(c)).peekable();
        let mut start = None;
        if letters.peek().is_none() {
            if token.is_empty() {
                bail!("empty aligned word");
            }
            let offset = text[cursor..]
                .find(token)
                .context("aligned punctuation does not match transcript")?;
            if text[cursor..cursor + offset].chars().any(is_word_char) {
                bail!("aligned punctuation skips transcript words");
            }
            start = Some(cursor + offset);
            cursor += offset + token.len();
        } else {
            for expected in letters {
                let (offset, actual) = text[cursor..]
                    .char_indices()
                    .find(|&(_, c)| is_word_char(c))
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
    if text[cursor..].chars().any(is_word_char) {
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
    /// Longest plausible duration of one aligned word.
    const MAX_WORD_SECONDS: f32 = 1.2;
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
        // The aligner can stretch a word's start back across a long pause (one word spanned
        // 18 s in noisy audio). No spoken word is that long; keep the end, which was right.
        word.start = clamp(word.start, previous_start);
        word.end = clamp(word.end, word.start);
        // Limit the audible interval. Padding beyond the audio must not push a valid
        // final word's start all the way to the end of the recording.
        if word.end - word.start > MAX_WORD_SECONDS {
            word.start = word.end - MAX_WORD_SECONDS;
        }
        previous_start = word.start;
    }
}

/// Words the aligner could not place in the audio: three or more in a row, all at one
/// instant (`normalize_word_timing` holds words that would start earlier at the previous
/// word's start). On the scraps of audio a muted speaker track leaves, Qwen3 invents whole
/// sentences ("The first two were released in 1998..."), and they all land there; real words
/// that close together come in pairs at most. Returns how many were dropped.
pub(crate) fn drop_unplaced_words(words: &mut Vec<Word>) -> usize {
    const RUN: usize = 3;
    let instant = |word: &Word| word.end - word.start < 0.005;
    let mut keep = vec![true; words.len()];
    let mut start = 0;
    while start < words.len() {
        let mut end = start + 1;
        if instant(&words[start]) {
            while end < words.len()
                && instant(&words[end])
                && words[end].start == words[start].start
            {
                end += 1;
            }
            if end - start >= RUN {
                keep[start..end].fill(false);
            }
        }
        start = end;
    }
    let before = words.len();
    let mut keep = keep.into_iter();
    words.retain(|_| keep.next().unwrap_or(true));
    before - words.len()
}

/// How a transcript is rendered to a file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderOptions {
    pub(crate) format: OutputFormat,
    pub(crate) timestamps: TimestampGranularity,
    /// Plain text only: one `[Speaker N]` paragraph per speaker turn instead of the exact
    /// transcript text. Timestamped formats always carry speaker labels.
    pub(crate) speaker_turns: bool,
}

/// Publish a completed temporary sibling atomically. Without `overwrite`, publication
/// must fail if another process has already created the destination.
pub(crate) fn write_output(
    doc: &TranscriptDocument,
    path: &Path,
    options: RenderOptions,
    overwrite: bool,
) -> Result<()> {
    doc.validate()?;
    publish_atomic(path, overwrite, |writer| {
        render_output_to(writer, doc, options)
    })
}

pub(crate) fn publish_atomic(
    path: &Path,
    overwrite: bool,
    write: impl FnOnce(&mut BufWriter<std::fs::File>) -> io::Result<()>,
) -> Result<()> {
    let Some(destination_name) = path.file_name() else {
        bail!("output path has no file name: {}", path.display());
    };
    let names_destination = |name: &str| {
        destination_name
            .to_str()
            .is_some_and(|destination| name.eq_ignore_ascii_case(destination))
    };
    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);
    static SWEPT: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());
    if let Some(folder) = path.parent()
        && let Ok(mut swept) = SWEPT.lock()
        && !swept.iter().any(|done| done == folder)
    {
        let directory = if folder.as_os_str().is_empty() {
            Path::new(".")
        } else {
            folder
        };
        crate::audio_convert::remove_stale_files(directory, |name| {
            if names_destination(name) {
                return None;
            }
            let owner = name.strip_prefix(".transcriberrr-")?.strip_suffix(".tmp")?;
            let pid = match owner.split_once('.') {
                Some((pid, serial)) => {
                    serial.parse::<usize>().ok()?;
                    pid
                }
                None => owner, // Older temporary names contained only the PID.
            };
            pid.parse().ok()
        });
        swept.push(folder.to_path_buf());
    }
    let (temporary, file) = loop {
        // Keep the name short even when the destination is near the file name length limit.
        let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let temporary_name = format!(".transcriberrr-{}.{serial}.tmp", std::process::id());
        if names_destination(&temporary_name) {
            continue;
        }
        let temporary = path.with_file_name(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("creating {}", temporary.display()));
            }
        }
    };
    let mut writer = BufWriter::new(file);
    let result = write(&mut writer).and_then(|()| writer.flush());
    drop(writer);
    let result = result.and_then(|()| {
        if overwrite {
            std::fs::rename(&temporary, path)
        } else {
            publish_without_overwrite(&temporary, path)
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

fn publish_without_overwrite(temporary: &Path, path: &Path) -> io::Result<()> {
    let result = std::fs::hard_link(temporary, path);
    #[cfg(target_os = "macos")]
    if let Err(error) = &result
        && error.kind() != io::ErrorKind::AlreadyExists
    {
        // Some volumes do not support hard links. Ask macOS for an exclusive rename.
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let source = CString::new(temporary.as_os_str().as_bytes())?;
        let destination = CString::new(path.as_os_str().as_bytes())?;
        // Both C strings live through the call. Their paths are relative to the same cwd.
        let status = unsafe {
            libc::renameatx_np(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                destination.as_ptr(),
                libc::RENAME_EXCL,
            )
        };
        return if status == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        };
    }
    result
}

#[cfg(test)]
pub(crate) fn render_output(doc: &TranscriptDocument, options: RenderOptions) -> Result<String> {
    doc.validate()?;
    let mut bytes = Vec::new();
    render_output_to(&mut bytes, doc, options)?;
    Ok(String::from_utf8(bytes)?)
}

fn render_output_to(
    output: &mut impl Write,
    doc: &TranscriptDocument,
    options: RenderOptions,
) -> io::Result<()> {
    let RenderOptions {
        format,
        timestamps,
        speaker_turns,
    } = options;
    if matches!(format, OutputFormat::Json) {
        serde_json::to_writer_pretty(&mut *output, doc)?;
        return output.write_all(b"\n");
    }
    if matches!(format, OutputFormat::Txt) && timestamps == TimestampGranularity::None {
        if speaker_turns && doc.segments.iter().any(|segment| segment.speaker.is_some()) {
            return output.write_all(speaker_turn_text(doc).as_bytes());
        }
        let mut ends_with_newline = false;
        for text in doc.segments.iter().flat_map(|segment| {
            segment
                .text
                .as_deref()
                .into_iter()
                .chain(segment.words.iter().map(|word| word.text.as_str()))
        }) {
            output.write_all(text.as_bytes())?;
            if !text.is_empty() {
                ends_with_newline = text.ends_with('\n');
            }
        }
        return if ends_with_newline {
            Ok(())
        } else {
            output.write_all(b"\n")
        };
    }
    let vtt = matches!(format, OutputFormat::Vtt);
    if vtt {
        output.write_all(b"WEBVTT\n\n")?;
    }
    let mut index = 0;
    let mut emit_cue = |start, end, text: &str, speaker: Option<usize>| {
        index += 1;
        let speaker = speaker
            .map(|id| format!("[{}] ", doc.speaker_label(id)))
            .unwrap_or_default();
        match format {
            OutputFormat::Txt => writeln!(
                output,
                "[{} --> {}] {}{}",
                timestamp(start, true),
                timestamp(end, true),
                speaker,
                text.split_whitespace().collect::<Vec<_>>().join(" ")
            ),
            OutputFormat::Srt | OutputFormat::Vtt => {
                let (start_ms, end_ms) = subtitle_interval(start, end);
                if !vtt {
                    writeln!(output, "{index}")?;
                }
                writeln!(
                    output,
                    "{} --> {}\n{}\n",
                    timestamp_millis(start_ms, vtt),
                    timestamp_millis(end_ms, vtt),
                    cue_text(&format!("{speaker}{text}"), vtt)
                )
            }
            OutputFormat::Json => unreachable!(),
        }
    };
    for segment in &doc.segments {
        if timestamps == TimestampGranularity::Words && !segment.words.is_empty() {
            for word in &segment.words {
                emit_cue(word.start, word.end, &word.text, word.speaker)?;
            }
        } else {
            emit_cue(
                segment.start,
                segment.end,
                &segment.content(),
                segment.speaker,
            )?;
        }
    }
    Ok(())
}

/// A subtitle cue's text. Blank lines delimit cues in SRT/WebVTT, so it is one line (JSON and
/// plain text keep exact whitespace); `-->` would read as a timing line, and WebVTT reads `<`
/// and `&` as markup ("a <b c" opened a bold tag that swallowed the rest).
fn cue_text(text: &str, vtt: bool) -> String {
    let line = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("-->", "->");
    if vtt {
        line.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    } else {
        line
    }
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
            output.push_str(&format!("[{}] ", doc.speaker_label(id)));
        }
        output.push_str(text.trim_start());
        previous = Some(segment.speaker);
    }
    output.truncate(output.trim_end().len());
    output + "\n"
}

fn timestamp(seconds: f32, vtt: bool) -> String {
    timestamp_millis(milliseconds(seconds), vtt)
}

fn milliseconds(seconds: f32) -> u64 {
    (f64::from(seconds.max(0.0)) * 1000.0).round() as u64
}

fn subtitle_interval(start: f32, end: f32) -> (u64, u64) {
    let start = milliseconds(start).min(u64::MAX - 1);
    (start, milliseconds(end).max(start + 1))
}

fn timestamp_millis(total_ms: u64, vtt: bool) -> String {
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
    fn drops_runs_of_words_the_aligner_could_not_place() {
        let mut words = words(&["So", "The", "first", "two", "were", "next", "a", "were"]);
        for (word, (start, end)) in words.iter_mut().zip([
            (1.0, 1.3),
            (1.3, 1.3),
            (1.3, 1.3),
            (1.3, 1.3),
            (1.3, 1.3),
            (2.0, 2.4),
            // A real pair at one instant stays.
            (3.0, 3.0),
            (3.0, 3.0),
        ]) {
            word.start = start;
            word.end = end;
        }
        assert_eq!(drop_unplaced_words(&mut words), 4);
        let text = words
            .iter()
            .map(|word| word.text.trim())
            .collect::<Vec<_>>();
        assert_eq!(text, ["So", "next", "a", "were"]);
    }

    #[test]
    fn caps_a_word_stretched_back_across_a_pause() {
        let mut words = words(&["Does", "anyone"]);
        words[0].start = 26.74;
        words[0].end = 45.22;
        words[1].start = 45.3;
        words[1].end = 45.7;
        normalize_word_timing(&mut words, 60.0);
        assert!((words[0].start - 44.02).abs() < 1e-3 && words[0].end == 45.22);
        assert!(words[1].start >= words[0].start);
    }

    #[test]
    fn padded_word_end_preserves_audible_final_word() {
        let mut words = words(&["end"]);
        words[0].start = 9.5;
        words[0].end = 15.0;
        normalize_word_timing(&mut words, 10.0);
        assert_eq!((words[0].start, words[0].end), (9.5, 10.0));
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
            speaker_names: Vec::new(),
            metadata: TranscriptMetadata {
                transcribed_with: "transcriberrr".to_owned(),
                version: "0".to_owned(),
                source_created_at: None,
                source_modified_at: None,
                transcribed_at: "2026-01-01T00:00:00Z".to_owned(),
                model: "m".to_owned(),
                diarization: None,
                merged_from: Vec::new(),
            },
        };
        doc.validate().expect("normalized timing must validate");
        let directory =
            std::env::temp_dir().join(format!("transcriberrr-write-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("out.txt");
        std::fs::write(&path, "stale").unwrap();
        let occupied = directory.join(format!(".transcriberrr-{}.tmp", std::process::id()));
        std::fs::write(&occupied, "unrelated").unwrap();
        write_output(&doc, &path, PLAIN, true).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "abcd\n");
        assert_eq!(std::fs::read_to_string(&occupied).unwrap(), "unrelated");
        std::fs::remove_file(&occupied).unwrap();
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
    fn subtitle_cues_escape_markup_and_timing_arrows() {
        assert_eq!(
            cue_text("[Alice <Mic>] Use x -->\n y & a <b", true),
            "[Alice &lt;Mic&gt;] Use x -&gt; y &amp; a &lt;b"
        );
        assert_eq!(cue_text("x --> y <b>", false), "x -> y <b>");
    }

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
            speaker_names: Vec::new(),
            metadata: TranscriptMetadata {
                transcribed_with: "transcriberrr".to_owned(),
                version: "0".to_owned(),
                source_created_at: None,
                source_modified_at: None,
                transcribed_at: "2026-01-01T00:00:00Z".to_owned(),
                model: "m".to_owned(),
                diarization: None,
                merged_from: Vec::new(),
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
    fn matches_devanagari_without_the_aligners_dropped_vowel_signs() {
        // The aligner strips combining marks: "मेरा" arrives as "मर".
        let mut words = words(&["नमसत", "मर", "नम"]);
        preserve_word_text("नमस्ते, मेरा नाम", &mut words).unwrap();
        assert_eq!(words[1].text, " मेरा");
        assert_eq!(words[2].text, " नाम");
    }

    #[test]
    fn sentences_do_not_split_inside_numbers() {
        let mut words = words(&["It's", "3.", "45pm.", "Next"]);
        for word in &mut words {
            word.speaker = None;
        }
        preserve_word_text("It's 3.45pm. Next", &mut words).unwrap();
        let segments = build_segments(words);
        assert_eq!(
            segments.iter().map(Segment::content).collect::<Vec<_>>(),
            vec!["It's 3.45pm.", " Next"]
        );
    }

    #[test]
    fn subtitle_intervals_survive_millisecond_rounding() {
        assert_eq!(subtitle_interval(1.0001, 1.0002), (1000, 1001));
        assert_eq!(subtitle_interval(5.0, 5.0), (5000, 5001));
        assert_eq!(timestamp(18_000.021, true), "05:00:00.021");
    }

    #[test]
    fn timestamps_round_across_hour_boundary() {
        assert_eq!(timestamp(3599.9998, false), "01:00:00,000");
    }
}
