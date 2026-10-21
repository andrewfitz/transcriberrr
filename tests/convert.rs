use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "transcriberrr-convert-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("input.json"), r#"{
          "source":"meeting.wav","model":"test","device":"cpu","language":"English","duration_seconds":3.0,
          "segments":[
            {"start":0.0,"end":1.0,"speaker":1,"words":[
              {"text":"“Hello,","start":0.0,"end":0.4,"speaker":1},
              {"text":" world!”","start":0.5,"end":1.0,"speaker":1}]},
            {"start":2.0,"end":3.0,"speaker":2,"words":[
              {"text":"\nYes—it's","start":2.0,"end":2.5,"speaker":2},
              {"text":" me.","start":2.6,"end":3.0,"speaker":2}]}],
          "metadata":{"transcribed-with":"transcriberrr","version":"0.1.0","source_created_at":null,"source_modified_at":null,"transcribed_at":"2026-10-17T00:00:00Z","model":"test"}
        }"#).unwrap();
        Self(path)
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
            .current_dir(&self.0)
            .arg("convert")
            .arg("input.json")
            .args(args)
            .output()
            .unwrap()
    }
    fn read(&self, name: &str) -> String {
        fs::read_to_string(self.0.join(name)).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn success(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn converts_word_json_to_timestamped_speaker_text_without_models() {
    let f = Fixture::new();
    success(f.run(&["--format", "txt", "-o", "out.txt"]));
    assert_eq!(
        f.read("out.txt"),
        "[00:00:00.000 --> 00:00:01.000] [Speaker 1] “Hello, world!”\n[00:00:02.000 --> 00:00:03.000] [Speaker 2] Yes—it's me.\n"
    );
}

#[test]
fn plain_text_reconstitutes_exact_spacing_and_punctuation() {
    let f = Fixture::new();
    success(f.run(&["--format", "txt", "--timestamps", "none", "-o", "plain.txt"]));
    assert_eq!(f.read("plain.txt"), "“Hello, world!”\nYes—it's me.\n");
}

#[test]
fn plain_text_can_label_speaker_turns() {
    let f = Fixture::new();
    success(f.run(&[
        "--format",
        "txt",
        "--timestamps",
        "none",
        "--speaker-turns",
        "-o",
        "turns.txt",
    ]));
    assert_eq!(
        f.read("turns.txt"),
        "[Speaker 1] “Hello, world!”\n[Speaker 2] Yes—it's me.\n"
    );
}

#[test]
fn converts_to_sentence_and_word_subtitles() {
    let f = Fixture::new();
    success(f.run(&["--format", "srt", "-o", "out.srt"]));
    assert_eq!(
        f.read("out.srt"),
        "1\n00:00:00,000 --> 00:00:01,000\n[Speaker 1] “Hello, world!”\n\n2\n00:00:02,000 --> 00:00:03,000\n[Speaker 2] Yes—it's me.\n\n"
    );
    success(f.run(&["--format", "vtt", "--timestamps", "words", "-o", "out.vtt"]));
    let vtt = f.read("out.vtt");
    assert!(vtt.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:00.400\n[Speaker 1] “Hello,\n\n"));
    assert!(vtt.contains("[Speaker 2] me."));
}

#[test]
fn json_stores_text_once_at_requested_granularity() {
    let f = Fixture::new();
    success(f.run(&[
        "--format",
        "json",
        "--timestamps",
        "words",
        "-o",
        "words.json",
    ]));
    let doc: serde_json::Value = serde_json::from_str(&f.read("words.json")).unwrap();
    assert!(doc.get("text").is_none());
    assert!(doc["segments"][0].get("text").is_none());
    assert_eq!(doc["segments"][1]["words"][0]["text"], "\nYes—it's");
    success(f.run(&["--format", "json", "-o", "segments.json"]));
    let doc: serde_json::Value = serde_json::from_str(&f.read("segments.json")).unwrap();
    assert!(doc.get("text").is_none());
    assert_eq!(doc["segments"][0]["text"], "“Hello, world!”");
    assert!(doc["segments"][0].get("words").is_none());
}

#[test]
fn refuses_overwrite_and_invalid_timings_without_writing() {
    let f = Fixture::new();
    fs::write(f.0.join("out.txt"), "keep").unwrap();
    assert!(!f.run(&["-o", "out.txt"]).status.success());
    assert_eq!(f.read("out.txt"), "keep");
    success(f.run(&["-o", "out.txt", "--overwrite"]));
    let input = f.read("input.json").replace("\"end\":0.4", "\"end\":-1.0");
    fs::write(f.0.join("input.json"), input).unwrap();
    assert!(!f.run(&["-o", "invalid.txt"]).status.success());
    assert!(!f.0.join("invalid.txt").exists());
}

#[test]
fn segment_json_roundtrip_keeps_transcript_and_metadata() {
    let f = Fixture::new();
    success(f.run(&["--format", "json", "-o", "segments.json"]));
    let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(&f.0)
        .args([
            "convert",
            "segments.json",
            "--timestamps",
            "none",
            "-o",
            "plain.txt",
        ])
        .output()
        .unwrap();
    success(output);
    assert_eq!(f.read("plain.txt"), "“Hello, world!”\nYes—it's me.\n");
    let doc: serde_json::Value = serde_json::from_str(&f.read("segments.json")).unwrap();
    assert_eq!(doc["metadata"]["transcribed_at"], "2026-10-17T00:00:00Z");
    assert_eq!(doc["segments"][1]["speaker"], 2);
    let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(&f.0)
        .args([
            "convert",
            "segments.json",
            "--timestamps",
            "words",
            "-o",
            "impossible.txt",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!f.0.join("impossible.txt").exists());
}

#[test]
fn invalid_speaker_and_duplicate_text_are_rejected() {
    let f = Fixture::new();
    let original = f.read("input.json");
    for input in [
        original.replacen("\"speaker\":1}", "\"speaker\":2}", 1),
        original.replacen(
            "\"start\":0.0,\"end\":1.0",
            "\"text\":\"duplicate\",\"start\":0.0,\"end\":1.0",
            1,
        ),
        original.replacen("\"start\":2.0,\"end\":3.0", "\"start\":-1.0,\"end\":3.0", 1),
    ] {
        fs::write(f.0.join("input.json"), input).unwrap();
        assert!(!f.run(&["-o", "invalid.txt"]).status.success());
        assert!(!f.0.join("invalid.txt").exists());
    }
}

#[test]
fn default_destination_preserves_source() {
    let f = Fixture::new();
    let original = f.read("input.json");
    success(f.run(&[]));
    assert!(f.0.join("input.converted.txt").is_file());
    assert_eq!(f.read("input.json"), original);
}

#[test]
fn apple_native_cli_uses_compact_json_and_shared_writer() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new();
    let helper = f.0.join("native-helper");
    fs::write(&helper, "#!/bin/sh\ncat \"$(dirname \"$0\")/input.json\"\n").unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(f.0.join("audio.wav"), []).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(&f.0)
        .env("TRANSCRIBERRR_NATIVE", &helper)
        .args([
            "audio.wav",
            "--engine",
            "apple-native",
            "--format",
            "txt",
            "--timestamps",
            "sentences",
        ])
        .output()
        .unwrap();
    success(output);
    assert_eq!(
        f.read("audio.txt"),
        "[00:00:00.000 --> 00:00:01.000] [Speaker 1] “Hello, world!”\n[00:00:02.000 --> 00:00:03.000] [Speaker 2] Yes—it's me.\n"
    );
}

#[test]
fn blank_lines_in_transcript_do_not_split_subtitle_cues() {
    let f = Fixture::new();
    let input = f.read("input.json").replace(" world!", "\\n\\nworld!");
    fs::write(f.0.join("input.json"), input).unwrap();
    for (format, output) in [("srt", "out.srt"), ("vtt", "out.vtt")] {
        success(f.run(&["--format", format, "-o", output]));
        let content = f.read(output);
        let body = content.strip_prefix("WEBVTT\n\n").unwrap_or(&content);
        assert_eq!(
            body.trim_end().split("\n\n").count(),
            2,
            "{format} must contain two complete cues"
        );
        assert!(body.contains("[Speaker 1] “Hello, world!”"));
    }
    success(f.run(&["--format", "txt", "--timestamps", "none", "-o", "plain.txt"]));
    assert_eq!(f.read("plain.txt"), "“Hello,\n\nworld!”\nYes—it's me.\n");
}
