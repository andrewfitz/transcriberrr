use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "transcriberrr-merge-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        let track = |source: &str, words: &str| {
            format!(
                r#"{{
          "source":"/zoom/Audio Record/{source}.m4a","model":"test","device":"cpu","language":"English","duration_seconds":10.0,
          "segments":[{words}],
          "metadata":{{"transcribed-with":"transcriberrr","version":"0.1.0","source_created_at":null,"source_modified_at":null,"transcribed_at":"2026-10-22T00:00:00Z","model":"test"}}
        }}"#
            )
        };
        fs::write(
            path.join("andrew.json"),
            track(
                "audioAndrewDaniel21067067067",
                r#"{"start":0.0,"end":1.0,"words":[
                  {"text":"Hello","start":0.0,"end":0.4},
                  {"text":" there.","start":0.5,"end":1.0}]},
                {"start":4.0,"end":5.0,"words":[{"text":" Shall we?","start":4.0,"end":5.0}]}"#,
            ),
        )
        .unwrap();
        fs::write(
            path.join("holly.json"),
            track(
                "audioHana31067067067",
                r#"{"start":0.6,"end":1.2,"words":[{"text":"Yeah.","start":0.6,"end":1.2}]}"#,
            ),
        )
        .unwrap();
        Self(path)
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
            .current_dir(&self.0)
            .arg("merge")
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
fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn merges_zoom_tracks_into_named_speaker_text() {
    let f = Fixture::new();
    let output = f.run(&[
        "andrew.json",
        "holly.json",
        "--format",
        "txt",
        "--timestamps",
        "sentences",
        "-o",
        "talk.txt",
    ]);
    success(&output);
    assert_eq!(
        f.read("talk.txt"),
        "[00:00:00.000 --> 00:00:01.000] [Andrew Daniel] Hello there.\n\
         [00:00:00.600 --> 00:00:01.200] [Hana] Yeah.\n\
         [00:00:04.000 --> 00:00:05.000] [Andrew Daniel] Shall we?\n"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Hana <- /zoom/Audio Record/audioHana31067067067.m4a"));
}

#[test]
fn word_timestamps_interleave_like_a_diarized_transcript() {
    let f = Fixture::new();
    success(&f.run(&[
        "andrew.json",
        "holly.json",
        "--format",
        "srt",
        "-o",
        "words.srt",
    ]));
    assert_eq!(
        f.read("words.srt"),
        "1\n00:00:00,000 --> 00:00:00,400\n[Andrew Daniel] Hello\n\n\
         2\n00:00:00,500 --> 00:00:01,000\n[Andrew Daniel] there.\n\n\
         3\n00:00:00,600 --> 00:00:01,200\n[Hana] Yeah.\n\n\
         4\n00:00:04,000 --> 00:00:05,000\n[Andrew Daniel] Shall we?\n\n"
    );
}

#[test]
fn merged_json_keeps_words_and_names_and_converts() {
    let f = Fixture::new();
    success(&f.run(&[
        "andrew.json",
        "holly.json",
        "--name",
        "Andrew",
        "-o",
        "talk.json",
    ]));
    let doc: serde_json::Value = serde_json::from_str(&f.read("talk.json")).unwrap();
    assert_eq!(doc["speaker_names"], serde_json::json!(["Andrew", "Hana"]));
    assert_eq!(doc["segments"][1]["words"][0]["text"], " Yeah.");
    assert_eq!(doc["segments"][1]["words"][0]["speaker"], 1);
    assert_eq!(doc["segments"][1]["speaker"], 1);
    assert_eq!(doc["metadata"]["diarization"]["detected_speakers"], 2);
    assert_eq!(doc["metadata"]["merged_from"][1]["speaker"], 1);
    // The merged transcript is an ordinary transcript: convert reads it and keeps the names.
    let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(&f.0)
        .args([
            "convert",
            "talk.json",
            "--timestamps",
            "none",
            "--speaker-turns",
            "-o",
            "turns.txt",
        ])
        .output()
        .unwrap();
    success(&output);
    assert_eq!(
        f.read("turns.txt"),
        "[Andrew] Hello there.\n[Hana] Yeah.\n[Andrew] Shall we?\n"
    );
}

#[test]
fn one_track_or_a_folder_finds_the_rest_of_the_session() {
    let f = Fixture::new();
    // Another meeting's track in the same folder must be left out.
    fs::write(
        f.0.join("other.json"),
        f.read("holly.json")
            .replace("audioHana31067067067", "audioSam12062062062"),
    )
    .unwrap();
    let output = f.run(&["holly.json", "-o", "talk.json"]);
    success(&output);
    let doc: serde_json::Value = serde_json::from_str(&f.read("talk.json")).unwrap();
    assert_eq!(
        doc["speaker_names"],
        serde_json::json!(["Andrew Daniel", "Hana"])
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("Zoom session 062062062"));

    // The folder now also holds talk.json, which must not be merged again.
    let output = f.run(&[".", "-o", "again.json"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("067067067 (Andrew Daniel, Hana)"),
        "{stderr}"
    );
    assert!(stderr.contains("062062062 (Sam)"), "{stderr}");
    let output = f.run(&[".", "--session", "1067067067", "-o", "again.json"]);
    success(&output);
    assert!(String::from_utf8_lossy(&output.stderr).contains("already a merged transcript"));
    assert!(
        !f.run(&[".", "--name", "A", "-o", "x.json"])
            .status
            .success()
    );
}

#[test]
fn refuses_existing_output() {
    let f = Fixture::new();
    success(&f.run(&["andrew.json", "holly.json", "-o", "talk.json"]));
    let again = f.run(&["andrew.json", "holly.json", "-o", "talk.json"]);
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("--overwrite"));
}
