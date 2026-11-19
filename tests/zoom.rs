use serde_json::Value;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "transcriberrr-zoom-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        Self(root)
    }

    fn file(&self, name: &str, contents: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, contents).unwrap();
        path
    }

    fn track(&self, name: &str) {
        // One second of PCM WAV. The decoder probes the header, so no media encoder is needed.
        let samples = vec![0u8; 32_000];
        let mut wav = Vec::from(b"RIFF".as_slice());
        wav.extend((36 + samples.len() as u32).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16u32.to_le_bytes());
        wav.extend(1u16.to_le_bytes());
        wav.extend(1u16.to_le_bytes());
        wav.extend(16_000u32.to_le_bytes());
        wav.extend(32_000u32.to_le_bytes());
        wav.extend(2u16.to_le_bytes());
        wav.extend(16u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((samples.len() as u32).to_le_bytes());
        wav.extend(samples);
        self.file(name, wav);
    }

    fn fake_native_helper(&self) -> PathBuf {
        self.file(
            "words.json",
            r#"{
              "source":"fake.wav","model":"test","device":"cpu","language":"English","duration_seconds":1.0,
              "segments":[{"start":0.1,"end":0.5,"words":[{"text":"Hello","start":0.1,"end":0.5}]}],
              "metadata":{"transcribed-with":"transcriberrr","version":"0.2.0","transcribed_at":"2026-11-18T00:00:00Z","model":"test"}
            }"#,
        );
        let helper = self.file(
            "speech-helper",
            "#!/bin/sh\nprintf 'called\\n' >> calls.log\nsleep 0.2\ncat words.json\n",
        );
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
        helper
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_transcriberrr"));
        command.current_dir(&self.0).args(args);
        command
    }

    fn plan(&self, args: &[&str]) -> Value {
        let output = self.command(args).arg("--plan").output().unwrap();
        success(&output);
        serde_json::from_slice(&output.stdout).unwrap()
    }

    fn calls(&self) -> usize {
        fs::read_to_string(self.0.join("calls.log"))
            .unwrap()
            .lines()
            .count()
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

fn destination(plan: &Value) -> &Path {
    Path::new(plan["meetings"][0]["output"].as_str().unwrap())
}

#[test]
fn plans_bulk_and_file_inputs_with_stable_destinations_and_id_matching() {
    let fixture = Fixture::new();
    for name in [
        "library/first/Same name/Audio Record/audioAlice11067067067.m4a",
        "library/first/Same name/Audio Record/audioBob21067067067.m4a",
        "library/first/Same name/audio1067067067.m4a",
        "library/first/Same name/video1067067067.mp4",
        "library/first/Same name/video1087087087.mp4",
        "library/first/Same name/unrelated.mp4",
        "library/second/Same name/Audio Record/audioCarol11067067067.m4a",
        "library/#recycle/Audio Record/audioOld11067067067.m4a",
    ] {
        fixture.file(name, []);
    }
    let from_bulk = fixture.plan(&["--input-dir", "library", "-o", "out", "--format", "txt"]);
    let from_file = fixture.plan(&[
        "library/first/Same name/video1067067067.mp4",
        "library/first/Same name/Audio Record/audioAlice11067067067.m4a",
        "-o",
        "out",
        "--format",
        "txt",
    ]);
    let from_folder = fixture.plan(&["library/first/Same name", "-o", "out", "--format", "txt"]);
    assert_eq!(from_bulk["meetings"].as_array().unwrap().len(), 2);
    assert_eq!(destination(&from_bulk), destination(&from_file));
    assert_eq!(destination(&from_folder), destination(&from_file));
    assert_ne!(
        from_bulk["meetings"][0]["output"],
        from_bulk["meetings"][1]["output"]
    );
    assert_eq!(from_file["meetings"][0]["participants"], 2);
    assert_eq!(from_file["files"].as_array().unwrap().len(), 4);
    for file in from_file["files"].as_array().unwrap() {
        assert_eq!(file["meeting"], from_file["meetings"][0]["id"]);
    }
    let from_directory = from_folder["files"].as_array().unwrap();
    for filename in ["video1087087087.mp4", "unrelated.mp4"] {
        assert!(from_directory.iter().any(
            |file| file["path"].as_str().unwrap().ends_with(filename)
                && file["meeting"].is_null()
                && file["skip"].is_null()
        ));
    }
}

#[test]
fn bulk_reruns_only_missing_outputs_and_defers_overlapping_runs() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    fixture.track("library/2026 Call/Audio Record/audioAlice11067067067.m4a");
    fixture.file("library/2026 Call/audio1067067067.m4a", []);
    fixture.file("library/2026 Call/video1067067067.mp4", []);
    let args = [
        "library",
        "-o",
        "out",
        "--engine",
        "apple-native",
        "--diarize",
    ];
    let first_destination = destination(&fixture.plan(&args)).to_path_buf();
    let start = || {
        fixture
            .command(&args)
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    let first = start();
    let overlapping = start();
    success(&first.wait_with_output().unwrap());
    success(&overlapping.wait_with_output().unwrap());
    assert_eq!(fixture.calls(), 1);
    let merged: Value = serde_json::from_slice(&fs::read(&first_destination).unwrap()).unwrap();
    assert_eq!(merged["speaker_names"], serde_json::json!(["Alice"]));
    assert_eq!(
        merged["metadata"]["merged_from"].as_array().unwrap().len(),
        1
    );
    assert_eq!(
        fs::read_dir(first_destination.parent().unwrap())
            .unwrap()
            .count(),
        1
    );
    assert_eq!(fixture.plan(&args)["meetings"][0]["done"], true);

    fixture.track("library/2026 Call/Audio Record/audioBob21067067067.m4a");
    fixture.track("library/2026 Review/audio1187878787.m4a");
    let fallback_args = ["library", "-o", "out", "--engine", "apple-native"];
    success(
        &fixture
            .command(&fallback_args)
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .output()
            .unwrap(),
    );
    assert_eq!(
        fixture.calls(),
        2,
        "only the new mixed-only meeting is transcribed"
    );

    // The destination determines completion, even after the participant files changed.
    assert_eq!(
        fs::read_dir(first_destination.parent().unwrap())
            .unwrap()
            .count(),
        1
    );
    fs::remove_file(&first_destination).unwrap();
    success(
        &fixture
            .command(&fallback_args)
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .output()
            .unwrap(),
    );
    assert_eq!(
        fixture.calls(),
        4,
        "both participant tracks are needed to recreate the merge"
    );
    let merged: Value = serde_json::from_slice(&fs::read(&first_destination).unwrap()).unwrap();
    assert_eq!(merged["speaker_names"], serde_json::json!(["Alice", "Bob"]));
}

#[test]
fn an_explicitly_missing_participant_cannot_produce_a_partial_merge() {
    let fixture = Fixture::new();
    let alice = "meeting/Audio Record/audioAlice11067067067.m4a";
    let bob = "meeting/Audio Record/audioBob21067067067.m4a";
    fixture.file(alice, []);
    fixture.file("meeting/audio1067067067.m4a", []);
    fixture.file("meeting/video1067067067.mp4", []);
    let plan = fixture.plan(&[alice, bob]);
    assert!(plan["meetings"].as_array().unwrap().is_empty());
    let files = plan["files"].as_array().unwrap();
    assert_eq!(files.len(), 4);
    assert!(files.iter().all(|file| file["error"].is_string()));
}

#[test]
fn ambiguous_and_unconverted_exports_do_not_block_other_meetings() {
    let fixture = Fixture::new();
    for file in [
        "library/Shared/Audio Record/audioAlice11067067067.wav",
        "library/Shared/Audio Record/audioBob11084084084.wav",
        "library/Shared/zoom_0.mp4",
        "library/Converting/audio1067067067.m4a",
        "library/Converting/double_click_to_convert_01.zoom",
    ] {
        fixture.file(file, []);
    }
    let plan = fixture.plan(&["library"]);
    assert_eq!(plan["meetings"].as_array().unwrap().len(), 2);
    let files = plan["files"].as_array().unwrap();
    let error_for = |suffix: &str| {
        files
            .iter()
            .find(|file| file["path"].as_str().unwrap().ends_with(suffix))
            .unwrap()["error"]
            .as_str()
            .unwrap()
    };
    assert!(error_for("Shared/zoom_0.mp4").contains("several meetings"));
    assert!(error_for("Converting/audio1067067067.m4a").contains("still converting"));
}

#[test]
fn keeps_every_fallback_segment_and_chooses_one_view() {
    let fixture = Fixture::new();
    let audio = "Call/audio1067067067.wav";
    let later_video = "Call/video2067067067.mp4";
    let alternate = "Call/shared_screen_with_speaker_view2067067067.mp4";
    for name in [audio, later_video, alternate] {
        fixture.track(name);
    }
    let helper = fixture.fake_native_helper();
    let args = ["Call", "-o", "out", "--engine", "apple-native"];
    let plan = fixture.plan(&args);
    let files = plan["files"].as_array().unwrap();
    let primary = files
        .iter()
        .find(|file| file["output"].is_string())
        .unwrap();
    let output = primary["output"].as_str().unwrap();
    assert!(primary["path"].as_str().unwrap().ends_with(audio));
    let reason_for = |name: &str| {
        files
            .iter()
            .find(|file| file["path"].as_str().unwrap().ends_with(name))
            .unwrap()["skip"]
            .as_str()
            .unwrap()
    };
    assert!(reason_for(later_video).contains("continues"));
    assert!(reason_for(alternate).contains("same recording segment"));
    success(
        &fixture
            .command(&args)
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .output()
            .unwrap(),
    );
    assert_eq!(fixture.calls(), 1);
    assert!(Path::new(output).is_file());
    fixture.track("Call/audio2067067067.wav");
    let later_plan = fixture.plan(&[later_video, "-o", "out"]);
    assert_eq!(later_plan["files"][0]["output"], output);
    assert_eq!(later_plan["files"][0]["done"], true);
}

#[test]
fn recognizes_mixed_mp4_and_leaves_generic_short_video_names_alone() {
    let fixture = Fixture::new();
    fixture.file("Zoom/Audio Record/audioAlice11067067067.wav", []);
    fixture.file("Zoom/audio1067067067.mp4", []);
    fixture.file("Zoom/._double_click_to_convert_01.zoom", []);
    fixture.file("Other/video1.mp4", []);
    let plan = fixture.plan(&["--input-dir", "."]);
    let files = plan["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(plan["meetings"].as_array().unwrap().len(), 1);
    assert!(files.iter().any(|file| {
        file["path"]
            .as_str()
            .unwrap()
            .ends_with("audio1067067067.mp4")
            && file["skip"]
                .as_str()
                .is_some_and(|skip| skip.contains("participant tracks"))
    }));
}

#[test]
fn rejects_overlapping_progress_logs_before_creating_or_appending_to_them() {
    let fixture = Fixture::new();
    let source = fixture.file(
        "Call/Audio Record/audioAlice11067067067.wav",
        "original audio",
    );
    let plan = fixture.plan(&["Call", "--output-location", "next-to-source"]);
    let output = destination(&plan);
    let args = [
        "Call",
        "--output-location",
        "next-to-source",
        "--engine",
        "apple-native",
        "--progress-file",
        output.to_str().unwrap(),
    ];
    let verify_rejected = |args| {
        let result = fixture.command(args).output().unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("progress file overlaps"));
    };
    verify_rejected(&args[..]);
    assert!(!output.exists());
    let relative = output
        .strip_prefix(fixture.0.canonicalize().unwrap())
        .unwrap();
    let alias = Path::new("missing-folder/..").join(relative);
    let mut alias_args = args;
    alias_args[6] = alias.to_str().unwrap();
    verify_rejected(&alias_args);
    assert!(!output.exists());
    assert!(!fixture.0.join("missing-folder").exists());
    fs::write(output, b"saved result").unwrap();
    verify_rejected(&args[..]);
    assert_eq!(fs::read(output).unwrap(), b"saved result");

    let alias = fixture.0.join("progress.jsonl");
    fs::hard_link(&source, &alias).unwrap();
    verify_rejected(&[
        "Call",
        "--engine",
        "apple-native",
        "--progress-file",
        alias.to_str().unwrap(),
    ]);
    assert_eq!(fs::read(source).unwrap(), b"original audio");
}

#[test]
fn duplicate_selections_cannot_bypass_a_conversion_error() {
    let fixture = Fixture::new();
    let audio = "Call/audio1067067067.m4a";
    fixture.file(audio, []);
    fixture.file("Call/double_click_to_convert_01.zoom", []);
    let plan = fixture.plan(&["Call", audio, "./Call/audio1067067067.m4a"]);
    let files = plan["files"].as_array().unwrap();
    assert_eq!(files.len(), 1);
    assert!(
        files[0]["error"]
            .as_str()
            .unwrap()
            .contains("still converting")
    );
}

#[test]
fn explicit_manifest_rejections_and_exclusions_survive_discovery() {
    let fixture = Fixture::new();
    fixture.file("Call/audio1067067067.m4a", []);
    fixture.file("Call/video1067067067.mp4", []);
    fixture.file(
        "manifest.json",
        r#"{"items":[
        {"input":"Call/video1067067067.mp4"},
        {"input":"Call/audio1067067067.m4a","speakers":900}
    ]}"#,
    );
    let plan = fixture.plan(&["--manifest", "manifest.json"]);
    let files = plan["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    assert!(files.iter().any(|file| {
        file["error"]
            .as_str()
            .is_some_and(|reason| reason.contains("invalid speakers"))
    }));

    fixture.file(
        "manifest.json",
        r#"{
      "excluded_inputs":["Call/audio1067067067.m4a"],
      "items":[{"input":"Call/video1067067067.mp4"}] 
    }"#,
    );
    let plan = fixture.plan(&["--manifest", "manifest.json"]);
    assert_eq!(plan["files"].as_array().unwrap().len(), 1);
    assert!(plan["files"][0]["path"].as_str().unwrap().ends_with(".mp4"));

    fixture.file("Call/Audio Record/audioAlice11067067067.m4a", []);
    fixture.file("Call/Audio Record/audioCarol31067067067.m4a", []);
    fixture.file(
        "manifest.json",
        r#"{"discover_zoom":false,"items":[
        {"input":"Call/Audio Record/audioAlice11067067067.m4a"},
        {"input":"Call/Audio Record/audioBob21067067067.m4a","iso_track":false}
    ]}"#,
    );
    let plan = fixture.plan(&["--manifest", "manifest.json"]);
    assert_eq!(plan["meetings"][0]["participants"], 1);
    let files = plan["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    assert!(files.iter().any(|file| file["error"].is_string()));
}

#[test]
fn preview_honors_resume_and_zoom_overwrite_forces_a_rerun() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    fixture.track("ordinary.wav");
    fixture.track("Call/Audio Record/audioAlice11067067067.wav");
    let args = [
        "ordinary.wav",
        "Call",
        "--engine",
        "apple-native",
        "-o",
        "out",
        "--progress-file",
        "progress.jsonl",
    ];
    let run = |flags: &[&str]| {
        fixture
            .command(&args)
            .args(flags)
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .output()
            .unwrap()
    };
    success(&run(&[]));
    assert_eq!(fixture.calls(), 2);
    let log = fs::read(fixture.0.join("progress.jsonl")).unwrap();
    let plan = fixture.plan(&[&args[..], &["--resume"]].concat());
    assert!(
        plan["files"]
            .as_array()
            .unwrap()
            .iter()
            .all(|file| file["done"] == true)
    );
    assert_eq!(fs::read(fixture.0.join("progress.jsonl")).unwrap(), log);
    success(&run(&["--resume", "--overwrite"]));
    assert_eq!(
        fixture.calls(),
        3,
        "overwrite forces the Zoom merge even with a resume log"
    );
}

#[test]
fn frozen_manifest_cannot_turn_a_lost_recording_part_into_a_complete_transcript() {
    let fixture = Fixture::new();
    let audio = "Call/audio1067067067.m4a";
    let later_video = "Call/video2067067067.mp4";
    fixture.track(audio);
    fixture.track(later_video);
    fixture.file(
        "manifest.json",
        serde_json::to_vec(&serde_json::json!({
            "discover_zoom": false,
            "items": [{ "input": audio }, { "input": later_video }],
        }))
        .unwrap(),
    );
    let plan = fixture.plan(&["--manifest", "manifest.json"]);
    assert_eq!(plan["files"].as_array().unwrap().len(), 2);
    assert!(plan["files"][0]["output"].is_string());
    fs::remove_file(fixture.0.join(later_video)).unwrap();
    let plan = fixture.plan(&["--manifest", "manifest.json"]);
    assert_eq!(plan["files"].as_array().unwrap().len(), 2);
    assert!(
        plan["files"]
            .as_array()
            .unwrap()
            .iter()
            .all(|file| file["error"].is_string())
    );
}

#[test]
fn manual_speaker_video_rewrites_its_intermediate_on_merge_retry() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    let video = "Call/video1067067067.mp4";
    fixture.track(video);
    fixture.track("Call/alice.wav");
    fixture.file("Call/audio1067067067.m4a", []);
    fixture.file(
        "manifest.json",
        r#"{"items":[
        {"input":"Call/video1067067067.mp4","iso_track":true},
        {"input":"Call/alice.wav","iso_track":true}
    ]}"#,
    );
    let args = [
        "--manifest",
        "manifest.json",
        "-o",
        "out",
        "--engine",
        "apple-native",
        "--progress-file",
        "progress.jsonl",
        "--resume",
    ];
    let plan = fixture.plan(&args);
    let files = plan["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    let video_plan = files
        .iter()
        .find(|file| file["path"].as_str().unwrap().ends_with(video))
        .unwrap();
    assert_eq!(video_plan["zoom_recording"], false);
    let intermediate = Path::new(video_plan["output"].as_str().unwrap());
    fs::create_dir_all(intermediate.parent().unwrap()).unwrap();
    fs::write(intermediate, b"interrupted transcript").unwrap();
    success(
        &fixture
            .command(&args)
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .output()
            .unwrap(),
    );
    assert_eq!(fixture.calls(), 2);
    assert!(fixture.0.join("out/Call.merged.json").is_file());
}

#[test]
fn inventory_preview_lists_ordinary_inputs_before_the_app_assigns_outputs() {
    let fixture = Fixture::new();
    fixture.file("library/One/talk.m4a", []);
    fixture.file("library/Two/talk.m4a", []);
    let inventory = fixture.plan(&["library", "--inventory-only"]);
    assert!(
        inventory["files"]
            .as_array()
            .unwrap()
            .iter()
            .all(|file| file["error"].is_null())
    );
    let checked = fixture.plan(&["library"]);
    assert!(
        checked["files"]
            .as_array()
            .unwrap()
            .iter()
            .all(|file| file["error"].is_string())
    );
}

#[test]
fn adding_an_equivalent_export_keeps_the_existing_fallback_destination() {
    let fixture = Fixture::new();
    let options = ["--output-location", "next-to-source"];
    fixture.track("Call/zoom_0.mp4");
    let plan = fixture.plan(&[&["Call"][..], &options].concat());
    let saved = Path::new(plan["files"][0]["output"].as_str().unwrap());
    fs::write(saved, b"saved id-less result").unwrap();
    fixture.track("Call/audio0067067067.m4a");
    let plan = fixture.plan(&[&["Call"][..], &options].concat());
    assert_eq!(plan["files"][0]["output"], saved.to_str().unwrap());
    assert_eq!(plan["files"][0]["done"], true);

    fixture.track("Other/video1067067067.mp4");
    let legacy = fixture.file("Other/video1067067067.json", "saved old video result");
    fixture.track("Other/audio1067067067.m4a");
    let plan = fixture.plan(&[&["Other"][..], &options].concat());
    assert_eq!(
        plan["files"][0]["output"],
        legacy.canonicalize().unwrap().to_str().unwrap()
    );
    assert_eq!(plan["files"][0]["done"], true);
}

#[test]
fn frozen_fallback_retains_the_meeting_id_when_an_equivalent_legacy_export_survives() {
    let fixture = Fixture::new();
    let mixed = "Call/audio0067067067.m4a";
    fixture.track(mixed);
    fixture.track("Call/zoom_0.mp4");
    fixture.file(
        "manifest.json",
        r#"{"discover_zoom":false,"items":[
            {"input":"Call/audio0067067067.m4a"},
            {"input":"Call/zoom_0.mp4"}
        ]}"#,
    );
    let args = [
        "--manifest",
        "manifest.json",
        "--output-location",
        "next-to-source",
    ];
    let before = fixture.plan(&args);
    let output = &before["files"][0]["output"];
    assert!(output.as_str().unwrap().contains("[067067067]"));
    fs::remove_file(fixture.0.join(mixed)).unwrap();
    let after = fixture.plan(&args);
    assert_eq!(after["files"][0]["output"], *output);
}

#[test]
fn protects_a_transcript_from_an_equivalent_unicode_progress_filename() {
    let fixture = Fixture::new();
    fixture.track("Café/Audio Record/audioAlice11067067067.wav");
    let plan = fixture.plan(&["Café", "--output-location", "next-to-source"]);
    let output = destination(&plan);
    let filename = output.file_name().unwrap().to_str().unwrap();
    let decomposed = output.with_file_name(filename.replace("é", "e\u{301}"));
    let result = fixture
        .command(&[
            "Café",
            "--output-location",
            "next-to-source",
            "--engine",
            "apple-native",
            "--progress-file",
            decomposed.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("progress file overlaps"));
    assert!(!output.exists() && !decomposed.exists());
}
