use serde_json::Value;
use std::fs;
use std::io::Write;
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
            "#!/bin/sh\nprintf 'called\\n' >> calls.log\nprintf '%s\\n' \"$@\" >> helper-arguments.log\nsleep 0.2\ncat words.json\n",
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

fn json_lines(output: &Output) -> Vec<Value> {
    std::str::from_utf8(&output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn capabilities_and_stdin_plan_are_read_only_and_report_offline_requirements() {
    let fixture = Fixture::new();
    fixture.track("voice memo.qta");
    let capabilities = fixture.command(&["capabilities"]).output().unwrap();
    success(&capabilities);
    let capabilities = &json_lines(&capabilities)[0];
    assert_eq!(capabilities["schema_version"], 1);
    assert!(
        capabilities["input_extensions"]
            .as_array()
            .unwrap()
            .contains(&Value::from("qta"))
    );
    assert_eq!(capabilities["automation"]["events"], "--jsonl");
    assert_eq!(
        capabilities["engines"][1]["included"],
        cfg!(feature = "parakeet")
    );

    let mut child = fixture
        .command(&[
            "--manifest",
            "-",
            "--plan",
            "--offline",
            "--model-cache",
            "cold-cache",
            "-o",
            "out",
            "--progress-file",
            "events/run.jsonl",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"items":[{"input":"voice memo.qta","output":"chosen.json"}]}"#)
        .unwrap();
    let plan = child.wait_with_output().unwrap();
    success(&plan);
    let plan = &json_lines(&plan)[0];
    assert_eq!(plan["summary"]["pending_units"], 1);
    assert!(
        plan["files"][0]["output"]
            .as_str()
            .unwrap()
            .ends_with("/chosen.json")
    );
    assert_eq!(plan["required_models"][0]["role"], "asr");
    assert_eq!(plan["required_models"][1]["role"], "aligner");
    assert_eq!(plan["required_models"][0]["cached"], false);
    assert!(
        plan["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue.as_str().unwrap().contains("--offline"))
    );
    for path in ["cold-cache", "chosen.json", "out", "events"] {
        assert!(!fixture.0.join(path).exists());
    }

    fixture.file(
        "typo.json",
        br#"{"items":[{"input":"voice memo.qta","diarzie":true}]}"#,
    );
    let output = fixture
        .command(&["--manifest", "typo.json", "--plan"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown field"));
}

#[test]
fn jsonl_tracks_multiple_passes_and_partial_failures() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    fixture.track("meeting/Audio Record/audioAlice11067067067.wav");
    fixture.track("meeting/Audio Record/audioBob21067067067.wav");
    fixture.file("meeting/video1067067067.mp4", []);
    let args = [
        "meeting",
        "missing.wav",
        "-o",
        "out",
        "--offline",
        "--pass",
        "qwen3",
        "--pass",
        "apple-native",
        "--model-cache",
        "cold-cache",
        "--progress-file",
        "run.jsonl",
    ];
    let plan = fixture.plan(&args);
    let qwen = destination(&plan["passes"][0]);
    let apple = destination(&plan["passes"][1]);
    fixture.file(qwen.to_str().unwrap(), "existing Qwen output");
    let result = fixture
        .command(&args)
        .arg("--jsonl")
        .env("TRANSCRIBERRR_NATIVE", &helper)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let events = json_lines(&result);
    assert!(events.iter().all(|event| event["schema_version"] == 1));
    assert_eq!(events[0]["event"], "run_started");
    assert_eq!(events.last().unwrap()["event"], "run_finished");
    assert_eq!(events.last().unwrap()["status"], "failed");
    assert!(events.iter().any(|event| event["kind"] == "merge"
        && event["status"] == "skipped"
        && event["pass"]["model_tag"] == "qw"
        && event["output"] == qwen.to_str().unwrap()));
    assert!(events.iter().any(|event| event["kind"] == "merge"
        && event["status"] == "completed"
        && event["pass"]["index"] == 2
        && event["output"] == apple.to_str().unwrap()));
    assert!(events.iter().any(|event| event["event"] == "progress"
        && event["status"] == "failed"
        && event["input"].as_str().unwrap().ends_with("/missing.wav")));
    let log = fs::read_to_string(fixture.0.join("run.ap.jsonl")).unwrap();
    assert!(log.lines().all(|line| {
        serde_json::from_str::<Value>(line)
            .unwrap()
            .get("event")
            .is_none()
    }));
    assert_eq!(fixture.calls(), 2);
    assert!(
        fs::read_to_string(fixture.0.join("helper-arguments.log"))
            .unwrap()
            .lines()
            .any(|arg| arg == "--offline")
    );
    assert!(!fixture.0.join("cold-cache").exists());
}

#[test]
fn offline_run_reports_missing_model_without_creating_a_cache() {
    let fixture = Fixture::new();
    fixture.track("test.wav");
    let result = fixture
        .command(&[
            "test.wav",
            "--jsonl",
            "--offline",
            "--model-cache",
            "cold-cache",
            "-o",
            "out",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let events = json_lines(&result);
    assert_eq!(events[2]["event"], "pass_finished");
    assert!(
        events[2]["error"]
            .as_str()
            .unwrap()
            .contains("not available locally")
    );
    assert_eq!(events[3]["status"], "failed");
    assert!(!fixture.0.join("cold-cache").exists());
    assert!(!fixture.0.join("out").exists());
}

#[test]
fn model_pass_plans_tag_regular_and_zoom_destinations() {
    let fixture = Fixture::new();
    fixture.track("library/interview.2026.wav");
    fixture.track("library/Call/Audio Record/audioAlice11067067067.m4a");
    fixture.file("library/Call/video1067067067.mp4", []);
    fixture.track("library/Review/audio1187878787.m4a");
    let untagged = fixture.plan(&["library", "-o", "out"]);
    assert!(destination(&untagged).ends_with("Call.merged.json"));
    let plan = fixture.plan(&[
        "library",
        "-o",
        "out",
        "--pass",
        "qwen3",
        "--pass",
        "parakeet",
        "--pass",
        "qwen3=qwen3-asr-0.6b",
    ]);
    for (pass, tag) in plan["passes"]
        .as_array()
        .unwrap()
        .iter()
        .zip(["qw", "pk", "qw06"])
    {
        assert_eq!(pass["model_tag"], tag);
        assert_eq!(pass["meetings"][0]["id"], untagged["meetings"][0]["id"]);
        assert!(destination(pass).ends_with(format!("Call.merged.{tag}.json")));
        let files = pass["files"].as_array().unwrap();
        for name in ["interview.2026", "Review.zoom"] {
            assert!(files.iter().any(|file| {
                file["output"]
                    .as_str()
                    .is_some_and(|path| Path::new(path).ends_with(format!("{name}.{tag}.json")))
            }));
        }
        assert!(files.iter().any(|file| {
            file["path"]
                .as_str()
                .is_some_and(|path| path.ends_with("video1067067067.mp4"))
                && file["skip"].is_string()
        }));
    }
    let single = fixture.plan(&[
        "library",
        "-o",
        "out",
        "--engine",
        "parakeet-v3",
        "--model-suffix",
    ]);
    assert_eq!(&single, &plan["passes"][1]);
}

#[test]
fn model_passes_resume_independently_and_reject_repeated_models() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    fixture.track("meeting/Audio Record/audioAlice11067067067.m4a");
    fixture.track("interview.wav");
    let duplicate = fixture
        .command(&[
            "meeting",
            "--pass",
            "qwen3",
            "--pass",
            "qwen3=qwen3-asr-1.7b",
        ])
        .output()
        .unwrap();
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("repeated identifier qw"));
    let incompatible = fixture
        .command(&[
            "interview.wav",
            "--pass",
            "parakeet=qwen3-asr-0.6b",
            "--plan",
        ])
        .output()
        .unwrap();
    assert!(!incompatible.status.success());
    assert!(String::from_utf8_lossy(&incompatible.stderr).contains("different engine"));

    let args = [
        "meeting",
        "interview.wav",
        "-o",
        "out",
        "--pass",
        "qwen3",
        "--pass",
        "apple-native",
        "--progress-file",
        "run.jsonl",
        "--resume",
    ];
    let plan = fixture.plan(&args);
    let qwen = destination(&plan["passes"][0]);
    let apple = destination(&plan["passes"][1]);
    fixture.file(qwen.to_str().unwrap(), b"existing Qwen result");
    fixture.file("out/interview.qw.json", b"existing Qwen interview");
    let run = || {
        success(
            &fixture
                .command(&args)
                .env("TRANSCRIBERRR_NATIVE", &helper)
                .output()
                .unwrap(),
        )
    };
    run();
    run();
    assert_eq!(fixture.calls(), 2);
    assert_eq!(fs::read(qwen).unwrap(), b"existing Qwen result");
    assert!(fixture.0.join("out/interview.ap.json").is_file());
    for (tag, output) in [("qw", qwen), ("ap", apple)] {
        let events = fs::read_to_string(fixture.0.join(format!("run.{tag}.jsonl"))).unwrap();
        assert!(events.contains(output.to_str().unwrap()));
    }
    assert!(!fixture.0.join("run.jsonl").exists());
    fs::remove_file(apple).unwrap();
    run();
    assert_eq!(fixture.calls(), 3);
    assert!(apple.is_file());

    let other_log = fixture.0.join("alias.qw.jsonl");
    let existing = fs::read(apple).unwrap();
    fs::hard_link(apple, other_log).unwrap();
    let collision = fixture
        .command(&[
            "meeting",
            "-o",
            "out",
            "--pass",
            "qwen3",
            "--pass",
            "apple-native",
            "--progress-file",
            "alias.jsonl",
            "--resume",
        ])
        .output()
        .unwrap();
    assert!(!collision.status.success());
    assert!(String::from_utf8_lossy(&collision.stderr).contains("progress file overlaps"));
    assert_eq!(fs::read(apple).unwrap(), existing);
    assert!(!fixture.0.join("alias.ap.jsonl").exists());
}

#[test]
fn overwrite_reprocesses_tagged_outputs_even_with_resume() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    fixture.track("interview.wav");
    let args = [
        "interview.wav",
        "--pass",
        "apple-native",
        "--progress-file",
        "run.jsonl",
        "--resume",
    ];
    let run = |extra: &[&str]| {
        success(
            &fixture
                .command(&args)
                .args(extra)
                .env("TRANSCRIBERRR_NATIVE", &helper)
                .output()
                .unwrap(),
        );
    };
    run(&[]);
    run(&[]);
    assert_eq!(fixture.calls(), 1);
    run(&["--overwrite"]);
    assert_eq!(fixture.calls(), 2);
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
fn identifies_renamed_zoom_video_by_its_complete_audio_stream() {
    let Some(ffmpeg) = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
        .into_iter()
        .find(|path| Path::new(path).is_file())
    else {
        eprintln!("skipping: ffmpeg not found");
        return;
    };
    let fixture = Fixture::new();
    let mixed = "Call/audio1067067067.m4a";
    let renamed = "Call/Renamed session.mp4";
    let unrelated = "Call/Other footage.mp4";
    fixture.track("Call/Audio Record/audioAlice11067067067.wav");
    let encode = |args: &[&str]| {
        success(
            &Command::new(ffmpeg)
                .current_dir(&fixture.0)
                .args(["-hide_banner", "-nostdin", "-loglevel", "error"])
                .args(args)
                .output()
                .unwrap(),
        );
    };
    encode(&[
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-c:a",
        "aac",
        mixed,
    ]);
    encode(&[
        "-f",
        "lavfi",
        "-i",
        "color=s=32x32:d=1",
        "-i",
        mixed,
        "-c:v",
        "mpeg4",
        "-c:a",
        "copy",
        renamed,
    ]);
    encode(&[
        "-f",
        "lavfi",
        "-i",
        "color=s=32x32:d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:duration=1",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        unrelated,
    ]);

    let plan = fixture.plan(&["Call", "-o", "out"]);
    let output = destination(&plan);
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    fs::write(output, b"existing merge").unwrap();
    for input in ["Call", renamed] {
        let plan = fixture.plan(&[input, "-o", "out"]);
        assert_eq!(destination(&plan), output);
        assert_eq!(plan["meetings"][0]["done"], true);
        let files = plan["files"].as_array().unwrap();
        let named = |name: &str| {
            files
                .iter()
                .find(|file| file["path"].as_str().unwrap().ends_with(name))
        };
        let video = named(renamed).unwrap();
        assert_eq!(video["meeting"], plan["meetings"][0]["id"]);
        assert!(
            video["skip"]
                .as_str()
                .unwrap()
                .contains("participant tracks")
        );
        assert!(video["output"].is_null());
        if input == "Call" {
            let other = named(unrelated).unwrap();
            assert!(other["skip"].is_null() && other["output"].is_string());
        } else {
            assert!(named(unrelated).is_none());
        }
    }

    let legacy = "Legacy/audio_only01.m4a";
    let legacy_video = "Legacy/Renamed legacy session.mp4";
    fs::create_dir(fixture.0.join("Legacy")).unwrap();
    fs::copy(fixture.0.join(mixed), fixture.0.join(legacy)).unwrap();
    fs::copy(fixture.0.join(renamed), fixture.0.join(legacy_video)).unwrap();
    let plan = fixture.plan(&[legacy_video, "-o", "out", "--model-suffix"]);
    let files = plan["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    assert!(files.iter().any(|file| {
        file["path"].as_str().unwrap().ends_with(legacy_video)
            && file["skip"]
                .as_str()
                .unwrap_or_default()
                .contains("identical audio")
    }));
    assert!(files.iter().any(|file| {
        file["path"].as_str().unwrap().ends_with(legacy)
            && file["output"]
                .as_str()
                .unwrap_or_default()
                .ends_with("audio_only01.qw.json")
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
        4,
        "overwrite reruns both outputs even with a resume log"
    );
}

#[test]
fn resume_does_not_claim_or_replace_a_different_destination() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    fixture.track("ordinary.wav");
    let args = [
        "ordinary.wav",
        "--engine",
        "apple-native",
        "--progress-file",
        "run.jsonl",
        "--resume",
    ];
    let run = |folder| {
        fixture
            .command(&args)
            .args(["-o", folder])
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .output()
            .unwrap()
    };
    success(&run("first"));
    fixture.file("second/ordinary.json", "a different output");
    let plan = fixture.plan(&[&args[..], &["-o", "second"]].concat());
    assert_eq!(plan["files"][0]["done"], false);
    for _ in 0..2 {
        let conflict = run("second");
        assert!(!conflict.status.success());
        assert!(String::from_utf8_lossy(&conflict.stderr).contains("output exists"));
    }
    assert_eq!(
        fs::read_to_string(fixture.0.join("second/ordinary.json")).unwrap(),
        "a different output"
    );
    success(&run("first"));
    assert_eq!(fixture.calls(), 1);
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

#[test]
fn skip_existing_keeps_another_sessions_track_transcripts() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    for session in ["A", "B"] {
        fixture.track(&format!("{session}/ep2/alice.wav"));
        fixture.track(&format!("{session}/ep2/bob.wav"));
    }
    let run = |session: &str| {
        fixture
            .command(&[
                &format!("{session}/ep2/alice.wav"),
                &format!("{session}/ep2/bob.wav"),
                "--iso-tracks",
                "-o",
                "shared",
                "--engine",
                "apple-native",
                "--skip-existing",
            ])
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .output()
            .unwrap()
    };
    success(&run("A"));
    let before = fs::read_to_string(fixture.0.join("shared/alice.json")).unwrap();
    assert!(before.contains("A/ep2/alice.wav"));
    let output = run("B");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("output exists"));
    assert_eq!(
        fs::read_to_string(fixture.0.join("shared/alice.json")).unwrap(),
        before
    );
}

#[test]
fn rerun_from_another_directory_skips_a_merged_session() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    fixture.track("proj/ep1/alice.wav");
    fixture.track("proj/ep1/bob.wav");
    fs::create_dir_all(fixture.0.join("elsewhere")).unwrap();
    let common = [
        "--iso-tracks",
        "--engine",
        "apple-native",
        "--skip-existing",
    ];
    success(
        &fixture
            .command(&["proj/ep1/alice.wav", "proj/ep1/bob.wav", "-o", "proj/out"])
            .args(common)
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .output()
            .unwrap(),
    );
    let absolute = |name: &str| fixture.0.join(name).display().to_string();
    let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(fixture.0.join("elsewhere"))
        .args([
            absolute("proj/ep1/alice.wav"),
            absolute("proj/ep1/bob.wav"),
            "-o".to_owned(),
            absolute("proj/out"),
        ])
        .args(common)
        .env("TRANSCRIBERRR_NATIVE", &helper)
        .output()
        .unwrap();
    success(&output);
    assert!(!fixture.0.join("elsewhere/calls.log").exists());
    assert!(!fixture.0.join("proj/out/ep1 2.merged.json").exists());
}

#[test]
fn app_rerun_of_a_manual_session_keeps_another_sessions_transcripts() {
    let fixture = Fixture::new();
    let helper = fixture.fake_native_helper();
    for session in ["A", "B"] {
        fixture.track(&format!("{session}/ep/alice.wav"));
        fixture.track(&format!("{session}/ep/bob.wav"));
    }
    let run = |session: &str, format: &str, log: &str| {
        let output = fixture
            .command(&[
                &format!("{session}/ep/alice.wav"),
                &format!("{session}/ep/bob.wav"),
                "--iso-tracks",
                "-o",
                &format!("out-{format}"),
                "--format",
                format,
                "--engine",
                "apple-native",
                "--progress-file",
                log,
                "--resume",
            ])
            .env("TRANSCRIBERRR_NATIVE", &helper)
            .output()
            .unwrap();
        success(&output);
    };
    // JSON tells its own merge: a re-run replaces it.
    run("A", "json", "p1.jsonl");
    run("A", "json", "p2.jsonl");
    assert!(fixture.0.join("out-json/ep.merged.json").is_file());
    assert!(!fixture.0.join("out-json/ep 2.merged.json").exists());
    // TXT cannot tell whose transcripts these are: another session with the same names
    // stops at its track files and replaces nothing of the first one's.
    run("A", "txt", "p3.jsonl");
    let first = fs::read_to_string(fixture.0.join("out-txt/ep.merged.txt")).unwrap();
    let track = fs::read_to_string(fixture.0.join("out-txt/alice.txt")).unwrap();
    let output = fixture
        .command(&[
            "B/ep/alice.wav",
            "B/ep/bob.wav",
            "--iso-tracks",
            "-o",
            "out-txt",
            "--format",
            "txt",
            "--engine",
            "apple-native",
            "--progress-file",
            "p4.jsonl",
            "--resume",
        ])
        .env("TRANSCRIBERRR_NATIVE", &helper)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("output exists"));
    assert_eq!(
        fs::read_to_string(fixture.0.join("out-txt/ep.merged.txt")).unwrap(),
        first
    );
    assert_eq!(
        fs::read_to_string(fixture.0.join("out-txt/alice.txt")).unwrap(),
        track
    );
    // The same session, resumed with its own log, still redoes its tracks and merge.
    run("A", "txt", "p3.jsonl");
}
