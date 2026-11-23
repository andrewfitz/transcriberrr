use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Workspace(PathBuf);

impl Workspace {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "transcriberrr-session-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn run(&self, args: &[&str]) -> Output {
        let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    fn write_json(&self, path: &str, value: &Value) {
        fs::write(self.0.join(path), serde_json::to_vec(value).unwrap()).unwrap();
    }

    fn read_json(&self, path: &str) -> Value {
        serde_json::from_slice(&fs::read(self.0.join(path)).unwrap()).unwrap()
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_wav(path: &Path, samples: &[i16]) {
    let size = (samples.len() * 2) as u32;
    let mut wav = Vec::new();
    wav.extend(b"RIFF");
    wav.extend((36 + size).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16u32.to_le_bytes());
    wav.extend(1u16.to_le_bytes());
    wav.extend(1u16.to_le_bytes());
    wav.extend(16_000u32.to_le_bytes());
    wav.extend(32_000u32.to_le_bytes());
    wav.extend(2u16.to_le_bytes());
    wav.extend(16u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend(size.to_le_bytes());
    wav.extend(samples.iter().flat_map(|&sample| sample.to_le_bytes()));
    fs::write(path, wav).unwrap();
}

#[test]
fn resumes_track_artifacts_and_merges_partial_coverage_without_models_or_wavs() {
    let workspace = Workspace::new();
    let mut samples = vec![0; 321];
    samples[320] = i16::MIN;
    write_wav(&workspace.0.join("click.wav"), &samples);
    let manifest = json!({
        "schema_version": 1, "job_id":"job", "fingerprint":"first",
        "output":"session.json", "engine": "parakeet-v3", "model": "/missing/model",
        "tracks": [{
            "track_id":"camera", "source_id":"source", "stream_evidence":"digest",
            "binding":{"stream_index":1, "origin":{"numerator":1,"denominator":48_000},
                "session_start":{"numerator":-1,"denominator":2}},
            "channel":0, "name":"Camera", "role":"unknown", "recognize":false, "start_frame":-8000,
            "audio":"click.wav", "result":"camera.json", "activity":"camera-activity.json",
            "recognition_key":"recognition-key"
        }, {
            "track_id":"silent-angle", "source_id":"silent", "binding":{
                "stream_index":null, "origin":{"numerator":0,"denominator":1},
                "session_start":{"numerator":0,"denominator":1}},
            "name":"Silent", "role":"unknown", "recognize":false, "start_frame":0
        }]
    });
    workspace.write_json("manifest.json", &manifest);
    workspace.run(&[
        "session",
        "--manifest",
        "manifest.json",
        "--offline",
        "--jsonl",
    ]);
    let activity = workspace.read_json("camera-activity.json");
    assert_eq!(activity["duration_samples"], 321);
    assert_eq!(
        activity["activity"][0],
        json!({"start_sample":0,"end_sample":320,"level":0.0})
    );
    assert_eq!(
        activity["activity"][1],
        json!({"start_sample":320,"end_sample":321,"level":1.0})
    );
    assert_eq!(workspace.read_json("session.json")["words"], json!([]));
    assert!(!workspace.0.join("camera.json").exists());
    fs::remove_file(workspace.0.join("click.wav")).unwrap();

    workspace.write_json(
        "camera.json",
        &json!({
            "version":1, "sample_rate":16_000, "recognition_key":"recognition-key",
            "stream_evidence":"digest", "duration_samples":321,
            "words":[{"id":"w1", "utterance_id":"u1", "text":" Hello!", "start_sample":8,
                "end_sample":302, "speaker_id":null, "precision":"word"}],
            "activity":[]
        }),
    );
    let mut updated = manifest;
    updated["fingerprint"] = json!("second");
    updated["tracks"][0]["recognize"] = json!(true);
    updated["tracks"][0]["start_frame"] = json!(-4000);
    updated["tracks"][0]["binding"]["session_start"] = json!({"numerator":-1,"denominator":4});
    updated["tracks"][0]
        .as_object_mut()
        .unwrap()
        .remove("audio");
    workspace.write_json("manifest.json", &updated);
    let output = workspace.run(&[
        "session",
        "--manifest",
        "manifest.json",
        "--reconcile-only",
        "--jsonl",
    ]);
    let events: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events.last().unwrap()["status"], "completed");
    assert_eq!(events.last().unwrap()["fingerprint"], "second");
    let result = workspace.read_json("session.json");
    assert_eq!(result["words"][0]["start_sample"], -3992);
    assert_eq!(result["words"][0]["observations"][0]["start_sample"], 8);
    assert_eq!(result["tracks"][0]["origin"]["denominator"], 48_000);
    assert_eq!(result["tracks"][1]["coverage"], "coverage_only");

    updated["tracks"][0]["stream_evidence"] = json!("replaced");
    workspace.write_json("manifest.json", &updated);
    let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(&workspace.0)
        .args([
            "session",
            "--manifest",
            "manifest.json",
            "--reconcile-only",
            "--jsonl",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let lines = String::from_utf8(output.stdout).unwrap();
    let events: Vec<Value> = lines
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events.last().unwrap()["status"], "failed");
    assert_eq!(events[1]["status"], "failed");
    assert_eq!(workspace.read_json("session.json"), result);

    updated["tracks"][0]["stream_evidence"] = json!("digest");
    updated["tracks"][0]["reuse"] = json!(true);
    updated["tracks"].as_array_mut().unwrap().push(json!({
        "track_id":"offline-angle", "source_id":"offline", "binding":{
            "origin":{"numerator":0,"denominator":1}, "session_start":{"numerator":0,"denominator":1}},
        "name":"Offline", "role":"mixed_audio", "recognize":true, "start_frame":0,
        "unavailable":"Source media is offline"
    }));
    workspace.write_json("manifest.json", &updated);
    let output = workspace.run(&["session", "--manifest", "manifest.json", "--offline"]);
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["status"], "partial");
    let result = workspace.read_json("session.json");
    assert_eq!(result["tracks"][0]["coverage"], "recognized");
    assert_eq!(result["tracks"][2]["coverage"], "failed");
    assert_eq!(result["tracks"][2]["error"], "Source media is offline");
}

#[test]
fn reads_mpcut_contract_fixture_without_losing_native_timing_or_evidence() {
    let workspace = Workspace::new();
    let fixture = include_bytes!("fixtures/session-result-v1.json");
    fs::write(workspace.0.join("fixture.json"), fixture).unwrap();
    workspace.run(&[
        "convert",
        "fixture.json",
        "--format",
        "json",
        "--output",
        "converted.json",
    ]);
    assert_eq!(
        workspace.read_json("converted.json"),
        serde_json::from_slice::<Value>(fixture).unwrap()
    );
}

#[test]
fn converts_presentation_cues_using_the_exact_source_origin() {
    let workspace = Workspace::new();
    let mut fixture: Value =
        serde_json::from_slice(include_bytes!("fixtures/session-result-v1.json")).unwrap();
    fixture["clock"] = json!("presentation");
    fixture["tracks"].as_array_mut().unwrap().truncate(1);
    fixture["tracks"][0]["origin"] = json!({"numerator":1,"denominator":2});
    fixture["words"][0]["start_sample"] = json!(1);
    fixture["words"][0]["end_sample"] = json!(16_003);
    workspace.write_json("fixture.json", &fixture);
    workspace.run(&[
        "convert",
        "fixture.json",
        "--format",
        "srt",
        "--output",
        "converted.srt",
    ]);
    let srt = fs::read_to_string(workspace.0.join("converted.srt")).unwrap();
    assert!(srt.contains("00:00:00,500 --> 00:00:01,500"), "{srt}");
}

#[test]
fn missing_diarization_preserves_silent_isolated_coverage_and_mixed_activity() {
    let workspace = Workspace::new();
    write_wav(&workspace.0.join("mix.wav"), &[2000; 320]);
    write_wav(&workspace.0.join("lav.wav"), &[0; 320]);
    let track = |id: &str, role: &str| {
        json!({
            "track_id":id,"source_id":id,"stream_evidence":"digest", "name":id,
            "binding":{"stream_index":0,"origin":{"numerator":0,"denominator":1},
                "session_start":{"numerator":0,"denominator":1}},
            "role":role,"recognize":true,"start_frame":0,
            "audio":format!("{id}.wav"),"result":format!("{id}.json")
        })
    };
    workspace.write_json(
        "manifest.json",
        &json!({
            "schema_version":1,"job_id":"job","fingerprint":"fingerprint",
            "engine":"apple-native","diarization":true,"output":"session.json",
            "tracks":[track("mix", "mixed_audio"),track("lav", "isolated_mic")],
        }),
    );
    let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(&workspace.0)
        // Planning checks for an executable. Diarization must fail before recognition.
        .env("TRANSCRIBERRR_NATIVE", "/usr/bin/false")
        .env("TRANSCRIBERRR_MODEL_CACHE", workspace.0.join("empty-cache"))
        .args(["session", "--manifest", "manifest.json", "--offline"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["status"], "partial");
    let tracks = workspace.read_json("session.json")["tracks"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(tracks[0]["coverage"], "failed");
    assert!(
        tracks[0]["error"].as_str().unwrap().contains("diarization"),
        "{}",
        tracks[0]
    );
    assert!(tracks[0]["activity"][0]["level"].as_f64().unwrap() > 0.0);
    assert_eq!(tracks[1]["coverage"], "recognized");
    assert_eq!(tracks[1]["words"], json!([]));
}

#[test]
fn protects_activity_artifact_for_unavailable_track() {
    let workspace = Workspace::new();
    let old_activity = json!({"previous":"activity"});
    workspace.write_json("activity.json", &old_activity);
    workspace.write_json(
        "manifest.json",
        &json!({
            "schema_version":1, "job_id":"job", "fingerprint":"session",
            "output":"activity.json", "tracks":[{
                "track_id":"camera", "source_id":"source", "name":"Camera",
                "binding":{"stream_index":0, "origin":{"numerator":0,"denominator":1},
                    "session_start":{"numerator":0,"denominator":1}},
                "role":"unknown", "recognize":false, "start_frame":0,
                "activity":"activity.json", "unavailable":"missing audio"
            }]
        }),
    );
    let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(&workspace.0)
        .args(["session", "--manifest", "manifest.json", "--reconcile-only"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("session output overlaps"));
    assert_eq!(workspace.read_json("activity.json"), old_activity);
}
