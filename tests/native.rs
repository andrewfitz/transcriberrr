use std::fs;
use std::io::{BufRead, BufReader};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::{sync::mpsc, time::Duration};

struct Fixture(PathBuf);

impl Fixture {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "transcriberrr-native-{name}-{}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }

    fn helper(&self, script: &str) -> PathBuf {
        let helper = self.0.join("native-helper");
        fs::write(&helper, script).unwrap();
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
        helper
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn apple_helper_exit_drains_ready_server() {
    let fixture = Fixture::new("realtime-helper-exit");
    let helper = fixture.helper(
        r#"#!/bin/sh
printf '%s\n' "$$" > helper.pid
printf '%s\n' '{"kind":"ready","id":0}'
read -r packet
"#,
    );
    let mut child = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(&fixture.0)
        .env("TRANSCRIBERRR_NATIVE", helper)
        .args([
            "serve",
            "--engine",
            "apple-native",
            "--offline",
            "--port",
            "0",
            "--exit-on-stdin-eof",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (events, output) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if events.send(line).is_err() {
                break;
            }
        }
    });
    let next_event = || -> serde_json::Value {
        serde_json::from_str(&output.recv_timeout(Duration::from_secs(15)).unwrap()).unwrap()
    };
    assert_eq!(next_event()["event"], "listening");
    assert_eq!(next_event()["event"], "ready");
    let pid = fs::read_to_string(fixture.0.join("helper.pid")).unwrap();
    assert!(
        Command::new("/bin/kill")
            .arg(pid.trim())
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(next_event()["event"], "draining");
    let failure = next_event();
    assert_eq!(failure["event"], "error");
    assert_eq!(failure["message"], "Apple Speech helper exited");
    assert!(!child.wait().unwrap().success());
}

#[test]
fn apple_native_resumes_after_publishing_transcript() {
    let fixture = Fixture::new("resume");
    fs::write(fixture.0.join("audio.wav"), []).unwrap();
    let helper = fixture.helper(
        r#"#!/bin/sh
cat <<'JSON'
{"source":"audio.wav","model":"test","device":"cpu","language":"English","duration_seconds":1.0,"segments":[{"start":0.0,"end":1.0,"text":"Hello."}],"metadata":{"transcribed-with":"transcriberrr","version":"0.1.0","transcribed_at":"2026-10-17T00:00:00Z","model":"test"}}
JSON
"#,
    );
    let run = |helper: &std::path::Path, resume| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_transcriberrr"));
        command
            .current_dir(&fixture.0)
            .env("TRANSCRIBERRR_NATIVE", helper)
            .args([
                "audio.wav",
                "--engine",
                "apple-native",
                "--format",
                "txt",
                "--progress-file",
                "progress.jsonl",
            ]);
        if resume {
            command.arg("--resume");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };

    run(&helper, false);
    let progress_path = fixture.0.join("progress.jsonl");
    let progress = fs::read_to_string(&progress_path).unwrap();
    let (before_completion, _) = progress.rsplit_once("\n{").unwrap();
    assert!(before_completion.contains(r#""phase":"Writing output""#));
    // Simulate stopping after the transcript was published but before completion was logged.
    fs::write(progress_path, before_completion).unwrap();
    run(std::path::Path::new("native-helper"), true);
    assert!(
        fs::read_to_string(fixture.0.join("audio.txt"))
            .unwrap()
            .contains("Hello.")
    );
}

#[test]
fn fail_fast_records_failed_session_when_a_track_fails() {
    let fixture = Fixture::new("fail-fast");
    // One second of silent PCM WAV. The mock helper fails after session comparison succeeds.
    let sample_rate = 16_000u32;
    let data_size = sample_rate * 2;
    let mut wav = Vec::from(b"RIFF".as_slice());
    wav.extend((36 + data_size).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16u32.to_le_bytes());
    wav.extend(1u16.to_le_bytes()); // PCM
    wav.extend(1u16.to_le_bytes()); // Mono
    wav.extend(sample_rate.to_le_bytes());
    wav.extend(data_size.to_le_bytes());
    wav.extend(2u16.to_le_bytes());
    wav.extend(16u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend(data_size.to_le_bytes());
    wav.resize(44 + data_size as usize, 0);
    fs::write(fixture.0.join("alice.wav"), &wav).unwrap();
    fs::write(fixture.0.join("bob.wav"), &wav).unwrap();
    let helper = fixture.helper("#!/bin/sh\necho 'recognizer failed' >&2\nexit 1\n");
    let output = Command::new(env!("CARGO_BIN_EXE_transcriberrr"))
        .current_dir(&fixture.0)
        .env("TRANSCRIBERRR_NATIVE", helper)
        .args([
            "alice.wav",
            "bob.wav",
            "--engine",
            "apple-native",
            "--iso-tracks",
            "--fail-fast",
            "--progress-file",
            "progress.jsonl",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let log = fs::read_to_string(fixture.0.join("progress.jsonl")).unwrap();
    let terminal = log.lines().last().unwrap();
    assert!(
        terminal.contains(r#""status":"failed""#)
            && terminal.contains(r#""kind":"merge""#)
            && terminal.contains("recognizer failed"),
        "{log}"
    );
}
