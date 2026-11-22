//! Persistent Apple Speech helper transport. Turn IDs are local to this process.
use super::recognizer::{
    Hypothesis, RecognitionOptions, RecognitionStream, Recognizer, SAMPLE_RATE, TimedWord,
};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::Shutdown;
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::watch;

// Match the persistent Swift helper's maximum number of analyzer streams.
pub(super) const MAX_STREAMS: usize = 64;
// Keep packets under the helper's 64 KiB framing limit.
const PCM_PACKET_SAMPLES: usize = SAMPLE_RATE as usize;
const HEADER_BYTES: usize = 13;
const OPEN: u8 = 1;
const APPEND: u8 = 2;
const COMMIT: u8 = 3;
const CANCEL: u8 = 4;
const OPEN_COMMITTED: u8 = 5;
const MAX_EVENT_BYTES: u64 = 4 * 1024 * 1024;
const WRITE_TIMEOUT: Duration = Duration::from_secs(2);
const FINAL_TIMEOUT: Duration = Duration::from_secs(5);

pub(super) fn available() -> bool {
    let Some(helper) = crate::apple_native_helper() else {
        return false;
    };
    let Ok(mut child) = Command::new(helper)
        .arg("--stream-supported")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    false
}

pub(super) struct AppleRecognizer {
    stdin: Mutex<UnixStream>,
    child: Child,
    reader: Option<thread::JoinHandle<()>>,
    mailbox: Arc<Mailbox>,
    next_stream: AtomicU64,
}

#[derive(Default)]
struct Mailbox {
    state: Mutex<MailState>,
    updated: Condvar,
    failure: watch::Sender<Option<String>>,
}

impl Mailbox {
    fn disconnect(&self, reason: String) {
        let mut state = self.state.lock().unwrap();
        if state.disconnected.is_none() {
            state.disconnected = Some(reason.clone());
            self.failure.send_replace(Some(reason));
            self.updated.notify_all();
        }
    }
}

#[derive(Default)]
struct MailState {
    turns: HashMap<u64, TurnEvents>,
    ready: bool,
    startup_error: Option<String>,
    disconnected: Option<String>,
}

#[derive(Default)]
struct TurnEvents {
    stable: Option<Hypothesis>,
    draft: Option<Hypothesis>,
    terminal: Option<Result<Hypothesis, String>>,
}

impl TurnEvents {
    fn take_partial(&mut self) -> Result<Option<Hypothesis>> {
        match &self.terminal {
            Some(Err(error)) => bail!("Apple Speech: {error}"),
            Some(Ok(_)) => Ok(None),
            None => Ok(self.stable.take().or_else(|| self.draft.take())),
        }
    }
}

#[derive(Deserialize)]
struct Message {
    kind: Kind,
    id: u64,
    text: Option<String>,
    words: Option<Vec<Word>>,
    error: Option<String>,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Ready,
    Error,
    Draft,
    Stable,
    Final,
}

#[derive(Deserialize)]
struct Word {
    text: String,
    start: f64,
    end: f64,
}

impl AppleRecognizer {
    pub(super) fn load(offline: bool) -> Result<Self> {
        let helper = crate::apple_native_helper().context(
            "Apple Speech helper is missing. Build the macOS app or set TRANSCRIBERRR_NATIVE",
        )?;
        let mut command = Command::new(helper);
        command.arg("stream");
        if offline {
            command.arg("--offline");
        }
        // A write deadline also bounds failures when the helper is alive but stops reading.
        let (stdin, helper_input) =
            UnixStream::pair().context("create Apple Speech input socket")?;
        stdin
            .set_write_timeout(Some(WRITE_TIMEOUT))
            .context("set Apple Speech input write timeout")?;
        let mut child = command
            .env(
                "TRANSCRIBERRR_NATIVE_PARENT_PID",
                std::process::id().to_string(),
            )
            .stdin(Stdio::from(OwnedFd::from(helper_input)))
            .stdout(Stdio::piped())
            .spawn()
            .context("start Apple Speech helper")?;
        let stdout = child.stdout.take().expect("piped Apple Speech stdout");
        let mailbox = Arc::new(Mailbox::default());
        let mut speech = Self {
            child,
            reader: None,
            stdin: Mutex::new(stdin),
            mailbox: Arc::clone(&mailbox),
            next_stream: AtomicU64::new(1),
        };
        speech.reader = Some(
            thread::Builder::new()
                .name("apple-speech-results".into())
                .spawn(move || read_results(stdout, &mailbox))?,
        );
        {
            let state = speech.mailbox.state.lock().unwrap();
            let (state, _) = speech
                .mailbox
                .updated
                .wait_timeout_while(state, Duration::from_secs(120), |state| {
                    !state.ready && state.startup_error.is_none() && state.disconnected.is_none()
                })
                .unwrap();
            if let Some(error) = &state.startup_error {
                bail!("Apple Speech helper failed to start: {error}");
            }
            if let Some(reason) = &state.disconnected {
                bail!("Apple Speech helper failed to start: {reason}");
            }
            if !state.ready {
                bail!("Apple Speech helper did not become ready. Check its stderr for details");
            }
        }
        Ok(speech)
    }

    fn control(&self, operation: u8, id: u64) -> Result<()> {
        self.send(id, &header(operation, id))
    }

    fn send(&self, id: u64, packet: &[u8]) -> Result<()> {
        {
            let state = self.mailbox.state.lock().unwrap();
            if let Some(Err(error)) = state.turns.get(&id).and_then(|turn| turn.terminal.as_ref()) {
                bail!("Apple Speech: {error}");
            }
            if let Some(reason) = &state.disconnected {
                bail!("{reason}");
            }
        }
        let mut stdin = self.stdin.lock().unwrap();
        if let Err(error) = stdin.write_all(packet) {
            // A partially written packet leaves no valid framing for later writes.
            let reason = format!("write Apple Speech packet: {error}");
            self.mailbox.disconnect(reason.clone());
            let _ = stdin.shutdown(Shutdown::Write);
            bail!("{reason}");
        }
        Ok(())
    }
}

impl Drop for AppleRecognizer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

impl AppleRecognizer {
    fn open(&self, operation: u8) -> Result<Box<dyn RecognitionStream + '_>> {
        let id = self.next_stream.fetch_add(1, Ordering::Relaxed);
        {
            let mut state = self.mailbox.state.lock().unwrap();
            if let Some(reason) = &state.disconnected {
                bail!("{reason}");
            }
            state.turns.insert(id, TurnEvents::default());
        }
        let stream = AppleStream {
            parent: self,
            id,
            committed: operation == OPEN_COMMITTED,
            packet: header(APPEND, id).to_vec(),
        };
        self.control(operation, id)?;
        Ok(Box::new(stream))
    }
}

impl Recognizer for AppleRecognizer {
    fn backend_failure(&self) -> Option<watch::Receiver<Option<String>>> {
        Some(self.mailbox.failure.subscribe())
    }

    fn start(&self, _options: &RecognitionOptions) -> Result<Box<dyn RecognitionStream + '_>> {
        self.open(OPEN)
    }

    fn start_committed(
        &self,
        _options: &RecognitionOptions,
    ) -> Result<Box<dyn RecognitionStream + '_>> {
        self.open(OPEN_COMMITTED)
    }
}

fn header(operation: u8, id: u64) -> [u8; HEADER_BYTES] {
    let mut header = [0; HEADER_BYTES];
    header[0] = operation;
    header[1..9].copy_from_slice(&id.to_le_bytes());
    header
}

struct AppleStream<'a> {
    parent: &'a AppleRecognizer,
    id: u64,
    committed: bool,
    packet: Vec<u8>,
}

impl AppleStream<'_> {
    fn flush(&mut self) -> Result<()> {
        let bytes = self.packet.len() - HEADER_BYTES;
        if bytes > 0 {
            self.packet[9..HEADER_BYTES].copy_from_slice(&(bytes as u32).to_le_bytes());
            self.parent.send(self.id, &self.packet)?;
            self.packet.truncate(HEADER_BYTES);
        }
        Ok(())
    }
}

impl RecognitionStream for AppleStream<'_> {
    fn push(&mut self, mut samples: &[f32]) -> Result<Option<Hypothesis>> {
        while !samples.is_empty() {
            let buffered = (self.packet.len() - HEADER_BYTES) / 2;
            let count = samples.len().min(PCM_PACKET_SAMPLES - buffered);
            let (next, rest) = samples.split_at(count);
            samples = rest;
            self.packet.reserve(count * 2);
            self.packet.extend(next.iter().flat_map(|sample| {
                let pcm = (sample * 32768.0).round() as i16;
                pcm.to_le_bytes()
            }));
            // Live audio goes straight through. Committed turns replay many short
            // spans after commit, so pack them without duplicating the entire turn.
            if !self.committed || buffered + count == PCM_PACKET_SAMPLES {
                self.flush()?;
            }
        }
        if self.committed {
            return Ok(None);
        }
        let mut state = self.parent.mailbox.state.lock().unwrap();
        let events = state.turns.get_mut(&self.id).expect("registered stream");
        let partial = events.take_partial()?;
        if let Some(reason) = &state.disconnected {
            bail!("{reason}");
        }
        Ok(partial)
    }

    fn finish(mut self: Box<Self>) -> Result<Hypothesis> {
        self.flush()?;
        let mailbox = &self.parent.mailbox;
        let awaiting_result = |state: &MailState| {
            state.disconnected.is_none() && state.turns[&self.id].terminal.is_none()
        };
        let mut state = mailbox.state.lock().unwrap();
        if awaiting_result(&state) {
            drop(state);
            let sent = self.parent.control(COMMIT, self.id);
            state = mailbox.state.lock().unwrap();
            if sent.is_ok() {
                (state, _) = mailbox
                    .updated
                    .wait_timeout_while(state, FINAL_TIMEOUT, |state| awaiting_result(state))
                    .unwrap();
            }
        }
        if let Some(result) = state
            .turns
            .get_mut(&self.id)
            .and_then(|events| events.terminal.take())
        {
            state.turns.remove(&self.id);
            return result.map_err(|error| anyhow::anyhow!("Apple Speech: {error}"));
        }
        if let Some(reason) = &state.disconnected {
            bail!("Apple Speech helper stopped before finalizing: {reason}");
        }
        bail!("Apple Speech helper timed out while finalizing");
    }
}

impl Drop for AppleStream<'_> {
    fn drop(&mut self) {
        let turn = self
            .parent
            .mailbox
            .state
            .lock()
            .unwrap()
            .turns
            .remove(&self.id);
        if turn.is_some_and(|events| events.terminal.is_none()) {
            let _ = self.parent.control(CANCEL, self.id);
        }
    }
}

fn read_results(stdout: impl Read, mailbox: &Mailbox) {
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let reason = loop {
        match (&mut reader).take(MAX_EVENT_BYTES).read_line(&mut line) {
            Ok(0) => break "Apple Speech helper exited".to_owned(),
            Ok(_) if !line.ends_with('\n') => {
                break "Apple Speech helper returned an incomplete or oversized result".to_owned();
            }
            Err(error) => break format!("read Apple Speech helper result: {error}"),
            Ok(_) => {}
        }
        let event = match serde_json::from_str::<Message>(&line) {
            Ok(event) => event,
            Err(error) => break format!("invalid Apple Speech helper result: {error}"),
        };
        line.clear();
        let mut state = mailbox.state.lock().unwrap();
        if event.kind == Kind::Ready {
            state.ready = true;
        } else if event.kind == Kind::Error && event.id == 0 {
            state.startup_error = Some(event.error.unwrap_or_else(|| "unknown error".into()));
        } else if let Some(turn) = state.turns.get_mut(&event.id)
            && turn.terminal.is_none()
        {
            if event.kind == Kind::Error {
                turn.terminal = Some(Err(event.error.unwrap_or_else(|| "unknown error".into())));
            } else {
                let text = event.text.unwrap_or_default();
                let hypothesis = Hypothesis {
                    words: timed_words(&text, event.words.unwrap_or_default()),
                    text,
                    stable: event.kind != Kind::Draft,
                    language: None,
                };
                if event.kind == Kind::Final {
                    turn.terminal = Some(Ok(hypothesis));
                } else if hypothesis.stable {
                    turn.stable = Some(hypothesis);
                    turn.draft = None;
                } else {
                    turn.draft = Some(hypothesis);
                }
            }
        }
        mailbox.updated.notify_all();
    };
    mailbox.disconnect(reason);
}

fn timed_words(text: &str, words: Vec<Word>) -> Vec<TimedWord> {
    let mut remaining = text;
    let mut previous_start = 0.0;
    let mut timed = Vec::with_capacity(words.len());
    for word in words {
        let Some(suffix) = remaining.strip_prefix(word.text.as_str()) else {
            return Vec::new();
        };
        if word.text.is_empty()
            || !word.start.is_finite()
            || !word.end.is_finite()
            || word.start < previous_start
            || word.end < word.start
        {
            return Vec::new();
        }
        timed.push(TimedWord {
            text: word.text,
            start_sample: (word.start * f64::from(SAMPLE_RATE)).round() as u64,
            end_sample: (word.end * f64::from(SAMPLE_RATE)).round() as u64,
        });
        remaining = suffix;
        previous_start = word.start;
    }
    if remaining.is_empty() {
        timed
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_audio_packs_short_spans_and_flushes_the_exact_pcm_tail() {
        let (stdin, mut peer) = UnixStream::pair().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mailbox = Arc::new(Mailbox::default());
        let recognizer = AppleRecognizer {
            stdin: Mutex::new(stdin),
            child: Command::new("/usr/bin/true").spawn().unwrap(),
            reader: None,
            mailbox: mailbox.clone(),
            next_stream: AtomicU64::new(1),
        };
        let worker = thread::spawn(move || {
            for (operation, samples) in [
                (OPEN_COMMITTED, 0),
                (APPEND, PCM_PACKET_SAMPLES),
                (APPEND, 3),
                (COMMIT, 0),
            ] {
                let mut header = [0; HEADER_BYTES];
                peer.read_exact(&mut header).unwrap();
                assert_eq!(header[0], operation);
                assert_eq!(u64::from_le_bytes(header[1..9].try_into().unwrap()), 1);
                let length = u32::from_le_bytes(header[9..].try_into().unwrap()) as usize;
                assert_eq!(length, 2 * samples);
                let mut payload = vec![0; length];
                peer.read_exact(&mut payload).unwrap();
                if samples == PCM_PACKET_SAMPLES {
                    let (samples, tail) = payload.as_chunks::<2>();
                    assert!(tail.is_empty() && samples.iter().all(|pcm| *pcm == [0, 64]));
                } else if samples == 3 {
                    assert_eq!(payload, [0, 128, 0, 0, 255, 127]);
                }
            }
            let mut state = mailbox.state.lock().unwrap();
            state.turns.get_mut(&1).unwrap().terminal = Some(Ok(Hypothesis::default()));
            mailbox.updated.notify_all();
        });
        let mut stream = recognizer.open(OPEN_COMMITTED).unwrap();
        for span in vec![0.5; PCM_PACKET_SAMPLES - 1].chunks(257) {
            assert!(stream.push(span).unwrap().is_none());
        }
        stream.push(&[0.5, -1.0, 0.0, 1.0]).unwrap();
        stream.finish().unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn helper_results_keep_latest_draft_and_independent_final_timing() {
        let mailbox = Mailbox::default();
        {
            let mut state = mailbox.state.lock().unwrap();
            state.turns.insert(1, TurnEvents::default());
            state.turns.insert(2, TurnEvents::default());
        }
        let trace = r#"{"kind":"ready","id":0}
{"kind":"draft","id":1,"text":"Old draft"}
{"kind":"draft","id":2,"text":"Other channel"}
{"kind":"stable","id":1,"text":"New"}
{"kind":"draft","id":1,"text":"New draft"}
{"kind":"final","id":2,"text":"Other channel","words":[{"text":"Other","start":0.125,"end":0.25},{"text":" channel","start":0.25,"end":0.5}]}
{"kind":"error","id":2,"error":"Late error"}
{"kind":"error","id":3,"error":"Canceled turn"}
"#;
        read_results(trace.as_bytes(), &mailbox);
        let mut state = mailbox.state.lock().unwrap();
        assert!(state.ready && state.disconnected.is_some());
        let turn = state.turns.get_mut(&1).unwrap();
        let stable = turn.take_partial().unwrap().unwrap();
        assert!(stable.stable);
        assert_eq!(stable.text, "New");
        let draft = turn.take_partial().unwrap().unwrap();
        assert_eq!(draft.text, "New draft");
        assert!(!draft.stable);
        let turn = state.turns.get_mut(&2).unwrap();
        assert!(turn.take_partial().unwrap().is_none());
        let final_result = turn.terminal.take().unwrap().unwrap();
        assert_eq!(final_result.text, "Other channel");
        assert!(final_result.stable);
        assert_eq!(final_result.words[0].start_sample, 2000);
        assert_eq!(final_result.words[1].end_sample, 8000);
        assert_eq!(final_result.words[1].text, " channel");
    }

    #[test]
    fn helper_preserves_turn_errors_and_drops_unreliable_timing() {
        let mailbox = Mailbox::default();
        {
            let mut state = mailbox.state.lock().unwrap();
            state.turns.insert(1, TurnEvents::default());
            state.turns.insert(2, TurnEvents::default());
            state.turns.insert(3, TurnEvents::default());
        }
        let trace = r#"{"kind":"error","id":1,"error":"Apple Speech input queue is full"}
{"kind":"final","id":1,"text":"Late final"}
{"kind":"final","id":2,"text":"First second","words":[{"text":"First","start":1.0,"end":1.5},{"text":" second","start":0.1,"end":0.25}]}
{"kind":"final","id":3,"text":"Partial coverage","words":[{"text":"Partial","start":0.0,"end":0.25}]}
partial result"#;
        read_results(trace.as_bytes(), &mailbox);
        let mut state = mailbox.state.lock().unwrap();
        assert_eq!(
            state.disconnected.as_deref(),
            Some("Apple Speech helper returned an incomplete or oversized result")
        );
        let turn = state.turns.get_mut(&1).unwrap();
        assert!(
            turn.take_partial()
                .unwrap_err()
                .to_string()
                .contains("input queue is full")
        );
        assert!(turn.terminal.take().unwrap().is_err());
        for (id, text) in [(2, "First second"), (3, "Partial coverage")] {
            let result = state
                .turns
                .get_mut(&id)
                .unwrap()
                .terminal
                .take()
                .unwrap()
                .unwrap();
            assert_eq!(result.text, text);
            assert!(result.words.is_empty());
        }
    }
}
