//! Persistent Apple Speech helper transport. Turn IDs are local to this process.
use super::recognizer::{
    Hypothesis, RecognitionOptions, RecognitionStream, Recognizer, SAMPLE_RATE, TimedWord,
};
use anyhow::{Context, Result, anyhow, bail};
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
use tokio::sync::{Notify, watch};

// Tracks the server admits. The Swift helper accepts a few more streams, because a
// bleed re-decode opens a committed stream beside a full set of live ones.
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
// A helper that misses this many finals in a row, with no final or turn error from it
// in between, is treated as dead. Each miss otherwise stalls the worker for a timeout.
const MAX_CONSECUTIVE_FINAL_TIMEOUTS: u32 = 3;
// A committed turn is decoded only after commit, so its wait also covers the audio.
// The slowest measured decode ran at about a tenth of the audio's duration.
const COMMITTED_AUDIO_PER_TIMEOUT_SECOND: u32 = 4;
const READY_TIMEOUT: Duration = Duration::from_secs(120);
// Without --offline the helper installs missing speech assets before it reports ready.
const INSTALL_TIMEOUT: Duration = Duration::from_secs(30 * 60);

pub(super) fn available() -> bool {
    let Some(helper) = crate::apple_native_helper() else {
        return false;
    };
    let Ok(mut child) = Command::new(helper)
        .arg("--stream-supported")
        .env_remove("TRANSCRIBERRR_API_KEY")
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
    /// Wait for a live turn's final. Tests shorten it per recognizer.
    final_timeout: Duration,
}

#[derive(Default)]
struct Mailbox {
    state: Mutex<MailState>,
    updated: Condvar,
    failure: watch::Sender<Option<String>>,
    /// Wakes the recognition worker when a requested final may have resolved.
    final_ready: Arc<Notify>,
}

impl Mailbox {
    fn disconnect(&self, reason: String) {
        let mut state = self.state.lock().unwrap();
        if state.disconnected.is_none() {
            state.disconnected = Some(reason.clone());
            self.failure.send_replace(Some(reason));
            self.updated.notify_all();
            self.final_ready.notify_one();
        }
    }
}

#[derive(Default)]
struct MailState {
    turns: HashMap<u64, TurnEvents>,
    ready: bool,
    startup_error: Option<String>,
    disconnected: Option<String>,
    consecutive_timeouts: u32,
    last_timeout: Option<Instant>,
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
        let stdout = stdin
            .try_clone()
            .context("clone Apple Speech result socket")?;
        let helper_output = helper_input
            .try_clone()
            .context("clone Apple Speech helper socket")?;
        let child = command
            // The helper has no use for the server's bearer key.
            .env_remove("TRANSCRIBERRR_API_KEY")
            .env(
                "TRANSCRIBERRR_NATIVE_PARENT_PID",
                std::process::id().to_string(),
            )
            .stdin(Stdio::from(OwnedFd::from(helper_input)))
            .stdout(Stdio::from(OwnedFd::from(helper_output)))
            .spawn()
            .context("start Apple Speech helper")?;
        let mailbox = Arc::new(Mailbox::default());
        let mut speech = Self {
            child,
            reader: None,
            stdin: Mutex::new(stdin),
            mailbox: Arc::clone(&mailbox),
            next_stream: AtomicU64::new(1),
            final_timeout: FINAL_TIMEOUT,
        };
        speech.reader = Some(
            thread::Builder::new()
                .name("apple-speech-results".into())
                .spawn(move || read_results(stdout, &mailbox))?,
        );
        {
            // Ending the wait early would kill the helper in the middle of a download,
            // and the next start would fail the same way.
            let timeout = if offline {
                READY_TIMEOUT
            } else {
                INSTALL_TIMEOUT
            };
            let state = speech.mailbox.state.lock().unwrap();
            let (state, _) = speech
                .mailbox
                .updated
                .wait_timeout_while(state, timeout, |state| {
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
        // A launcher can leave its stdout open in a descendant. Interrupt our
        // reader locally so joining it never depends on that descendant exiting.
        let _ = self.stdin.get_mut().unwrap().shutdown(Shutdown::Both);
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
            samples: 0,
            packet: header(APPEND, id).to_vec(),
            deadline: None,
        };
        self.control(operation, id)?;
        Ok(Box::new(stream))
    }
}

impl Recognizer for AppleRecognizer {
    fn backend_failure(&self) -> Option<watch::Receiver<Option<String>>> {
        Some(self.mailbox.failure.subscribe())
    }

    fn final_notify(&self) -> Option<Arc<Notify>> {
        Some(self.mailbox.final_ready.clone())
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
    samples: u64,
    packet: Vec<u8>,
    /// Set once the final has been requested.
    deadline: Option<Instant>,
}

impl AppleStream<'_> {
    fn final_timeout(&self) -> Duration {
        if self.committed {
            let rate = u64::from(SAMPLE_RATE * COMMITTED_AUDIO_PER_TIMEOUT_SECOND);
            self.parent.final_timeout + Duration::from_millis(self.samples * 1000 / rate)
        } else {
            self.parent.final_timeout
        }
    }

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
        self.samples += samples.len() as u64;
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
        let deadline = self
            .request_final()?
            .expect("Apple finals are asynchronous");
        let parent = self.parent;
        loop {
            if let Some(result) = self.try_final(Instant::now()) {
                return result;
            }
            let state = parent.mailbox.state.lock().unwrap();
            let _ = parent
                .mailbox
                .updated
                .wait_timeout_while(
                    state,
                    deadline.saturating_duration_since(Instant::now()),
                    |state| {
                        state.disconnected.is_none()
                            && state
                                .turns
                                .get(&self.id)
                                .is_some_and(|turn| turn.terminal.is_none())
                    },
                )
                .unwrap();
        }
    }

    fn request_final(&mut self) -> Result<Option<Instant>> {
        if let Some(deadline) = self.deadline {
            return Ok(Some(deadline));
        }
        self.flush()?;
        let awaiting = {
            let state = self.parent.mailbox.state.lock().unwrap();
            state.disconnected.is_none() && state.turns[&self.id].terminal.is_none()
        };
        if awaiting {
            // A failed send leaves a disconnect or a turn error for try_final to report.
            let _ = self.parent.control(COMMIT, self.id);
        }
        let deadline = Instant::now() + self.final_timeout();
        self.deadline = Some(deadline);
        Ok(Some(deadline))
    }

    fn try_final(&mut self, now: Instant) -> Option<Result<Hypothesis>> {
        let deadline = self.deadline?;
        let mailbox = &self.parent.mailbox;
        let mut state = mailbox.state.lock().unwrap();
        if let Some(result) = state
            .turns
            .get_mut(&self.id)
            .and_then(|events| events.terminal.take())
        {
            state.turns.remove(&self.id);
            return Some(result.map_err(|error| anyhow!("Apple Speech: {error}")));
        }
        if let Some(reason) = &state.disconnected {
            return Some(Err(anyhow!(
                "Apple Speech helper stopped before finalizing: {reason}"
            )));
        }
        if now < deadline {
            return None;
        }
        // Finals that were already waiting when another timed out share its stall.
        // Only a final requested after the last timeout counts as the next in a row.
        let requested = deadline.checked_sub(self.final_timeout());
        let fresh = state
            .last_timeout
            .zip(requested)
            .is_none_or(|(last, requested)| requested >= last);
        if fresh {
            // An uncounted timeout must not move the mark, or two tracks whose
            // turns overlap would keep each other from ever counting.
            state.last_timeout = Some(now);
            state.consecutive_timeouts += 1;
        }
        if state.consecutive_timeouts < MAX_CONSECUTIVE_FINAL_TIMEOUTS {
            return Some(Err(anyhow!(
                "Apple Speech helper timed out while finalizing"
            )));
        }
        drop(state);
        // Report it as a lost helper: sessions get the reason and the server exits
        // instead of stalling every later turn. Closing input lets a reading helper exit.
        let reason = format!(
            "Apple Speech helper stopped responding: \
             {MAX_CONSECUTIVE_FINAL_TIMEOUTS} turns in a row timed out while finalizing"
        );
        mailbox.disconnect(reason.clone());
        let _ = self.parent.stdin.lock().unwrap().shutdown(Shutdown::Write);
        Some(Err(anyhow!("{reason}")))
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
        if matches!(event.kind, Kind::Final | Kind::Error) && event.id != 0 {
            // Any terminal reply, even a late one for an abandoned turn, shows the
            // helper is still finalizing.
            state.consecutive_timeouts = 0;
        }
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
        let terminal = matches!(event.kind, Kind::Final | Kind::Error) && event.id != 0;
        drop(state);
        mailbox.updated.notify_all();
        if terminal {
            mailbox.final_ready.notify_one();
        }
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
            spoken: None,
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
    fn dropping_recognizer_interrupts_reader_while_helper_output_remains_open() {
        let (stdin, peer) = UnixStream::pair().unwrap();
        let results = stdin.try_clone().unwrap();
        let mailbox = Arc::new(Mailbox::default());
        let reader = {
            let mailbox = mailbox.clone();
            thread::spawn(move || read_results(results, &mailbox))
        };
        let recognizer = AppleRecognizer {
            stdin: Mutex::new(stdin),
            child: Command::new("/usr/bin/true").spawn().unwrap(),
            reader: Some(reader),
            mailbox,
            next_stream: AtomicU64::new(1),
            final_timeout: FINAL_TIMEOUT,
        };
        let (done, dropped) = std::sync::mpsc::channel();
        let shutdown = thread::spawn(move || {
            drop(recognizer);
            let _ = done.send(());
        });
        let result = dropped.recv_timeout(Duration::from_secs(2));
        drop(peer);
        shutdown.join().unwrap();
        result.expect("recognizer shutdown waited for a descendant to close stdout");
    }

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
            final_timeout: FINAL_TIMEOUT,
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
    fn a_committed_turn_waits_longer_for_its_final_as_it_carries_more_audio() {
        let stream = |committed, seconds: u64| {
            AppleStream {
                parent: &AppleRecognizer {
                    stdin: Mutex::new(UnixStream::pair().unwrap().0),
                    child: Command::new("/usr/bin/true").spawn().unwrap(),
                    reader: None,
                    mailbox: Arc::new(Mailbox::default()),
                    next_stream: AtomicU64::new(1),
                    final_timeout: FINAL_TIMEOUT,
                },
                id: 1,
                committed,
                samples: seconds * u64::from(SAMPLE_RATE),
                packet: header(APPEND, 1).to_vec(),
                deadline: None,
            }
            .final_timeout()
        };
        assert_eq!(stream(true, 0), FINAL_TIMEOUT);
        assert_eq!(stream(true, 60), FINAL_TIMEOUT + Duration::from_secs(15));
        // A live turn has already been decoded up to its tail.
        assert_eq!(stream(false, 60), FINAL_TIMEOUT);
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

    /// Closes the fake helper's input before the recognizer joins its reader, so a
    /// failed assertion ends the test instead of hanging it.
    struct CloseInput<'a>(&'a AppleRecognizer);

    impl Drop for CloseInput<'_> {
        fn drop(&mut self) {
            let _ = self.0.stdin.lock().unwrap().shutdown(Shutdown::Write);
        }
    }

    #[test]
    fn finals_that_time_out_together_count_as_one_stall() {
        let (stdin, mut peer) = UnixStream::pair().unwrap();
        let (results, helper_output) = UnixStream::pair().unwrap();
        let mailbox = Arc::new(Mailbox::default());
        let reader = {
            let mailbox = mailbox.clone();
            thread::spawn(move || read_results(results, &mailbox))
        };
        // The fake helper reads its input and never answers.
        let helper = thread::spawn(move || {
            let _output = helper_output;
            let mut sink = Vec::new();
            let _ = peer.read_to_end(&mut sink);
        });
        let recognizer = AppleRecognizer {
            stdin: Mutex::new(stdin),
            child: Command::new("/usr/bin/true").spawn().unwrap(),
            reader: Some(reader),
            mailbox: mailbox.clone(),
            next_stream: AtomicU64::new(1),
            final_timeout: Duration::from_millis(20),
        };
        let close = CloseInput(&recognizer);
        let timed_out = |stream: &mut Box<dyn RecognitionStream + '_>, deadline: Instant| {
            thread::sleep(deadline.saturating_duration_since(Instant::now()));
            let outcome = stream.try_final(Instant::now()).unwrap();
            outcome.unwrap_err().to_string()
        };

        // Four tracks wait for their finals at once, as the worker now allows.
        let mut together: Vec<_> = (0..4)
            .map(|_| recognizer.start(&RecognitionOptions::default()).unwrap())
            .collect();
        let mut latest = Instant::now();
        for stream in &mut together {
            stream.push(&[0.25; 160]).unwrap();
            latest = stream.request_final().unwrap().unwrap();
        }
        for stream in &mut together {
            assert_eq!(
                timed_out(stream, latest),
                "Apple Speech helper timed out while finalizing"
            );
        }
        {
            let state = mailbox.state.lock().unwrap();
            assert_eq!(state.consecutive_timeouts, 1);
            assert!(state.disconnected.is_none());
        }
        drop(together);

        // Finals requested after a timeout are the next in a row.
        let mut last = String::new();
        for _ in 0..2 {
            let mut next = recognizer.start(&RecognitionOptions::default()).unwrap();
            next.push(&[0.25; 160]).unwrap();
            let deadline = next.request_final().unwrap().unwrap();
            last = timed_out(&mut next, deadline);
        }
        assert!(last.contains("stopped responding"), "{last}");
        assert!(mailbox.state.lock().unwrap().disconnected.is_some());
        drop(close);
        drop(recognizer);
        helper.join().unwrap();
    }

    #[test]
    fn staggered_tracks_still_escalate_when_the_helper_never_answers() {
        let (stdin, mut peer) = UnixStream::pair().unwrap();
        let (results, helper_output) = UnixStream::pair().unwrap();
        let mailbox = Arc::new(Mailbox::default());
        let reader = {
            let mailbox = mailbox.clone();
            thread::spawn(move || read_results(results, &mailbox))
        };
        let helper = thread::spawn(move || {
            let _output = helper_output;
            let mut sink = Vec::new();
            let _ = peer.read_to_end(&mut sink);
        });
        let recognizer = AppleRecognizer {
            stdin: Mutex::new(stdin),
            child: Command::new("/usr/bin/true").spawn().unwrap(),
            reader: Some(reader),
            mailbox: mailbox.clone(),
            next_stream: AtomicU64::new(1),
            final_timeout: Duration::from_millis(50),
        };
        let close = CloseInput(&recognizer);
        // Requests a final and returns the stream with its deadline, unless the
        // backend has already failed.
        let request = || {
            let mut stream = recognizer.start(&RecognitionOptions::default()).ok()?;
            stream.push(&[0.25; 160]).ok()?;
            let deadline = stream.request_final().unwrap().unwrap();
            Some((stream, deadline))
        };
        let failed = || mailbox.state.lock().unwrap().disconnected.is_some();

        // Two microphones whose turns overlap: each final is requested while the
        // other track's is still waiting, so their timeouts interleave.
        let mut tracks = [request(), None];
        thread::sleep(Duration::from_millis(40));
        tracks[1] = request();
        'rounds: for _ in 0..6 {
            for track in &mut tracks {
                let Some((mut stream, deadline)) = track.take() else {
                    break 'rounds;
                };
                thread::sleep(deadline.saturating_duration_since(Instant::now()));
                assert!(stream.try_final(Instant::now()).unwrap().is_err());
                drop(stream);
                if failed() {
                    break 'rounds;
                }
                *track = request();
            }
        }
        assert!(
            failed(),
            "a helper that never answers must fail the backend"
        );
        drop(tracks);
        drop(close);
        drop(recognizer);
        helper.join().unwrap();
    }

    #[test]
    fn a_requested_final_is_polled_without_blocking_and_its_reply_wakes_the_worker() {
        let (stdin, mut peer) = UnixStream::pair().unwrap();
        let (results, mut helper_output) = UnixStream::pair().unwrap();
        let mailbox = Arc::new(Mailbox::default());
        let reader = {
            let mailbox = mailbox.clone();
            thread::spawn(move || read_results(results, &mailbox))
        };
        // The fake helper answers only turn 1, when it is committed.
        let helper = thread::spawn(move || {
            let mut header = [0; HEADER_BYTES];
            while peer.read_exact(&mut header).is_ok() {
                let id = u64::from_le_bytes(header[1..9].try_into().unwrap());
                let length = u32::from_le_bytes(header[9..].try_into().unwrap()) as usize;
                let mut payload = vec![0; length];
                peer.read_exact(&mut payload).unwrap();
                if header[0] == COMMIT && id == 1 {
                    helper_output
                        .write_all(b"{\"kind\":\"final\",\"id\":1,\"text\":\"Answered\"}\n")
                        .unwrap();
                }
            }
        });
        let recognizer = AppleRecognizer {
            stdin: Mutex::new(stdin),
            child: Command::new("/usr/bin/true").spawn().unwrap(),
            reader: Some(reader),
            mailbox: mailbox.clone(),
            next_stream: AtomicU64::new(1),
            final_timeout: Duration::from_secs(60),
        };
        let close = CloseInput(&recognizer);
        let notify = recognizer.final_notify().unwrap();
        let mut answered = recognizer.start(&RecognitionOptions::default()).unwrap();
        let mut unanswered = recognizer.start(&RecognitionOptions::default()).unwrap();
        answered.push(&[0.25; 160]).unwrap();
        unanswered.push(&[0.25; 160]).unwrap();

        let requested = Instant::now();
        let deadline = unanswered.request_final().unwrap().unwrap();
        assert!(unanswered.try_final(Instant::now()).is_none());
        assert!(requested.elapsed() < Duration::from_secs(5));
        // A second request neither resends the commit nor moves the deadline.
        assert_eq!(unanswered.request_final().unwrap(), Some(deadline));

        answered.request_final().unwrap().unwrap();
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(async {
                tokio::time::timeout(Duration::from_secs(10), notify.notified()).await
            })
            .expect("the helper's final did not wake the worker");
        let text = answered.try_final(Instant::now()).unwrap().unwrap().text;
        let timed_out = unanswered.try_final(deadline).unwrap().unwrap_err();
        drop((answered, unanswered, close));
        drop(recognizer);
        helper.join().unwrap();

        assert_eq!(text, "Answered");
        assert_eq!(
            timed_out.to_string(),
            "Apple Speech helper timed out while finalizing"
        );
    }

    #[test]
    fn consecutive_finalize_timeouts_fail_the_backend_and_a_helper_reply_resets_the_count() {
        let (stdin, mut peer) = UnixStream::pair().unwrap();
        let (results, mut helper_output) = UnixStream::pair().unwrap();
        let mailbox = Arc::new(Mailbox::default());
        let reader = {
            let mailbox = mailbox.clone();
            thread::spawn(move || read_results(results, &mailbox))
        };
        // The fake helper keeps reading every packet, so writes never fail, and
        // answers only turn 3, as soon as it opens.
        let helper = thread::spawn(move || {
            let mut header = [0; HEADER_BYTES];
            while peer.read_exact(&mut header).is_ok() {
                let id = u64::from_le_bytes(header[1..9].try_into().unwrap());
                let length = u32::from_le_bytes(header[9..].try_into().unwrap()) as usize;
                let mut payload = vec![0; length];
                peer.read_exact(&mut payload).unwrap();
                if header[0] == OPEN && id == 3 {
                    helper_output
                        .write_all(b"{\"kind\":\"final\",\"id\":3,\"text\":\"Answered\"}\n")
                        .unwrap();
                }
            }
        });
        let recognizer = AppleRecognizer {
            stdin: Mutex::new(stdin),
            child: Command::new("/usr/bin/true").spawn().unwrap(),
            reader: Some(reader),
            mailbox: mailbox.clone(),
            next_stream: AtomicU64::new(1),
            final_timeout: Duration::from_millis(50),
        };
        // Releases the fake helper even when the backend was never failed.
        let close = CloseInput(&recognizer);
        let failure = recognizer.backend_failure().unwrap();
        let mut outcomes = Vec::new();
        for id in 1..=6 {
            let outcome = recognizer
                .start(&RecognitionOptions::default())
                .and_then(|mut stream| {
                    if id == 3 {
                        let deadline = Instant::now() + Duration::from_secs(10);
                        while mailbox.state.lock().unwrap().turns[&3].terminal.is_none() {
                            assert!(Instant::now() < deadline, "the fake helper never answered");
                            thread::sleep(Duration::from_millis(5));
                        }
                    }
                    stream.push(&[0.25; 160])?;
                    stream.finish()
                })
                .map(|hypothesis| hypothesis.text)
                .map_err(|error| error.to_string());
            outcomes.push((outcome, failure.borrow().clone()));
        }
        let later = recognizer
            .start(&RecognitionOptions::default())
            .err()
            .map(|error| error.to_string());
        drop(close);
        drop(recognizer);
        helper.join().unwrap();

        let timed_out = || {
            (
                Err("Apple Speech helper timed out while finalizing".into()),
                None,
            )
        };
        assert_eq!(outcomes[0], timed_out());
        assert_eq!(outcomes[1], timed_out());
        assert_eq!(outcomes[2], (Ok("Answered".into()), None));
        // The reply reset the count, so two more timeouts still leave the helper in use.
        assert_eq!(outcomes[3], timed_out());
        assert_eq!(outcomes[4], timed_out());
        let (Err(error), Some(reason)) = &outcomes[5] else {
            panic!("the third consecutive timeout did not fail the backend: {outcomes:?}");
        };
        assert!(reason.contains("stopped responding"), "{reason}");
        assert!(error.contains(reason.as_str()), "{error}");
        assert_eq!(later.as_ref(), Some(reason));
    }
}
