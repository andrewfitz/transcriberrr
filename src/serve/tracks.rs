//! Multiplexed capture sessions and bounded result replay.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::ws::{CloseFrame, Message, Utf8Bytes, WebSocket};
use futures_util::{Sink, SinkExt, StreamExt};
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch};

pub(super) const RESUME_WINDOW: Duration = Duration::from_secs(30);
const NEGOTIATION_WINDOW: Duration = Duration::from_millis(250);
const SEND_TIMEOUT: Duration = Duration::from_secs(2);
const CLOSE_TIMEOUT: Duration = Duration::from_secs(1);
const GAP_WAIT: Duration = Duration::from_millis(250);
const MAX_PACKET_FRAMES: usize = 24_000;
const MAX_PENDING_FRAMES: usize = 48_000;
const MAX_PENDING_PACKETS: usize = 128;
const MAX_EVENTS: usize = 256;
const MAX_REPLAY_BYTES: usize = 4 * 1024 * 1024;
const OUTBOUND_EVENTS: usize = 64;
const OUTBOUND_BYTES: usize = 4 * 1024 * 1024;
const LAG_FRAMES: usize = 24_000;
const GLOBAL_LAG_COMMANDS: usize = 64;
const COMMAND_QUEUE: usize = 16;
const SLOW_READER: u16 = 4008;
const TAKEN_OVER: u16 = 4009;

#[derive(Default)]
pub(super) struct Registry {
    inner: Mutex<RegistryState>,
}

#[derive(Default)]
struct RegistryState {
    sessions: HashMap<String, SessionEndpoint>,
    completed: VecDeque<String>,
}

impl Registry {
    fn get(&self, id: &str) -> Option<SessionEndpoint> {
        self.inner.lock().unwrap().sessions.get(id).cloned()
    }

    fn contains(&self, id: &str) -> bool {
        self.inner.lock().unwrap().sessions.contains_key(id)
    }

    fn insert(&self, id: String, sender: SessionEndpoint) {
        self.inner.lock().unwrap().sessions.insert(id, sender);
    }

    fn complete(&self, id: &str, limit: usize) {
        let mut state = self.inner.lock().unwrap();
        state.completed.push_back(id.to_owned());
        while state.completed.len() > limit
            && let Some(oldest) = state.completed.pop_front()
        {
            state.sessions.remove(&oldest);
        }
    }

    fn remove(&self, id: &str) {
        let mut state = self.inner.lock().unwrap();
        state.sessions.remove(id);
        state.completed.retain(|completed| completed != id);
    }
}

fn random_id() -> anyhow::Result<String> {
    use std::fmt::Write as _;
    let mut bytes = [0; 32];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| anyhow::anyhow!("could not generate session credential"))?;
    let mut id = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(id, "{byte:02x}");
    }
    Ok(id)
}

fn error(code: &str, message: &str) -> Value {
    json!({"type": "error", "error": {"type": "invalid_request_error", "code": code, "message": message}})
}

struct Replay {
    next: u64,
    bytes: usize,
    events: VecDeque<(u64, Utf8Bytes)>,
}

impl Default for Replay {
    fn default() -> Self {
        Self {
            next: 1,
            bytes: 0,
            events: VecDeque::new(),
        }
    }
}

impl Replay {
    fn emit(&mut self, mut event: Value) -> Utf8Bytes {
        let sequence = self.next;
        self.next += 1;
        event["sequence"] = sequence.into();
        let text: Utf8Bytes = event.to_string().into();
        self.bytes += text.len();
        self.events.push_back((sequence, text.clone()));
        while self.events.len() > MAX_EVENTS || self.bytes > MAX_REPLAY_BYTES {
            self.drop_first();
        }
        text
    }

    fn drop_first(&mut self) {
        if let Some((_, text)) = self.events.pop_front() {
            self.bytes -= text.len();
        }
    }

    fn earliest(&self) -> u64 {
        self.events.front().map_or(self.next, |(seq, _)| *seq)
    }

    fn ack(&mut self, cursor: u64) -> Result<(), Value> {
        self.validate(cursor)?;
        while self.events.front().is_some_and(|(seq, _)| *seq <= cursor) {
            self.drop_first();
        }
        Ok(())
    }

    fn validate(&self, cursor: u64) -> Result<(), Value> {
        if cursor >= self.next {
            return Err(error(
                "invalid_sequence",
                "The cursor is ahead of the session.",
            ));
        }
        Ok(())
    }

    fn resume(&self, cursor: u64, accept_loss: bool) -> Result<ReplayPlan, Value> {
        self.validate(cursor)?;
        let earliest = self.earliest();
        let lost = (cursor + 1 < earliest).then(|| [cursor + 1, earliest - 1]);
        if lost.is_some() && !accept_loss {
            let mut event = error(
                "replay_unavailable",
                "The requested events are outside replay history.",
            );
            event["error"]["earliest_replayable_sequence"] = earliest.into();
            return Err(event);
        }
        let events = self
            .events
            .iter()
            .filter(|(seq, _)| *seq > cursor)
            .map(|(_, text)| text.clone())
            .collect();
        Ok(ReplayPlan {
            earliest,
            lost,
            events,
        })
    }
}

struct ReplayPlan {
    earliest: u64,
    lost: Option<[u64; 2]>,
    events: Vec<Utf8Bytes>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Isolated,
    Mixed,
}

#[derive(Default)]
struct Clock {
    expected: u64,
    pending: BTreeMap<u64, PendingPacket>,
    deadline: Option<Instant>,
    pending_frames: usize,
}

struct PendingPacket {
    samples: Vec<i16>,
    deadline: Instant,
}

enum Audio {
    Samples {
        start: u64,
        samples: Vec<i16>,
    },
    Gap {
        start: u64,
        end: u64,
        reason: &'static str,
    },
}

impl Clock {
    fn append(&mut self, start: u64, samples: Vec<i16>, now: Instant) -> Vec<Audio> {
        let mut ready = Vec::new();
        if start <= self.expected {
            self.accept(start, samples, &mut ready);
        } else {
            let deadline = now + GAP_WAIT;
            if let Some(prior) = self.pending.get_mut(&start) {
                if samples.len() > prior.samples.len() {
                    self.pending_frames += samples.len() - prior.samples.len();
                    prior.samples = samples;
                }
            } else {
                self.pending_frames += samples.len();
                self.pending
                    .insert(start, PendingPacket { samples, deadline });
                self.deadline.get_or_insert(deadline);
            }
        }
        self.drain(now, false, &mut ready);
        ready
    }

    fn accept(&mut self, start: u64, samples: Vec<i16>, ready: &mut Vec<Audio>) {
        let received_end = start + samples.len() as u64;
        if received_end <= self.expected {
            return;
        }
        let accepted_start = self.expected;
        let offset = (self.expected - start) as usize;
        self.expected = received_end;
        let samples = if offset == 0 {
            samples
        } else {
            samples[offset..].to_vec()
        };
        ready.push(Audio::Samples {
            start: accepted_start,
            samples,
        });
    }

    fn resume(&mut self, now: Instant) {
        let deadline = now + GAP_WAIT;
        for packet in self.pending.values_mut() {
            packet.deadline = deadline;
        }
        self.deadline = (!self.pending.is_empty()).then_some(deadline);
    }

    fn delay_gap_deadlines(&mut self, paused_at: Instant, now: Instant) {
        // Count only time during which this connection could submit a repair.
        // A takeover can grant a fresh GAP_WAIT during the suspension.
        for packet in self.pending.values_mut() {
            let remaining = packet
                .deadline
                .saturating_duration_since(paused_at)
                .min(GAP_WAIT);
            packet.deadline = now + remaining;
        }
        self.deadline = self.pending.values().map(|packet| packet.deadline).min();
    }

    fn skip_to(&mut self, end: u64, reason: &'static str, ready: &mut Vec<Audio>) {
        if end > self.expected {
            let start = self.expected;
            self.expected = end;
            ready.push(Audio::Gap { start, end, reason });
        }
    }

    fn drain(&mut self, now: Instant, force: bool, ready: &mut Vec<Audio>) {
        let expired = self.deadline.is_some_and(|deadline| deadline <= now);
        let through = if force {
            u64::MAX
        } else if expired {
            self.pending
                .iter()
                .rev()
                .find_map(|(&start, packet)| (packet.deadline <= now).then_some(start))
                .unwrap_or(self.expected)
        } else {
            self.expected
        };
        self.drain_through(through, "missing_audio", ready);
        if expired {
            self.deadline = self.pending.values().map(|packet| packet.deadline).min();
        }
    }

    fn drain_through(&mut self, through: u64, reason: &'static str, ready: &mut Vec<Audio>) {
        while let Some((&start, _)) = self.pending.first_key_value() {
            if start > self.expected {
                if start > through
                    && self.pending_frames <= MAX_PENDING_FRAMES
                    && self.pending.len() <= MAX_PENDING_PACKETS
                {
                    break;
                }
                self.skip_to(start, reason, ready);
            }
            let (_, packet) = self.pending.pop_first().expect("first pending packet");
            self.pending_frames -= packet.samples.len();
            self.accept(start, packet.samples, ready);
        }
        if self.pending.is_empty() {
            self.deadline = None;
        }
    }

    fn client_gap(&mut self, end: u64, ready: &mut Vec<Audio>) {
        // The client can no longer retransmit through this frame. Keep samples
        // that already reached the reorder buffer before the disconnect.
        self.drain_through(end, "client_gap", ready);
        self.skip_to(end, "client_gap", ready);
    }
}

struct QueuedEvent {
    text: Utf8Bytes,
    _bytes: OwnedSemaphorePermit,
}

struct Client {
    owner: u64,
    events: Option<mpsc::Sender<QueuedEvent>>,
    bytes: Arc<Semaphore>,
    close: watch::Sender<Option<(u16, &'static str)>>,
}

impl Client {
    fn send(&self, text: Utf8Bytes) -> bool {
        let Some(events) = &self.events else {
            return true;
        };
        let Ok(size) = u32::try_from(text.len()) else {
            return false;
        };
        let Ok(bytes) = Arc::clone(&self.bytes).try_acquire_many_owned(size) else {
            return false;
        };
        events
            .try_send(QueuedEvent {
                text,
                _bytes: bytes,
            })
            .is_ok()
    }
}

#[derive(serde::Deserialize)]
struct Resume {
    session_id: String,
    resume_credential: String,
    last_sequence: u64,
    #[serde(default)]
    accept_loss: bool,
}

struct Attachment {
    owner: u64,
    announcement: Value,
    replay: Vec<Utf8Bytes>,
}

struct Attach {
    resume: Resume,
    client: Client,
    reply: oneshot::Sender<Result<Attachment, Value>>,
}

#[derive(Clone)]
struct SessionEndpoint {
    input: mpsc::Sender<Command>,
    takeover: mpsc::Sender<Attach>,
}

impl SessionEndpoint {
    fn channel() -> (Self, mpsc::Receiver<Command>, mpsc::Receiver<Attach>) {
        let (input, commands) = mpsc::channel(COMMAND_QUEUE);
        let (takeover, attachments) = mpsc::channel(1);
        (Self { input, takeover }, commands, attachments)
    }
}

enum Command {
    Input { owner: u64, message: Value },
    Invalid { owner: u64, failure: Value },
    Detach { owner: u64 },
}

type EventReceiver = mpsc::Receiver<QueuedEvent>;
type CloseReceiver = watch::Receiver<Option<(u16, &'static str)>>;

fn connection() -> (Client, EventReceiver, CloseReceiver) {
    let (events, receiver) = mpsc::channel(OUTBOUND_EVENTS);
    let (close, close_receiver) = watch::channel(None);
    (
        Client {
            owner: 1,
            events: Some(events),
            bytes: Arc::new(Semaphore::new(OUTBOUND_BYTES)),
            close,
        },
        receiver,
        close_receiver,
    )
}

enum Input {
    Text(Value),
    Invalid(Value),
    Ignore,
    PeerClose,
    Closed,
}

fn parse_message(message: Result<Message, axum::Error>) -> Input {
    match message {
        Ok(Message::Text(text)) => match serde_json::from_str::<Value>(&text) {
            Ok(value) if value.is_object() => Input::Text(value),
            _ => Input::Invalid(error("invalid_json", "Expected a JSON event object.")),
        },
        Ok(Message::Binary(_)) => {
            Input::Invalid(error("invalid_message", "Expected a text message."))
        }
        Ok(Message::Close(_)) => Input::PeerClose,
        Err(_) => Input::Closed,
        _ => Input::Ignore,
    }
}

pub(super) async fn run(mut socket: WebSocket, state: Arc<super::AppState>, resume_only: bool) {
    let mut shutdown = state.shutdown.clone();
    let deadline = tokio::time::sleep(if resume_only {
        RESUME_WINDOW
    } else {
        NEGOTIATION_WINDOW
    });
    tokio::pin!(deadline);
    let mut resuming = resume_only;
    loop {
        if *shutdown.borrow() {
            return;
        }
        let initial = tokio::select! {
            _ = &mut deadline => break,
            _ = shutdown.changed() => return,
            incoming = socket.recv() => incoming,
        };
        match initial.map(parse_message) {
            Some(Input::Text(message))
                if event_type(&message) == "transcriberrr.session.resume" =>
            {
                if !resuming {
                    resuming = true;
                    deadline.set(tokio::time::sleep(RESUME_WINDOW));
                }
                match serde_json::from_value::<Resume>(message) {
                    Ok(resume) => {
                        let sender = state.native.get(&resume.session_id);
                        if let Some(sender) = sender {
                            let (client, events, close) = connection();
                            let (reply, response) = oneshot::channel();
                            let command = Attach {
                                resume,
                                client,
                                reply,
                            };
                            let attach = async {
                                sender.takeover.send(command).await.ok()?;
                                response.await.ok()
                            };
                            match tokio::select! {
                                _ = &mut deadline => return,
                                _ = shutdown.changed() => return,
                                result = attach => result,
                            } {
                                Some(Ok(attachment)) => {
                                    connected(socket, sender.input, attachment, events, close)
                                        .await;
                                    return;
                                }
                                Some(Err(failure)) => {
                                    if send_json(&mut socket, &failure).await.is_err() {
                                        return;
                                    }
                                    // A client can explicitly accept replay loss on this socket.
                                    continue;
                                }
                                None => {}
                            }
                        }
                        if send_json(
                            &mut socket,
                            &error("session_expired", "The session is no longer available."),
                        )
                        .await
                        .is_err()
                        {
                            return;
                        }
                    }
                    Err(_) => {
                        if send_json(
                            &mut socket,
                            &error(
                                "invalid_resume",
                                "Expected a session ID, credential, and last sequence.",
                            ),
                        )
                        .await
                        .is_err()
                        {
                            return;
                        }
                    }
                }
            }
            Some(Input::Text(_)) if resume_only => {
                let _ = send_json(
                    &mut socket,
                    &error(
                        "expected_resume",
                        "Send a session resume event on this socket.",
                    ),
                )
                .await;
                return;
            }
            Some(Input::Text(_)) if resuming => return,
            Some(Input::Text(message)) => {
                start_session(socket, state, Some(message)).await;
                return;
            }
            Some(Input::Invalid(failure)) => {
                if send_json(&mut socket, &failure).await.is_err() {
                    return;
                }
            }
            Some(Input::Ignore) => {}
            Some(Input::PeerClose) => {
                let _ = tokio::time::timeout(CLOSE_TIMEOUT, socket.flush()).await;
                return;
            }
            Some(Input::Closed) | None => return,
        }
    }
    if !resuming {
        start_session(socket, state, None).await;
    }
}

enum SendFailure {
    Stalled,
    Closed,
}

async fn send_text(
    socket: &mut (impl Sink<Message, Error = axum::Error> + Unpin),
    text: Utf8Bytes,
) -> Result<(), SendFailure> {
    match tokio::time::timeout(SEND_TIMEOUT, socket.send(Message::Text(text))).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) => Err(SendFailure::Closed),
        Err(_) => Err(SendFailure::Stalled),
    }
}

async fn send_json(socket: &mut WebSocket, value: &Value) -> Result<(), SendFailure> {
    send_text(socket, value.to_string().into()).await
}

async fn start_session(mut socket: WebSocket, state: Arc<super::AppState>, initial: Option<Value>) {
    let Ok(permit) = Arc::clone(&state.active_sessions).try_acquire_owned() else {
        let _ = send_json(
            &mut socket,
            &error("session_limit", "The server is at its session limit."),
        )
        .await;
        return;
    };
    let (Ok(id), Ok(credential)) = (random_id(), random_id()) else {
        let _ = send_json(
            &mut socket,
            &error("internal_error", "The session could not be created."),
        )
        .await;
        return;
    };
    let (sender, receiver, attachments) = SessionEndpoint::channel();
    let (client, events, close) = connection();
    let mut session = Session::new(id, credential, client, state.clone());
    session.permit = Some(permit);
    let announcement = session.initial_announcement();
    state.native.insert(session.id.clone(), sender.clone());
    if let Some(message) = initial {
        session.input(message);
    }
    tokio::spawn(async move {
        session.run(receiver, attachments).await;
        session.state.native.remove(&session.id);
    });
    connected(
        socket,
        sender.input,
        Attachment {
            owner: 1,
            announcement,
            replay: Vec::new(),
        },
        events,
        close,
    )
    .await;
}

async fn connected(
    socket: WebSocket,
    sender: mpsc::Sender<Command>,
    attachment: Attachment,
    mut events: EventReceiver,
    mut close: CloseReceiver,
) {
    let (mut writer, mut reader) = socket.split();
    let owner = attachment.owner;
    let reading = async {
        while let Some(message) = reader.next().await {
            let command = match parse_message(message) {
                Input::Text(message) => Command::Input { owner, message },
                Input::Invalid(failure) => Command::Invalid { owner, failure },
                Input::Ignore => continue,
                Input::PeerClose => return true,
                Input::Closed => return false,
            };
            if sender.send(command).await.is_err() {
                return false;
            }
        }
        false
    };
    let writing = async {
        let delivery = async {
            send_text(&mut writer, attachment.announcement.to_string().into()).await?;
            for event in attachment.replay {
                send_text(&mut writer, event).await?;
            }
            while let Some(event) = events.recv().await {
                send_text(&mut writer, event.text).await?;
            }
            Ok(())
        };
        let closing = async {
            if close.wait_for(Option::is_some).await.is_err() {
                std::future::pending::<()>().await;
            }
            (*close.borrow()).expect("close reason")
        };
        tokio::select! {
            biased;
            notice = closing => Some(notice),
            result = delivery => match result {
                Ok(()) => Some((1000, "session ended")),
                Err(SendFailure::Stalled) => Some((SLOW_READER, "outbound delivery stalled")),
                Err(SendFailure::Closed) => None,
            }
        }
    };
    let (peer_closed, close_notice) = tokio::select! {
        peer_closed = reading => (peer_closed, None),
        notice = writing => (false, notice),
    };
    let closing = async {
        if let Some((code, reason)) = close_notice {
            let frame = CloseFrame {
                code,
                reason: reason.into(),
            };
            let _ = writer.send(Message::Close(Some(frame))).await;
        } else if peer_closed {
            // Tungstenite queues the close echo when reading a peer's close frame.
            let _ = writer.flush().await;
        }
    };
    let _ = tokio::join!(
        sender.send(Command::Detach { owner }),
        tokio::time::timeout(CLOSE_TIMEOUT, closing),
    );
}

fn event_type(message: &Value) -> &str {
    message
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

use super::live_bleed::{Context, DRAFT_LOOKAHEAD, FINAL_LOOKAHEAD, LiveBleed, Quality};
use super::pipeline::{
    MAX_NATIVE_FRAMES, MAX_QUEUED_FRAMES, PipelineEvent, PipelineUpdate, QueueError,
    TIMED_ROLLOVER_TAIL_FRAMES, TrackHandle, TrackSettings, UNTIMED_ROLLOVER_TAIL_FRAMES,
};
use super::recognizer::{Hypothesis, RecognitionOptions};

const MAX_CAPTURE_QUEUE: usize = 128;
const MAX_RAW_PACKETS: usize = 512;
const MAX_RAW_FRAMES: usize = 2 * MAX_NATIVE_FRAMES as usize;
const FINAL_WAIT: Duration = Duration::from_millis(1_250);
const DIARIZATION_WAIT: Duration = Duration::from_secs(2);

enum CaptureAction {
    Audio(Audio),
    Commit { end: bool },
}

struct Capture {
    action: CaptureAction,
    deadline: Instant,
}

struct Output {
    event: PipelineEvent,
    deadline: Instant,
    redo: Option<Redo>,
    quality: Quality,
}

struct Redo {
    worker: TrackHandle,
    start: u64,
    remaining: VecDeque<Vec<i16>>,
    committed: bool,
}

struct RawPacket {
    start: u64,
    original: Vec<i16>,
    // Only allocate a second copy for packets changed by the draft gate.
    gated: Option<Vec<i16>>,
}

impl RawPacket {
    fn end(&self) -> u64 {
        self.start + self.original.len() as u64
    }
}

struct Track {
    key: u64,
    role: Role,
    speaker: Option<String>,
    clock: Clock,
    handle: TrackHandle,
    markers: VecDeque<Marker>,
    next_marker: u64,
    ending: bool,
    draft: Option<(u64, u64)>,
    last_terminal: Option<u64>,
    capture: VecDeque<Capture>,
    capture_frames: usize,
    raw: VecDeque<RawPacket>,
    raw_frames: usize,
    outputs: VecDeque<Output>,
    lagging: bool,
    #[cfg(feature = "diarization")]
    diarization: Option<super::diarization::Stream>,
    diarize: bool,
}

enum MarkerAction {
    Commit,
    End,
    Gap {
        start: u64,
        end: u64,
        reason: &'static str,
    },
}

struct Marker {
    id: u64,
    action: MarkerAction,
}

impl Track {
    fn close_item(&mut self, turn: u64) {
        self.draft = None;
        self.last_terminal = Some(turn);
    }

    fn check_control_capacity(&self) -> Result<(), Value> {
        if self.markers.len() + self.capture.len() >= MAX_CAPTURE_QUEUE {
            return Err(error(
                "track_busy",
                "The track is waiting for prior audio and control events.",
            ));
        }
        Ok(())
    }
}

struct Session {
    state: Arc<super::AppState>,
    id: String,
    credential: String,
    client: Option<Client>,
    owner: u64,
    detached_at: Option<Instant>,
    replay: Replay,
    tracks: BTreeMap<String, Track>,
    frame_zero_time: Option<String>,
    ending: bool,
    ended: bool,
    shutdown_deadline: Option<Instant>,
    input_paused_at: Option<Instant>,
    permit: Option<OwnedSemaphorePermit>,
    bleed: LiveBleed,
    next_key: u64,
    ended_isolated: VecDeque<u64>,
}

impl Session {
    fn new(id: String, credential: String, client: Client, state: Arc<super::AppState>) -> Self {
        Self {
            state,
            id,
            credential,
            client: Some(client),
            owner: 1,
            detached_at: None,
            replay: Replay::default(),
            tracks: BTreeMap::new(),
            frame_zero_time: None,
            ending: false,
            ended: false,
            shutdown_deadline: None,
            input_paused_at: None,
            permit: None,
            bleed: LiveBleed::default(),
            next_key: 0,
            ended_isolated: VecDeque::new(),
        }
    }

    fn initial_announcement(&self) -> Value {
        json!({
            "type": "transcriberrr.session.created",
            "version": 1,
            "session_id": self.id,
            "resume_credential": self.credential,
            "engine": self.state.engine,
            "model": self.state.model,
            "sample_rate": 24_000,
            "resume_window_ms": RESUME_WINDOW.as_millis() as u64,
            "capabilities": {
                "drafts": self.state.capabilities.stable_deltas || self.state.capabilities.revisable_drafts,
                "revisable_drafts": self.state.capabilities.revisable_drafts,
                "stable_deltas": self.state.capabilities.stable_deltas,
                "word_timing": self.state.capabilities.word_timing,
                "word_timing_granularity": self.state.capabilities.word_timing_granularity,
                "draft_word_timing": self.state.capabilities.draft_word_timing,
                "languages": self.state.capabilities.languages,
                "diarization": self.diarization_available(),
                "max_diarized_tracks": usize::from(self.diarization_available()),
            },
            "limits": {
                "max_tracks": self.state.max_tracks,
                "max_sessions": self.state.max_sessions,
                "max_completed_sessions": self.state.max_sessions,
                "max_append_frames": MAX_PACKET_FRAMES,
                "max_item_frames": MAX_NATIVE_FRAMES,
                "timed_rollover_tail_frames": TIMED_ROLLOVER_TAIL_FRAMES,
                "untimed_rollover_tail_frames": UNTIMED_ROLLOVER_TAIL_FRAMES,
                "gap_wait_ms": GAP_WAIT.as_millis() as u64,
                "max_pending_audio_frames": MAX_PENDING_FRAMES,
                "max_pending_audio_packets": MAX_PENDING_PACKETS,
                "max_replay_events": MAX_EVENTS,
                "max_replay_bytes": MAX_REPLAY_BYTES,
                "max_outbound_events": OUTBOUND_EVENTS,
                "max_outbound_bytes": OUTBOUND_BYTES,
                "backlog_warning_frames": LAG_FRAMES,
                "max_recognition_queue_frames": MAX_QUEUED_FRAMES,
                "input_pause_capture_frames": MAX_PENDING_FRAMES,
                "input_pause_capture_packets": MAX_CAPTURE_QUEUE,
            },
        })
    }

    async fn run(
        &mut self,
        mut commands: mpsc::Receiver<Command>,
        mut attachments: mpsc::Receiver<Attach>,
    ) {
        let mut tick = tokio::time::interval(Duration::from_millis(10));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut shutdown = self.state.shutdown.clone();
        loop {
            let now = Instant::now();
            if self.is_expired(now) {
                break;
            }
            if *shutdown.borrow() && self.shutdown_deadline.is_none() {
                if self.client.is_none() {
                    break;
                }
                self.shutdown_deadline = Some(now + Duration::from_secs(8));
                self.end();
            }
            if self
                .shutdown_deadline
                .is_some_and(|deadline| now >= deadline || self.ended)
            {
                break;
            }
            let input_paused = self.update_input_pressure(now);
            tokio::select! {
                _ = tick.tick() => self.poll(),
                _ = shutdown.changed(), if self.shutdown_deadline.is_none() => {}
                attachment = attachments.recv() => match attachment {
                    Some(Attach { resume, client, reply }) => self.attach(resume, client, reply),
                    None => break,
                },
                command = commands.recv(), if !input_paused => match command {
                    Some(Command::Input { owner, message }) if self.is_owner(owner) => self.input(message),
                    Some(Command::Invalid { owner, failure }) if self.is_owner(owner) => self.fail(failure),
                    Some(Command::Detach { owner }) if self.is_owner(owner) => {
                        self.client = None;
                        self.detached_at.get_or_insert_with(Instant::now);
                    }
                    Some(_) => {}
                    None => break,
                },
            }
        }
    }

    fn input_paused(&self) -> bool {
        self.tracks.values().any(|track| {
            track.capture_frames >= MAX_PENDING_FRAMES
                || track.capture.len() + track.markers.len() >= MAX_CAPTURE_QUEUE
        })
    }

    fn update_input_pressure(&mut self, now: Instant) -> bool {
        let paused = self.input_paused();
        if paused {
            self.input_paused_at.get_or_insert(now);
        } else if let Some(paused_at) = self.input_paused_at.take() {
            for track in self.tracks.values_mut() {
                track.clock.delay_gap_deadlines(paused_at, now);
            }
        }
        paused
    }

    fn is_owner(&self, owner: u64) -> bool {
        self.client
            .as_ref()
            .is_some_and(|client| client.owner == owner)
    }

    fn is_expired(&self, now: Instant) -> bool {
        self.detached_at
            .is_some_and(|at| now.duration_since(at) >= RESUME_WINDOW)
            || (self.ended && !self.state.native.contains(&self.id))
    }

    fn attach(
        &mut self,
        resume: Resume,
        mut client: Client,
        reply: oneshot::Sender<Result<Attachment, Value>>,
    ) {
        let now = Instant::now();
        if self.is_expired(now) {
            let _ = reply.send(Err(error(
                "session_expired",
                "The session is no longer available.",
            )));
            return;
        }
        let credential_matches = {
            use ring::hmac;
            let key = hmac::Key::new(hmac::HMAC_SHA256, self.credential.as_bytes());
            let tag = hmac::sign(&key, b"transcriberrr resume");
            let candidate = hmac::Key::new(hmac::HMAC_SHA256, resume.resume_credential.as_bytes());
            hmac::verify(&candidate, b"transcriberrr resume", tag.as_ref()).is_ok()
        };
        if !credential_matches {
            let _ = reply.send(Err(error(
                "invalid_resume_credential",
                "The resume credential is invalid.",
            )));
            return;
        }
        let ReplayPlan {
            earliest,
            lost,
            events,
        } = match self.replay.resume(resume.last_sequence, resume.accept_loss) {
            Ok(plan) => plan,
            Err(failure) => {
                let _ = reply.send(Err(failure));
                return;
            }
        };
        let owner = self.owner + 1;
        let next_frames: BTreeMap<_, _> = self
            .tracks
            .iter()
            .map(|(id, track)| (id, track.clock.expected))
            .collect();
        let announcement = json!({
            "type": "transcriberrr.session.resumed",
            "session_id": self.id,
            "earliest_replayable_sequence": earliest,
            "next_expected_frames": next_frames,
            "lost_range": lost,
        });
        if reply
            .send(Ok(Attachment {
                owner,
                announcement,
                replay: events,
            }))
            .is_err()
        {
            return;
        }
        if let Some(old) = self.client.take() {
            let _ = old.close.send(Some((TAKEN_OVER, "session taken over")));
        }
        client.owner = owner;
        self.owner = owner;
        if self.ended {
            client.events = None;
        } else {
            for track in self.tracks.values_mut() {
                track.clock.resume(now);
            }
            self.detached_at = None;
        }
        self.client = Some(client);
    }

    fn send(&mut self, text: Utf8Bytes) {
        if self
            .client
            .as_ref()
            .is_some_and(|client| !client.send(text))
            && let Some(client) = self.client.take()
        {
            let _ = client
                .close
                .send(Some((SLOW_READER, "outbound queue full")));
            self.detached_at = Some(Instant::now());
        }
    }

    fn emit(&mut self, event: Value) {
        let text = self.replay.emit(event);
        self.send(text);
    }

    fn fail(&mut self, failure: Value) {
        self.send(failure.to_string().into());
    }

    fn input(&mut self, message: Value) {
        if let Err(failure) = self.handle(&message) {
            self.fail(failure);
        }
    }

    fn handle(&mut self, message: &Value) -> Result<(), Value> {
        let kind = event_type(message);
        if kind == "transcriberrr.session.ack" {
            return self.replay.ack(unsigned(message, "sequence")?);
        }
        // Ended sessions retain their final events until the client detaches and the resume window ends.
        if self.ending {
            return Err(error("session_ending", "The session is ending."));
        }
        match kind {
            "transcriberrr.session.start" => self.set_clock(message)?,
            "transcriberrr.session.end" => self.end(),
            "transcriberrr.track.register" => self.register(message)?,
            "transcriberrr.track.append" => self.append(message)?,
            "transcriberrr.track.commit" => self.commit(track_id(message)?, false)?,
            "transcriberrr.track.end" => self.commit(track_id(message)?, true)?,
            "transcriberrr.track.gap" => self.explicit_gap(message)?,
            _ => {
                return Err(error(
                    "unsupported_event",
                    "The event type is not supported.",
                ));
            }
        }
        Ok(())
    }

    fn set_clock(&mut self, message: &Value) -> Result<(), Value> {
        let frame_zero_time = optional_string(message, "frame_zero_time")?.ok_or_else(|| {
            error(
                "invalid_clock",
                "Expected frame_zero_time in RFC 3339 format.",
            )
        })?;
        if self.next_key != 0
            || self.frame_zero_time.is_some()
            || chrono::DateTime::parse_from_rfc3339(frame_zero_time).is_err()
        {
            return Err(error(
                "invalid_clock",
                "Expected one RFC 3339 frame_zero_time before track registration.",
            ));
        }
        self.frame_zero_time = Some(frame_zero_time.into());
        self.emit(
            json!({"type":"transcriberrr.session.started", "frame_zero_time": frame_zero_time}),
        );
        Ok(())
    }

    fn register(&mut self, message: &Value) -> Result<(), Value> {
        let id = track_id(message)?;
        if self.tracks.contains_key(id) {
            return Err(error("duplicate_track", "The track is already registered."));
        }
        if self.tracks.len() >= self.state.max_tracks {
            return Err(error("track_limit", "The session is at its track limit."));
        }
        let role = match message.get("role").and_then(Value::as_str) {
            Some("isolated_mic") => Role::Isolated,
            Some("mixed_audio") => Role::Mixed,
            _ => {
                return Err(error(
                    "invalid_role",
                    "Expected role isolated_mic or mixed_audio.",
                ));
            }
        };
        let speaker = optional_string(message, "speaker")?;
        if speaker.is_some_and(|name| name.len() > 256) {
            return Err(error(
                "invalid_speaker",
                "Speaker labels may contain up to 256 UTF-8 bytes.",
            ));
        }
        let diarize = match message.get("diarization") {
            None => false,
            Some(value) => value.as_bool().ok_or_else(|| {
                error(
                    "invalid_diarization",
                    "Expected a boolean diarization option.",
                )
            })?,
        };
        if diarize && (role != Role::Mixed || !self.diarization_available()) {
            return Err(error(
                "unsupported_diarization",
                "Live diarization is unavailable for this track.",
            ));
        }
        #[cfg(feature = "diarization")]
        let diarization = if diarize {
            Some(
                self.state
                    .diarization
                    .as_ref()
                    .and_then(|worker| worker.reserve())
                    .ok_or_else(|| {
                        error("diarization_busy", "The live diarizer is already reserved.")
                    })?,
            )
        } else {
            None
        };
        let expected = if message.get("start_frame").is_some() {
            unsigned(message, "start_frame")?
        } else {
            0
        };
        let handle = self.state.worker.track(TrackSettings {
            native: true,
            streaming: true,
            options: RecognitionOptions {
                prompt: None,
                languages: Vec::new(),
            },
        });
        let key = self.next_key;
        self.next_key += 1;
        if role == Role::Isolated {
            self.bleed.register(key, expected);
        }
        self.tracks.insert(
            id.into(),
            Track {
                key,
                role,
                speaker: speaker.map(str::to_owned),
                clock: Clock {
                    expected,
                    ..Clock::default()
                },
                handle,
                markers: VecDeque::new(),
                next_marker: 0,
                ending: false,
                draft: None,
                last_terminal: None,
                capture: VecDeque::new(),
                capture_frames: 0,
                raw: VecDeque::new(),
                raw_frames: 0,
                outputs: VecDeque::new(),
                lagging: false,
                diarize,
                #[cfg(feature = "diarization")]
                diarization,
            },
        );
        self.emit(json!({
            "type": "transcriberrr.track.registered",
            "track_id": id,
            "role": message["role"],
            "speaker": speaker,
            "next_expected_frame": expected,
        }));
        Ok(())
    }

    fn get_track(&mut self, id: &str) -> Result<&mut Track, Value> {
        self.tracks
            .get_mut(id)
            .filter(|track| !track.ending)
            .ok_or_else(|| error("unknown_track", "The track is unknown or ending."))
    }

    fn append(&mut self, message: &Value) -> Result<(), Value> {
        let id = track_id(message)?;
        let start = unsigned(message, "start_frame")?;
        let samples = pcm(message)?;
        if start.checked_add(samples.len() as u64).is_none() {
            return Err(error(
                "invalid_frame",
                "The end frame exceeds the session clock.",
            ));
        }
        let track = self.get_track(id)?;
        // A discontinuity may add a control event when its reorder deadline expires.
        if start > track.clock.expected {
            track.check_control_capacity()?;
        }
        let ready = track.clock.append(start, samples, Instant::now());
        self.audio(id, ready);
        Ok(())
    }

    fn explicit_gap(&mut self, message: &Value) -> Result<(), Value> {
        let id = track_id(message)?;
        let end = unsigned(message, "end_frame")?;
        let track = self.get_track(id)?;
        track.check_control_capacity()?;
        if end <= track.clock.expected {
            return Err(error(
                "invalid_frame",
                "The gap must end after the expected frame.",
            ));
        }
        let mut ready = Vec::new();
        track.clock.client_gap(end, &mut ready);
        track.clock.drain(Instant::now(), false, &mut ready);
        self.audio(id, ready);
        Ok(())
    }

    fn diarization_available(&self) -> bool {
        #[cfg(feature = "diarization")]
        {
            self.state.diarization.is_some()
        }
        #[cfg(not(feature = "diarization"))]
        {
            false
        }
    }

    fn commit(&mut self, id: &str, end: bool) -> Result<(), Value> {
        let track = self.get_track(id)?;
        if !end {
            track.check_control_capacity()?;
        }
        let mut ready = Vec::new();
        track.clock.drain(Instant::now(), true, &mut ready);
        self.audio(id, ready);
        self.queue_commit(id, end);
        Ok(())
    }

    fn queue_commit(&mut self, id: &str, end: bool) {
        let track = self.tracks.get_mut(id).expect("registered track");
        track.ending = end;
        if end && track.role == Role::Isolated {
            self.bleed.end(track.key);
        }
        track.capture.push_back(Capture {
            action: CaptureAction::Commit { end },
            deadline: Instant::now(),
        });
        self.feed(id, true);
    }

    fn end(&mut self) {
        if self.ending {
            return;
        }
        self.ending = true;
        let tracks: Vec<_> = self.tracks.keys().cloned().collect();
        for id in tracks {
            if self.tracks.get(&id).is_some_and(|track| !track.ending) {
                let mut ready = Vec::new();
                self.tracks
                    .get_mut(&id)
                    .unwrap()
                    .clock
                    .drain(Instant::now(), true, &mut ready);
                self.audio(&id, ready);
                self.queue_commit(&id, true);
            }
        }
        self.finish_if_drained();
    }

    fn finish_if_drained(&mut self) {
        if self.ending && !self.ended && self.tracks.is_empty() {
            self.ended = true;
            self.emit(json!({"type":"transcriberrr.session.ended"}));
            self.state
                .native
                .complete(&self.id, self.state.max_sessions);
            self.permit.take();
            // Close the live queue after the terminal event. Keep the close
            // sender so a resume can take over while this socket is still draining.
            if let Some(client) = &mut self.client {
                client.events = None;
            }
            self.detached_at.get_or_insert_with(Instant::now);
        }
    }
}

fn unsigned(message: &Value, field: &str) -> Result<u64, Value> {
    message.get(field).and_then(Value::as_u64).ok_or_else(|| {
        error(
            "invalid_field",
            &format!("Expected unsigned integer field {field}."),
        )
    })
}

fn optional_string<'a>(message: &'a Value, field: &str) -> Result<Option<&'a str>, Value> {
    message
        .get(field)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| error("invalid_field", &format!("Expected string field {field}.")))
        })
        .transpose()
}

fn track_id(message: &Value) -> Result<&str, Value> {
    message
        .get("track_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty() && id.len() <= 128)
        .ok_or_else(|| {
            error(
                "invalid_track_id",
                "Expected a track ID of 1 to 128 UTF-8 bytes.",
            )
        })
}

fn pcm(message: &Value) -> Result<Vec<i16>, Value> {
    let invalid = || {
        error(
            "invalid_audio",
            "Expected up to 24000 frames of base64 little-endian PCM16 audio.",
        )
    };
    let encoded = message
        .get("audio")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty() && text.len() <= (MAX_PACKET_FRAMES * 2).div_ceil(3) * 4)
        .ok_or_else(invalid)?;
    let samples = super::openai::decode_pcm(encoded).map_err(|_| invalid())?;
    if samples.is_empty() || samples.len() > MAX_PACKET_FRAMES {
        return Err(invalid());
    }
    Ok(samples)
}

impl Track {
    fn append_raw(&mut self, start: u64, original: Vec<i16>, gated: &[i16]) {
        self.raw_frames += original.len();
        let changed = original != gated;
        if let Some(tail) = self.raw.back_mut()
            && tail.end() == start
            && tail.original.len() + original.len() <= MAX_PACKET_FRAMES
        {
            if changed && tail.gated.is_none() {
                tail.gated = Some(tail.original.clone());
            }
            if let Some(samples) = &mut tail.gated {
                samples.extend_from_slice(gated);
            }
            tail.original.extend(original);
        } else {
            self.raw.push_back(RawPacket {
                start,
                original,
                gated: changed.then(|| gated.to_vec()),
            });
        }
        // Count stored audio so a forward clock jump can't evict the PCM
        // for a final still waiting to publish.
        while self.raw.len() > MAX_RAW_PACKETS || self.raw_frames > MAX_RAW_FRAMES {
            let old = self.raw.pop_front().expect("retained audio");
            self.raw_frames -= old.original.len();
        }
    }

    fn interval(&self, start: u64, end: u64) -> Option<(Vec<i16>, Vec<i16>)> {
        if end < start || end - start > MAX_NATIVE_FRAMES {
            return None;
        }
        let len = (end - start) as usize;
        let mut original = Vec::with_capacity(len);
        let mut gated = Vec::with_capacity(len);
        let mut next = start;
        for packet in &self.raw {
            let packet_end = packet.end();
            if packet_end <= next {
                continue;
            }
            if packet.start > next || next == end {
                break;
            }
            let through = end.min(packet_end);
            let range = (next - packet.start) as usize..(through - packet.start) as usize;
            let draft = packet.gated.as_deref().unwrap_or(&packet.original);
            original.extend_from_slice(&packet.original[range.clone()]);
            gated.extend_from_slice(&draft[range]);
            next = through;
        }
        (next == end).then_some((original, gated))
    }
}

impl Session {
    fn audio(&mut self, id: &str, audio: Vec<Audio>) {
        let now = Instant::now();
        for action in audio {
            let track = self.tracks.get_mut(id).expect("registered track");
            match &action {
                Audio::Samples { start, samples } => {
                    if track.role == Role::Isolated {
                        self.bleed.ingest(track.key, *start, samples);
                    }
                    track.capture_frames += samples.len();
                }
                Audio::Gap { end, .. } if track.role == Role::Isolated => {
                    self.bleed.gap(track.key, *end);
                }
                Audio::Gap { .. } => {}
            }
            track.capture.push_back(Capture {
                action: CaptureAction::Audio(action),
                deadline: now + GAP_WAIT,
            });
            if track.capture.len() >= MAX_CAPTURE_QUEUE
                || track.capture_frames >= MAX_PENDING_FRAMES
            {
                self.feed(id, true);
            }
        }
        self.feed(id, false);
    }

    fn feed(&mut self, id: &str, force: bool) {
        let Some(track) = self.tracks.get_mut(id) else {
            return;
        };
        let now = Instant::now();
        let mut blocked = false;
        while let Some(front) = track.capture.front() {
            match &front.action {
                CaptureAction::Audio(Audio::Samples { start, samples }) => {
                    if track.role == Role::Isolated
                        && !force
                        && front.deadline > now
                        && !self.bleed.ready_through(
                            track.key,
                            start + samples.len() as u64,
                            DRAFT_LOOKAHEAD,
                        )
                    {
                        break;
                    }
                    // Reserve before popping or muting the packet. A full worker leaves
                    // the original PCM at the head, ahead of every later gap or commit.
                    let Ok(permit) = track.handle.reserve(samples.len()) else {
                        blocked = true;
                        break;
                    };
                    #[cfg(feature = "diarization")]
                    if let Some(stream) = &track.diarization
                        && !stream.try_append(*start, samples)
                    {
                        blocked = true;
                        break;
                    }
                    let CaptureAction::Audio(Audio::Samples { start, mut samples }) =
                        track.capture.pop_front().expect("capture head").action
                    else {
                        unreachable!();
                    };
                    track.capture_frames -= samples.len();
                    if track.role == Role::Isolated {
                        let original = samples.clone();
                        self.bleed
                            .apply(track.key, start, &mut samples, Context::Draft);
                        track.append_raw(start, original, &samples);
                    }
                    if track
                        .handle
                        .append_reserved(permit, start, samples, 0)
                        .is_err()
                    {
                        // The worker closed after reservation. poll_track reports its termination.
                        break;
                    }
                }
                action => {
                    let marker = track.next_marker + 1;
                    let (result, action) = match action {
                        CaptureAction::Audio(Audio::Gap { start, end, reason }) => (
                            track.handle.gap(marker, false),
                            MarkerAction::Gap {
                                start: *start,
                                end: *end,
                                reason,
                            },
                        ),
                        CaptureAction::Commit { end } => (
                            track.handle.commit(marker),
                            if *end {
                                MarkerAction::End
                            } else {
                                MarkerAction::Commit
                            },
                        ),
                        CaptureAction::Audio(Audio::Samples { .. }) => unreachable!(),
                    };
                    if result.is_err() {
                        blocked = true;
                        break;
                    }
                    #[cfg(feature = "diarization")]
                    if matches!(action, MarkerAction::End)
                        && let Some(stream) = &mut track.diarization
                    {
                        stream.finish();
                    }
                    track.capture.pop_front();
                    track.next_marker = marker;
                    track.markers.push_back(Marker { id: marker, action });
                }
            }
        }

        let queued_frames = track.handle.backlog_frames() + track.capture_frames;
        let lag = blocked
            || self.state.worker.backlog() >= GLOBAL_LAG_COMMANDS
            || queued_frames >= LAG_FRAMES;
        if lag && !track.lagging {
            track.lagging = true;
            self.emit(json!({
                "type": "transcriberrr.track.backlog",
                "track_id": id,
                "queue_depth": self.state.worker.backlog(),
                "queued_frames": queued_frames,
            }));
        } else if !lag {
            track.lagging = false;
        }
    }

    fn poll(&mut self) {
        let input_paused = self.update_input_pressure(Instant::now());
        let track_ids: Vec<_> = self.tracks.keys().cloned().collect();
        for id in track_ids {
            let mut ready = Vec::new();
            if !input_paused && self.detached_at.is_none() {
                self.tracks
                    .get_mut(&id)
                    .unwrap()
                    .clock
                    .drain(Instant::now(), false, &mut ready);
            }
            self.audio(&id, ready);
            self.poll_track(&id);
        }
        self.finish_if_drained();
    }

    fn poll_track(&mut self, id: &str) {
        let mut track = self.tracks.remove(id).expect("registered track");
        let now = Instant::now();
        let mut closed = false;
        while track.outputs.len() < OUTBOUND_EVENTS {
            match track.handle.try_recv() {
                Ok(event) => {
                    track.outputs.push_back(Output {
                        event,
                        deadline: now
                            + if track.diarize {
                                DIARIZATION_WAIT
                            } else {
                                FINAL_WAIT
                            },
                        redo: None,
                        quality: Quality::default(),
                    });
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    closed = true;
                    break;
                }
            }
        }
        while let Some(mut output) = track.outputs.pop_front() {
            match self.publish(id, &mut track, &mut output, now) {
                Published::Yes => {}
                Published::Wait => {
                    track.outputs.push_front(output);
                    break;
                }
                Published::End => {
                    self.retire(&track);
                    return;
                }
            }
        }
        if closed && track.outputs.is_empty() {
            if let Some((turn, _)) = track.draft.take() {
                self.retract(
                    id,
                    &format!("item_{}_{turn}", track.key),
                    "recognizer_closed",
                );
            }
            self.emit(json!({
                "type": "transcriberrr.track.ended",
                "track_id": id,
                "reason": "recognizer_closed",
            }));
            self.retire(&track);
        } else {
            self.tracks.insert(id.into(), track);
        }
    }

    fn retire(&mut self, track: &Track) {
        if track.role == Role::Isolated {
            self.bleed.end(track.key);
            self.ended_isolated.push_back(track.key);
            if self.ended_isolated.len() > self.state.max_tracks
                && let Some(old) = self.ended_isolated.pop_front()
            {
                self.bleed.remove(old);
            }
        }
    }

    fn publish(
        &mut self,
        id: &str,
        track: &mut Track,
        output: &mut Output,
        now: Instant,
    ) -> Published {
        let overload = matches!(output.event.update, PipelineUpdate::OverloadFinal(_));
        let PipelineEvent {
            turn,
            start,
            end,
            update,
        } = &mut output.event;
        // Native turn numbers never repeat on a track, including across decoder resets.
        if !matches!(update, PipelineUpdate::Committed { .. })
            && track.last_terminal.is_some_and(|closed| *turn <= closed)
        {
            return Published::Yes;
        }
        let item_id = format!("item_{}_{}", track.key, turn);
        match update {
            PipelineUpdate::Draft(hypothesis) => {
                let revision = match track.draft {
                    Some((active_turn, revision)) if active_turn == *turn => revision + 1,
                    _ => 1,
                };
                track.draft = Some((*turn, revision));
                self.emit(json!({
                    "type": "transcriberrr.transcript.draft", "track_id": id, "item_id": item_id,
                    "start_frame": start, "end_frame": end, "revision": revision, "text": hypothesis.text,
                    "words": words(hypothesis),
                    "speaker": track.speaker,
                    "speaker_source": track.speaker.as_ref().map(|_| "client"),
                }));
            }
            PipelineUpdate::Final(hypothesis) | PipelineUpdate::OverloadFinal(hypothesis) => {
                // A stable decoder prefix can include part of a word whose timing
                // has not settled yet. An overload final can only confirm timed words.
                if overload && !hypothesis.words.is_empty() {
                    rebuild_text(hypothesis);
                }
                if track.role == Role::Isolated && overload {
                    if !self.bleed_stable(track, *start, *end, hypothesis, &mut output.quality) {
                        track.close_item(*turn);
                        self.retract(id, &item_id, "overload");
                        return Published::Yes;
                    }
                } else if track.role == Role::Isolated {
                    let ready = self.bleed.ready_through(track.key, *end, FINAL_LOOKAHEAD);
                    if !ready && now < output.deadline && output.redo.is_none() {
                        return Published::Wait;
                    }
                    output.quality.missing_peer |= !ready;
                    let result = self.bleed_final(
                        track,
                        *start..*end,
                        hypothesis,
                        &mut output.redo,
                        &mut output.quality,
                        now < output.deadline + FINAL_WAIT,
                    );
                    match result {
                        BleedFinal::Wait => return Published::Wait,
                        BleedFinal::Muted => {
                            track.close_item(*turn);
                            self.retract(id, &item_id, "bleed");
                            return Published::Yes;
                        }
                        BleedFinal::Unavailable => {
                            track.close_item(*turn);
                            self.retract(id, &item_id, "bleed_redecode_unavailable");
                            return Published::Yes;
                        }
                        BleedFinal::Ready => {}
                    }
                }
                let Some(mut event) = final_event(
                    id,
                    &item_id,
                    track,
                    *start,
                    *end,
                    hypothesis,
                    now >= output.deadline,
                ) else {
                    return Published::Wait;
                };
                if track.role == Role::Isolated {
                    event["bleed_uncalibrated"] = output.quality.uncalibrated.into();
                    event["bleed_missing_peer"] = output.quality.missing_peer.into();
                }
                track.close_item(*turn);
                self.emit(event);
            }
            PipelineUpdate::Failed(reason) => {
                track.close_item(*turn);
                self.retract(id, &item_id, reason);
            }
            PipelineUpdate::Committed { marker } => {
                #[cfg(feature = "diarization")]
                if track
                    .markers
                    .front()
                    .is_some_and(|head| matches!(head.action, MarkerAction::End))
                    && track
                        .diarization
                        .as_ref()
                        .is_some_and(|stream| !stream.is_finished())
                {
                    return Published::Wait;
                }
                let Some(head) = track.markers.pop_front() else {
                    return Published::Yes;
                };
                debug_assert_eq!(head.id, *marker);
                match head.action {
                    MarkerAction::Commit => {
                        self.emit(json!({"type": "transcriberrr.track.committed", "track_id": id}))
                    }
                    MarkerAction::End => {
                        self.emit(json!({"type": "transcriberrr.track.ended", "track_id": id}));
                        return Published::End;
                    }
                    MarkerAction::Gap { start, end, reason } => self.emit(json!({
                        "type": "transcriberrr.track.gap", "track_id": id, "start_frame": start,
                        "end_frame": end, "reason": reason,
                    })),
                }
            }
        }
        Published::Yes
    }

    fn retract(&mut self, track_id: &str, item_id: &str, reason: &str) {
        self.emit(json!({
            "type": "transcriberrr.transcript.retracted",
            "track_id": track_id,
            "item_id": item_id,
            "reason": reason,
        }));
    }

    fn bleed_stable(
        &mut self,
        track: &Track,
        start: u64,
        end: u64,
        hypothesis: &mut Hypothesis,
        quality: &mut Quality,
    ) -> bool {
        if !hypothesis.words.is_empty() {
            hypothesis.words.retain(|word| {
                if !self
                    .bleed
                    .ready_through(track.key, word.end_sample, FINAL_LOOKAHEAD)
                {
                    return false;
                }
                let decision = self.bleed.decision(
                    track.key,
                    word.start_sample,
                    word.end_sample,
                    Context::Final,
                );
                quality.merge(decision.quality);
                !decision.quality.missing_peer && !decision.muted
            });
            rebuild_text(hypothesis);
            return !hypothesis.words.is_empty();
        }
        if !self.bleed.ready_through(track.key, end, FINAL_LOOKAHEAD) {
            return false;
        }
        let Some((mut final_samples, draft_samples)) = track.interval(start, end) else {
            return false;
        };
        quality.merge(
            self.bleed
                .apply(track.key, start, &mut final_samples, Context::Final),
        );
        !quality.missing_peer && final_samples == draft_samples && !hypothesis.text.is_empty()
    }

    fn bleed_final(
        &mut self,
        track: &Track,
        interval: Range<u64>,
        hypothesis: &mut Hypothesis,
        redo: &mut Option<Redo>,
        quality: &mut Quality,
        within_deadline: bool,
    ) -> BleedFinal {
        let Range { start, end } = interval;
        if !hypothesis.words.is_empty() {
            let count = hypothesis.words.len();
            hypothesis.words.retain(|word| {
                let decision = self.bleed.decision(
                    track.key,
                    word.start_sample,
                    word.end_sample,
                    Context::Final,
                );
                quality.merge(decision.quality);
                !decision.muted
            });
            if hypothesis.words.is_empty() {
                return BleedFinal::Muted;
            }
            if hypothesis.words.len() != count {
                rebuild_text(hypothesis);
            }
            return BleedFinal::Ready;
        }
        if hypothesis.text.is_empty() {
            return BleedFinal::Ready;
        }
        if redo.is_none() {
            let decision = self.bleed.decision(track.key, start, end, Context::Final);
            quality.merge(decision.quality);
            let Some((mut samples, draft_samples)) = track.interval(start, end) else {
                return if decision.muted {
                    BleedFinal::Muted
                } else {
                    BleedFinal::Unavailable
                };
            };
            quality.merge(
                self.bleed
                    .apply(track.key, start, &mut samples, Context::Final),
            );
            if samples == draft_samples {
                return BleedFinal::Ready;
            }
            if samples.iter().all(|&sample| sample == 0) {
                return BleedFinal::Muted;
            }
            let worker = self.state.worker.track(TrackSettings {
                native: false,
                streaming: false,
                options: RecognitionOptions::default(),
            });
            let remaining = samples
                .chunks(MAX_PACKET_FRAMES)
                .map(<[i16]>::to_vec)
                .collect();
            *redo = Some(Redo {
                worker,
                start,
                remaining,
                committed: false,
            });
        }
        redo.as_mut().unwrap().poll(hypothesis, within_deadline)
    }
}

enum Published {
    Yes,
    Wait,
    End,
}
enum BleedFinal {
    Ready,
    Wait,
    Muted,
    Unavailable,
}

impl Redo {
    fn poll(&mut self, hypothesis: &mut Hypothesis, within_deadline: bool) -> BleedFinal {
        if !within_deadline {
            return BleedFinal::Unavailable;
        }
        // Submit a small number of packets per tick so another track can enqueue its audio.
        for _ in 0..2 {
            let Some(packet) = self.remaining.front() else {
                break;
            };
            match self.worker.append(self.start, packet.clone(), 1) {
                Ok(()) => {
                    self.start += packet.len() as u64;
                    self.remaining.pop_front();
                }
                Err(QueueError::Full) => return BleedFinal::Wait,
                Err(QueueError::Closed) => return BleedFinal::Unavailable,
            }
        }
        if self.remaining.is_empty() && !self.committed {
            match self.worker.commit(1) {
                Ok(()) => self.committed = true,
                Err(QueueError::Full) => return BleedFinal::Wait,
                Err(QueueError::Closed) => return BleedFinal::Unavailable,
            }
        }
        loop {
            match self.worker.try_recv() {
                Ok(PipelineEvent {
                    update: PipelineUpdate::Final(result),
                    ..
                }) => {
                    if result.text.is_empty() {
                        return BleedFinal::Muted;
                    }
                    *hypothesis = result;
                    return BleedFinal::Ready;
                }
                Ok(PipelineEvent {
                    update: PipelineUpdate::Failed(_),
                    ..
                })
                | Err(mpsc::error::TryRecvError::Disconnected) => return BleedFinal::Unavailable,
                Ok(_) => {}
                Err(mpsc::error::TryRecvError::Empty) => return BleedFinal::Wait,
            }
        }
    }
}

fn rebuild_text(hypothesis: &mut Hypothesis) {
    hypothesis.text = fragment_text(hypothesis).trim_start().to_owned();
}

fn fragment_text(hypothesis: &Hypothesis) -> String {
    hypothesis
        .words
        .iter()
        .map(|word| word.text.as_str())
        .collect()
}

fn words(hypothesis: &Hypothesis) -> Vec<Value> {
    hypothesis.words.iter().map(|word| {
        json!({"text": word.text, "start_frame": word.start_sample, "end_frame": word.end_sample})
    }).collect()
}

fn final_event(
    id: &str,
    item_id: &str,
    track: &Track,
    start: u64,
    end: u64,
    hypothesis: &Hypothesis,
    expired: bool,
) -> Option<Value> {
    let mut event = json!({
        "type": "transcriberrr.transcript.final",
        "track_id": id,
        "item_id": item_id,
        "start_frame": start,
        "end_frame": end,
        "text": hypothesis.text,
        "speaker": track.speaker,
        "speaker_source": track.speaker.as_ref().map(|_| "client"),
        "speaker_spans": [],
    });
    event["words"] = Value::Array(words(hypothesis));
    if !track.diarize {
        return Some(event);
    }
    #[cfg(feature = "diarization")]
    if let Some(stream) = &track.diarization {
        let intervals: Vec<_> = hypothesis
            .words
            .iter()
            .map(|word| (word.start_sample, word.end_sample))
            .collect();
        if let Some(attribution) = stream.attribution(start, end, &intervals) {
            let fully_aligned = !hypothesis.words.is_empty()
                && fragment_text(hypothesis).trim() == hypothesis.text.trim();
            event["speaker"] = json!(attribution.speaker.filter(|_| fully_aligned));
            event["speaker_source"] = json!("diarization");
            event["speaker_spans"] = json!(attribution.speaker_spans);
            event["text_attribution"] = json!(if fully_aligned
                && attribution.words.iter().any(Option::is_some)
            {
                "word"
            } else {
                "unknown"
            });
            for (word, speaker) in event["words"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .zip(attribution.words)
            {
                word["speaker"] = json!(speaker);
            }
            return Some(event);
        }
    }
    if !expired {
        return None;
    }
    #[cfg(feature = "diarization")]
    if let Some(stream) = &track.diarization {
        stream.report_timeout(start, end);
    }
    event["speaker"] = Value::Null;
    event["speaker_source"] = Value::Null;
    event["text_attribution"] = json!("unknown");
    event["diarization_unavailable"] = json!(true);
    for word in event["words"].as_array_mut().unwrap() {
        word["speaker"] = Value::Null;
    }
    Some(event)
}

#[cfg(all(test, feature = "diarization"))]
mod mixed_tests {
    use super::*;
    use crate::serve::diarization::{SpeakerSpan, Worker};
    use crate::serve::recognizer::TimedWord;

    const TRACK: &str = "mix";
    const START: u64 = 48_000;
    const END: u64 = 62_000;

    async fn mixed_track() -> (Session, EventReceiver) {
        let (mut session, mut events, _close, _shutdown) = super::tests::session_fixture().await;
        Arc::get_mut(&mut session.state).unwrap().diarization = Some(Worker::scripted());
        session
            .register(&json!({
                "track_id": TRACK,
                "role": "mixed_audio",
                "start_frame": START,
                "diarization": true
            }))
            .unwrap();
        assert_eq!(
            next_event(&mut events)["type"],
            "transcriberrr.track.registered"
        );
        (session, events)
    }

    fn next_event(events: &mut EventReceiver) -> Value {
        serde_json::from_str(&events.try_recv().unwrap().text).unwrap()
    }

    fn script(session: &Session, spans: Vec<SpeakerSpan>) {
        session.tracks[TRACK]
            .diarization
            .as_ref()
            .unwrap()
            .script_coverage((START, END), spans);
    }

    fn span(speaker: &str, start_frame: u64, end_frame: u64) -> SpeakerSpan {
        SpeakerSpan {
            speaker: speaker.into(),
            start_frame,
            end_frame,
        }
    }

    fn caption(session: &mut Session, timed: bool) {
        let words = if timed {
            [
                ("red", START + 1_000, START + 3_000),
                (" amber", START + 7_000, START + 9_000),
                (" blue", START + 10_000, START + 13_000),
            ]
            .into_iter()
            .map(|(text, start_sample, end_sample)| TimedWord {
                text: text.into(),
                start_sample,
                end_sample,
            })
            .collect()
        } else {
            Vec::new()
        };
        queue_caption(
            session,
            PipelineUpdate::Final(Hypothesis {
                text: "red amber blue".into(),
                stable: true,
                words,
                ..Hypothesis::default()
            }),
        );
    }

    fn queue_caption(session: &mut Session, update: PipelineUpdate) {
        session
            .tracks
            .get_mut(TRACK)
            .unwrap()
            .outputs
            .push_back(Output {
                event: PipelineEvent {
                    turn: 1,
                    start: START,
                    end: END,
                    update,
                },
                deadline: Instant::now() + DIARIZATION_WAIT,
                redo: None,
                quality: Quality::default(),
            });
    }

    fn partly_timed_caption() -> Hypothesis {
        Hypothesis {
            text: "The cedar trees. We carried".into(),
            stable: true,
            words: [
                ("The", 1_000, 2_000),
                (" cedar", 2_000, 3_000),
                (" trees.", 3_000, 4_000),
                (" We", 5_000, 7_000),
            ]
            .into_iter()
            .map(|(text, start, end)| TimedWord {
                text: text.into(),
                start_sample: START + start,
                end_sample: START + end,
            })
            .collect(),
            ..Hypothesis::default()
        }
    }

    #[tokio::test]
    async fn mixed_overload_final_excludes_unconfirmed_suffix() {
        let (mut session, mut events) = mixed_track().await;
        script(&session, vec![span("speaker_1", START, END)]);
        queue_caption(
            &mut session,
            PipelineUpdate::OverloadFinal(partly_timed_caption()),
        );
        session.poll_track(TRACK);

        let event = next_event(&mut events);
        assert_eq!(event["type"], "transcriberrr.transcript.final");
        assert_eq!(event["text"], "The cedar trees. We");
        assert_eq!(
            event["words"]
                .as_array()
                .unwrap()
                .iter()
                .map(|word| word["text"].as_str().unwrap())
                .collect::<String>(),
            event["text"]
        );
        assert_eq!(event["speaker"], "speaker_1");
        assert_eq!(event["text_attribution"], "word");
    }

    #[tokio::test]
    async fn incomplete_word_alignment_does_not_label_the_entire_mixed_final() {
        let (mut session, mut events) = mixed_track().await;
        script(&session, vec![span("speaker_1", START, END)]);
        queue_caption(&mut session, PipelineUpdate::Final(partly_timed_caption()));
        session.poll_track(TRACK);

        let event = next_event(&mut events);
        assert_eq!(event["text"], "The cedar trees. We carried");
        assert_eq!(event["speaker"], Value::Null);
        assert_eq!(event["text_attribution"], "unknown");
        assert_eq!(event["words"][0]["speaker"], "speaker_1");
    }

    #[tokio::test]
    async fn mixed_handoff_preserves_overlap_and_leaves_ambiguous_word_unlabeled() {
        let (mut session, mut events) = mixed_track().await;
        caption(&mut session, true);
        session.poll_track(TRACK);
        assert!(matches!(
            events.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        script(
            &session,
            vec![
                span("speaker_1", START, START + 9_000),
                span("speaker_2", START + 7_000, END),
            ],
        );
        session.poll_track(TRACK);

        let final_event = next_event(&mut events);
        assert_eq!(final_event["type"], "transcriberrr.transcript.final");
        assert_eq!(final_event["start_frame"], START);
        assert_eq!(final_event["end_frame"], END);
        assert_eq!(final_event["speaker"], Value::Null);
        assert_eq!(final_event["speaker_source"], "diarization");
        assert_eq!(final_event["text_attribution"], "word");
        assert_eq!(
            final_event["speaker_spans"],
            json!([
                {"speaker": "speaker_1", "start_frame": START, "end_frame": START + 9_000},
                {"speaker": "speaker_2", "start_frame": START + 7_000, "end_frame": END}
            ])
        );
        assert_eq!(
            final_event["words"],
            json!([
                {"text": "red", "start_frame": START + 1_000, "end_frame": START + 3_000, "speaker": "speaker_1"},
                {"text": " amber", "start_frame": START + 7_000, "end_frame": START + 9_000, "speaker": null},
                {"text": " blue", "start_frame": START + 10_000, "end_frame": START + 13_000, "speaker": "speaker_2"}
            ])
        );
    }

    #[tokio::test]
    async fn mixed_untimed_final_retains_intervals_and_reports_unknown_text_attribution() {
        let (mut session, mut events) = mixed_track().await;
        script(&session, vec![span("speaker_1", START, END)]);
        caption(&mut session, false);
        session.poll_track(TRACK);
        let final_event = next_event(&mut events);
        assert_eq!(final_event["type"], "transcriberrr.transcript.final");
        assert_eq!(final_event["text"], "red amber blue");
        assert_eq!(final_event["speaker"], Value::Null);
        assert_eq!(final_event["speaker_source"], "diarization");
        assert_eq!(
            final_event["speaker_spans"],
            json!([
                {"speaker": "speaker_1", "start_frame": START, "end_frame": END}
            ])
        );
        assert_eq!(final_event["words"], json!([]));
        assert_eq!(final_event["text_attribution"], "unknown");
        assert!(final_event.get("diarization_unavailable").is_none());
    }

    #[tokio::test]
    async fn mixed_final_with_coverage_and_no_speakers_is_unlabeled_without_timeout() {
        let (mut session, mut events) = mixed_track().await;
        script(&session, Vec::new());
        caption(&mut session, true);
        session.poll_track(TRACK);
        let final_event = next_event(&mut events);
        assert_eq!(final_event["type"], "transcriberrr.transcript.final");
        assert_eq!(final_event["speaker"], Value::Null);
        assert_eq!(final_event["speaker_source"], "diarization");
        assert_eq!(final_event["speaker_spans"], json!([]));
        assert_eq!(final_event["text_attribution"], "unknown");
        assert!(final_event.get("diarization_unavailable").is_none());
    }

    #[tokio::test]
    async fn mixed_final_waits_then_expires_explicitly_unlabeled() {
        let (mut session, mut events) = mixed_track().await;
        caption(&mut session, true);
        session.poll_track(TRACK);
        assert!(matches!(
            events.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        session
            .tracks
            .get_mut(TRACK)
            .unwrap()
            .outputs
            .front_mut()
            .unwrap()
            .deadline = Instant::now();
        session.poll_track(TRACK);
        let final_event = next_event(&mut events);
        assert_eq!(final_event["type"], "transcriberrr.transcript.final");
        assert_eq!(final_event["text"], "red amber blue");
        assert_eq!(final_event["speaker"], Value::Null);
        assert_eq!(final_event["speaker_source"], Value::Null);
        assert_eq!(final_event["speaker_spans"], json!([]));
        assert_eq!(final_event["text_attribution"], "unknown");
        assert_eq!(final_event["diarization_unavailable"], true);
        assert!(
            final_event["words"]
                .as_array()
                .unwrap()
                .iter()
                .all(|word| word.get("speaker") == Some(&Value::Null))
        );

        script(&session, vec![span("speaker_1", START, END)]);
        session.poll_track(TRACK);
        assert!(matches!(
            events.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_reports_exact_loss_then_delivers_remaining_events_in_order() {
        let mut replay = Replay::default();
        for index in 1..=100 {
            replay.emit(json!({"type": "test", "index": index}));
        }
        replay.ack(39).unwrap();
        let error = replay.resume(38, false).err().unwrap();
        assert_eq!(error["error"]["code"], "replay_unavailable");
        assert_eq!(error["error"]["earliest_replayable_sequence"], 40);
        assert_eq!(replay.resume(39, false).unwrap().lost, None);
        let plan = replay.resume(0, true).unwrap();
        assert_eq!(plan.lost, Some([1, 39]));
        assert_eq!(plan.events.len(), 61);
        let plan = replay.resume(55, false).unwrap();
        let sequence: Vec<_> = plan
            .events
            .iter()
            .map(|text| {
                serde_json::from_str::<Value>(text).unwrap()["sequence"]
                    .as_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(sequence, (56..=100).collect::<Vec<_>>());
        assert!(replay.resume(101, true).is_err());
    }

    #[test]
    fn replay_stays_bounded_by_events_and_bytes() {
        let mut replay = Replay::default();
        for _ in 0..(MAX_EVENTS + 1) {
            replay.emit(json!({"type": "test"}));
        }
        assert_eq!(replay.earliest(), 2);
        assert_eq!(replay.events.len(), MAX_EVENTS);
        let big = "x".repeat(MAX_REPLAY_BYTES / 4);
        for _ in 0..5 {
            replay.emit(json!({"type": "test", "text": big}));
        }
        assert!(replay.bytes <= MAX_REPLAY_BYTES);
        replay.ack(replay.next - 1).unwrap();
        assert_eq!(replay.earliest(), replay.next);
        assert_eq!(replay.bytes, 0);
    }

    #[test]
    fn track_clock_trims_retransmits_and_reports_exact_gaps() {
        let now = Instant::now();
        let mut clock = Clock::default();
        assert!(clock.append(10, vec![10, 11, 12], now).is_empty());
        let ready = clock.append(0, (0..11).collect(), now);
        assert!(matches!(ready.as_slice(), [
            Audio::Samples { start: 0, samples: first },
            Audio::Samples { start: 11, samples: second },
        ] if first.len() == 11 && second == &[11, 12]));
        assert_eq!(clock.expected, 13);
        assert!(clock.append(0, vec![0; 13], now).is_empty());
        assert!(clock.append(20, vec![20], now).is_empty());
        let mut ready = Vec::new();
        clock.drain(now, false, &mut ready);
        assert!(ready.is_empty());
        clock.drain(now + GAP_WAIT, false, &mut ready);
        assert!(matches!(ready.as_slice(), [
            Audio::Gap { start: 13, end: 20, reason: "missing_audio" },
            Audio::Samples { start: 20, samples },
        ] if samples == &[20]));
        assert_eq!(clock.expected, 21);
        let ready = clock.append(19, vec![19, 20, 21], now);
        assert!(matches!(ready.as_slice(), [
            Audio::Samples { start: 21, samples },
        ] if samples == &[21]));
    }

    #[test]
    fn expiring_one_gap_allows_retransmission_for_a_newer_gap() {
        let now = Instant::now();
        let mut clock = Clock::default();
        assert!(clock.append(10, vec![10], now).is_empty());
        assert!(clock.append(20, vec![20], now + GAP_WAIT / 2).is_empty());

        let mut ready = Vec::new();
        clock.drain(now + GAP_WAIT, false, &mut ready);
        assert!(matches!(
            ready.as_slice(),
            [
                Audio::Gap {
                    start: 0,
                    end: 10,
                    ..
                },
                Audio::Samples { start: 10, .. }
            ]
        ));
        let ready = clock.append(11, vec![1; 9], now + GAP_WAIT);
        assert!(matches!(
            ready.as_slice(),
            [
                Audio::Samples { start: 11, .. },
                Audio::Samples { start: 20, .. }
            ]
        ));
    }

    #[test]
    fn reorder_pressure_skips_only_enough_audio_to_restore_the_bound() {
        let now = Instant::now();
        let mut clock = Clock::default();
        for index in 1..=MAX_PENDING_PACKETS as u64 {
            assert!(clock.append(index * 2, vec![0], now).is_empty());
        }
        let ready = clock.append(2 * (MAX_PENDING_PACKETS as u64 + 1), vec![0], now);
        assert!(matches!(
            ready.as_slice(),
            [
                Audio::Gap {
                    start: 0,
                    end: 2,
                    ..
                },
                Audio::Samples { start: 2, .. }
            ]
        ));
        assert_eq!(clock.pending.len(), MAX_PENDING_PACKETS);
        assert_eq!(clock.pending_frames, MAX_PENDING_PACKETS);
    }

    #[test]
    fn client_gap_preserves_buffered_samples_and_reports_only_missing_frames() {
        let now = Instant::now();
        let mut clock = Clock::default();
        assert!(clock.append(10, vec![10, 11], now).is_empty());
        assert!(clock.append(20, vec![20, 21], now).is_empty());
        let mut ready = Vec::new();
        clock.client_gap(21, &mut ready);
        assert!(matches!(
            ready.as_slice(),
            [
                Audio::Gap {
                    start: 0,
                    end: 10,
                    reason: "client_gap"
                },
                Audio::Samples { start: 10, .. },
                Audio::Gap {
                    start: 12,
                    end: 20,
                    reason: "client_gap"
                },
                Audio::Samples { start: 20, .. },
            ]
        ));
        assert_eq!(clock.expected, 22);
        assert_eq!(clock.pending_frames, 0);
    }

    pub(super) async fn session_fixture()
    -> (Session, EventReceiver, CloseReceiver, watch::Sender<bool>) {
        use super::super::recognizer::{RecognitionStream, Recognizer};

        struct UnusedRecognizer;

        impl Recognizer for UnusedRecognizer {
            fn start(
                &self,
                _: &RecognitionOptions,
            ) -> anyhow::Result<Box<dyn RecognitionStream + '_>> {
                anyhow::bail!("unexpected recognition in protocol-only test");
            }
        }

        session_with_recognizer(|| Ok(Box::new(UnusedRecognizer)), 1).await
    }

    async fn session_with_recognizer(
        load: impl FnOnce() -> anyhow::Result<Box<dyn super::super::recognizer::Recognizer>>
        + Send
        + 'static,
        max_tracks: usize,
    ) -> (Session, EventReceiver, CloseReceiver, watch::Sender<bool>) {
        let worker = super::super::pipeline::RecognitionWorker::start(load)
            .await
            .unwrap();
        let (shutdown_tx, shutdown) = watch::channel(false);
        let state = Arc::new(super::super::AppState {
            model: "test-model".into(),
            engine: "test-engine".into(),
            worker,
            capabilities: super::super::recognizer::capabilities(crate::EngineKind::Qwen3),
            shutdown,
            active_sessions: Arc::new(tokio::sync::Semaphore::new(1)),
            max_sessions: 1,
            max_tracks,
            native: Registry::default(),
            #[cfg(feature = "diarization")]
            diarization: None,
        });
        let (client, events, close) = connection();
        (
            Session::new("session".into(), "credential".into(), client, state),
            events,
            close,
            shutdown_tx,
        )
    }

    #[tokio::test]
    async fn invalid_or_abandoned_takeover_leaves_active_owner() {
        let (mut session, _events, mut close, _shutdown) = session_fixture().await;
        session.emit(json!({"type": "test"}));
        session.replay.ack(1).unwrap();
        let resume = |last_sequence| Resume {
            session_id: "session".into(),
            resume_credential: "credential".into(),
            last_sequence,
            accept_loss: false,
        };
        let (client, _events, _close) = connection();
        let (reply, response) = oneshot::channel();
        session.attach(resume(0), client, reply);
        assert_eq!(
            response.await.unwrap().err().unwrap()["error"]["code"],
            "replay_unavailable"
        );
        assert!(session.is_owner(1));
        assert!(!close.has_changed().unwrap());

        let (client, _events, _close) = connection();
        let (reply, response) = oneshot::channel();
        drop(response);
        session.attach(resume(1), client, reply);
        assert!(session.is_owner(1));
        assert!(!close.has_changed().unwrap());

        let (client, _events, _close) = connection();
        let (reply, response) = oneshot::channel();
        session.attach(resume(1), client, reply);
        let attached = response.await.unwrap().unwrap();
        assert_eq!(attached.owner, 2);
        assert!(session.is_owner(2));
        assert!(!session.is_owner(1));
        close.changed().await.unwrap();
        assert_eq!(close.borrow().unwrap().0, TAKEN_OVER);
    }

    #[tokio::test]
    async fn live_takeover_allows_retransmission_before_expiring_pending_gap() {
        let (mut session, _events, _close, _shutdown) = session_fixture().await;
        session
            .register(&json!({"track_id": "mic", "role": "mixed_audio"}))
            .unwrap();
        let clock = &mut session.tracks.get_mut("mic").unwrap().clock;
        let received_at = Instant::now() - GAP_WAIT - Duration::from_millis(1);
        assert!(clock.append(2, vec![0], received_at).is_empty());

        let (client, _events, _close) = connection();
        let (reply, response) = oneshot::channel();
        session.attach(
            Resume {
                session_id: session.id.clone(),
                resume_credential: session.credential.clone(),
                last_sequence: 1,
                accept_loss: false,
            },
            client,
            reply,
        );
        let attached = response.await.unwrap().unwrap();
        assert_eq!(attached.announcement["next_expected_frames"]["mic"], 0);
        session.poll();
        assert_eq!(session.tracks["mic"].clock.expected, 0);
        session
            .append(&json!({"track_id": "mic", "start_frame": 0, "audio": "AAAAAA=="}))
            .unwrap();
        let track = &session.tracks["mic"];
        assert_eq!(track.clock.expected, 3);
        assert!(track.clock.pending.is_empty());
        assert!(track.markers.is_empty());
    }

    #[tokio::test]
    async fn slow_reader_detaches_but_preserves_results_for_resume() {
        let (mut session, _events, mut close, _shutdown) = session_fixture().await;
        for _ in 0..=OUTBOUND_EVENTS {
            session.emit(json!({"type": "test"}));
        }
        assert!(session.detached_at.is_some());
        assert!(!session.is_owner(1));
        close.changed().await.unwrap();
        assert_eq!(close.borrow().unwrap().0, SLOW_READER);
        let plan = session.replay.resume(0, false).unwrap();
        assert_eq!(plan.events.len(), OUTBOUND_EVENTS + 1);
    }

    #[tokio::test]
    async fn overflowing_frame_is_rejected_without_advancing_track() {
        let (mut session, _events, _close, _shutdown) = session_fixture().await;
        session
            .register(&json!({
                "track_id": "mic", "role": "isolated_mic", "start_frame": u64::MAX - 1,
            }))
            .unwrap();
        let failure = session
            .append(&json!({
                "track_id": "mic", "start_frame": u64::MAX - 1, "audio": "AAAAAA==",
            }))
            .unwrap_err();
        assert_eq!(failure["error"]["code"], "invalid_frame");
        assert_eq!(session.tracks["mic"].clock.expected, u64::MAX - 1);
    }

    #[tokio::test]
    async fn clock_origin_cannot_be_set_after_all_registered_tracks_end() {
        let (mut session, _events, _close, _shutdown) = session_fixture().await;
        session
            .register(&json!({"track_id": "mic", "role": "mixed_audio"}))
            .unwrap();
        session.commit("mic", true).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while !session.tracks.is_empty() {
                session.poll();
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        let failure = session
            .set_clock(&json!({"frame_zero_time": "2026-11-21T00:00:00Z"}))
            .unwrap_err();
        assert_eq!(failure["error"]["code"], "invalid_clock");
    }

    #[tokio::test]
    async fn completed_sessions_release_admission_and_bound_replay_retention() {
        let (mut first, _events, _close, _shutdown) = session_fixture().await;
        let state = Arc::clone(&first.state);
        first.permit = Some(
            Arc::clone(&state.active_sessions)
                .try_acquire_owned()
                .unwrap(),
        );
        let (first_sender, _, _) = SessionEndpoint::channel();
        state.native.insert(first.id.clone(), first_sender);
        assert_eq!(state.active_sessions.available_permits(), 0);
        first.end();
        assert_eq!(state.active_sessions.available_permits(), 1);
        assert!(state.native.get(&first.id).is_some());

        let (client, _events, _close) = connection();
        let (second_sender, _, _) = SessionEndpoint::channel();
        let mut second = Session::new(
            "second".into(),
            "other credential".into(),
            client,
            state.clone(),
        );
        second.permit = Some(
            Arc::clone(&state.active_sessions)
                .try_acquire_owned()
                .unwrap(),
        );
        state.native.insert(second.id.clone(), second_sender);
        second.end();
        assert!(state.native.get(&second.id).is_some());
        assert!(state.native.get(&first.id).is_none());
        assert_eq!(state.active_sessions.available_permits(), 1);

        let (client, _events, _close) = connection();
        let (reply, response) = oneshot::channel();
        first.attach(
            Resume {
                session_id: first.id.clone(),
                resume_credential: first.credential.clone(),
                last_sequence: 0,
                accept_loss: false,
            },
            client,
            reply,
        );
        assert_eq!(
            response.await.unwrap().err().unwrap()["error"]["code"],
            "session_expired"
        );
    }

    #[tokio::test]
    async fn completed_replay_can_be_taken_over_while_delivery_is_in_progress() {
        let (mut session, mut events, mut close, _shutdown) = session_fixture().await;
        let (sender, _, _) = SessionEndpoint::channel();
        session.state.native.insert(session.id.clone(), sender);
        session.end();
        assert_eq!(
            next_event(&mut events)["type"],
            "transcriberrr.session.ended"
        );
        assert!(events.recv().await.is_none());
        let ended_at = session.detached_at;

        let resume = || Resume {
            session_id: "session".into(),
            resume_credential: "credential".into(),
            last_sequence: 0,
            accept_loss: false,
        };
        let (client, mut replay_events, mut replay_close) = connection();
        let (reply, response) = oneshot::channel();
        session.attach(resume(), client, reply);
        let attached = response.await.unwrap().unwrap();
        assert_eq!(attached.owner, 2);
        assert_eq!(attached.replay.len(), 1);
        assert!(replay_events.recv().await.is_none());
        close.changed().await.unwrap();
        assert_eq!(close.borrow().unwrap().0, TAKEN_OVER);

        let (client, _events, _close) = connection();
        let (reply, response) = oneshot::channel();
        session.attach(resume(), client, reply);
        assert_eq!(response.await.unwrap().unwrap().owner, 3);
        replay_close.changed().await.unwrap();
        assert_eq!(replay_close.borrow().unwrap().0, TAKEN_OVER);
        assert!(session.is_owner(3));
        assert!(!session.is_owner(2));
        assert_eq!(session.detached_at, ended_at);
    }

    #[tokio::test]
    async fn socket_backpressure_freezes_reorder_wait_and_keeps_remaining_budget() {
        let (mut session, _events, _close, _shutdown) = session_fixture().await;
        session
            .register(&json!({"track_id": "mic", "role": "mixed_audio"}))
            .unwrap();
        let now = Instant::now();
        let paused_at = now - 2 * GAP_WAIT;
        let track = session.tracks.get_mut("mic").unwrap();
        let mut pcm = vec![0i16; MAX_PENDING_FRAMES];
        for start in [0, MAX_PACKET_FRAMES] {
            track.capture.push_back(Capture {
                action: CaptureAction::Audio(Audio::Samples {
                    start: start as u64,
                    samples: pcm.split_off(pcm.len() - MAX_PACKET_FRAMES),
                }),
                deadline: paused_at,
            });
        }
        track.capture_frames = MAX_PENDING_FRAMES;
        track.clock.expected = MAX_PENDING_FRAMES as u64;
        let expected = track.clock.expected;
        let later = expected + 8;
        assert!(
            track
                .clock
                .append(later, vec![1, 2], paused_at - GAP_WAIT / 2)
                .is_empty()
        );
        assert!(session.update_input_pressure(paused_at));
        session.poll();
        assert_eq!(session.tracks["mic"].clock.expected, expected);
        // The poll just moved the capture into the worker and reopened input.
        assert!(!session.update_input_pressure(now));
        let clock = &mut session.tracks.get_mut("mic").unwrap().clock;
        assert_eq!(clock.deadline, Some(now + GAP_WAIT / 2));
        let repair = clock.append(expected, vec![0; 8], now);
        assert!(matches!(
            repair.as_slice(),
            [Audio::Samples { .. }, Audio::Samples { .. }]
        ));
        assert_eq!(clock.expected, later + 2);

        // Real missing audio still expires using the remaining active wait.
        assert!(clock.append(later + 4, vec![3], now).is_empty());
        let mut expired = Vec::new();
        clock.drain(now + GAP_WAIT, false, &mut expired);
        assert!(matches!(
            expired.as_slice(),
            [
                Audio::Gap {
                    reason: "missing_audio",
                    ..
                },
                Audio::Samples { .. }
            ]
        ));
    }

    #[tokio::test]
    async fn tiny_packets_backpressure_and_drain_before_session_end() {
        use super::super::recognizer::{RecognitionStream, Recognizer};

        struct BlockedRecognizer {
            entered: Mutex<Option<oneshot::Sender<()>>>,
            release: std::sync::mpsc::Receiver<()>,
        }

        struct CountSamples(usize);
        impl RecognitionStream for CountSamples {
            fn push(&mut self, samples: &[f32]) -> anyhow::Result<Option<Hypothesis>> {
                self.0 += samples.len();
                Ok(None)
            }

            fn finish(self: Box<Self>) -> anyhow::Result<Hypothesis> {
                Ok(Hypothesis {
                    text: self.0.to_string(),
                    ..Hypothesis::default()
                })
            }
        }

        impl Recognizer for BlockedRecognizer {
            fn start(
                &self,
                _: &RecognitionOptions,
            ) -> anyhow::Result<Box<dyn RecognitionStream + '_>> {
                if let Some(entered) = self.entered.lock().unwrap().take() {
                    let _ = entered.send(());
                    self.release.recv_timeout(Duration::from_secs(3))?;
                }
                Ok(Box::new(CountSamples(0)))
            }
        }

        let (entered, started) = oneshot::channel();
        let (released, release) = std::sync::mpsc::channel();
        let (mut session, mut events, _close, _shutdown) = session_with_recognizer(
            move || {
                Ok(Box::new(BlockedRecognizer {
                    entered: Mutex::new(Some(entered)),
                    release,
                }))
            },
            1,
        )
        .await;
        session
            .register(&json!({"track_id": "mic", "role": "mixed_audio"}))
            .unwrap();
        let mut frames = 12_000;
        session.audio(
            "mic",
            vec![Audio::Samples {
                start: 0,
                samples: vec![12_000; frames],
            }],
        );
        tokio::time::timeout(Duration::from_secs(2), started)
            .await
            .unwrap()
            .unwrap();
        while !session.input_paused() {
            session.audio(
                "mic",
                vec![Audio::Samples {
                    start: frames as u64,
                    samples: vec![12_000; 3],
                }],
            );
            frames += 3;
            assert!(frames < 13_000, "paused by packet count before frame bound");
        }
        let track = &session.tracks["mic"];
        assert_eq!(track.capture.len(), MAX_CAPTURE_QUEUE);
        assert_eq!(track.capture_frames, 3 * MAX_CAPTURE_QUEUE);
        // Shutdown can arrive while normal socket input is paused. Its end marker
        // must follow all accepted samples through the decoder.
        session.end();
        released.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while !session.ended {
                session.poll();
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        let mut final_samples = None;
        while let Ok(event) = events.try_recv() {
            let event: Value = serde_json::from_str(event.text.as_ref()).unwrap();
            assert_ne!(event["type"], "transcriberrr.track.gap");
            if event["type"] == "transcriberrr.transcript.final" {
                final_samples = event["text"].as_str().unwrap().parse::<usize>().ok();
                assert_eq!(event["end_frame"], frames as u64);
            }
        }
        assert_eq!(final_samples, Some(frames * 2 / 3));
    }

    const BLOCK_FRAMES: u64 = 480;
    const OWN_START: u64 = 120 * BLOCK_FRAMES;
    const BLEED_START: u64 = 140 * BLOCK_FRAMES;
    const ITEM_END: u64 = 160 * BLOCK_FRAMES;

    fn tone(block: u64, amplitude: f32) -> Vec<i16> {
        let start = block * BLOCK_FRAMES;
        (start..start + BLOCK_FRAMES)
            .map(|frame| {
                let phase = frame as f32 * std::f32::consts::TAU * 200.0 / 24_000.0;
                (amplitude * phase.sin()) as i16
            })
            .collect()
    }

    fn prepare_bleed_item(session: &mut Session) {
        for track_id in ["mic", "peer"] {
            session
                .register(&json!({"track_id": track_id, "role": "isolated_mic"}))
                .unwrap();
        }
        let mut track = session.tracks.remove("mic").unwrap();
        let peer = session.tracks["peer"].key;
        // Calibrate with the peer leading. The mic then speaks independently before
        // picking up the peer. Hold that last peer packet through the draft decision.
        let mut original = Vec::new();
        for block in 0..230 {
            let amplitude = match block {
                0..40 | 160.. => 0.,
                120..140 => 18_000.,
                _ => 1_000.,
            };
            let start = block * BLOCK_FRAMES;
            let samples = tone(block, amplitude);
            session.bleed.ingest(track.key, start, &samples);
            if (OWN_START..ITEM_END).contains(&start) {
                original.extend_from_slice(&samples);
            }
            if block < 140 {
                let amplitude = if (40..120).contains(&block) {
                    10_000.
                } else {
                    0.
                };
                session.bleed.ingest(peer, start, &tone(block, amplitude));
            }
        }
        let mut draft_audio = original.clone();
        let quality = session
            .bleed
            .apply(track.key, OWN_START, &mut draft_audio, Context::Draft);
        assert!(quality.missing_peer);
        assert_eq!(
            &draft_audio[(BLEED_START - OWN_START) as usize..],
            &original[(BLEED_START - OWN_START) as usize..]
        );
        track.append_raw(OWN_START, original, &draft_audio);
        for block in 140..230 {
            let amplitude = if block < 160 { 10_000. } else { 0. };
            session
                .bleed
                .ingest(peer, block * BLOCK_FRAMES, &tone(block, amplitude));
        }
        assert!(
            session
                .bleed
                .ready_through(track.key, ITEM_END, FINAL_LOOKAHEAD)
        );
        session.tracks.insert("mic".into(), track);
    }

    fn scripted(turn: u64, start: u64, end: u64, update: PipelineUpdate) -> Output {
        Output {
            event: PipelineEvent {
                turn,
                start,
                end,
                update,
            },
            deadline: Instant::now() + FINAL_WAIT,
            redo: None,
            quality: Quality::default(),
        }
    }

    fn deliver_scripted(session: &mut Session, output: Output) {
        session
            .tracks
            .get_mut("mic")
            .unwrap()
            .outputs
            .push_back(output);
        session.poll_track("mic");
    }

    fn next_event(events: &mut EventReceiver) -> Value {
        serde_json::from_str(&events.try_recv().unwrap().text).unwrap()
    }

    #[tokio::test]
    async fn timed_bleed_filters_words_and_terminal_replay_is_immutable() {
        use super::super::recognizer::TimedWord;

        let (mut session, mut events, _close, _shutdown) =
            session_with_recognizer(|| Ok(Box::new(RedecodeRecognizer::default())), 2).await;
        Arc::get_mut(&mut session.state).unwrap().capabilities =
            super::super::recognizer::capabilities(crate::EngineKind::AppleNative);
        prepare_bleed_item(&mut session);
        next_event(&mut events);
        next_event(&mut events);
        let own = TimedWord {
            text: "Own speech.".into(),
            start_sample: OWN_START + 5 * BLOCK_FRAMES,
            end_sample: BLEED_START - 5 * BLOCK_FRAMES,
        };
        let leaked = TimedWord {
            text: " Leaked speech.".into(),
            start_sample: BLEED_START + 7 * BLOCK_FRAMES,
            end_sample: ITEM_END - 3 * BLOCK_FRAMES,
        };
        let hypothesis = Hypothesis {
            text: "Own speech. Leaked speech.".into(),
            words: vec![own.clone(), leaked.clone()],
            ..Hypothesis::default()
        };

        for (revision, text) in [
            (1, "Own speech."),
            (2, ""),
            (3, "Own speech. Leaked speech."),
        ] {
            let output = scripted(
                1,
                OWN_START,
                ITEM_END,
                PipelineUpdate::Draft(Hypothesis {
                    text: text.into(),
                    ..Hypothesis::default()
                }),
            );
            deliver_scripted(&mut session, output);
            let event = next_event(&mut events);
            assert_eq!(event["type"], "transcriberrr.transcript.draft");
            assert_eq!(event["revision"], revision);
            assert_eq!(event["text"], text);
            assert_eq!(event["item_id"], "item_0_1");
        }

        let output = scripted(1, OWN_START, ITEM_END, PipelineUpdate::Final(hypothesis));
        deliver_scripted(&mut session, output);
        let final_event = next_event(&mut events);
        assert_eq!(final_event["type"], "transcriberrr.transcript.final");
        assert_eq!(final_event["text"], own.text);
        assert_eq!(final_event["bleed_missing_peer"], false);
        assert_eq!(
            final_event["words"],
            json!([{
                "text": own.text, "start_frame": own.start_sample, "end_frame": own.end_sample,
            }])
        );

        let leaked_only = Hypothesis {
            text: leaked.text.clone(),
            words: vec![leaked],
            ..Hypothesis::default()
        };
        for update in [
            PipelineUpdate::Draft(leaked_only.clone()),
            PipelineUpdate::Final(leaked_only.clone()),
        ] {
            let output = scripted(2, BLEED_START, ITEM_END, update);
            deliver_scripted(&mut session, output);
        }
        let draft = next_event(&mut events);
        let retraction = next_event(&mut events);
        assert_eq!(draft["item_id"], retraction["item_id"]);
        assert_eq!(retraction["type"], "transcriberrr.transcript.retracted");
        assert_eq!(retraction["reason"], "bleed");

        for update in [
            PipelineUpdate::Draft(leaked_only.clone()),
            PipelineUpdate::Final(leaked_only),
        ] {
            let late = scripted(2, BLEED_START, ITEM_END, update);
            deliver_scripted(&mut session, late);
        }
        assert!(matches!(
            events.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        let (client, _events, _close) = connection();
        let (reply, response) = oneshot::channel();
        session.attach(
            Resume {
                session_id: session.id.clone(),
                resume_credential: session.credential.clone(),
                last_sequence: draft["sequence"].as_u64().unwrap(),
                accept_loss: false,
            },
            client,
            reply,
        );
        let resumed = response.await.unwrap().unwrap();
        assert_eq!(resumed.replay.len(), 1);
        let replayed: Value = serde_json::from_str(&resumed.replay[0]).unwrap();
        assert_eq!(replayed, retraction);
        let unique: std::collections::HashSet<_> = [
            retraction["sequence"].as_u64().unwrap(),
            replayed["sequence"].as_u64().unwrap(),
        ]
        .into_iter()
        .collect();
        assert_eq!(unique.len(), 1);
    }

    #[derive(Default)]
    struct RedecodeRecognizer(Arc<std::sync::atomic::AtomicUsize>);

    struct RedecodeStream {
        finishes: Arc<std::sync::atomic::AtomicUsize>,
        samples: Vec<f32>,
    }

    impl super::super::recognizer::Recognizer for RedecodeRecognizer {
        fn start(
            &self,
            _: &RecognitionOptions,
        ) -> anyhow::Result<Box<dyn super::super::recognizer::RecognitionStream + '_>> {
            Ok(Box::new(RedecodeStream {
                finishes: self.0.clone(),
                samples: Vec::new(),
            }))
        }
    }

    impl super::super::recognizer::RecognitionStream for RedecodeStream {
        fn push(&mut self, samples: &[f32]) -> anyhow::Result<Option<Hypothesis>> {
            self.samples.extend_from_slice(samples);
            Ok(None)
        }

        fn finish(self: Box<Self>) -> anyhow::Result<Hypothesis> {
            let last_own = self
                .samples
                .iter()
                .rposition(|sample| sample.abs() > 0.1)
                .unwrap();
            // Ignore the transition after the mic stops speaking. The test's leaked
            // word is deep enough into the peer's turn for calibration to mute it.
            let leaked = self
                .samples
                .iter()
                .skip(last_own + 4_800)
                .filter(|sample| sample.abs() > 0.003)
                .count()
                > 200;
            self.finishes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Hypothesis {
                text: if leaked {
                    "Own speech. Leaked speech."
                } else {
                    "Own speech."
                }
                .into(),
                ..Hypothesis::default()
            })
        }
    }

    #[tokio::test]
    async fn untimed_bleed_redecodes_final_pcm() {
        let fake = RedecodeRecognizer::default();
        let finishes = fake.0.clone();
        let (mut session, mut events, _close, _shutdown) =
            session_with_recognizer(move || Ok(Box::new(fake)), 2).await;
        prepare_bleed_item(&mut session);
        next_event(&mut events);
        next_event(&mut events);
        let output = scripted(
            1,
            OWN_START,
            ITEM_END,
            PipelineUpdate::Final(Hypothesis {
                text: "Own speech. Leaked speech.".into(),
                ..Hypothesis::default()
            }),
        );
        deliver_scripted(&mut session, output);
        tokio::time::timeout(Duration::from_secs(3), async {
            while !session.tracks["mic"].outputs.is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
                session.poll_track("mic");
            }
        })
        .await
        .unwrap();
        let event = next_event(&mut events);
        assert_eq!(event["type"], "transcriberrr.transcript.final");
        assert_eq!(event["text"], "Own speech.");
        assert_eq!(event["bleed_missing_peer"], false);
        assert_eq!(event["words"], json!([]));
        assert_eq!(finishes.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn untimed_bleed_compares_final_pcm_to_the_actual_decoder_input() {
        let (mut session, _events, _close, _shutdown) = session_fixture().await;
        session
            .register(&json!({"track_id": "mic", "role": "isolated_mic"}))
            .unwrap();
        let mut track = session.tracks.remove("mic").unwrap();
        for packet in 0..3 {
            let start = packet * MAX_PACKET_FRAMES as u64;
            let original = vec![12_000; MAX_PACKET_FRAMES];
            session.bleed.ingest(track.key, start, &original);
            // The decoder saw the middle packet muted. Reapplying the current
            // draft decision would pass all three and miss the change.
            let gated = if packet == 1 {
                vec![0; MAX_PACKET_FRAMES]
            } else {
                original.clone()
            };
            track.append_raw(start, original, &gated);
        }
        let mut hypothesis = Hypothesis {
            text: "Provisional text.".into(),
            ..Hypothesis::default()
        };
        let mut redo = None;
        let result = session.bleed_final(
            &track,
            0..3 * MAX_PACKET_FRAMES as u64,
            &mut hypothesis,
            &mut redo,
            &mut Quality::default(),
            true,
        );
        assert!(matches!(result, BleedFinal::Wait));
        assert!(redo.is_some());
    }

    #[tokio::test]
    async fn audio_clock_jump_keeps_raw_pcm_for_a_pending_final() {
        let (mut session, _events, _close, _shutdown) = session_fixture().await;
        session
            .register(&json!({"track_id": "mic", "role": "isolated_mic"}))
            .unwrap();
        let track = session.tracks.get_mut("mic").unwrap();
        track.append_raw(0, vec![1; 6], &[1, 1, 0, 0, 1, 1]);
        let next = 3 * MAX_NATIVE_FRAMES;
        track.append_raw(next, vec![2; 6], &[2; 6]);
        assert_eq!(track.interval(1, 5), Some((vec![1; 4], vec![1, 0, 0, 1])));
        assert!(track.interval(5, 7).is_none());
        assert!(track.interval(next, next + 6).is_some());
    }
}
