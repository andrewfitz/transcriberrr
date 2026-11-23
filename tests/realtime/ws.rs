use std::{
    net::SocketAddr,
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

use axum::serve;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{Semaphore, oneshot, watch},
    task::JoinHandle,
    time::timeout,
};
use tokio_tungstenite::{
    WebSocketStream, connect_async,
    tungstenite::{
        Error, Message,
        client::IntoClientRequest,
        http::{StatusCode, header},
    },
};

use super::{
    AppState, ListenerState, pipeline, recognizer,
    recognizer::{Hypothesis, RecognitionOptions, RecognitionStream, Recognizer, TimedWord},
    router, tracks,
};
use crate::EngineKind;

const TEST_KEY: &str = "test-only-key";
const ORIGIN: &str = "http://localhost:3000";
const WAIT: Duration = Duration::from_secs(5);
const COMPLETED: &str = "conversation.item.input_audio_transcription.completed";
const DELTA: &str = "conversation.item.input_audio_transcription.delta";
const FAILED: &str = "conversation.item.input_audio_transcription.failed";

type Socket = WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>;

struct TestServer {
    address: SocketAddr,
    _app_tx: watch::Sender<Option<Arc<AppState>>>,
    shutdown: watch::Sender<bool>,
    task: JoinHandle<()>,
    blockade: Arc<Blockade>,
}

impl TestServer {
    async fn start(max_sessions: usize, max_tracks: usize) -> Self {
        Self::with_engine(max_sessions, max_tracks, EngineKind::ParakeetV3).await
    }

    async fn with_engine(max_sessions: usize, max_tracks: usize, engine: EngineKind) -> Self {
        let capabilities = recognizer::capabilities(engine);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown, shutdown_rx) = watch::channel(false);
        let blockade = Arc::new(Blockade::default());
        let model_blockade = blockade.clone();
        let (app_tx, app) = watch::channel(Some(Arc::new(AppState {
            model: "test-model".into(),
            engine: "test-engine".into(),
            worker: pipeline::RecognitionWorker::start(move || {
                Ok(Box::new(FakeRecognizer {
                    blockade: model_blockade,
                    revisable: capabilities.revisable_drafts,
                    timed: capabilities.word_timing,
                }))
            })
            .await
            .unwrap(),
            capabilities,
            shutdown: shutdown_rx.clone(),
            active_sessions: Arc::new(Semaphore::new(max_sessions)),
            max_sessions,
            max_tracks,
            native: tracks::Registry::default(),
            #[cfg(feature = "diarization")]
            diarization: None,
        })));
        let state = ListenerState {
            app,
            shutdown: shutdown_rx,
            allowed_origins: vec![ORIGIN.into()],
            key: Some(ring::digest::digest(
                &ring::digest::SHA256,
                TEST_KEY.as_bytes(),
            )),
            sockets: Arc::new(Semaphore::new(16)),
            socket_limit: 16,
            max_sessions,
        };
        let task =
            tokio::spawn(async move { serve(listener, router(Arc::new(state))).await.unwrap() });
        Self {
            address,
            _app_tx: app_tx,
            shutdown,
            task,
            blockade,
        }
    }

    async fn connect(
        &self,
        version: &str,
        auth: Option<&str>,
        origin: Option<&str>,
        protocols: &str,
    ) -> Result<Socket, Error> {
        let url = format!(
            "ws://{}/v1/realtime?intent=transcription{version}",
            self.address
        );
        let mut request = url.into_client_request()?;
        let headers = request.headers_mut();
        if let Some(auth) = auth {
            headers.insert(header::AUTHORIZATION, auth.parse().unwrap());
        }
        if let Some(origin) = origin {
            headers.insert(header::ORIGIN, origin.parse().unwrap());
        }
        headers.insert(header::SEC_WEBSOCKET_PROTOCOL, protocols.parse().unwrap());
        let (socket, response) = timeout(WAIT, connect_async(request)).await.unwrap()?;
        assert_eq!(
            response.headers()[header::SEC_WEBSOCKET_PROTOCOL],
            "realtime"
        );
        Ok(socket)
    }

    async fn models(&self) -> Value {
        let mut stream = TcpStream::connect(self.address).await.unwrap();
        let request = format!(
            "GET /v1/models HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {TEST_KEY}\r\nConnection: close\r\n\r\n",
            self.address
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = String::new();
        timeout(WAIT, stream.read_to_string(&mut response))
            .await
            .unwrap()
            .unwrap();
        let (headers, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(headers.starts_with("HTTP/1.1 200 "), "{headers}");
        serde_json::from_str(body).unwrap()
    }

    async fn socket(&self, native: bool) -> Socket {
        let version = if native { "&transcriberrr=1" } else { "" };
        self.connect(
            version,
            Some(&format!("Bearer {TEST_KEY}")),
            None,
            "realtime",
        )
        .await
        .unwrap()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.shutdown.send(true);
        self.task.abort();
    }
}

#[derive(Default)]
struct Blockade {
    state: Mutex<BlockadeState>,
    released: Condvar,
}

#[derive(Default)]
struct BlockadeState {
    armed: Option<oneshot::Sender<()>>,
    held: bool,
}

impl Blockade {
    fn arm(&self) -> (oneshot::Receiver<()>, Release<'_>) {
        let (sender, started) = oneshot::channel();
        self.state.lock().unwrap().armed = Some(sender);
        (started, Release(self))
    }

    fn wait_if_armed(&self, samples: &[f32]) {
        // Only the intentionally louder overload packet can hold the worker.
        // A previous packet can still be processing after its first draft.
        if !samples.iter().any(|sample| sample.abs() > 0.6) {
            return;
        }
        let mut state = self.state.lock().unwrap();
        if let Some(sender) = state.armed.take() {
            state.held = true;
            let _ = sender.send(());
            let (_state, _) = self
                .released
                .wait_timeout_while(state, WAIT, |state| state.held)
                .unwrap();
        }
    }
}

struct Release<'a>(&'a Blockade);

impl Drop for Release<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap();
        state.armed = None;
        state.held = false;
        self.0.released.notify_one();
    }
}

struct FakeRecognizer {
    blockade: Arc<Blockade>,
    revisable: bool,
    timed: bool,
}
struct FakeStream {
    text: String,
    blockade: Arc<Blockade>,
    revisable: bool,
    timed: bool,
}

impl Recognizer for FakeRecognizer {
    fn start(&self, _: &RecognitionOptions) -> anyhow::Result<Box<dyn RecognitionStream + '_>> {
        Ok(Box::new(FakeStream {
            text: String::new(),
            blockade: self.blockade.clone(),
            revisable: self.revisable,
            timed: self.timed,
        }))
    }
}

impl RecognitionStream for FakeStream {
    fn push(&mut self, samples: &[f32]) -> anyhow::Result<Option<Hypothesis>> {
        self.blockade.wait_if_armed(samples);
        if !self.revisable && !self.text.is_empty() {
            return Ok(None);
        }
        let mut speech = samples.iter().filter(|sample| sample.abs() > 0.2);
        let sample = if self.revisable {
            speech.next_back()
        } else {
            speech.next()
        };
        let Some(first) = sample else {
            return Ok(None);
        };
        let next = match (self.revisable, *first > 0.0) {
            (true, true) => "we need a map",
            (true, false) => "we need the map",
            (false, true) => "positive",
            (false, false) => "negative",
        };
        if self.text == next {
            return Ok(None);
        }
        self.text = next.into();
        Ok(Some(self.hypothesis(!self.revisable)))
    }

    fn finish(self: Box<Self>) -> anyhow::Result<Hypothesis> {
        Ok(self.hypothesis(true))
    }
}

impl FakeStream {
    fn hypothesis(&self, stable: bool) -> Hypothesis {
        let words = (self.timed && !self.text.is_empty())
            .then(|| TimedWord {
                text: self.text.clone(),
                start_sample: 0,
                end_sample: 320,
                spoken: None,
            })
            .into_iter()
            .collect();
        Hypothesis {
            text: self.text.clone(),
            stable,
            words,
            ..Default::default()
        }
    }
}

async fn send(socket: &mut Socket, event: Value) {
    socket
        .send(Message::Text(event.to_string().into()))
        .await
        .unwrap();
}

async fn receive(socket: &mut Socket) -> Value {
    let received = timeout(WAIT, socket.next())
        .await
        .expect("server event timed out");
    let Message::Text(text) = received.expect("unexpected EOF").expect("WebSocket error") else {
        panic!("expected a text event");
    };
    serde_json::from_str(&text).unwrap()
}

async fn event(socket: &mut Socket, kind: &str) -> Value {
    let received = receive(socket).await;
    assert_eq!(received["type"], kind, "{received}");
    received
}

fn pcm(value: i16, frames: usize) -> String {
    STANDARD.encode(value.to_le_bytes().repeat(frames))
}

async fn append(socket: &mut Socket, value: i16, frames: usize) {
    send(
        socket,
        json!({"type":"input_audio_buffer.append", "audio":pcm(value,frames)}),
    )
    .await;
}

fn settings(model: &str) -> Value {
    json!({
        "type":"session.update", "session":{
            "type":"transcription", "audio":{"input":{
                "format":{"type":"audio/pcm","rate":24000},
                "transcription":{"model":model},
                "turn_detection":null
            }}
        }
    })
}

async fn commit(socket: &mut Socket) -> Value {
    send(socket, json!({"type":"input_audio_buffer.commit"})).await;
    let ack = event(socket, "input_audio_buffer.committed").await;
    let item = event(socket, "conversation.item.added").await;
    assert_eq!(ack["item_id"], item["item"]["id"]);
    ack
}

#[tokio::test]
async fn admission_and_browser_handshake() {
    let server = TestServer::start(1, 1).await;
    for (version, auth, origin, expected) in [
        ("", None, None, StatusCode::UNAUTHORIZED),
        ("", Some("Bearer wrong"), None, StatusCode::UNAUTHORIZED),
        (
            "",
            Some("Bearer test-only-key"),
            Some("https://disallowed.example"),
            StatusCode::FORBIDDEN,
        ),
        (
            "&transcriberrr=2",
            Some("Bearer test-only-key"),
            None,
            StatusCode::BAD_REQUEST,
        ),
    ] {
        let failure = server
            .connect(version, auth, origin, "realtime")
            .await
            .unwrap_err();
        let Error::Http(response) = failure else {
            panic!("expected HTTP rejection, got {failure}");
        };
        assert_eq!(response.status(), expected);
        if expected == StatusCode::UNAUTHORIZED {
            assert_eq!(response.headers()[header::WWW_AUTHENTICATE], "Bearer");
        }
    }
    let key_protocol = format!("openai-insecure-api-key.{TEST_KEY}");
    let error = server
        .connect("", None, Some(ORIGIN), &key_protocol)
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Http(response) if response.status() == StatusCode::UNAUTHORIZED)
    );
    let mut authorized = server
        .connect("", None, Some(ORIGIN), &format!("realtime, {key_protocol}"))
        .await
        .unwrap();
    event(&mut authorized, "session.created").await;
    let error = server
        .connect("", Some("Bearer   test-only-key"), None, "realtime")
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Http(response) if response.status() == StatusCode::TOO_MANY_REQUESTS)
    );
}

#[tokio::test]
async fn listener_rejects_upgrades_while_draining() {
    let server = TestServer::start(1, 1).await;
    server.shutdown.send_replace(true);
    for version in ["", "&transcriberrr=1"] {
        let result = server
            .connect(version, Some("Bearer test-only-key"), None, "realtime")
            .await;
        assert!(
            matches!(result, Err(Error::Http(response)) if response.status() == StatusCode::SERVICE_UNAVAILABLE)
        );
    }
}

#[tokio::test]
async fn standard_turns_clear_validation_and_concurrent_isolation() {
    let server = TestServer::start(2, 2).await;
    let (mut first, mut second) = tokio::join!(server.socket(false), server.socket(false));
    for socket in [&mut first, &mut second] {
        event(socket, "session.created").await;
        send(socket, settings("gpt-live-transcribe")).await;
        let updated = event(socket, "session.updated").await;
        assert_eq!(
            updated["session"]["audio"]["input"]["transcription"]["model"],
            "gpt-live-transcribe"
        );
    }
    for (option, value) in [
        ("keywords", json!(["word"])),
        ("delay", json!(1)),
        ("languages", json!(["en", "fr"])),
        ("prompt", json!("hint")),
    ] {
        let mut invalid = settings("gpt-live-transcribe");
        invalid["session"]["audio"]["input"]["transcription"][option] = value;
        invalid["event_id"] = json!(option);
        send(&mut first, invalid).await;
        assert_eq!(
            event(&mut first, "error").await["error"]["event_id"],
            option
        );
    }
    let mut invalid = settings("gpt-live-transcribe");
    invalid["session"]["audio"]["input"]["turn_detection"] = json!({"type":"server_vad"});
    send(&mut first, invalid).await;
    event(&mut first, "error").await;
    send(
        &mut first,
        json!({"type":"input_audio_buffer.append","audio":"AA=="}),
    )
    .await;
    event(&mut first, "error").await;
    send(&mut first, json!({"type":"input_audio_buffer.commit"})).await;
    assert_eq!(
        event(&mut first, "error").await["error"]["code"],
        "input_audio_buffer_commit_empty"
    );

    tokio::join!(
        append(&mut first, 12000, 12000),
        append(&mut second, -12000, 12000)
    );
    let (a, b) = tokio::join!(event(&mut first, DELTA), event(&mut second, DELTA));
    assert_eq!(a["delta"], "positive");
    assert_eq!(b["delta"], "negative");
    assert_ne!(a["item_id"], b["item_id"]);
    send(&mut first, json!({"type":"input_audio_buffer.clear"})).await;
    event(&mut first, "input_audio_buffer.cleared").await;

    let ack = commit(&mut second).await;
    assert_eq!(ack["item_id"], b["item_id"]);
    let complete = event(&mut second, COMPLETED).await;
    assert_eq!(complete["item_id"], b["item_id"]);
    assert_eq!(complete["transcript"], "negative");

    append(&mut first, 0, 12000).await;
    let ack = commit(&mut first).await;
    assert_ne!(ack["item_id"], a["item_id"]);
    let complete = event(&mut first, COMPLETED).await;
    assert_eq!(complete["item_id"], ack["item_id"]);
    assert_eq!(complete["transcript"], "");

    send(&mut first, settings("gpt-transcribe")).await;
    event(&mut first, "session.updated").await;
    append(&mut first, -12000, 12000).await;
    let ack = commit(&mut first).await;
    let complete = event(&mut first, COMPLETED).await;
    assert_eq!(complete["transcript"], "negative");
    assert_eq!(complete["item_id"], ack["item_id"]);
}

async fn register(socket: &mut Socket, track: &str) -> Value {
    register_role(socket, track, "isolated_mic").await
}

async fn register_role(socket: &mut Socket, track: &str, role: &str) -> Value {
    send(
        socket,
        json!({
            "type":"transcriberrr.track.register", "track_id":track,
            "role":role, "speaker":track
        }),
    )
    .await;
    let event = event(socket, "transcriberrr.track.registered").await;
    assert_eq!(event["track_id"], track);
    event
}

async fn append_track(socket: &mut Socket, track: &str, start: u64, frames: usize, amplitude: i16) {
    send(
        socket,
        json!({
            "type":"transcriberrr.track.append", "track_id":track,
            "start_frame":start, "audio":pcm(amplitude, frames)
        }),
    )
    .await;
}

async fn resume(socket: &mut Socket, greeting: &Value, last: u64, accept_loss: bool) {
    send(
        socket,
        json!({
            "type":"transcriberrr.session.resume", "session_id":greeting["session_id"],
            "resume_credential":greeting["resume_credential"],
            "last_sequence":last, "accept_loss":accept_loss
        }),
    )
    .await;
}

#[tokio::test]
async fn explicit_resume_waits_for_credentials_with_all_session_slots_occupied() {
    let server = TestServer::start(1, 1).await;
    let mut original = server.socket(true).await;
    let greeting = event(&mut original, "transcriberrr.session.created").await;
    let track = register(&mut original, "mic").await;

    let mut next = server
        .connect(
            "&transcriberrr=1&resume=1",
            Some(&format!("Bearer {TEST_KEY}")),
            None,
            "realtime",
        )
        .await
        .unwrap();
    // Wait past the automatic new-session greeting on an ordinary connection.
    assert!(
        timeout(Duration::from_millis(400), next.next())
            .await
            .is_err()
    );
    resume(&mut next, &greeting, 0, false).await;
    let resumed = event(&mut next, "transcriberrr.session.resumed").await;
    assert_eq!(resumed["session_id"], greeting["session_id"]);
    assert_eq!(resumed["next_expected_frames"], json!({"mic": 0}));
    assert_eq!(receive(&mut next).await, track);
}

#[tokio::test]
async fn native_echoes_client_close_during_negotiation_and_after_registration() {
    use tokio_tungstenite::tungstenite::protocol::{CloseFrame, frame::coding::CloseCode};

    let server = TestServer::start(2, 1).await;
    for registered in [false, true] {
        let mut socket = server.socket(true).await;
        if registered {
            event(&mut socket, "transcriberrr.session.created").await;
            register(&mut socket, "mic").await;
        }
        let frame = CloseFrame {
            code: CloseCode::Normal,
            reason: "capture finished".into(),
        };
        socket
            .send(Message::Close(Some(frame.clone())))
            .await
            .unwrap();
        let reply = timeout(WAIT, async {
            loop {
                match socket.next().await {
                    Some(Ok(Message::Close(frame))) => break frame,
                    Some(Ok(_)) => continue,
                    other => panic!("connection ended without close reply: {other:?}"),
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(reply, Some(frame));
    }
}

#[tokio::test]
async fn revisable_drafts_replace_text_and_qwen_only_accepts_committed_alias() {
    let server = TestServer::with_engine(2, 1, EngineKind::Qwen3).await;
    let models = server.models().await;
    assert_eq!(
        models["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["test-model", "gpt-transcribe"]
    );
    let mut native = server.socket(true).await;
    let greeting = event(&mut native, "transcriberrr.session.created").await;
    assert_eq!(greeting["engine"], "test-engine");
    assert_eq!(greeting["model"], "test-model");
    assert_eq!(greeting["capabilities"]["revisable_drafts"], true);
    assert_eq!(greeting["capabilities"]["stable_deltas"], false);
    assert_eq!(greeting["capabilities"]["word_timing"], false);
    assert!(greeting["capabilities"]["word_timing_granularity"].is_null());
    send(
        &mut native,
        json!({
            "type": "transcriberrr.track.register", "track_id": "mic", "role": "mixed_audio",
            "start_frame": 24_000
        }),
    )
    .await;
    let registered = event(&mut native, "transcriberrr.track.registered").await;
    assert_eq!(registered["next_expected_frame"], 24_000);
    append_track(&mut native, "mic", 24_000, 12_000, 12_000).await;
    let first = event(&mut native, "transcriberrr.transcript.draft").await;
    assert_eq!(first["text"], "we need a map");
    assert_eq!(first["revision"], 1);
    assert_eq!(first["words"], json!([]));
    append_track(&mut native, "mic", 36_000, 12_000, -12_000).await;
    let replacement = event(&mut native, "transcriberrr.transcript.draft").await;
    assert_eq!(replacement["item_id"], first["item_id"]);
    assert_eq!(replacement["revision"], 2);
    assert_eq!(replacement["text"], "we need the map");
    send(
        &mut native,
        json!({"type":"transcriberrr.track.commit", "track_id":"mic"}),
    )
    .await;
    let final_caption = event(&mut native, "transcriberrr.transcript.final").await;
    assert_eq!(final_caption["item_id"], first["item_id"]);
    assert_eq!(final_caption["text"], "we need the map");
    assert_eq!(final_caption["words"], json!([]));
    assert_eq!(final_caption["start_frame"], 24_000);
    assert_eq!(final_caption["end_frame"], 48_000);

    let mut standard = server.socket(false).await;
    event(&mut standard, "session.created").await;
    send(&mut standard, settings("gpt-live-transcribe")).await;
    assert_eq!(
        event(&mut standard, "error").await["error"]["code"],
        "model_not_found"
    );
    send(&mut standard, settings("gpt-transcribe")).await;
    let updated = event(&mut standard, "session.updated").await;
    assert_eq!(
        updated["session"]["audio"]["input"]["transcription"]["model"],
        "gpt-transcribe"
    );
    append(&mut standard, 12_000, 12_000).await;
    append(&mut standard, -12_000, 12_000).await;
    let ack = commit(&mut standard).await;
    let completion = event(&mut standard, COMPLETED).await;
    assert_eq!(completion["item_id"], ack["item_id"]);
    assert_eq!(completion["transcript"], "we need the map");
}

#[tokio::test]
async fn native_replay_loss_takeover_and_duplicate_frames() {
    let server = TestServer::start(1, 2).await;
    let mut old = server.socket(true).await;
    let greeting = event(&mut old, "transcriberrr.session.created").await;
    assert_eq!(greeting["model"], "test-model");
    assert_eq!(greeting["engine"], "test-engine");
    assert_eq!(greeting["capabilities"]["word_timing"], true);
    assert_eq!(greeting["capabilities"]["word_timing_granularity"], "word");
    let a = register(&mut old, "alice").await;
    let b = register(&mut old, "bob").await;
    assert_eq!(a["sequence"], 1);
    assert_eq!(b["sequence"], 2);
    send(
        &mut old,
        json!({
            "type":"transcriberrr.track.register", "track_id":"overflow", "role":"isolated_mic"
        }),
    )
    .await;
    event(&mut old, "error").await;
    append_track(&mut old, "alice", 0, 960, 0).await;
    append_track(&mut old, "alice", 0, 960, 0).await;
    append_track(&mut old, "bob", 0, 480, 0).await;
    send(
        &mut old,
        json!({"type":"transcriberrr.session.ack","sequence":1}),
    )
    .await;
    // An invalid acknowledgement acts as a barrier after the successful one.
    send(
        &mut old,
        json!({"type":"transcriberrr.session.ack","sequence":900}),
    )
    .await;
    assert_eq!(
        event(&mut old, "error").await["error"]["code"],
        "invalid_sequence"
    );

    let mut next = server.socket(true).await;
    resume(&mut next, &greeting, 0, false).await;
    let failure = event(&mut next, "error").await;
    assert_eq!(failure["error"]["code"], "replay_unavailable");
    assert_eq!(failure["error"]["earliest_replayable_sequence"], 2);

    // A rejected resume must leave ownership with the old socket.
    send(
        &mut old,
        json!({"type":"transcriberrr.session.ack","sequence":901}),
    )
    .await;
    assert_eq!(
        event(&mut old, "error").await["error"]["code"],
        "invalid_sequence"
    );
    resume(&mut next, &greeting, 0, true).await;
    let ack = event(&mut next, "transcriberrr.session.resumed").await;
    assert_eq!(ack["lost_range"], json!([1, 1]));
    assert_eq!(ack["next_expected_frames"], json!({"alice":960,"bob":480}));
    assert_eq!(ack["earliest_replayable_sequence"], 2);
    assert_eq!(ack.get("sequence"), None);
    assert_eq!(receive(&mut next).await, b);

    let Message::Close(Some(close)) = timeout(WAIT, old.next()).await.unwrap().unwrap().unwrap()
    else {
        panic!("expected the old socket to close after takeover");
    };
    assert_eq!(u16::from(close.code), 4009);

    next.close(None).await.unwrap();
    let mut resumed = server.socket(true).await;
    resume(&mut resumed, &greeting, 1, false).await;
    assert_eq!(
        event(&mut resumed, "transcriberrr.session.resumed").await["lost_range"],
        Value::Null
    );
    assert_eq!(receive(&mut resumed).await, b);
    send(&mut resumed, json!({"type":"transcriberrr.session.end"})).await;
    loop {
        let item = receive(&mut resumed).await;
        assert_ne!(item["type"], "error", "{item}");
        if item["type"] == "transcriberrr.session.ended" {
            break;
        }
    }
}

#[tokio::test]
async fn standard_overload_after_delta_recovers_on_next_turn() {
    let server = TestServer::start(1, 1).await;
    let mut socket = server.socket(false).await;
    event(&mut socket, "session.created").await;
    send(&mut socket, settings("gpt-live-transcribe")).await;
    event(&mut socket, "session.updated").await;
    append(&mut socket, 12000, 12000).await;
    let partial = event(&mut socket, DELTA).await;

    let (started, release) = server.blockade.arm();
    append(&mut socket, 24000, 24000).await;
    timeout(WAIT, started).await.unwrap().unwrap();
    for _ in 0..4 {
        append(&mut socket, 12000, 24000).await;
    }
    let failure = event(&mut socket, FAILED).await;
    assert_eq!(failure["item_id"], partial["item_id"]);
    assert_eq!(failure["error"]["code"], "overload");
    append(&mut socket, -12000, 24000).await;
    assert_eq!(commit(&mut socket).await["item_id"], partial["item_id"]);
    drop(release);

    // Sentinels verify the connection continues processing client messages,
    // and no completion or second failure was emitted for the discarded turn.
    send(&mut socket, settings("gpt-transcribe")).await;
    event(&mut socket, "session.updated").await;
    append(&mut socket, -12000, 12000).await;
    let fresh = commit(&mut socket).await;
    let completion = event(&mut socket, COMPLETED).await;
    assert_ne!(fresh["item_id"], partial["item_id"]);
    assert_eq!(completion["item_id"], fresh["item_id"]);
    assert_eq!(completion["transcript"], "negative");
}

#[tokio::test]
async fn native_gap_and_terminal_caption_replay() {
    let server = TestServer::start(1, 2).await;
    let mut socket = server.socket(true).await;
    let greeting = event(&mut socket, "transcriberrr.session.created").await;
    register(&mut socket, "alice").await;
    register(&mut socket, "bob").await;

    // Both tracks start together. Bob then drops exactly 1,200 frames.
    append_track(&mut socket, "alice", 0, 12000, 12000).await;
    append_track(&mut socket, "bob", 0, 12000, -12000).await;
    append_track(&mut socket, "bob", 13200, 12000, -12000).await;
    append_track(&mut socket, "alice", 12000, 13200, 12000).await;
    let mut saw_gap = false;
    let mut alice_draft = None;
    while !saw_gap || alice_draft.is_none() {
        let result = receive(&mut socket).await;
        match result["type"].as_str().unwrap() {
            "transcriberrr.transcript.draft" if result["track_id"] == "alice" => {
                assert_eq!(result["text"], "positive");
                alice_draft = result["item_id"].as_str().map(str::to_owned);
            }
            "transcriberrr.track.gap" => {
                assert_eq!(result["track_id"], "bob");
                assert_eq!(result["start_frame"], 12000);
                assert_eq!(result["end_frame"], 13200);
                assert_eq!(result["reason"], "missing_audio");
                saw_gap = true;
            }
            "error" => panic!("unexpected protocol error: {result}"),
            _ => {}
        }
    }
    send(
        &mut socket,
        json!({"type":"transcriberrr.track.commit","track_id":"alice"}),
    )
    .await;
    let final_caption = loop {
        let result = receive(&mut socket).await;
        if result["type"] == "transcriberrr.transcript.final" && result["track_id"] == "alice" {
            break result;
        }
        assert_ne!(result["type"], "error");
    };
    assert_eq!(final_caption["item_id"], alice_draft.unwrap());
    assert_eq!(final_caption["text"], "positive");
    assert_eq!(final_caption["speaker"], "alice");
    assert_eq!(final_caption["start_frame"], 0);
    assert!(final_caption["end_frame"].as_u64().unwrap() <= 25200);
    assert_eq!(
        final_caption["words"],
        json!([
            {"text":"positive","start_frame":0,"end_frame":480}
        ])
    );
    let sequence = final_caption["sequence"].as_u64().unwrap();

    socket.close(None).await.unwrap();
    let mut next = server.socket(true).await;
    resume(&mut next, &greeting, sequence - 1, false).await;
    let ack = event(&mut next, "transcriberrr.session.resumed").await;
    assert_eq!(
        ack["next_expected_frames"],
        json!({"alice":25200,"bob":25200})
    );
    assert_eq!(receive(&mut next).await, final_caption);
    send(&mut next, json!({"type":"transcriberrr.session.end"})).await;
    let mut last_sequence = sequence;
    loop {
        let result = receive(&mut next).await;
        assert_eq!(result["sequence"], last_sequence + 1, "{result}");
        last_sequence += 1;
        assert_ne!(result["item_id"], final_caption["item_id"]);
        if result["type"] == "transcriberrr.session.ended" {
            break;
        }
    }
}

#[tokio::test]
async fn native_backpressure_preserves_both_tracks_through_end() {
    let server = TestServer::with_engine(1, 2, EngineKind::Qwen3).await;
    let mut socket = server.socket(true).await;
    event(&mut socket, "transcriberrr.session.created").await;
    register_role(&mut socket, "busy", "mixed_audio").await;
    register_role(&mut socket, "peer", "mixed_audio").await;
    append_track(&mut socket, "busy", 0, 12_000, 12_000).await;
    let first = event(&mut socket, "transcriberrr.transcript.draft").await;
    assert_eq!(first["text"], "we need a map");

    let (started, release) = server.blockade.arm();
    append_track(&mut socket, "busy", 12_000, 24_000, 24_000).await;
    timeout(WAIT, started).await.unwrap().unwrap();
    append_track(&mut socket, "busy", 36_000, 24_000, 12_000).await;
    append_track(&mut socket, "busy", 60_000, 24_000, 12_000).await;
    assert_eq!(
        event(&mut socket, "transcriberrr.track.backlog").await["track_id"],
        "busy"
    );

    // The last packet revises the caption. Its text and frame bounds prove
    // that the worker consumed audio submitted after its input queue filled.
    append_track(&mut socket, "busy", 84_000, 12_000, -12_000).await;
    append_track(&mut socket, "peer", 0, 12_000, 12_000).await;
    send(&mut socket, json!({"type":"transcriberrr.session.end"})).await;
    drop(release);

    let mut finals = std::collections::BTreeMap::new();
    loop {
        let item = receive(&mut socket).await;
        match item["type"].as_str().unwrap() {
            "transcriberrr.transcript.final" => {
                let track = item["track_id"].as_str().unwrap().to_owned();
                assert!(finals.insert(track, item).is_none());
            }
            "transcriberrr.session.ended" => break,
            "error" | "transcriberrr.track.gap" | "transcriberrr.transcript.retracted" => {
                panic!("unexpected event while draining audio: {item}")
            }
            _ => {}
        }
    }
    assert_eq!(finals.len(), 2);
    assert_eq!(finals["busy"]["item_id"], first["item_id"]);
    assert_eq!(finals["busy"]["text"], "we need the map");
    assert_eq!(finals["busy"]["start_frame"], 0);
    assert_eq!(finals["busy"]["end_frame"], 96_000);
    assert_eq!(finals["peer"]["text"], "we need a map");
    assert_eq!(finals["peer"]["end_frame"], 12_000);
}

#[tokio::test]
async fn native_resume_preempts_stalled_input_and_recovers_from_server_cursor() {
    let server = TestServer::with_engine(1, 1, EngineKind::Qwen3).await;
    let mut old = server.socket(true).await;
    let greeting = event(&mut old, "transcriberrr.session.created").await;
    register_role(&mut old, "mic", "mixed_audio").await;
    let (started, release) = server.blockade.arm();
    append_track(&mut old, "mic", 0, 24_000, 24_000).await;
    timeout(WAIT, started).await.unwrap().unwrap();
    const PACKET: u64 = 24_000;
    const BEFORE_TAKEOVER: u64 = 8 * PACKET;
    for start in (PACKET..BEFORE_TAKEOVER).step_by(PACKET as usize) {
        append_track(&mut old, "mic", start, PACKET as usize, 12_000).await;
    }
    let backlog = event(&mut old, "transcriberrr.track.backlog").await;
    let mut next = server
        .connect(
            "&transcriberrr=1&resume=1",
            Some(&format!("Bearer {TEST_KEY}")),
            None,
            "realtime",
        )
        .await
        .unwrap();
    resume(
        &mut next,
        &greeting,
        backlog["sequence"].as_u64().unwrap(),
        false,
    )
    .await;
    let ack = event(&mut next, "transcriberrr.session.resumed").await;
    let accepted = ack["next_expected_frames"]["mic"].as_u64().unwrap();
    assert!(
        (PACKET..BEFORE_TAKEOVER).contains(&accepted),
        "expected part of the original input to still be pending: {ack}"
    );
    drop(release);
    for start in (accepted..BEFORE_TAKEOVER).step_by(PACKET as usize) {
        append_track(&mut next, "mic", start, PACKET as usize, 12_000).await;
    }
    append_track(&mut next, "mic", BEFORE_TAKEOVER, PACKET as usize, -12_000).await;
    send(&mut next, json!({"type":"transcriberrr.session.end"})).await;
    let mut finals = 0;
    loop {
        let item = receive(&mut next).await;
        match item["type"].as_str().unwrap() {
            "transcriberrr.transcript.final" => {
                assert_eq!(item["text"], "we need the map");
                assert_eq!(item["start_frame"], 0);
                assert_eq!(item["end_frame"], BEFORE_TAKEOVER + PACKET);
                finals += 1;
            }
            "transcriberrr.session.ended" => break,
            "error" | "transcriberrr.track.gap" | "transcriberrr.transcript.retracted" => {
                panic!("unexpected recovery event: {item}")
            }
            _ => {}
        }
    }
    assert_eq!(finals, 1);
}

#[tokio::test]
async fn standard_short_appends_survive_a_stalled_worker_through_commit() {
    let server = TestServer::start(1, 1).await;
    let mut socket = server.socket(false).await;
    event(&mut socket, "session.created").await;
    send(&mut socket, settings("gpt-live-transcribe")).await;
    event(&mut socket, "session.updated").await;
    append(&mut socket, 12000, 12000).await;
    let partial = event(&mut socket, DELTA).await;

    let (started, release) = server.blockade.arm();
    append(&mut socket, 24000, 2400).await;
    timeout(WAIT, started).await.unwrap().unwrap();
    // 0.8 s of audio in 20 ms packets: more messages than the queue admits
    // per track, but well inside its two seconds of audio.
    for _ in 0..40 {
        append(&mut socket, 12000, 480).await;
    }
    assert_eq!(commit(&mut socket).await["item_id"], partial["item_id"]);
    drop(release);
    let completion = event(&mut socket, COMPLETED).await;
    assert_eq!(completion["item_id"], partial["item_id"]);
    assert_eq!(completion["transcript"], "positive");
    assert_eq!(completion["usage"]["seconds"], 33_600.0 / 24_000.0);
}

#[tokio::test]
async fn standard_append_over_the_queue_limit_is_rejected_without_failing_the_turn() {
    let server = TestServer::start(1, 1).await;
    let mut socket = server.socket(false).await;
    event(&mut socket, "session.created").await;
    send(&mut socket, settings("gpt-live-transcribe")).await;
    event(&mut socket, "session.updated").await;
    append(&mut socket, 12000, 48_001).await;
    let rejected = event(&mut socket, "error").await;
    assert_eq!(rejected["error"]["param"], "audio");

    append(&mut socket, 12000, 48_000).await;
    let partial = event(&mut socket, DELTA).await;
    assert_eq!(commit(&mut socket).await["item_id"], partial["item_id"]);
    let completion = event(&mut socket, COMPLETED).await;
    assert_eq!(completion["transcript"], "positive");
    assert_eq!(completion["usage"]["seconds"], 2.0);
}

#[tokio::test]
async fn standard_replies_to_a_pipelined_burst_reach_a_reading_client() {
    let server = TestServer::start(1, 1).await;
    let mut socket = server.socket(false).await;
    event(&mut socket, "session.created").await;
    let clear = json!({"type":"input_audio_buffer.clear"}).to_string();
    for _ in 0..500 {
        socket
            .feed(Message::Text(clear.clone().into()))
            .await
            .unwrap();
    }
    socket.flush().await.unwrap();
    for _ in 0..500 {
        event(&mut socket, "input_audio_buffer.cleared").await;
    }
}

#[tokio::test]
async fn standard_socket_pings_and_drops_a_peer_that_stops_answering() {
    let server = TestServer::start(1, 1).await;
    let app = server._app_tx.borrow().clone().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let route = axum::routing::get(move |ws: axum::extract::WebSocketUpgrade| async move {
        ws.on_upgrade(move |socket| {
            super::openai::run_with_liveness(
                socket,
                app,
                Duration::from_millis(100),
                Duration::from_secs(1),
            )
        })
    });
    let task = tokio::spawn(async move {
        serve(listener, axum::Router::new().route("/", route))
            .await
            .unwrap()
    });
    let (mut socket, _) = connect_async(format!("ws://{address}/")).await.unwrap();
    event(&mut socket, "session.created").await;

    // Reading answers each ping, so the session outlives the idle timeout.
    let mut pings = 0;
    let closed = timeout(Duration::from_millis(1_500), async {
        loop {
            match socket.next().await {
                Some(Ok(Message::Ping(_))) => pings += 1,
                other => break other,
            }
        }
    })
    .await;
    assert!(closed.is_err(), "{closed:?}");
    assert!(pings > 0);

    // A peer that stops reading sends no pongs and loses the session.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let end = timeout(WAIT, async {
        loop {
            match socket.next().await {
                Some(Ok(Message::Ping(_))) => {}
                other => break other,
            }
        }
    })
    .await
    .expect("the idle session stayed open");
    assert!(
        !matches!(end, Some(Ok(Message::Text(_) | Message::Binary(_)))),
        "{end:?}"
    );
    task.abort();
}

#[tokio::test]
async fn frame_errors_close_the_standard_socket_with_a_reason() {
    use tokio_tungstenite::MaybeTlsStream;

    let server = TestServer::start(3, 1).await;
    // Masked client frames, written below the client library so it cannot refuse
    // them. The oversized frame sends only its header, so nothing is left unread.
    let mask = [0x11, 0x22, 0x33, 0x44];
    let mut oversized = vec![0x81, 0x80 | 127];
    oversized.extend_from_slice(&(super::MAX_MESSAGE_BYTES as u64 + 1).to_be_bytes());
    oversized.extend_from_slice(&mask);
    let mut reserved_bits = vec![0xC1, 0x80];
    reserved_bits.extend_from_slice(&mask);
    let mut invalid_text = vec![0x81, 0x80 | 2];
    invalid_text.extend_from_slice(&mask);
    invalid_text.extend([0xFF ^ mask[0], 0xFE ^ mask[1]]);
    for (frame, close_code, code) in [
        (oversized, 1009, "message_too_large"),
        (reserved_bits, 1002, "protocol_error"),
        (invalid_text, 1007, "invalid_utf8"),
    ] {
        let mut socket = server.socket(false).await;
        event(&mut socket, "session.created").await;
        let MaybeTlsStream::Plain(stream) = socket.get_mut() else {
            panic!("expected a plain TCP stream");
        };
        stream.write_all(&frame).await.unwrap();
        let failure = event(&mut socket, "error").await;
        assert_eq!(failure["error"]["code"], code, "{failure}");
        let reply = timeout(WAIT, socket.next()).await.unwrap();
        let Some(Ok(Message::Close(Some(close)))) = reply else {
            panic!("expected a close frame, got {reply:?}");
        };
        assert_eq!(u16::from(close.code), close_code);
    }
}

#[tokio::test]
async fn native_oversized_frame_closes_with_a_reason_and_leaves_the_session_resumable() {
    use tokio_tungstenite::MaybeTlsStream;

    let server = TestServer::start(1, 1).await;
    // Only the header of a masked text frame one byte over the limit.
    let mut oversized = vec![0x81, 0x80 | 127];
    oversized.extend_from_slice(&(super::MAX_MESSAGE_BYTES as u64 + 1).to_be_bytes());
    oversized.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
    async fn write_raw(socket: &mut Socket, frame: &[u8]) {
        let MaybeTlsStream::Plain(stream) = socket.get_mut() else {
            panic!("expected a plain TCP stream");
        };
        stream.write_all(frame).await.unwrap();
    }
    async fn close_code(socket: &mut Socket) -> u16 {
        let reply = timeout(WAIT, socket.next()).await.unwrap();
        let Some(Ok(Message::Close(Some(close)))) = reply else {
            panic!("expected a close frame, got {reply:?}");
        };
        close.code.into()
    }

    // In a session, the close code carries the reason and the session survives.
    let mut socket = server.socket(true).await;
    let greeting = event(&mut socket, "transcriberrr.session.created").await;
    register(&mut socket, "mic").await;
    write_raw(&mut socket, &oversized).await;
    assert_eq!(close_code(&mut socket).await, 1009);

    // Before a session, the socket can also say why.
    let mut resumed = server
        .connect(
            "&transcriberrr=1&resume=1",
            Some(&format!("Bearer {TEST_KEY}")),
            None,
            "realtime",
        )
        .await
        .unwrap();
    write_raw(&mut resumed, &oversized).await;
    let failure = event(&mut resumed, "error").await;
    assert_eq!(failure["error"]["code"], "message_too_large", "{failure}");
    assert_eq!(close_code(&mut resumed).await, 1009);

    let mut resumed = server
        .connect(
            "&transcriberrr=1&resume=1",
            Some(&format!("Bearer {TEST_KEY}")),
            None,
            "realtime",
        )
        .await
        .unwrap();
    send(
        &mut resumed,
        json!({
            "type":"transcriberrr.session.resume", "session_id":greeting["session_id"],
            "resume_credential":greeting["resume_credential"], "last_sequence":0
        }),
    )
    .await;
    event(&mut resumed, "transcriberrr.session.resumed").await;
}

#[tokio::test]
async fn models_advertise_the_enforced_standard_limits() {
    let server = TestServer::start(1, 1).await;
    let limits = server.models().await["transcriberrr"]["limits"].clone();
    assert_eq!(limits["sample_rate"], 24_000);
    assert_eq!(limits["max_message_bytes"], 256 * 1024);
    assert_eq!(limits["max_append_frames"], 2 * 24_000);
    assert_eq!(limits["max_uncommitted_frames"], 60 * 24_000);
    assert_eq!(limits["ping_interval_ms"], 20_000);
    assert_eq!(limits["idle_timeout_ms"], 60_000);
    assert_eq!(
        super::engine_capabilities(EngineKind::Qwen3)["limits"],
        limits
    );
}

#[tokio::test]
async fn native_shutdown_ends_the_session_for_a_client_that_is_still_sending() {
    let server = TestServer::start(1, 1).await;
    let mut socket = server.socket(true).await;
    event(&mut socket, "transcriberrr.session.created").await;
    register(&mut socket, "mic").await;
    let (mut writer, mut reader) = socket.split();
    let sending = tokio::spawn(async move {
        // Audio keeps arriving after the session has ended and stopped reading.
        for packet in 0..5_000_u64 {
            let append = json!({
                "type":"transcriberrr.track.append", "track_id":"mic",
                "start_frame":packet * 480, "audio":pcm(0, 480)
            });
            if writer
                .send(Message::Text(append.to_string().into()))
                .await
                .is_err()
            {
                break;
            }
        }
    });
    server.shutdown.send_replace(true);

    let mut ended = false;
    let close = timeout(WAIT, async {
        loop {
            match reader.next().await {
                Some(Ok(Message::Text(text))) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    ended |= event["type"] == "transcriberrr.session.ended";
                }
                Some(Ok(Message::Close(frame))) => break frame,
                Some(Ok(_)) => {}
                other => panic!("connection ended without a close frame: {other:?}"),
            }
        }
    })
    .await
    .unwrap();
    assert!(ended);
    assert_eq!(u16::from(close.unwrap().code), 1000);
    sending.abort();
}

async fn native_liveness_route(
    server: &TestServer,
    ping_interval: Duration,
    idle_timeout: Duration,
) -> (String, JoinHandle<()>) {
    let app = server._app_tx.borrow().clone().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let route = axum::routing::get(move |ws: axum::extract::WebSocketUpgrade| async move {
        ws.on_upgrade(move |socket| {
            tracks::run_with_liveness(socket, app, false, ping_interval, idle_timeout)
        })
    });
    let task = tokio::spawn(async move {
        serve(listener, axum::Router::new().route("/", route))
            .await
            .unwrap()
    });
    (format!("ws://{address}/"), task)
}

#[tokio::test]
async fn native_socket_pings_and_detaches_a_peer_that_stops_answering() {
    let server = TestServer::start(1, 1).await;
    let (url, task) =
        native_liveness_route(&server, Duration::from_millis(100), Duration::from_secs(1)).await;
    let (mut socket, _) = connect_async(&url).await.unwrap();
    let greeting = event(&mut socket, "transcriberrr.session.created").await;
    let track = register(&mut socket, "mic").await;

    // Reading answers each ping, so the socket outlives the idle timeout.
    let mut pings = 0;
    let closed = timeout(Duration::from_millis(1_500), async {
        loop {
            match socket.next().await {
                Some(Ok(Message::Ping(_))) => pings += 1,
                other => break other,
            }
        }
    })
    .await;
    assert!(closed.is_err(), "{closed:?}");
    assert!(pings > 0);

    // A peer that stops reading sends no pongs and loses the socket.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let end = timeout(WAIT, async {
        loop {
            match socket.next().await {
                Some(Ok(Message::Ping(_))) => {}
                other => break other,
            }
        }
    })
    .await
    .expect("the idle socket stayed open");
    match &end {
        Some(Ok(Message::Close(Some(frame)))) => assert_eq!(u16::from(frame.code), 1001),
        Some(Ok(Message::Text(_) | Message::Binary(_))) => panic!("{end:?}"),
        _ => {}
    }

    // The session detached as for any other disconnect, so it can resume.
    let (mut next, _) = connect_async(&url).await.unwrap();
    resume(&mut next, &greeting, 0, false).await;
    let resumed = event(&mut next, "transcriberrr.session.resumed").await;
    assert_eq!(resumed["next_expected_frames"], json!({"mic": 0}));
    assert_eq!(receive(&mut next).await, track);
    task.abort();
}

#[tokio::test]
async fn native_idle_timeout_excludes_time_the_server_stops_reading() {
    let server = TestServer::with_engine(1, 1, EngineKind::Qwen3).await;
    let (url, task) =
        native_liveness_route(&server, Duration::from_millis(100), Duration::from_secs(2)).await;
    let (mut socket, _) = connect_async(&url).await.unwrap();
    event(&mut socket, "transcriberrr.session.created").await;
    register_role(&mut socket, "mic", "mixed_audio").await;
    let (started, release) = server.blockade.arm();
    append_track(&mut socket, "mic", 0, 24_000, 24_000).await;
    timeout(WAIT, started).await.unwrap().unwrap();

    // Fill the worker queue, the capture buffer, and the session's command
    // queue, so the server stops reading this socket and its pongs.
    const PACKET: u64 = 24_000;
    const PACKETS: u64 = 40;
    let (mut writer, mut reader) = socket.split();
    let sending = tokio::spawn(async move {
        for start in (1..PACKETS).map(|packet| packet * PACKET) {
            let append = json!({
                "type":"transcriberrr.track.append", "track_id":"mic",
                "start_frame":start, "audio":pcm(12_000, PACKET as usize)
            });
            writer
                .send(Message::Text(append.to_string().into()))
                .await
                .unwrap();
        }
        let end = json!({"type":"transcriberrr.session.end"});
        writer
            .send(Message::Text(end.to_string().into()))
            .await
            .unwrap();
        writer
    });
    let mut backlog = false;
    let held = timeout(Duration::from_secs(4), async {
        loop {
            match reader.next().await {
                Some(Ok(Message::Text(text))) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    assert_ne!(event["type"], "error", "{event}");
                    backlog |= event["type"] == "transcriberrr.track.backlog";
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                other => break other,
            }
        }
    })
    .await;
    assert!(held.is_err(), "paused socket was dropped: {held:?}");
    assert!(backlog);
    drop(release);

    // Forty seconds of queued audio take a while to drain on a loaded machine.
    let ended = timeout(4 * WAIT, async {
        loop {
            match reader.next().await {
                Some(Ok(Message::Text(text))) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    assert_ne!(event["type"], "error", "{event}");
                    if event["type"] == "transcriberrr.session.ended" {
                        break;
                    }
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                other => panic!("connection ended before the session: {other:?}"),
            }
        }
    })
    .await;
    assert!(ended.is_ok(), "the session did not end after reads resumed");
    let _writer = timeout(WAIT, sending).await.unwrap().unwrap();
    task.abort();
}
