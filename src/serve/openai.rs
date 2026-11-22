use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures_util::{SinkExt, StreamExt, stream::SplitSink};
use tokio::{
    sync::{mpsc, oneshot},
    time::{sleep, timeout},
};

use super::{
    AppState,
    pipeline::{
        PipelineEvent, PipelineUpdate, QueueError, RecognitionWorker, TrackHandle, TrackSettings,
    },
    recognizer::{Capabilities, RecognitionOptions},
};

use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};

pub(super) const STREAMING_ALIAS: &str = "gpt-live-transcribe";
pub(super) const COMMITTED_ALIAS: &str = "gpt-transcribe";

#[derive(Clone, Debug)]
struct Settings {
    model: String,
    prompt: Option<String>,
    languages: Vec<String>,
}

impl Settings {
    fn wire(&self, session_id: &str) -> Value {
        let mut transcription = json!({"model": self.model});
        if let Some(prompt) = &self.prompt {
            transcription["prompt"] = json!(prompt);
        }
        if !self.languages.is_empty() {
            transcription["languages"] = json!(self.languages);
        }
        json!({
            "type": "transcription",
            "object": "realtime.transcription_session",
            "id": session_id,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": null,
                    "transcription": transcription,
                },
            },
        })
    }

    fn updated(
        &self,
        session: &Value,
        actual_model: &str,
        stable: bool,
        supports_prompt: bool,
        supported_languages: &[&str],
    ) -> Result<Self, ClientError> {
        let mut updated = self.clone();
        fields(session, "session", &["type", "audio", "include"])?;
        if session["type"].as_str() != Some("transcription") {
            return Err(ClientError::invalid(
                "session.type",
                "Expected transcription",
            ));
        }
        if let Some(include) = session.get("include")
            && !include.as_array().is_some_and(Vec::is_empty)
        {
            return Err(ClientError::unsupported("session.include"));
        }
        let Some(audio) = session.get("audio") else {
            return Ok(updated);
        };
        fields(audio, "session.audio", &["input"])?;
        let Some(input) = audio.get("input") else {
            return Ok(updated);
        };
        fields(
            input,
            "session.audio.input",
            &["format", "turn_detection", "transcription"],
        )?;
        if !input["turn_detection"].is_null() {
            return Err(ClientError::unsupported(
                "session.audio.input.turn_detection",
            ));
        }
        if let Some(format) = input.get("format") {
            fields(format, "session.audio.input.format", &["type", "rate"])?;
            if format["type"].as_str() != Some("audio/pcm")
                || format["rate"].as_u64() != Some(24_000)
            {
                return Err(ClientError::invalid(
                    "session.audio.input.format",
                    "Expected 24 kHz mono signed little-endian PCM16",
                ));
            }
        }
        let Some(transcription) = input.get("transcription") else {
            return Ok(updated);
        };
        let path = "session.audio.input.transcription";
        fields(transcription, path, &["model", "prompt", "languages"])?;
        if let Some(model) = transcription.get("model") {
            let name = model.as_str().ok_or_else(|| {
                ClientError::invalid(format!("{path}.model"), "Expected a model name")
            })?;
            if name != actual_model
                && name != COMMITTED_ALIAS
                && (name != STREAMING_ALIAS || !stable)
            {
                return Err(ClientError::new(
                    "model_not_found",
                    format!("{path}.model"),
                    "The requested model is unavailable on this server",
                ));
            }
            updated.model = name.to_owned();
        }
        if let Some(prompt) = transcription.get("prompt") {
            updated.prompt = match prompt {
                Value::Null => None,
                Value::String(prompt) if supports_prompt => Some(prompt.clone()),
                Value::String(_) => return Err(ClientError::unsupported(format!("{path}.prompt"))),
                _ => {
                    return Err(ClientError::invalid(
                        format!("{path}.prompt"),
                        "Expected a string or null",
                    ));
                }
            };
        }
        if let Some(languages) = transcription.get("languages") {
            let values = languages.as_array().ok_or_else(|| {
                ClientError::invalid(
                    format!("{path}.languages"),
                    "Expected an array of language codes",
                )
            })?;
            updated.languages = values
                .iter()
                .map(|value| {
                    let code = value.as_str().ok_or_else(|| {
                        ClientError::invalid(
                            format!("{path}.languages"),
                            "Expected a language code",
                        )
                    })?;
                    if !supported_languages.contains(&code) {
                        return Err(ClientError::new(
                            "unsupported_language",
                            format!("{path}.languages"),
                            "The engine does not support a requested language",
                        ));
                    }
                    Ok(code.to_owned())
                })
                .collect::<Result<_, _>>()?;
        }
        Ok(updated)
    }
}

#[derive(Debug)]
struct ClientError {
    code: &'static str,
    param: String,
    message: &'static str,
}

impl ClientError {
    fn new(code: &'static str, param: impl Into<String>, message: &'static str) -> Self {
        Self {
            code,
            param: param.into(),
            message,
        }
    }

    fn invalid(param: impl Into<String>, message: &'static str) -> Self {
        Self::new("invalid_value", param, message)
    }

    fn unsupported(param: impl Into<String>) -> Self {
        Self::new(
            "unsupported_option",
            param,
            "The requested option is unsupported",
        )
    }

    fn wire(&self, client_event_id: Option<&str>) -> Value {
        json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": self.code,
                "message": self.message,
                "param": (!self.param.is_empty()).then_some(self.param.as_str()),
                "event_id": client_event_id,
            },
        })
    }
}

fn fields(value: &Value, path: &str, allowed: &[&str]) -> Result<(), ClientError> {
    let object = value
        .as_object()
        .ok_or_else(|| ClientError::invalid(path, "Expected an object"))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        let param = if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };
        return Err(ClientError::unsupported(param));
    }
    Ok(())
}

#[derive(Debug)]
pub(super) enum PcmError {
    Base64,
    PartialSample,
}

pub(super) fn decode_pcm(encoded: &str) -> Result<Vec<i16>, PcmError> {
    let bytes = STANDARD.decode(encoded).map_err(|_| PcmError::Base64)?;
    let (samples, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(PcmError::PartialSample);
    }
    Ok(samples.iter().copied().map(i16::from_le_bytes).collect())
}

fn audio(value: &Value) -> Result<Vec<i16>, ClientError> {
    let encoded = value
        .as_str()
        .ok_or_else(|| ClientError::invalid("audio", "Expected base64 PCM16 audio"))?;
    decode_pcm(encoded).map_err(|error| {
        let message = match error {
            PcmError::Base64 => "Invalid base64 audio",
            PcmError::PartialSample => "PCM16 audio must contain whole samples",
        };
        ClientError::invalid("audio", message)
    })
}

const MAX_TURN_FRAMES: u64 = 60 * 24_000;
const MAX_PENDING_TURNS: usize = 64;
const OUTBOUND_EVENTS: usize = 64;
pub(super) const SLOW_READER: u16 = 4008;

#[derive(Default)]
struct Turn {
    frames: u64,
    stable: String,
    failed: bool,
}

struct Session {
    id: u64,
    next_turn: u64,
    active: Option<(u64, Turn)>,
    pending: HashMap<u64, Turn>,
    previous: Option<String>,
    settings: Settings,
    capabilities: Capabilities,
    actual_model: String,
    worker: RecognitionWorker,
    track: TrackHandle,
}

impl Session {
    fn new(worker: RecognitionWorker, capabilities: Capabilities, model: &str) -> Self {
        let settings = Settings {
            model: model.to_owned(),
            prompt: None,
            languages: Vec::new(),
        };
        Self {
            id: next_id(),
            next_turn: 1,
            active: None,
            pending: HashMap::new(),
            previous: None,
            track: worker.track(track_settings(&settings, capabilities)),
            settings,
            capabilities,
            actual_model: model.to_owned(),
            worker,
        }
    }

    fn session_event(&self, event_type: &str) -> Value {
        json!({
            "type": event_type,
            "session": self.settings.wire(&format!("sess_{}", self.id)),
        })
    }

    fn handle_message(&mut self, text: &str) -> Vec<Value> {
        let value: Value = match serde_json::from_str(text) {
            Ok(value) => value,
            Err(_) => {
                return vec![ClientError::new("invalid_json", "", "Invalid JSON event").wire(None)];
            }
        };
        let event_id = value.get("event_id").and_then(Value::as_str);
        self.handle_event(&value)
            .unwrap_or_else(|error| vec![error.wire(event_id)])
    }

    fn handle_event(&mut self, event: &Value) -> Result<Vec<Value>, ClientError> {
        if let Some(id) = event.get("event_id")
            && !id.is_string()
        {
            return Err(ClientError::invalid(
                "event_id",
                "Expected an event ID string",
            ));
        }
        match event["type"].as_str() {
            Some("session.update") => {
                fields(event, "", &["type", "event_id", "session"])?;
                self.update_settings(&event["session"])
            }
            Some("input_audio_buffer.append") => {
                fields(event, "", &["type", "event_id", "audio"])?;
                if self.active.as_ref().is_some_and(|(_, turn)| turn.failed) {
                    return Ok(Vec::new());
                }
                let samples = audio(&event["audio"])?;
                Ok(self.append(samples).into_iter().collect())
            }
            Some("input_audio_buffer.clear") => {
                fields(event, "", &["type", "event_id"])?;
                self.clear();
                Ok(vec![json!({"type": "input_audio_buffer.cleared"})])
            }
            Some("input_audio_buffer.commit") => {
                fields(event, "", &["type", "event_id"])?;
                self.commit()
            }
            _ => Err(ClientError::invalid(
                "type",
                "Unsupported client event type",
            )),
        }
    }

    fn update_settings(&mut self, update: &Value) -> Result<Vec<Value>, ClientError> {
        let settings = self.settings.updated(
            update,
            &self.actual_model,
            self.capabilities.stable_deltas,
            self.capabilities.prompt,
            self.capabilities.languages,
        )?;
        if settings.model != self.settings.model
            || settings.prompt != self.settings.prompt
            || settings.languages != self.settings.languages
        {
            if self.active.is_some() || !self.pending.is_empty() {
                return Err(ClientError::new(
                    "session_busy",
                    "session.audio.input.transcription",
                    "Wait for all current turns to finish before changing the model or hints",
                ));
            }
            self.track = self
                .worker
                .track(track_settings(&settings, self.capabilities));
            self.settings = settings;
        }
        Ok(vec![self.session_event("session.updated")])
    }

    fn append(&mut self, samples: Vec<i16>) -> Option<Value> {
        if samples.is_empty() {
            return None;
        }
        let (id, turn) = self.active.get_or_insert_with(|| {
            let id = self.next_turn;
            self.next_turn += 1;
            (id, Turn::default())
        });
        let id = *id;
        let start = turn.frames;
        turn.frames += samples.len() as u64;
        let failure = if turn.frames > MAX_TURN_FRAMES {
            Some((
                "input_audio_buffer_limit",
                "Uncommitted audio exceeded the 60-second limit",
            ))
        } else {
            self.track.append(start, samples, id).err().map(queue_error)
        };
        failure.map(|(code, message)| {
            turn.failed = true;
            let _ = self.track.clear(id);
            failed(self.id, id, code, message)
        })
    }

    fn clear(&mut self) {
        if let Some((id, turn)) = self.active.take()
            && !turn.failed
        {
            let _ = self.track.clear(id);
        }
    }

    fn commit(&mut self) -> Result<Vec<Value>, ClientError> {
        let (id, turn) = self.active.take().ok_or_else(|| {
            ClientError::new(
                "input_audio_buffer_commit_empty",
                "audio",
                "Cannot commit an empty audio buffer",
            )
        })?;
        let item = item_id(self.id, id);
        let previous = self.previous.replace(item.clone());
        let mut events = vec![
            json!({"type": "input_audio_buffer.committed", "item_id": item, "previous_item_id": previous}),
            json!({
                "type": "conversation.item.added",
                "previous_item_id": previous,
                "item": {
                    "id": item,
                    "type": "message",
                    "object": "realtime.item",
                    "status": "completed",
                    "role": "user",
                    "content": [{"type": "input_audio", "transcript": null}],
                },
            }),
        ];
        if !turn.failed {
            let committed = if self.pending.len() >= MAX_PENDING_TURNS {
                Err(QueueError::Full)
            } else {
                self.track.commit(id)
            };
            match committed {
                Ok(()) => {
                    self.pending.insert(id, turn);
                }
                Err(error) => {
                    let (code, message) = queue_error(error);
                    let _ = self.track.clear(id);
                    events.push(failed(self.id, id, code, message));
                }
            }
        }
        Ok(events)
    }

    fn recognition(&mut self, event: PipelineEvent) -> Option<Value> {
        let id = event.turn;
        let item = item_id(self.id, id);
        let active = self
            .active
            .as_mut()
            .and_then(|(active, turn)| (*active == id && !turn.failed).then_some(turn));
        match event.update {
            PipelineUpdate::Draft(hypothesis) => {
                if !hypothesis.stable || !streaming(&self.settings, self.capabilities) {
                    return None;
                }
                let turn = active.or_else(|| self.pending.get_mut(&id))?;
                let delta = hypothesis.text.strip_prefix(&turn.stable)?;
                if delta.is_empty() {
                    return None;
                }
                let event = json!({
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": item,
                    "content_index": 0,
                    "delta": delta,
                });
                turn.stable = hypothesis.text;
                Some(event)
            }
            PipelineUpdate::Final(hypothesis) => {
                let turn = self.pending.remove(&id)?;
                let mut event = json!({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item,
                    "content_index": 0,
                    "transcript": hypothesis.text,
                    "usage": {"type": "duration", "seconds": turn.frames as f64 / 24_000.0},
                });
                if self.settings.model == COMMITTED_ALIAS {
                    event["languages"] = match hypothesis.language {
                        Some(code) => json!([{"code": code}]),
                        None => json!([]),
                    };
                }
                Some(event)
            }
            PipelineUpdate::Failed(reason) => {
                if let Some(turn) = active {
                    turn.failed = true;
                    let _ = self.track.clear(id);
                } else {
                    self.pending.remove(&id)?;
                }
                Some(failed(self.id, id, "recognition_failed", &reason))
            }
            _ => None,
        }
    }

    fn shutdown(&mut self) -> Option<Value> {
        let (turn, active) = self.active.as_ref()?;
        let event = (!active.failed).then(|| {
            failed(
                self.id,
                *turn,
                "server_shutdown",
                "The server shut down before the item was committed",
            )
        });
        self.clear();
        event
    }
}

fn streaming(settings: &Settings, capabilities: Capabilities) -> bool {
    capabilities.stable_deltas && settings.model != COMMITTED_ALIAS
}

fn track_settings(settings: &Settings, capabilities: Capabilities) -> TrackSettings {
    TrackSettings {
        native: false,
        streaming: streaming(settings, capabilities),
        options: RecognitionOptions {
            prompt: settings.prompt.clone(),
            languages: settings.languages.clone(),
        },
    }
}

fn next_id() -> u64 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn item_id(session: u64, turn: u64) -> String {
    format!("item_{session}_{turn}")
}

fn failed(session: u64, turn: u64, code: &str, message: &str) -> Value {
    json!({
        "type": "conversation.item.input_audio_transcription.failed",
        "item_id": item_id(session, turn),
        "content_index": 0,
        "error": {
            "type": "transcription_error",
            "code": code,
            "message": message,
            "param": null,
        },
    })
}

fn queue_error(error: QueueError) -> (&'static str, &'static str) {
    match error {
        QueueError::Full => ("overload", "The recognition queue reached its hard limit"),
        QueueError::Closed => (
            "recognition_unavailable",
            "The recognition worker is unavailable",
        ),
    }
}

pub(super) async fn run(socket: WebSocket, state: Arc<AppState>) {
    let (sink, mut incoming) = socket.split();
    let (outbound, events) = mpsc::channel(OUTBOUND_EVENTS);
    let (close, closing) = oneshot::channel();
    let writer = tokio::spawn(write_events(sink, events, closing));
    let mut session = Session::new(state.worker.clone(), state.capabilities, &state.model);
    let mut shutdown = state.shutdown.clone();
    let drain_deadline = sleep(Duration::from_secs(8));
    tokio::pin!(drain_deadline);

    let mut output = vec![session.session_event("session.created")];
    let mut draining = false;
    loop {
        if !send_events(&outbound, &mut output, draining).await {
            let _ = close.send(WriterEnd::SlowReader);
            break;
        }
        if draining && session.pending.is_empty() {
            break;
        }
        tokio::select! {
            _ = outbound.closed() => break,
            _ = &mut drain_deadline, if draining => {
                let unfinished = std::mem::take(&mut session.pending);
                output.extend(unfinished.keys().map(|id| {
                    failed(session.id, *id, "server_shutdown", "The shutdown deadline elapsed")
                }));
            }
            _ = shutdown.changed(), if !draining => {
                draining = true;
                drain_deadline.as_mut().reset(tokio::time::Instant::now() + Duration::from_secs(8));
                output.extend(session.shutdown());
            }
            update = session.track.recv() => {
                match update {
                    Some(event) => output.extend(session.recognition(event)),
                    None => {
                        let unfinished = std::mem::take(&mut session.pending);
                        output.extend(unfinished.keys().map(|id| {
                            failed(session.id, *id, "recognition_unavailable", "The recognition worker is unavailable")
                        }));
                        if let Some((id, turn)) = &mut session.active
                            && !turn.failed
                        {
                            turn.failed = true;
                            output.push(failed(session.id, *id, "recognition_unavailable", "The recognition worker is unavailable"));
                        }
                        draining = true;
                    }
                }
            }
            message = incoming.next() => {
                match message {
                    Some(Ok(Message::Text(text))) if !draining => output = session.handle_message(&text),
                    Some(Ok(Message::Binary(_))) if !draining => {
                        output.push(ClientError::invalid("type", "Expected a JSON text message").wire(None));
                    }
                    Some(Ok(Message::Close(_))) => {
                        let _ = close.send(WriterEnd::PeerClose);
                        break;
                    }
                    Some(Err(_)) | None => break,
                    Some(Ok(_)) => {}
                }
            }
        }
    }
    drop(session);
    drop(outbound);
    let _ = writer.await;
}

async fn send_events(
    outbound: &mpsc::Sender<Message>,
    events: &mut Vec<Value>,
    draining: bool,
) -> bool {
    for mut event in events.drain(..) {
        event["event_id"] = json!(format!("event_{}", next_id()));
        let message = Message::Text(event.to_string().into());
        // Shutdown or worker loss can fail every outstanding turn in one batch.
        // Give the bounded writer time to deliver those terminal events.
        let sent = if draining {
            outbound.send(message).await.is_ok()
        } else {
            outbound.try_send(message).is_ok()
        };
        if !sent {
            return false;
        }
    }
    true
}

enum WriterEnd {
    Drained,
    PeerClose,
    SlowReader,
    Disconnected,
}

async fn write_events(
    mut sink: SplitSink<WebSocket, Message>,
    mut events: mpsc::Receiver<Message>,
    closing: oneshot::Receiver<WriterEnd>,
) {
    let delivery = async {
        while let Some(event) = events.recv().await {
            match timeout(Duration::from_secs(2), sink.send(event)).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => return WriterEnd::Disconnected,
                Err(_) => return WriterEnd::SlowReader,
            }
        }
        WriterEnd::Drained
    };
    let end = tokio::select! {
        biased;
        reason = closing => reason.unwrap_or(WriterEnd::Disconnected),
        end = delivery => end,
    };
    let _ = timeout(Duration::from_millis(250), async {
        let (code, reason) = match end {
            WriterEnd::PeerClose | WriterEnd::Disconnected => {
                // The WebSocket queues the peer's echo. A cancelled send may
                // still occupy SplitSink's slot and fail before that flush.
                if sink.flush().await.is_err() {
                    let _ = sink.close().await;
                }
                return;
            }
            WriterEnd::SlowReader => (SLOW_READER, "slow reader"),
            WriterEnd::Drained => (1001, "connection closing"),
        };
        let _ = sink
            .send(Message::Close(Some(CloseFrame {
                code,
                reason: reason.into(),
            })))
            .await;
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::watch;

    #[tokio::test]
    async fn draining_delivers_every_terminal_event_through_a_full_queue() {
        let (outbound, mut incoming) = mpsc::channel(1);
        outbound.try_send(Message::Text("queued".into())).unwrap();
        let mut failures = (1..=MAX_PENDING_TURNS as u64 + 1)
            .map(|turn| failed(1, turn, "server_shutdown", "Stopping"))
            .collect();
        let sending = async {
            let delivered = send_events(&outbound, &mut failures, true).await;
            drop(outbound);
            delivered
        };
        let reading = async {
            incoming.recv().await.unwrap();
            let mut ids = Vec::new();
            while let Some(Message::Text(text)) = incoming.recv().await {
                let event: Value = serde_json::from_str(&text).unwrap();
                ids.push(event["item_id"].as_str().unwrap().to_owned());
            }
            ids
        };
        let (delivered, ids) = tokio::join!(sending, reading);
        assert!(delivered);
        let expected: Vec<_> = (1..=MAX_PENDING_TURNS as u64 + 1)
            .map(|turn| item_id(1, turn))
            .collect();
        assert_eq!(ids, expected);
    }

    fn settings() -> Settings {
        Settings {
            model: "local-model".into(),
            prompt: None,
            languages: Vec::new(),
        }
    }

    fn update(transcription: Value) -> Value {
        json!({
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": null,
                "transcription": transcription,
            }},
        })
    }

    #[test]
    fn accepts_aliases_and_echoes_the_chosen_one() {
        let settings = settings();
        for model in [COMMITTED_ALIAS, STREAMING_ALIAS, "local-model"] {
            let options = settings
                .updated(
                    &update(json!({"model": model})),
                    "local-model",
                    true,
                    false,
                    &["en"],
                )
                .unwrap();
            assert_eq!(
                options.wire("sess_1")["audio"]["input"]["transcription"]["model"],
                model
            );
        }

        let error = settings
            .updated(
                &update(json!({"model": STREAMING_ALIAS})),
                "local-model",
                false,
                false,
                &["en"],
            )
            .unwrap_err();
        assert_eq!(error.code, "model_not_found");
    }

    #[test]
    fn validates_audio_format_and_requires_client_commits() {
        let mut event = update(json!({}));
        for detector in ["server_vad", "semantic_vad"] {
            event["audio"]["input"]["turn_detection"] = json!({"type": detector});
            let error = settings()
                .updated(&event, "local-model", true, false, &["en"])
                .unwrap_err();
            assert_eq!(error.param, "session.audio.input.turn_detection");
        }
        event["audio"]["input"]["turn_detection"] = Value::Null;
        event["audio"]["input"]["format"]["rate"] = json!(16_000);
        let error = settings()
            .updated(&event, "local-model", true, false, &["en"])
            .unwrap_err();
        assert_eq!(error.param, "session.audio.input.format");
    }

    #[test]
    fn validates_hints_against_capabilities() {
        let multilingual = update(json!({"prompt": "Local vocabulary", "languages": ["en", "fr"]}));
        let allowed = settings()
            .updated(&multilingual, "local-model", false, true, &["en", "fr"])
            .unwrap();
        assert_eq!(allowed.prompt.as_deref(), Some("Local vocabulary"));
        assert_eq!(allowed.languages, ["en", "fr"]);

        let error = settings()
            .updated(&multilingual, "local-model", false, false, &["en"])
            .unwrap_err();
        assert_eq!(error.param, "session.audio.input.transcription.prompt");
        let error = settings()
            .updated(&multilingual, "local-model", false, true, &["en"])
            .unwrap_err();
        assert_eq!(error.code, "unsupported_language");

        for unsupported in ["keywords", "delay"] {
            let mut request = json!({});
            request[unsupported] = json!("example");
            let error = settings()
                .updated(&update(request), "local-model", false, true, &["en"])
                .unwrap_err();
            assert_eq!(
                error.param,
                format!("session.audio.input.transcription.{unsupported}")
            );
        }
    }

    #[test]
    fn decodes_signed_little_endian_pcm_and_rejects_partial_samples() {
        assert_eq!(audio(&json!("AID/fw==")).unwrap(), vec![i16::MIN, i16::MAX]);
        assert!(audio(&json!("AA==")).is_err());
        assert!(audio(&json!("#")).is_err());
    }
    struct UnusedRecognizer;

    impl super::super::recognizer::Recognizer for UnusedRecognizer {
        fn start(
            &self,
            _options: &RecognitionOptions,
        ) -> anyhow::Result<Box<dyn super::super::recognizer::RecognitionStream + '_>> {
            unreachable!("The protocol test supplies results directly")
        }
    }

    #[tokio::test]
    async fn echoes_a_client_close_frame() {
        use axum::{Router, extract::WebSocketUpgrade, routing::get};
        use tokio::{net::TcpListener, sync::Semaphore};
        use tokio_tungstenite::{connect_async, tungstenite::Message as ClientMessage};

        let worker = RecognitionWorker::start(|| Ok(Box::new(UnusedRecognizer)))
            .await
            .unwrap();
        let (_shutdown, shutdown) = watch::channel(false);
        let state = Arc::new(AppState {
            model: "local-model".into(),
            engine: "test-engine".into(),
            worker,
            capabilities: super::super::recognizer::capabilities(crate::EngineKind::ParakeetV3),
            shutdown,
            active_sessions: Arc::new(Semaphore::new(1)),
            max_sessions: 1,
            max_tracks: 1,
            native: Default::default(),
            #[cfg(feature = "diarization")]
            diarization: None,
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = Router::new().route(
            "/",
            get(move |ws: WebSocketUpgrade| async move {
                ws.on_upgrade(move |socket| run(socket, state))
            }),
        );
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let (mut client, _) = connect_async(format!("ws://{addr}/")).await.unwrap();
        let greeting = timeout(Duration::from_secs(5), client.next())
            .await
            .unwrap();
        assert!(matches!(greeting, Some(Ok(ClientMessage::Text(_)))));

        let frame = tokio_tungstenite::tungstenite::protocol::CloseFrame {
            code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Normal,
            reason: "finished".into(),
        };
        client.close(Some(frame.clone())).await.unwrap();
        let reply = timeout(Duration::from_secs(5), client.next())
            .await
            .unwrap();
        task.abort();
        assert!(matches!(reply, Some(Ok(ClientMessage::Close(Some(echo)))) if echo == frame));
    }

    fn hypothesis(text: &str, stable: bool) -> super::super::recognizer::Hypothesis {
        super::super::recognizer::Hypothesis {
            text: text.to_owned(),
            stable,
            ..Default::default()
        }
    }

    fn recognition(turn: u64, update: PipelineUpdate) -> PipelineEvent {
        PipelineEvent {
            turn,
            start: 0,
            end: 0,
            update,
        }
    }

    #[tokio::test]
    async fn preserves_a_delta_item_id_when_finals_arrive_out_of_order() {
        let worker = RecognitionWorker::start(|| Ok(Box::new(UnusedRecognizer)))
            .await
            .unwrap();
        let capabilities = super::super::recognizer::capabilities(crate::EngineKind::ParakeetV3);
        let mut session = Session::new(worker, capabilities, "local-model");
        session.active = Some((1, Turn::default()));
        let delta = session
            .recognition(recognition(
                1,
                PipelineUpdate::Draft(hypothesis("Hello", true)),
            ))
            .unwrap();
        assert_eq!(delta["delta"], "Hello");
        assert!(
            session
                .recognition(recognition(
                    1,
                    PipelineUpdate::Draft(hypothesis("Hey", false))
                ))
                .is_none()
        );
        let delta2 = session
            .recognition(recognition(
                1,
                PipelineUpdate::Draft(hypothesis("Hello, world", true)),
            ))
            .unwrap();
        assert_eq!(delta2["delta"], ", world");
        assert_eq!(delta2["item_id"], delta["item_id"]);

        let (_, first) = session.active.take().unwrap();
        session.pending.insert(1, first);
        session.pending.insert(2, Turn::default());
        let second_final = session
            .recognition(recognition(
                2,
                PipelineUpdate::Final(hypothesis("Second turn", true)),
            ))
            .unwrap();
        assert_eq!(second_final["item_id"], item_id(session.id, 2));
        let first_final = session
            .recognition(recognition(
                1,
                PipelineUpdate::Final(hypothesis("Hello, world!", true)),
            ))
            .unwrap();
        assert_eq!(first_final["item_id"], delta["item_id"]);
        assert_eq!(first_final["transcript"], "Hello, world!");
        assert!(
            session
                .recognition(recognition(
                    1,
                    PipelineUpdate::Final(hypothesis("Duplicate", true))
                ))
                .is_none()
        );
    }
    #[tokio::test]
    async fn committed_alias_reports_only_detected_languages() {
        let worker = RecognitionWorker::start(|| Ok(Box::new(UnusedRecognizer)))
            .await
            .unwrap();
        let capabilities = super::super::recognizer::capabilities(crate::EngineKind::Qwen3);
        let mut session = Session::new(worker, capabilities, "local-model");
        session
            .update_settings(&update(
                json!({"model": COMMITTED_ALIAS, "languages": ["fr"]}),
            ))
            .unwrap();

        session.pending.insert(1, Turn::default());
        let detected = PipelineUpdate::Final(super::super::recognizer::Hypothesis {
            language: Some("en".to_owned()),
            ..hypothesis("Hello", true)
        });
        let final_event = session.recognition(recognition(1, detected)).unwrap();
        assert_eq!(final_event["languages"], json!([{"code": "en"}]));

        session.pending.insert(2, Turn::default());
        let undetected = PipelineUpdate::Final(hypothesis("", true));
        let final_event = session.recognition(recognition(2, undetected)).unwrap();
        assert_eq!(final_event["languages"], json!([]));
    }
}
