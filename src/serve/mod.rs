use std::{
    fs, io,
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    extract::{Query, State, WebSocketUpgrade},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use clap::Args;
use ring::digest::{self, Digest};
use serde::Deserialize;
use serde_json::json;
use subtle::ConstantTimeEq;
use tokio::{
    net::TcpListener,
    signal::unix::{SignalKind, signal},
    sync::{Semaphore, oneshot, watch},
};

use crate::{DeviceKind, EngineKind, default_asr, engine_name, model_cache, write_json_line};

mod apple;
#[cfg(feature = "diarization")]
mod diarization;
mod live_bleed;
mod openai;
#[cfg(feature = "parakeet")]
mod parakeet_profile;
mod pipeline;
mod recognizer;
mod tracks;

const DEFAULT_MAX_SESSIONS: u32 = 2;
const DEFAULT_MAX_TRACKS: u32 = 2;
const MAX_MESSAGE_BYTES: usize = 256 * 1024;
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_ENGINE: EngineKind = if cfg!(feature = "parakeet") {
    EngineKind::ParakeetV3
} else {
    EngineKind::Qwen3
};

fn default_model(engine: EngineKind) -> &'static str {
    match engine {
        EngineKind::Qwen3 => model_cache::QWEN3_ASR_06B,
        other => default_asr(other),
    }
}

pub(crate) fn engine_capabilities(engine: EngineKind) -> serde_json::Value {
    let included = match engine {
        EngineKind::Qwen3 => true,
        EngineKind::AppleNative => apple::available(),
        EngineKind::ParakeetV3 => cfg!(feature = "parakeet"),
    };
    let caps = recognizer::capabilities(engine);
    let mut info = recognition_metadata(caps);
    info["default_model"] = json!(default_model(engine));
    info["included"] = json!(included);
    info["max_concurrent_streams"] = json!(if included {
        DEFAULT_MAX_SESSIONS * DEFAULT_MAX_TRACKS
    } else {
        0
    });
    info["aliases"] = json!(model_aliases(caps).filter(|_| included).collect::<Vec<_>>());
    info
}

fn recognition_metadata(caps: recognizer::Capabilities) -> serde_json::Value {
    json!({
        "stable_deltas": caps.stable_deltas,
        "revisable_drafts": caps.revisable_drafts,
        "word_timing_granularity": caps.word_timing_granularity,
        "draft_word_timing": caps.draft_word_timing,
        "languages": caps.languages,
        "prompt": caps.prompt,
    })
}

fn model_aliases(caps: recognizer::Capabilities) -> impl Iterator<Item = &'static str> {
    std::iter::once(openai::COMMITTED_ALIAS)
        .chain(caps.stable_deltas.then_some(openai::STREAMING_ALIAS))
}

#[derive(Clone, Debug, Args)]
pub(crate) struct ServeArgs {
    /// IP address to listen on. Non-loopback addresses require an API key.
    #[arg(long, default_value = "127.0.0.1")]
    bind: IpAddr,
    /// HTTP/WebSocket port. Pass 0 to have the OS choose an available port.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Backend to keep loaded for all sessions.
    #[arg(long, value_enum, default_value_t = DEFAULT_ENGINE)]
    engine: EngineKind,
    /// Hugging Face model ID or local model directory.
    #[arg(long)]
    model: Option<String>,
    /// Optional Qwen3 forced aligner for final word timings.
    #[arg(long)]
    forced_aligner: Option<String>,
    /// Device to use for inference.
    #[arg(long, value_enum, default_value_t = DeviceKind::Metal)]
    device: DeviceKind,
    /// Use only local models and installed Apple speech assets.
    #[arg(long)]
    offline: bool,
    /// Root directory for downloaded model snapshots.
    #[arg(long)]
    model_cache: Option<PathBuf>,
    /// Enable live mixed-track diarization using this Sortformer model path or cache ID.
    #[arg(long)]
    diarization_model: Option<PathBuf>,
    /// Maximum number of active and resumable sessions.
    #[arg(long, default_value_t = DEFAULT_MAX_SESSIONS, value_parser = clap::value_parser!(u32).range(1..=1024))]
    max_sessions: u32,
    /// Maximum number of tracks per extension session.
    #[arg(long, default_value_t = DEFAULT_MAX_TRACKS, value_parser = clap::value_parser!(u32).range(1..=64))]
    max_tracks: u32,
    /// Read the API key from this file instead of TRANSCRIBERRR_API_KEY.
    #[arg(long)]
    api_key_file: Option<PathBuf>,
    /// Allow WebSocket upgrades from this exact browser origin. Repeat for multiple origins.
    #[arg(long)]
    allow_origin: Vec<String>,
    /// Begin draining on EOF from stdin, for app-managed servers.
    #[arg(long)]
    exit_on_stdin_eof: bool,
}

struct AppState {
    model: String,
    engine: String,
    worker: pipeline::RecognitionWorker,
    capabilities: recognizer::Capabilities,
    shutdown: watch::Receiver<bool>,
    active_sessions: Arc<Semaphore>,
    max_sessions: usize,
    max_tracks: usize,
    native: tracks::Registry,
    #[cfg(feature = "diarization")]
    diarization: Option<Arc<diarization::Worker>>,
}

struct ListenerState {
    app: watch::Receiver<Option<Arc<AppState>>>,
    shutdown: watch::Receiver<bool>,
    allowed_origins: Vec<String>,
    key: Option<Digest>,
    sockets: Arc<Semaphore>,
    socket_limit: u32,
    max_sessions: usize,
}

pub(crate) fn run(args: ServeArgs) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("create server runtime")?;
    let mut address = SocketAddr::new(args.bind, args.port);
    let result = runtime.block_on(run_server(args, &mut address));
    runtime.shutdown_background();
    if let Err(error) = &result {
        write_json_line(&json!({
            "event": "error", "address": address.to_string(), "message": format!("{error:#}"),
        }))?;
    }
    result
}

async fn run_server(args: ServeArgs, address: &mut SocketAddr) -> Result<()> {
    if matches!(args.device, DeviceKind::Coreml)
        && (!matches!(args.engine, EngineKind::AppleNative) || args.diarization_model.is_some())
    {
        bail!(
            "CoreML/ANE is unavailable for the Rust recognition and diarization engines; use --device metal or --device cpu"
        );
    }
    if matches!(args.engine, EngineKind::AppleNative)
        && args.max_sessions as usize * args.max_tracks as usize > apple::MAX_STREAMS
    {
        bail!(
            "apple-native supports at most {} streams; reduce --max-sessions or --max-tracks",
            apple::MAX_STREAMS
        );
    }
    let model = match args.model {
        Some(_) if matches!(args.engine, EngineKind::AppleNative) => {
            bail!("apple-native uses the system model and takes no model override");
        }
        Some(ref model) if model.trim().is_empty() => bail!("model must not be empty"),
        Some(ref model) if Path::new(model).exists() => model.clone(),
        Some(ref model) => model_cache::canonical_id(model),
        None => default_model(args.engine).to_owned(),
    };
    let with_aligner = args.forced_aligner.is_some();
    if with_aligner && !matches!(args.engine, EngineKind::Qwen3) {
        bail!("--forced-aligner requires the qwen3 engine");
    }
    #[cfg(not(feature = "diarization"))]
    if args.diarization_model.is_some() {
        bail!("live diarization is not included in this build");
    }
    let key = load_key(args.api_key_file)?;
    if !args.bind.is_loopback() && key.is_none() {
        bail!("a non-loopback bind requires TRANSCRIBERRR_API_KEY or --api-key-file");
    }

    let mut interrupt = signal(SignalKind::interrupt()).context("install SIGINT handler")?;
    let mut terminate = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let stdin = stdin_eof(args.exit_on_stdin_eof)?;
    tokio::pin!(stdin);

    let listener = TcpListener::bind(*address)
        .await
        .with_context(|| format!("bind server to {address}"))?;
    *address = listener.local_addr().context("read listening address")?;
    announce("listening", *address)?;

    let (app_tx, app) = watch::channel(None);
    let (shutdown_tx, shutdown) = watch::channel(false);
    // Native clients may need an extra socket for a takeover while all session slots are in use.
    let socket_limit = args.max_sessions * 2 + 2;
    let state = Arc::new(ListenerState {
        app,
        shutdown: shutdown.clone(),
        allowed_origins: args.allow_origin,
        key,
        sockets: Arc::new(Semaphore::new(socket_limit as usize)),
        socket_limit,
        max_sessions: args.max_sessions as usize,
    });
    let mut listener_shutdown = shutdown.clone();
    let app = router(state.clone());
    let mut http = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = listener_shutdown.wait_for(|stopping| *stopping).await;
            })
            .await
    });

    let load = async {
        #[cfg(feature = "diarization")]
        let diarization = if let Some(path) = args.diarization_model {
            let path = crate::expand_home(&path);
            let cache = model_cache::ModelCache::new(args.model_cache.clone(), args.offline);
            let device = args.device;
            Some(
                tokio::task::spawn_blocking(move || {
                    let path = if path.is_file() {
                        path
                    } else {
                        cache.ensure(&path.to_string_lossy())?
                    };
                    diarization::Worker::start(path, device)
                })
                .await
                .context("join diarization startup")?
                .context("load live diarization model")?,
            )
        } else {
            None
        };

        let worker = pipeline::RecognitionWorker::spawn(
            args.engine,
            model.clone(),
            args.device,
            args.offline,
            args.model_cache,
            args.forced_aligner,
        )
        .await
        .context("load recognition model")?;
        Ok::<_, anyhow::Error>(AppState {
            model,
            engine: engine_name(args.engine).to_owned(),
            worker,
            capabilities: recognizer::capabilities_with_aligner(args.engine, with_aligner),
            shutdown,
            active_sessions: Arc::new(Semaphore::new(args.max_sessions as usize)),
            max_sessions: args.max_sessions as usize,
            max_tracks: args.max_tracks as usize,
            native: tracks::Registry::default(),
            #[cfg(feature = "diarization")]
            diarization,
        })
    };
    let loaded = tokio::select! {
        biased;
        _ = &mut stdin => None,
        _ = interrupt.recv() => None,
        _ = terminate.recv() => None,
        result = &mut http => return result.context("join HTTP server")?.context("run HTTP server"),
        result = load => Some(result?),
    };
    let mut worker_error = None;
    if let Some(app) = loaded {
        let worker = app.worker.clone();
        app_tx.send_replace(Some(Arc::new(app)));
        announce("ready", *address)?;

        tokio::select! {
            _ = &mut stdin => {}
            _ = interrupt.recv() => {}
            _ = terminate.recv() => {}
            reason = worker.closed() => { worker_error = Some(reason); }
            result = &mut http => return result.context("join HTTP server")?.context("run HTTP server"),
        }
    }

    shutdown_tx.send_replace(true);
    let announcement = announce("draining", *address);
    let drain = async {
        http.await
            .context("join HTTP server")?
            .context("run HTTP server")?;
        let _sockets = state.sockets.acquire_many(state.socket_limit).await?;
        Ok::<_, anyhow::Error>(())
    };
    tokio::select! {
        _ = interrupt.recv() => {}
        _ = terminate.recv() => {}
        result = tokio::time::timeout(DRAIN_TIMEOUT, drain) => {
            if let Ok(result) = result {
                result?;
            }
        }
    }
    announcement?;
    if let Some(reason) = worker_error {
        bail!("{reason}");
    }
    Ok(())
}

fn announce(event: &str, address: SocketAddr) -> Result<()> {
    write_json_line(&json!({"event": event, "address": address.to_string()}))
}

fn stdin_eof(enabled: bool) -> Result<impl Future<Output = ()>> {
    let receiver = if enabled {
        let (send, receive) = oneshot::channel();
        std::thread::Builder::new()
            .name("serve-stdin".into())
            .spawn(move || {
                let _ = io::copy(&mut io::stdin().lock(), &mut io::sink());
                let _ = send.send(());
            })
            .context("watch server stdin")?;
        Some(receive)
    } else {
        None
    };
    Ok(async move {
        if let Some(receiver) = receiver {
            let _ = receiver.await;
        } else {
            std::future::pending::<()>().await;
        }
    })
}

fn load_key(file: Option<PathBuf>) -> Result<Option<Digest>> {
    let value = match file {
        Some(file) => {
            let file = crate::expand_home(&file);
            Some(
                fs::read_to_string(&file)
                    .with_context(|| format!("read API key file {}", file.display()))?,
            )
        }
        None => match std::env::var("TRANSCRIBERRR_API_KEY") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => bail!("TRANSCRIBERRR_API_KEY must contain a UTF-8 value"),
        },
    };
    let Some(value) = value else { return Ok(None) };
    let token = value.trim_end_matches(['\r', '\n']);
    if token.is_empty() || !token.bytes().all(|byte| byte.is_ascii_graphic()) {
        bail!("API key must contain a nonempty printable ASCII token without whitespace");
    }
    Ok(Some(digest::digest(&digest::SHA256, token.as_bytes())))
}

fn router(state: Arc<ListenerState>) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/models", get(models))
        .route("/v1/realtime", get(upgrade))
        .with_state(state)
}

async fn health(State(state): State<Arc<ListenerState>>) -> Json<serde_json::Value> {
    let app = state.app.borrow();
    let stopping = *state.shutdown.borrow();
    let status = if stopping {
        "draining"
    } else if app.is_some() {
        "ready"
    } else {
        "loading"
    };
    Json(json!({
        "status": status,
        "ready": status == "ready",
        "active_sessions": app.as_ref().map_or(0, |app| {
            app.max_sessions - app.active_sessions.available_permits()
        }),
        "max_sessions": state.max_sessions,
        "backlog": app.as_ref().map_or(0, |app| app.worker.backlog()),
    }))
}

fn api_error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"error": {"code": code, "message": message}})),
    )
        .into_response()
}

fn unavailable() -> Response {
    api_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_unavailable",
        "Server is loading or draining",
    )
}

fn unauthorized() -> Response {
    let mut response = api_error(
        StatusCode::UNAUTHORIZED,
        "unauthorized",
        "A valid API key is required",
    );
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        header::HeaderValue::from_static("Bearer"),
    );
    response
}

fn token_matches(expected: &Digest, token: &str) -> bool {
    let candidate = digest::digest(&digest::SHA256, token.as_bytes());
    bool::from(expected.as_ref().ct_eq(candidate.as_ref()))
}

fn authorized(state: &ListenerState, headers: &HeaderMap, ws: Option<&WebSocketUpgrade>) -> bool {
    let Some(expected) = &state.key else {
        return true;
    };
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
        .is_some_and(|(_, token)| token_matches(expected, token.trim_start_matches(' ')));
    bearer
        || ws.is_some_and(|ws| {
            ws.selected_protocol().is_some()
                && ws
                    .requested_protocols()
                    .filter_map(|value| value.to_str().ok())
                    .filter_map(|value| value.strip_prefix("openai-insecure-api-key."))
                    .any(|token| token_matches(expected, token))
        })
}

async fn models(State(state): State<Arc<ListenerState>>, headers: HeaderMap) -> Response {
    if !authorized(&state, &headers, None) {
        return unauthorized();
    }
    let app = state.app.borrow();
    let Some(app) = app.as_ref().filter(|_| !*state.shutdown.borrow()) else {
        return unavailable();
    };
    let model_entry =
        |id: &str| json!({"id": id, "object": "model", "created": 0, "owned_by": "local"});
    let data: Vec<_> = std::iter::once(model_entry(&app.model))
        .chain(model_aliases(app.capabilities).map(model_entry))
        .collect();
    let mut metadata = recognition_metadata(app.capabilities);
    metadata["engine"] = json!(app.engine);
    metadata["max_concurrent_streams"] = json!(app.max_sessions * app.max_tracks);
    Json(json!({"object": "list", "data": data, "transcriberrr": metadata})).into_response()
}

#[derive(Deserialize)]
struct UpgradeQuery {
    intent: Option<String>,
    transcriberrr: Option<String>,
    resume: Option<String>,
}

async fn upgrade(
    State(state): State<Arc<ListenerState>>,
    Query(query): Query<UpgradeQuery>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let mut origins = headers.get_all(header::ORIGIN).iter();
    if let Some(origin) = origins.next()
        && (origins.next().is_some()
            || !origin.to_str().is_ok_and(|origin| {
                state
                    .allowed_origins
                    .iter()
                    .any(|allowed| allowed == origin)
            }))
    {
        return api_error(
            StatusCode::FORBIDDEN,
            "origin_not_allowed",
            "Origin is not allowed",
        );
    }
    let ws = ws.protocols(["realtime"]);
    if !authorized(&state, &headers, Some(&ws)) {
        return unauthorized();
    }
    if query.intent.as_deref() != Some("transcription") {
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_intent",
            "Expected intent=transcription",
        );
    }
    let native = match query.transcriberrr.as_deref() {
        None => false,
        Some("1") => true,
        Some(_) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "unsupported_version",
                "Expected transcriberrr=1",
            );
        }
    };
    let resume_only = match query.resume.as_deref() {
        None => false,
        Some("1") if native => true,
        Some(_) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "unsupported_resume",
                "Expected resume=1 with transcriberrr=1",
            );
        }
    };
    let app = state.app.borrow().clone();
    // Serialize admission with shutdown so the drain can account for every upgrade.
    let stopping = state.shutdown.borrow();
    let Some(app) = app.filter(|_| !*stopping) else {
        return unavailable();
    };
    let Ok(socket_slot) = state.sockets.clone().try_acquire_owned() else {
        return api_error(
            StatusCode::TOO_MANY_REQUESTS,
            "socket_limit",
            "Server connection limit reached",
        );
    };
    let session_slot = if native {
        None
    } else {
        let Ok(slot) = app.active_sessions.clone().try_acquire_owned() else {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "session_limit",
                "Server session limit reached",
            );
        };
        Some(slot)
    };

    let response = ws
        .max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |ws| async move {
            let _socket_slot = socket_slot;
            let _session_slot = session_slot;
            if native {
                tracks::run(ws, app, resume_only).await;
            } else {
                openai::run(ws, app).await;
            }
        });
    drop(stopping);
    response
}

#[cfg(test)]
#[path = "../../tests/realtime/ws.rs"]
mod ws_tests;
