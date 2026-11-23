use std::{
    fs, io,
    net::{IpAddr, Ipv6Addr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use axum::{
    Json, Router,
    extract::{Query, State, WebSocketUpgrade},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use clap::Args;
use hyper_util::{
    rt::{TokioIo, TokioTimer},
    service::TowerToHyperService,
};
use ring::digest::{self, Digest};
use serde::Deserialize;
use serde_json::json;
use subtle::ConstantTimeEq;
use tokio::{
    net::{TcpListener, TcpStream},
    signal::unix::{SignalKind, signal},
    sync::{OwnedSemaphorePermit, Semaphore, oneshot, watch},
};

use crate::{DeviceKind, EngineKind, default_asr, engine_name, model_cache, write_json_line};

mod apple;
#[cfg(feature = "diarization")]
mod diarization;
mod live_bleed;
mod openai;
#[cfg(feature = "parakeet")]
mod parakeet_profile;
#[cfg(feature = "parakeet")]
mod parakeet_stream;
mod pipeline;
mod recognizer;
mod trace;
mod tracks;

const DEFAULT_MAX_SESSIONS: u32 = 2;
const DEFAULT_MAX_TRACKS: u32 = 2;
const MAX_MESSAGE_BYTES: usize = 256 * 1024;
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
// A connection must send each request head within this time, including the first
// one after it connects or goes idle. It stays below the drain deadline.
const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(5);
// Connections that have not upgraded to a WebSocket. Upgraded sockets are counted
// by `ListenerState::sockets`. Further clients wait in the accept backlog.
const MAX_PENDING_CONNECTIONS: usize = 64;
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
    info["limits"] = openai::limits();
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
    let allowed_origins = args
        .allow_origin
        .iter()
        .map(|origin| browser_origin(origin))
        .collect::<Result<Vec<_>>>()?;

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
        allowed_origins,
        key,
        sockets: Arc::new(Semaphore::new(socket_limit as usize)),
        socket_limit,
        max_sessions: args.max_sessions as usize,
    });
    let app = router(state.clone());
    let mut http = tokio::spawn(serve_http(
        listener,
        app,
        shutdown.clone(),
        HttpLimits {
            pending_connections: MAX_PENDING_CONNECTIONS,
            header_read_timeout: HEADER_READ_TIMEOUT,
        },
    ));

    let load = async {
        // Each model loads on its own thread and owns its device context.
        #[cfg(feature = "diarization")]
        let diarization = async {
            let Some(path) = args.diarization_model else {
                return Ok(None);
            };
            let path = crate::expand_home(&path);
            let cache = model_cache::ModelCache::new(args.model_cache.clone(), args.offline);
            let device = args.device;
            let worker = tokio::task::spawn_blocking(move || {
                let path = if path.is_file() {
                    path
                } else {
                    cache.ensure(&path.to_string_lossy())?
                };
                diarization::Worker::start(path, device)
            })
            .await
            .context("join diarization startup")?
            .context("load live diarization model")?;
            Ok(Some(worker))
        };
        #[cfg(not(feature = "diarization"))]
        let diarization = async { Ok(()) };
        let recognition = async {
            pipeline::RecognitionWorker::spawn(
                args.engine,
                model.clone(),
                args.device,
                args.offline,
                args.model_cache.clone(),
                args.forced_aligner.clone(),
            )
            .await
            .context("load recognition model")
        };
        #[cfg_attr(not(feature = "diarization"), allow(unused_variables))]
        let (diarization, worker) = load_concurrently(diarization, recognition).await?;
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

/// Starts both loads at once and returns the first failure with its own context.
async fn load_concurrently<A, B>(
    first: impl Future<Output = Result<A>>,
    second: impl Future<Output = Result<B>>,
) -> Result<(A, B)> {
    tokio::try_join!(first, second)
}

#[derive(Clone, Copy)]
struct HttpLimits {
    pending_connections: usize,
    header_read_timeout: Duration,
}

/// Serves HTTP/1.1 until `shutdown` is set, then lets in-flight requests and
/// upgrades finish. `axum::serve` sets no header-read timeout, so a client that
/// connected and never finished a request would hold its descriptor forever.
async fn serve_http(
    listener: TcpListener,
    app: Router,
    mut shutdown: watch::Receiver<bool>,
    limits: HttpLimits,
) -> io::Result<()> {
    let pending = Arc::new(Semaphore::new(limits.pending_connections));
    // Every connection task holds a receiver. The sender closes when all have ended.
    let (close_tx, close_rx) = watch::channel(());
    loop {
        let (stream, slot) = tokio::select! {
            _ = shutdown.wait_for(|stopping| *stopping) => break,
            admitted = admit(&listener, &pending) => admitted,
        };
        // Events are small writes that the client does not answer. Nagle's algorithm
        // would hold each one for the acknowledgement of the last.
        let _ = stream.set_nodelay(true);
        let service = TowerToHyperService::new(app.clone());
        let mut shutdown = shutdown.clone();
        let close_rx = close_rx.clone();
        tokio::spawn(async move {
            let _close = close_rx;
            // The connection future ends at an upgrade, which releases the slot.
            let _slot = slot;
            let mut builder = hyper::server::conn::http1::Builder::new();
            builder
                .timer(TokioTimer::new())
                .header_read_timeout(limits.header_read_timeout);
            let connection = builder
                .serve_connection(TokioIo::new(stream), service)
                .with_upgrades();
            tokio::pin!(connection);
            let mut draining = false;
            loop {
                tokio::select! {
                    _ = connection.as_mut() => break,
                    _ = shutdown.wait_for(|stopping| *stopping), if !draining => {
                        draining = true;
                        connection.as_mut().graceful_shutdown();
                    }
                }
            }
        });
    }
    drop(close_rx);
    drop(listener);
    close_tx.closed().await;
    Ok(())
}

async fn admit(
    listener: &TcpListener,
    pending: &Arc<Semaphore>,
) -> (TcpStream, OwnedSemaphorePermit) {
    let slot = pending
        .clone()
        .acquire_owned()
        .await
        .expect("the connection semaphore is never closed");
    loop {
        match listener.accept().await {
            Ok((stream, _)) => return (stream, slot),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::ConnectionRefused
                        | io::ErrorKind::ConnectionAborted
                        | io::ErrorKind::ConnectionReset
                ) => {}
            // For example, the process ran out of file descriptors.
            Err(_) => tokio::time::sleep(Duration::from_secs(1)).await,
        }
    }
}

/// Normalizes an `--allow-origin` value to the form a browser sends in `Origin`:
/// `scheme://host[:port]`, without a default port, path, or trailing slash.
fn browser_origin(value: &str) -> Result<String> {
    let invalid = |problem: &str| {
        anyhow!(
            "--allow-origin {value:?} {problem}. Pass the exact origin a browser sends, such as https://example.com or http://localhost:3000"
        )
    };
    // Sandboxed frames and local files send this literal value.
    if value == "null" {
        return Ok(value.to_owned());
    }
    if value == "*" {
        return Err(invalid(
            "is not supported because origins are matched exactly",
        ));
    }
    if value.is_empty() || value.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(invalid("is empty or contains whitespace"));
    }
    if !value.is_ascii() {
        return Err(invalid(
            "contains non-ASCII text; use the punycode host name",
        ));
    }
    let Some((scheme, rest)) = value.split_once("://") else {
        return Err(invalid("has no scheme"));
    };
    let scheme = scheme.to_ascii_lowercase();
    if !scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        || !scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    {
        return Err(invalid("has an invalid scheme"));
    }
    let authority = rest.strip_suffix('/').unwrap_or(rest);
    if authority.contains(['/', '?', '#']) {
        return Err(invalid("has a path, query, or fragment"));
    }
    if authority.contains('@') {
        return Err(invalid("has user information"));
    }
    let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
        let Some((address, port)) = bracketed.split_once(']') else {
            return Err(invalid("has an unclosed IPv6 address"));
        };
        let address: Ipv6Addr = address
            .parse()
            .map_err(|_| invalid("has an invalid IPv6 address"))?;
        let port = match port {
            "" => None,
            port => Some(
                port.strip_prefix(':')
                    .ok_or_else(|| invalid("has text after its IPv6 address"))?,
            ),
        };
        (format!("[{address}]"), port)
    } else {
        let (host, port) = match authority.split_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        };
        if host.is_empty()
            || !host
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'))
        {
            return Err(invalid("has an invalid host"));
        }
        // Browsers lowercase hosts only for web schemes. Extension IDs keep their case.
        let special = matches!(scheme.as_str(), "http" | "https" | "ws" | "wss");
        let host = if special {
            host.to_ascii_lowercase()
        } else {
            host.to_owned()
        };
        (host, port)
    };
    let port = match port {
        None => String::new(),
        Some(port) => {
            let number = port
                .parse::<u16>()
                .ok()
                .filter(|number| *number > 0 && port.bytes().all(|b| b.is_ascii_digit()))
                .ok_or_else(|| invalid("has an invalid port"))?;
            let default = match scheme.as_str() {
                "http" | "ws" => Some(80),
                "https" | "wss" => Some(443),
                _ => None,
            };
            if default == Some(number) {
                String::new()
            } else {
                format!(":{number}")
            }
        }
    };
    Ok(format!("{scheme}://{host}{port}"))
}

/// The close for an incoming frame the WebSocket layer refused.
pub(super) struct FrameFailure {
    pub(super) close_code: u16,
    pub(super) reason: &'static str,
    pub(super) code: &'static str,
    pub(super) message: &'static str,
}

/// Classifies a WebSocket read error. Returns `None` when the transport failed,
/// because the peer can no longer be told. The socket stays writable after the
/// other errors, but not readable.
pub(super) fn frame_failure(error: &axum::Error) -> Option<FrameFailure> {
    // axum does not expose tungstenite's error type, so match its messages.
    // `frame_errors_close_the_standard_socket_with_a_reason` pins them.
    let text = error.to_string();
    let (close_code, reason, code, message) = if text.starts_with("Space limit exceeded") {
        (
            1009,
            "message too big",
            "message_too_large",
            "A WebSocket message or frame can carry at most 256 KiB",
        )
    } else if text.starts_with("UTF-8 encoding error") {
        (
            1007,
            "invalid UTF-8",
            "invalid_utf8",
            "Text messages must be valid UTF-8",
        )
    } else if text.starts_with("WebSocket protocol error") && !text.contains("closing handshake") {
        (
            1002,
            "protocol error",
            "protocol_error",
            "The frame violated the WebSocket protocol",
        )
    } else {
        return None;
    };
    Some(FrameFailure {
        close_code,
        reason,
        code,
        message,
    })
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
    metadata["limits"] = openai::limits();
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
        // Tungstenite eagerly allocates this for every socket, including idle
        // resume sockets. Larger messages can still grow the buffer on demand.
        .read_buffer_size(16 * 1024)
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

#[cfg(test)]
mod listener_tests {
    use std::time::Instant;

    use futures_util::{SinkExt, StreamExt};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        time::timeout,
    };
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    use super::*;

    const WAIT: Duration = Duration::from_secs(5);
    const REQUEST: &[u8] = b"GET /ok HTTP/1.1\r\nHost: test\r\n\r\n";

    struct Server {
        address: SocketAddr,
        shutdown: watch::Sender<bool>,
        task: tokio::task::JoinHandle<io::Result<()>>,
    }

    async fn start(pending_connections: usize, header_read_timeout: Duration) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route("/ok", get(|| async { "ok" })).route(
            "/echo",
            get(|ws: WebSocketUpgrade| async {
                ws.on_upgrade(|mut socket| async move {
                    while let Some(Ok(message)) = socket.recv().await {
                        if socket.send(message).await.is_err() {
                            break;
                        }
                    }
                })
            }),
        );
        let (shutdown, receiver) = watch::channel(false);
        let limits = HttpLimits {
            pending_connections,
            header_read_timeout,
        };
        let task = tokio::spawn(serve_http(listener, app, receiver, limits));
        Server {
            address,
            shutdown,
            task,
        }
    }

    /// Reads until the server closes the connection and returns what it sent.
    async fn read_to_close(stream: &mut TcpStream) -> String {
        let mut received = Vec::new();
        let mut buffer = [0; 1024];
        timeout(WAIT, async {
            loop {
                match stream.read(&mut buffer).await {
                    Ok(0) | Err(_) => break,
                    Ok(count) => received.extend_from_slice(&buffer[..count]),
                }
            }
        })
        .await
        .expect("the server kept the connection open");
        String::from_utf8_lossy(&received).into_owned()
    }

    #[tokio::test]
    async fn silent_trickling_and_idle_connections_are_closed() {
        let server = start(8, Duration::from_millis(300)).await;

        let started = Instant::now();
        let mut silent = TcpStream::connect(server.address).await.unwrap();
        read_to_close(&mut silent).await;
        assert!(started.elapsed() >= Duration::from_millis(250));

        // A request head that never ends, one header line at a time.
        let mut trickle = TcpStream::connect(server.address).await.unwrap();
        trickle.write_all(b"GET /ok HTTP/1.1\r\n").await.unwrap();
        let writer = async {
            for line in 0..40 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                let header = format!("X-Line-{line}: 1\r\n");
                if trickle.write_all(header.as_bytes()).await.is_err() {
                    return false;
                }
            }
            true
        };
        assert!(!writer.await, "the server read a 2-second request head");

        // A finished request leaves the connection idle, and the timer restarts.
        let mut idle = TcpStream::connect(server.address).await.unwrap();
        idle.write_all(REQUEST).await.unwrap();
        let response = read_to_close(&mut idle).await;
        assert!(response.starts_with("HTTP/1.1 200 "), "{response}");
        assert!(response.ends_with("ok"), "{response}");
    }

    #[tokio::test]
    async fn pending_connections_wait_for_a_free_slot() {
        let server = start(1, WAIT * 2).await;
        let silent = TcpStream::connect(server.address).await.unwrap();
        let mut waiting = TcpStream::connect(server.address).await.unwrap();
        waiting.write_all(REQUEST).await.unwrap();
        let mut first = [0; 64];
        assert!(
            timeout(Duration::from_millis(300), waiting.read(&mut first))
                .await
                .is_err(),
            "a connection beyond the limit was served"
        );
        drop(silent);
        let count = timeout(WAIT, waiting.read(&mut first))
            .await
            .unwrap()
            .unwrap();
        assert!(first[..count].starts_with(b"HTTP/1.1 200 "));
    }

    #[tokio::test]
    async fn an_upgraded_socket_frees_its_slot_and_outlives_the_header_timeout() {
        let server = start(1, Duration::from_millis(200)).await;
        let (mut socket, _) = connect_async(format!("ws://{}/echo", server.address))
            .await
            .unwrap();
        let mut plain = TcpStream::connect(server.address).await.unwrap();
        plain.write_all(REQUEST).await.unwrap();
        let mut first = [0; 64];
        let count = timeout(WAIT, plain.read(&mut first))
            .await
            .unwrap()
            .unwrap();
        assert!(first[..count].starts_with(b"HTTP/1.1 200 "));

        tokio::time::sleep(Duration::from_millis(500)).await;
        socket.send(Message::text("still here")).await.unwrap();
        let echo = timeout(WAIT, socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(echo, Message::text("still here"));
    }

    #[tokio::test]
    async fn shutdown_closes_waiting_connections_and_returns() {
        let server = start(4, WAIT * 2).await;
        let mut silent = TcpStream::connect(server.address).await.unwrap();
        let mut idle = TcpStream::connect(server.address).await.unwrap();
        idle.write_all(b"GET /ok HTTP/1.1\r\nHost: test\r\n\r\n")
            .await
            .unwrap();
        let mut response = [0; 256];
        timeout(WAIT, idle.read(&mut response))
            .await
            .unwrap()
            .unwrap();

        server.shutdown.send_replace(true);
        timeout(WAIT, server.task).await.unwrap().unwrap().unwrap();
        read_to_close(&mut silent).await;
        // The task returning shows the listener is gone. Connecting to its old port
        // proves nothing: another test may already have been given that port.
        read_to_close(&mut idle).await;
    }

    #[tokio::test]
    async fn loads_start_together_and_keep_their_own_errors() {
        let barrier = tokio::sync::Barrier::new(2);
        let both = load_concurrently(
            async {
                barrier.wait().await;
                Ok(1)
            },
            async {
                barrier.wait().await;
                Ok("two")
            },
        );
        assert_eq!(timeout(WAIT, both).await.unwrap().unwrap(), (1, "two"));

        let failed = load_concurrently(
            async {
                Err::<(), _>(anyhow!("missing weights")).context("load live diarization model")
            },
            std::future::pending::<Result<()>>(),
        );
        let error = timeout(WAIT, failed).await.unwrap().unwrap_err();
        assert_eq!(
            format!("{error:#}"),
            "load live diarization model: missing weights"
        );
    }

    #[test]
    fn allowed_origins_are_normalized_or_rejected() {
        for (input, expected) in [
            ("https://example.com", "https://example.com"),
            ("https://Example.COM/", "https://example.com"),
            ("HTTP://localhost:3000", "http://localhost:3000"),
            ("https://example.com:443", "https://example.com"),
            ("http://example.com:80/", "http://example.com"),
            ("http://example.com:8443", "http://example.com:8443"),
            ("http://[0:0::1]:3000", "http://[::1]:3000"),
            (
                "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
                "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
            ),
            (
                "safari-web-extension://E2B3C4D5-AAAA",
                "safari-web-extension://E2B3C4D5-AAAA",
            ),
            ("null", "null"),
        ] {
            assert_eq!(browser_origin(input).unwrap(), expected, "{input}");
        }
        for (input, problem) in [
            ("example.com", "has no scheme"),
            ("localhost:3000", "has no scheme"),
            ("https://example.com/app", "path"),
            ("https://example.com//", "path"),
            ("https://example.com?x=1", "query"),
            ("https://user@example.com", "user information"),
            (" https://example.com", "whitespace"),
            ("https://example.com ", "whitespace"),
            ("", "empty"),
            ("*", "matched exactly"),
            ("https://", "invalid host"),
            ("https://example.com:", "invalid port"),
            ("https://example.com:0", "invalid port"),
            ("https://example.com:+80", "invalid port"),
            ("https://example.com:99999", "invalid port"),
            ("https://bücher.example", "punycode"),
            ("http://[::1", "unclosed"),
            ("http://[::1]x", "after its IPv6"),
            ("1http://example.com", "invalid scheme"),
        ] {
            let message = browser_origin(input).unwrap_err().to_string();
            assert!(message.contains(problem), "{input}: {message}");
            assert!(message.contains("https://example.com"), "{message}");
        }
    }
}
