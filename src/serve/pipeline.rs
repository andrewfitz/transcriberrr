//! Resampling, speech gating and recognition on a model-owning thread.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use anyhow::{Context, Result, anyhow};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use rustvani_vad::{BYTES_PER_WINDOW, SAMPLES_PER_WINDOW, SileroVad};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch};

use super::recognizer::{
    self, Hypothesis, RecognitionOptions, RecognitionStream, Recognizer, TimedWord,
};
use crate::{DeviceKind, EngineKind};

const QUEUE_DEPTH: usize = 128;
const RESULT_DEPTH: usize = 128;
const PER_TRACK_QUEUE_DEPTH: usize = 32;
pub(super) const MAX_QUEUED_FRAMES: usize = 48_000;
const WORK_QUANTUM_FRAMES: usize = 2_400;
const INPUT_BLOCK: usize = 480;
const WINDOW: usize = SAMPLES_PER_WINDOW;
const PRE_ROLL: usize = 8;
const ONSET_WINDOWS: usize = 3;
const PADDING_WINDOWS: usize = 5;
const PAUSE_WINDOWS: usize = 14;
// A Parakeet draft can lag by one 1.44 s chunk plus 160 ms of right
// context. Allow time for the trailing incomplete word as well.
pub(super) const TIMED_ROLLOVER_TAIL_FRAMES: u64 = 57_600;
pub(super) const UNTIMED_ROLLOVER_TAIL_FRAMES: u64 = 24_000;
pub(super) const MAX_NATIVE_FRAMES: u64 = 30 * 24_000;

pub(super) struct TrackSettings {
    pub native: bool,
    pub streaming: bool,
    pub options: RecognitionOptions,
}

pub(super) struct PipelineEvent {
    pub turn: u64,
    pub start: u64,
    pub end: u64,
    pub update: PipelineUpdate,
}

pub(super) enum PipelineUpdate {
    Draft(Hypothesis),
    Final(Hypothesis),
    OverloadFinal(Hypothesis),
    Failed(String),
    Committed { marker: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QueueError {
    Full,
    Closed,
}

#[derive(Clone)]
pub(super) struct RecognitionWorker {
    commands: mpsc::Sender<Command>,
    next_id: Arc<AtomicU64>,
    capacity: Arc<Semaphore>,
    cleanup_pending: Arc<AtomicBool>,
    backend_failure: Option<watch::Receiver<Option<String>>>,
}

pub(super) struct TrackHandle {
    identity: Arc<TrackIdentity>,
    commands: mpsc::Sender<Command>,
    results: mpsc::Receiver<PipelineEvent>,
    capacity: Arc<Semaphore>,
    cleanup_pending: Arc<AtomicBool>,
}

struct TrackIdentity {
    id: u64,
    settings: TrackSettings,
    output: mpsc::Sender<PipelineEvent>,
    cancelled: Mutex<HashSet<u64>>,
    accepted: AtomicU64,
    queued_frames: Arc<AtomicUsize>,
    capacity: Arc<Semaphore>,
    closed: AtomicBool,
    closed_signal: Notify,
}

enum Operation {
    Append {
        start: u64,
        samples: Vec<i16>,
        turn: u64,
        offset: usize,
    },
    Commit(u64),
    Gap {
        marker: u64,
        overload: bool,
    },
}

pub(super) struct Permit {
    _global: OwnedSemaphorePermit,
    _track: OwnedSemaphorePermit,
    queued_frames: Arc<AtomicUsize>,
    frame_count: usize,
}

impl Permit {
    fn consume(&mut self, frames: usize) {
        self.queued_frames.fetch_sub(frames, Ordering::AcqRel);
        self.frame_count -= frames;
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.consume(self.frame_count);
    }
}

struct Work {
    id: u64,
    identity: Weak<TrackIdentity>,
    operation: Operation,
    permit: Permit,
}

enum Command {
    Track(Work),
    Cleanup(Arc<AtomicBool>),
}

impl RecognitionWorker {
    pub(super) async fn spawn(
        engine: EngineKind,
        model: String,
        device: DeviceKind,
        offline: bool,
        model_cache: Option<PathBuf>,
        forced_aligner: Option<String>,
    ) -> Result<Self> {
        let word_timing = recognizer::capabilities(engine).word_timing;
        Self::start_worker(word_timing, false, move || {
            recognizer::load_with_aligner(
                engine,
                &model,
                device,
                offline,
                model_cache,
                forced_aligner.as_deref(),
            )
        })
        .await
    }

    #[cfg(test)]
    pub(super) async fn start(
        load: impl FnOnce() -> Result<Box<dyn Recognizer>> + Send + 'static,
    ) -> Result<Self> {
        Self::start_worker(true, true, load).await
    }

    async fn start_worker(
        word_timing: bool,
        test_gate: bool,
        load: impl FnOnce() -> Result<Box<dyn Recognizer>> + Send + 'static,
    ) -> Result<Self> {
        let (worker, mut input) = Self::channel();
        let (ready, startup) = oneshot::channel();
        std::thread::Builder::new()
            .name("transcriberrr-recognition".into())
            .spawn(move || {
                let model = match load() {
                    Ok(model) => model,
                    Err(error) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                };
                if ready.send(Ok(model.backend_failure())).is_err() {
                    return;
                }
                run_worker(model.as_ref(), &mut input, word_timing, test_gate);
            })
            .context("starting recognition worker")?;
        let backend_failure = startup
            .await
            .context("recognition worker exited during startup")??;
        Ok(Self {
            backend_failure,
            ..worker
        })
    }

    fn channel() -> (Self, mpsc::Receiver<Command>) {
        // Track work holds a global permit. Leave one slot for a cleanup wakeup.
        let (commands, input) = mpsc::channel(QUEUE_DEPTH + 1);
        (
            Self {
                commands,
                next_id: Arc::new(AtomicU64::new(1)),
                capacity: Arc::new(Semaphore::new(QUEUE_DEPTH)),
                cleanup_pending: Arc::new(AtomicBool::new(false)),
                backend_failure: None,
            },
            input,
        )
    }

    pub(super) async fn closed(&self) -> String {
        let backend_failure = async {
            if let Some(mut failure) = self.backend_failure.clone()
                && let Ok(reason) = failure.wait_for(Option::is_some).await
            {
                return reason.as_ref().expect("published failure reason").clone();
            }
            std::future::pending().await
        };
        tokio::select! {
            reason = backend_failure => reason,
            _ = self.commands.closed() => "recognition worker exited unexpectedly".into(),
        }
    }

    pub(super) fn backlog(&self) -> usize {
        QUEUE_DEPTH - self.capacity.available_permits()
    }

    pub(super) fn track(&self, settings: TrackSettings) -> TrackHandle {
        let (output, results) = mpsc::channel(RESULT_DEPTH);
        TrackHandle {
            identity: Arc::new(TrackIdentity {
                id: self.next_id.fetch_add(1, Ordering::Relaxed),
                settings,
                output,
                cancelled: Mutex::new(HashSet::new()),
                accepted: AtomicU64::new(0),
                queued_frames: Arc::new(AtomicUsize::new(0)),
                capacity: Arc::new(Semaphore::new(PER_TRACK_QUEUE_DEPTH)),
                closed: AtomicBool::new(false),
                closed_signal: Notify::new(),
            }),
            commands: self.commands.clone(),
            results,
            capacity: self.capacity.clone(),
            cleanup_pending: self.cleanup_pending.clone(),
        }
    }
}

impl TrackHandle {
    pub(super) fn reserve(&self, frame_count: usize) -> Result<Permit, QueueError> {
        if self.identity.closed.load(Ordering::Acquire) {
            return Err(QueueError::Closed);
        }
        let global = self
            .capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| QueueError::Full)?;
        let track = self
            .identity
            .capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| QueueError::Full)?;
        self.identity
            .queued_frames
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                (used + frame_count <= MAX_QUEUED_FRAMES).then_some(used + frame_count)
            })
            .map_err(|_| QueueError::Full)?;
        Ok(Permit {
            _global: global,
            _track: track,
            queued_frames: self.identity.queued_frames.clone(),
            frame_count,
        })
    }

    fn send(&self, permit: Permit, operation: Operation) -> Result<(), QueueError> {
        let work = Work {
            id: self.identity.id,
            identity: Arc::downgrade(&self.identity),
            operation,
            permit,
        };
        self.commands
            .try_send(Command::Track(work))
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => QueueError::Full,
                mpsc::error::TrySendError::Closed(_) => QueueError::Closed,
            })
    }

    pub(super) fn append(
        &self,
        start: u64,
        samples: Vec<i16>,
        turn: u64,
    ) -> Result<(), QueueError> {
        self.append_reserved(self.reserve(samples.len())?, start, samples, turn)
    }

    pub(super) fn append_reserved(
        &self,
        permit: Permit,
        start: u64,
        samples: Vec<i16>,
        turn: u64,
    ) -> Result<(), QueueError> {
        debug_assert_eq!(permit.frame_count, samples.len());
        self.send(
            permit,
            Operation::Append {
                start,
                samples,
                turn,
                offset: 0,
            },
        )?;
        self.identity.accepted.fetch_max(turn, Ordering::Release);
        Ok(())
    }

    pub(super) fn commit(&self, marker: u64) -> Result<(), QueueError> {
        self.send(self.reserve(0)?, Operation::Commit(marker))
    }

    pub(super) fn backlog_frames(&self) -> usize {
        self.identity.queued_frames.load(Ordering::Acquire)
    }

    pub(super) fn gap(&self, marker: u64, overload: bool) -> Result<(), QueueError> {
        self.send(self.reserve(0)?, Operation::Gap { marker, overload })
    }

    pub(super) fn clear(&self, turn: u64) -> Result<(), QueueError> {
        if turn <= self.identity.accepted.load(Ordering::Acquire) {
            self.identity.cancelled.lock().unwrap().insert(turn);
        }
        self.signal_cleanup();
        if self.commands.is_closed() {
            Err(QueueError::Closed)
        } else {
            Ok(())
        }
    }

    fn signal_cleanup(&self) {
        if !self.cleanup_pending.swap(true, Ordering::AcqRel) {
            let _ = self
                .commands
                .try_send(Command::Cleanup(self.cleanup_pending.clone()));
        }
    }

    pub(super) async fn recv(&mut self) -> Option<PipelineEvent> {
        if self.identity.closed.load(Ordering::Acquire) {
            return self.results.try_recv().ok();
        }
        tokio::select! {
            result = self.results.recv() => result,
            _ = self.commands.closed() => self.results.try_recv().ok(),
            _ = self.identity.closed_signal.notified() => self.results.try_recv().ok(),
        }
    }
    pub(super) fn try_recv(
        &mut self,
    ) -> std::result::Result<PipelineEvent, mpsc::error::TryRecvError> {
        match self.results.try_recv() {
            Err(mpsc::error::TryRecvError::Empty)
                if self.identity.closed.load(Ordering::Acquire) || self.commands.is_closed() =>
            {
                Err(mpsc::error::TryRecvError::Disconnected)
            }
            result => result,
        }
    }
}

impl Drop for TrackHandle {
    fn drop(&mut self) {
        self.identity.closed.store(true, Ordering::Release);
        self.signal_cleanup();
    }
}

impl TrackIdentity {
    fn cancelled(&self, turn: u64) -> bool {
        !self.settings.native && self.cancelled.lock().unwrap().contains(&turn)
    }

    fn emit(&self, event: PipelineEvent) {
        if !self.closed.load(Ordering::Acquire)
            && !self.cancelled(event.turn)
            && self.output.try_send(event).is_err()
        {
            self.close_results();
        }
    }

    fn close_results(&self) {
        self.closed.store(true, Ordering::Release);
        self.closed_signal.notify_one();
    }
}

fn run_worker(
    model: &dyn Recognizer,
    input: &mut mpsc::Receiver<Command>,
    word_timing: bool,
    test_gate: bool,
) {
    let mut tracks: HashMap<u64, (Weak<TrackIdentity>, Track<'_>)> = HashMap::new();
    let mut pending: HashMap<u64, VecDeque<Work>> = HashMap::new();
    let mut ready = VecDeque::new();
    loop {
        if ready.is_empty() {
            let Some(command) = input.blocking_recv() else {
                break;
            };
            enqueue(command, &mut pending, &mut ready);
        }
        for _ in 0..QUEUE_DEPTH {
            let Ok(command) = input.try_recv() else { break };
            enqueue(command, &mut pending, &mut ready);
        }
        tracks.retain(|_, (identity, track)| {
            identity.upgrade().is_some_and(|identity| {
                if track
                    .standard_turn
                    .is_some_and(|turn| identity.cancelled(turn))
                {
                    track.reset();
                }
                !identity.closed.load(Ordering::Acquire)
            })
        });
        let Some(id) = ready.pop_front() else {
            continue;
        };
        let queue = pending.get_mut(&id).unwrap();
        let mut work = queue.pop_front().unwrap();
        if process_work(&mut work, &mut tracks, model, word_timing, test_gate) {
            queue.push_front(work);
        }
        if queue.is_empty() {
            pending.remove(&id);
        } else {
            ready.push_back(id);
        }
    }
}

fn enqueue(
    command: Command,
    pending: &mut HashMap<u64, VecDeque<Work>>,
    ready: &mut VecDeque<u64>,
) {
    match command {
        Command::Track(work) => {
            let queue = pending.entry(work.id).or_insert_with(|| {
                ready.push_back(work.id);
                VecDeque::new()
            });
            queue.push_back(work);
        }
        Command::Cleanup(pending) => {
            // The next scan observes cleanup requests coalesced into this wakeup.
            pending.swap(false, Ordering::AcqRel);
        }
    }
}

/// Handle one quantum, keeping work for any remaining PCM at the head of the track's queue.
fn process_work<'model>(
    work: &mut Work,
    tracks: &mut HashMap<u64, (Weak<TrackIdentity>, Track<'model>)>,
    model: &'model dyn Recognizer,
    word_timing: bool,
    test_gate: bool,
) -> bool {
    let Some(owner) = work.identity.upgrade() else {
        return false;
    };
    if owner.closed.load(Ordering::Acquire) {
        return false;
    }
    let track = match tracks.entry(owner.id) {
        std::collections::hash_map::Entry::Occupied(entry) => &mut entry.into_mut().1,
        std::collections::hash_map::Entry::Vacant(entry) => {
            match Track::new(model, word_timing, test_gate) {
                Ok(track) => &mut entry.insert((work.identity.clone(), track)).1,
                Err(error) => {
                    let (turn, barrier) = match work.operation {
                        Operation::Append { turn, .. } => (turn, false),
                        Operation::Commit(marker) | Operation::Gap { marker, .. } => (marker, true),
                    };
                    owner.emit(PipelineEvent {
                        turn,
                        start: 0,
                        end: 0,
                        update: PipelineUpdate::Failed(error.to_string()),
                    });
                    if owner.settings.native && barrier {
                        owner.emit(committed(turn));
                    }
                    return false;
                }
            }
        }
    };
    match &mut work.operation {
        Operation::Append {
            start,
            samples,
            turn,
            offset,
        } => {
            if !owner.settings.native {
                let mut cancelled = owner.cancelled.lock().unwrap();
                // Per-track work is ordered. Any earlier append and its commit,
                // if present, have already been handled when we see this turn.
                cancelled.retain(|&id| id >= *turn);
                if cancelled.contains(turn) {
                    return false;
                }
            }
            let end = (*offset + WORK_QUANTUM_FRAMES).min(samples.len());
            let outcome = start
                .checked_add(*offset as u64)
                .context("audio frame overflow")
                .and_then(|start| track.append(&owner, start, &samples[*offset..end], *turn));
            if let Err(error) = outcome {
                track.fail(&owner, error);
                return false;
            }
            work.permit.consume(end - *offset);
            *offset = end;
            end != samples.len()
        }
        Operation::Commit(marker) => {
            track.commit(&owner, *marker);
            false
        }
        Operation::Gap { marker, overload } => {
            track.gap(&owner, *marker, *overload);
            false
        }
    }
}

fn committed(marker: u64) -> PipelineEvent {
    PipelineEvent {
        turn: 0,
        start: 0,
        end: 0,
        update: PipelineUpdate::Committed { marker },
    }
}

/// Fixed input blocks make signal and boundaries independent of the websocket packet size.
pub(super) struct RateConverter {
    sinc: SincFixedIn<f32>,
    input: Vec<f32>,
    output: Vec<f32>,
    total_input: usize,
    emitted: usize,
}

impl RateConverter {
    pub(super) fn new() -> Result<Self> {
        let sinc = SincFixedIn::new(
            2.0 / 3.0,
            1.0,
            SincInterpolationParameters {
                sinc_len: 64,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 32,
                window: WindowFunction::BlackmanHarris2,
            },
            INPUT_BLOCK,
            1,
        )?;
        Ok(Self {
            sinc,
            input: Vec::with_capacity(INPUT_BLOCK),
            output: Vec::new(),
            total_input: 0,
            emitted: 0,
        })
    }

    pub(super) fn reset(&mut self) {
        self.sinc.reset();
        self.input.clear();
        self.total_input = 0;
        self.emitted = 0;
    }

    pub(super) fn push(&mut self, mut samples: &[i16]) -> Result<&[f32]> {
        let mut written = 0;
        self.total_input += samples.len();
        while !samples.is_empty() {
            let take = (INPUT_BLOCK - self.input.len()).min(samples.len());
            let (block, rest) = samples.split_at(take);
            self.input
                .extend(block.iter().map(|&sample| f32::from(sample) / 32768.0));
            samples = rest;
            if self.input.len() == INPUT_BLOCK {
                written = self.convert_block(written, usize::MAX)?;
            }
        }
        Ok(&self.output[..written])
    }

    pub(super) fn finish(&mut self) -> Result<&[f32]> {
        let mut written = 0;
        let target = self.total_input * 2 / 3;
        while self.emitted < target {
            self.input.resize(INPUT_BLOCK, 0.0);
            written = self.convert_block(written, target)?;
        }
        Ok(&self.output[..written])
    }

    fn convert_block(&mut self, start: usize, limit: usize) -> Result<usize> {
        let needed = start + self.sinc.output_frames_next();
        if needed > self.output.len() {
            self.output.resize(needed, 0.0);
        }
        let (_, produced) = self.sinc.process_into_buffer(
            &[self.input.as_slice()],
            &mut [&mut self.output[start..]],
            None,
        )?;
        self.input.clear();
        // SincFixedIn holds the filter lookahead before yielding output. Its
        // first output is already centered on the start of the input clock.
        // Commit pads the lookahead and caps output at the original duration.
        let take = produced.min(limit - self.emitted);
        self.emitted += take;
        Ok(start + take)
    }
}

struct SpeechGate {
    vad: SileroVad,
    previous_input: f32,
    previous_filtered: f32,
    peak: f32,
    partial: Vec<f32>,
    before: VecDeque<AudioSpan>,
    next_sample: u64,
    speech_run: usize,
    quiet_run: usize,
    speaking: bool,
    test_gate: bool,
}

impl SpeechGate {
    fn new(test_gate: bool) -> Result<Self> {
        Ok(Self {
            vad: SileroVad::new()
                .map_err(|error| anyhow!("loading the voice activity model: {error}"))?,
            previous_input: 0.0,
            previous_filtered: 0.0,
            peak: 0.0,
            partial: Vec::with_capacity(WINDOW),
            before: VecDeque::with_capacity(PRE_ROLL),
            next_sample: 0,
            speech_run: 0,
            quiet_run: 0,
            speaking: false,
            test_gate,
        })
    }

    fn reset(&mut self) {
        self.vad.reset_state();
        self.previous_input = 0.0;
        self.previous_filtered = 0.0;
        self.peak = 0.0;
        self.partial.clear();
        self.before.clear();
        self.next_sample = 0;
        self.speech_run = 0;
        self.quiet_run = 0;
        self.speaking = false;
    }

    fn probability(&mut self, samples: &[f32]) -> f32 {
        if cfg!(test) && self.test_gate {
            return if samples.iter().any(|sample| sample.abs() >= 0.01) {
                1.0
            } else {
                0.0
            };
        }
        let mut filtered = [0.0; WINDOW];
        let mut peak = 0.0f32;
        for (output, sample) in filtered.iter_mut().zip(samples) {
            self.previous_filtered = sample - self.previous_input + 0.943 * self.previous_filtered;
            self.previous_input = *sample;
            *output = self.previous_filtered;
            peak = peak.max(output.abs());
        }
        self.peak = (self.peak * 0.96).max(peak);
        let gain = 0.7 / self.peak.max(0.0005);
        let mut bytes = [0; BYTES_PER_WINDOW];
        for (pair, sample) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(filtered) {
            let pcm = ((sample * gain).clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
            pair.copy_from_slice(&pcm.to_le_bytes());
        }
        self.vad.infer(&bytes).unwrap_or(1.0)
    }

    fn process(&mut self, mut samples: &[f32], flush: bool) -> Vec<GateEvent> {
        let mut events = Vec::new();
        while !samples.is_empty() {
            let take = (WINDOW - self.partial.len()).min(samples.len());
            self.partial.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            if self.partial.len() == WINDOW {
                let window = std::mem::replace(&mut self.partial, Vec::with_capacity(WINDOW));
                self.window(window, &mut events);
            }
        }
        if flush && !self.partial.is_empty() {
            let window = std::mem::replace(&mut self.partial, Vec::with_capacity(WINDOW));
            self.window(window, &mut events);
        }
        events
    }

    fn window(&mut self, samples: Vec<f32>, events: &mut Vec<GateEvent>) {
        let speech = self.probability(&samples) >= 0.5;
        let start = self.next_sample;
        self.next_sample += samples.len() as u64;
        let span = AudioSpan { start, samples };
        if self.speaking {
            self.quiet_run = if speech { 0 } else { self.quiet_run + 1 };
            if self.quiet_run <= PADDING_WINDOWS {
                events.extend(self.before.drain(..).map(GateEvent::Audio));
                events.push(GateEvent::Audio(span));
                return;
            }
            if self.quiet_run == PAUSE_WINDOWS {
                self.speaking = false;
                self.speech_run = 0;
                events.push(GateEvent::Pause);
            }
        }
        if self.before.len() == PRE_ROLL {
            self.before.pop_front();
        }
        self.before.push_back(span);
        if !self.speaking {
            self.speech_run = if speech { self.speech_run + 1 } else { 0 };
            if self.speech_run == ONSET_WINDOWS {
                self.speaking = true;
                self.quiet_run = 0;
                events.extend(self.before.drain(..).map(GateEvent::Audio));
            }
        }
    }
}

enum GateEvent {
    Audio(AudioSpan),
    Pause,
}

struct AudioSpan {
    start: u64,
    samples: Vec<f32>,
}

impl AudioSpan {
    fn end(&self) -> u64 {
        self.start + self.samples.len() as u64
    }

    fn split(self, at: u64) -> (Option<Self>, Option<Self>) {
        if at <= self.start {
            return (None, Some(self));
        }
        if at >= self.end() {
            return (Some(self), None);
        }
        let mut head = self;
        let tail = head.samples.split_off((at - head.start) as usize);
        (
            Some(head),
            Some(Self {
                start: at,
                samples: tail,
            }),
        )
    }
}

fn frame(origin: u64, sample: u64) -> u64 {
    origin.saturating_add(sample.saturating_mul(3).div_ceil(2))
}

/// Only the words independently confirmed on the old and new decoders can
/// establish ownership. Parakeet timestamps are emission times on an 80 ms
/// grid, so the last two lexical anchors may move by one encoder frame.
struct RolloverContext {
    cut: u64,
    confirmed: Vec<(String, u64, u64)>,
}

impl RolloverContext {
    fn new(cut: u64, start: u64, words: &[TimedWord]) -> Option<Self> {
        let confirmed: Vec<_> = words
            .iter()
            .filter(|word| word.end_sample > start)
            .filter_map(lexical_anchor)
            .collect();
        (confirmed.len() >= 2).then_some(Self { cut, confirmed })
    }

    fn owned(&self, mut hypothesis: Hypothesis) -> std::result::Result<Option<Hypothesis>, ()> {
        // Each timed fragment must be entirely on one side of the seam, with
        // coverage of the original text. A draft may omit its incomplete word.
        let prefix_end = hypothesis
            .words
            .iter()
            .take_while(|word| word.end_sample <= self.cut)
            .count();
        let right = &hypothesis.words[prefix_end..];
        if right
            .iter()
            .any(|word| word.start_sample < self.cut || word.end_sample <= self.cut)
        {
            return Err(());
        }
        if !right.iter().any(|word| lexical_anchor(word).is_some()) {
            return Ok(None);
        }

        let old: Vec<_> = hypothesis.words[..prefix_end]
            .iter()
            .filter_map(lexical_anchor)
            .collect();
        // A lexical match alone could delete a legitimate repeat in the new
        // item. Match the old suffix only on the clock-confirmed old side.
        let Some(expected) = self
            .confirmed
            .get(self.confirmed.len().saturating_sub(old.len())..)
        else {
            return Err(());
        };
        if old.len() < 2
            || old.len() != expected.len()
            || old
                .iter()
                .zip(expected)
                .any(|(old, expected)| old.0 != expected.0)
            || old
                .iter()
                .zip(expected)
                .rev()
                .take(2)
                .any(|(old, expected)| {
                    old.1.abs_diff(expected.1) > 1_920 || old.2.abs_diff(expected.2) > 1_920
                })
        {
            return Err(());
        }

        let timed: String = hypothesis
            .words
            .iter()
            .map(|word| word.text.as_str())
            .collect();
        if !hypothesis.text.starts_with(&timed) {
            return Err(());
        }
        // Punctuation without a leading word belongs to the old context. Do
        // not send it or any trailing untimed characters as a native draft.
        let owned_start = prefix_end
            + right
                .iter()
                .take_while(|word| lexical_anchor(word).is_none())
                .count();
        hypothesis.words.drain(..owned_start);
        if let Some(first) = hypothesis.words.first_mut() {
            first.text = first.text.trim_start().to_owned();
        }
        hypothesis.text = hypothesis
            .words
            .iter()
            .map(|word| word.text.as_str())
            .collect();
        Ok(Some(hypothesis))
    }
}

fn lexical_anchor(word: &TimedWord) -> Option<(String, u64, u64)> {
    let text: String = word
        .text
        .chars()
        .filter(|letter| letter.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    (!text.is_empty()).then_some((text, word.start_sample, word.end_sample))
}

struct Item<'model> {
    turn: u64,
    origin: u64,
    audio: Vec<AudioSpan>,
    stream: Option<Box<dyn RecognitionStream + 'model>>,
    stable: Option<Hypothesis>,
    decoded: usize,
    has_source_gaps: bool,
    preroll: Option<RolloverContext>,
}

impl<'model> Item<'model> {
    fn new(turn: u64, origin: u64) -> Self {
        Self {
            turn,
            origin,
            audio: Vec::new(),
            stream: None,
            stable: None,
            decoded: 0,
            has_source_gaps: false,
            preroll: None,
        }
    }

    fn first_owned_sample(&self) -> Option<u64> {
        self.audio
            .get(usize::from(self.preroll.is_some()))
            .map(|span| span.start)
    }

    fn interval(&self) -> (u64, u64) {
        (
            self.first_owned_sample()
                .map_or(self.origin, |sample| frame(self.origin, sample)),
            self.audio
                .last()
                .map_or(self.origin, |span| frame(self.origin, span.end())),
        )
    }

    fn emit(&self, owner: &TrackIdentity, update: PipelineUpdate) {
        let (start, end) = self.interval();
        owner.emit(PipelineEvent {
            turn: self.turn,
            start,
            end,
            update,
        });
    }

    fn push(
        &mut self,
        audio: AudioSpan,
        model: &'model dyn Recognizer,
        owner: &TrackIdentity,
        emit_draft: bool,
    ) -> Result<()> {
        self.remember(audio);
        if !owner.settings.streaming {
            return Ok(());
        }
        let mut next = self.decode_pending(model, owner)?;
        if let Some(context) = &self.preroll
            && let Some(hypothesis) = next.take()
        {
            next = match context.owned(hypothesis) {
                Ok(owned) => owned,
                Err(()) => {
                    self.remove_preroll();
                    self.decode_pending(model, owner)?
                }
            };
        }
        if let Some(hypothesis) = next {
            if hypothesis.stable {
                self.stable = Some(hypothesis.clone());
            }
            if emit_draft {
                self.emit(owner, PipelineUpdate::Draft(hypothesis));
            }
        }
        Ok(())
    }

    fn decode_pending(
        &mut self,
        model: &'model dyn Recognizer,
        owner: &TrackIdentity,
    ) -> Result<Option<Hypothesis>> {
        if self.stream.is_none() {
            self.stream = Some(model.start(&owner.settings.options)?);
        }
        let stream = self.stream.as_mut().unwrap();
        let mut latest = None;
        for span in &self.audio[self.decoded..] {
            if let Some(result) = stream.push(&span.samples)? {
                latest = Some(result);
            }
        }
        self.decoded = self.audio.len();
        Ok(latest.map(|hypothesis| self.map(hypothesis)))
    }

    fn remove_preroll(&mut self) {
        self.preroll = None;
        self.audio.remove(0);
        self.stream = None;
        self.stable = None;
        self.decoded = 0;
    }

    fn finish(mut self, model: &'model dyn Recognizer, owner: &TrackIdentity) {
        if self.preroll.is_some() && self.first_owned_sample().is_none() {
            return;
        }
        let outcome = self.finish_decoder(model, owner).and_then(|hypothesis| {
            if let Some(context) = &self.preroll {
                // A finalized suffix must have complete timing. If ownership
                // cannot be proved, decode the owned audio without context.
                if hypothesis
                    .words
                    .iter()
                    .map(|word| word.text.as_str())
                    .collect::<String>()
                    == hypothesis.text
                    && let Ok(Some(owned)) = context.owned(hypothesis)
                {
                    return Ok(owned);
                }
                self.remove_preroll();
                self.finish_decoder(model, owner)
            } else {
                Ok(hypothesis)
            }
        });
        match outcome {
            Ok(hypothesis) => self.emit(owner, PipelineUpdate::Final(hypothesis)),
            Err(error) => self.emit(owner, PipelineUpdate::Failed(format!("{error:#}"))),
        }
    }

    fn finish_decoder(
        &mut self,
        model: &'model dyn Recognizer,
        owner: &TrackIdentity,
    ) -> Result<Hypothesis> {
        if self.audio.is_empty() {
            return Ok(Hypothesis {
                stable: true,
                ..Hypothesis::default()
            });
        }
        if self.stream.is_none() {
            self.stream = Some(model.start_committed(&owner.settings.options)?);
        }
        let mut stream = self.stream.take().unwrap();
        for span in &self.audio[self.decoded..] {
            stream.push(&span.samples)?;
        }
        Ok(self.map(stream.finish()?))
    }

    fn preceding_audio(&self, boundary: u64, samples: u64) -> Option<AudioSpan> {
        let start = boundary.checked_sub(samples)?;
        if samples == 0 || start < self.first_owned_sample()? {
            return None;
        }
        let mut context = Vec::with_capacity(samples as usize);
        let first = self.audio.partition_point(|span| span.end() <= start);
        for span in self.audio[first..]
            .iter()
            .take_while(|span| span.start < boundary)
        {
            let from = span.start.max(start);
            if from != start + context.len() as u64 {
                return None;
            }
            let end = span.end().min(boundary);
            context.extend_from_slice(
                &span.samples[(from - span.start) as usize..(end - span.start) as usize],
            );
        }
        (context.len() as u64 == samples).then_some(AudioSpan {
            start,
            samples: context,
        })
    }

    fn take_tail(&mut self, boundary: u64) -> Vec<AudioSpan> {
        let first = self.audio.partition_point(|span| span.start < boundary);
        let mut tail = self.audio.split_off(first);
        if let Some(last) = self.audio.last_mut()
            && boundary < last.end()
        {
            let samples = last.samples.split_off((boundary - last.start) as usize);
            tail.insert(
                0,
                AudioSpan {
                    start: boundary,
                    samples,
                },
            );
        }
        tail
    }

    fn map(&self, mut hypothesis: Hypothesis) -> Hypothesis {
        for word in &mut hypothesis.words {
            let start = self.source_sample(word.start_sample, false);
            let end = self.source_sample(word.end_sample, true).max(start);
            word.start_sample = frame(self.origin, start);
            word.end_sample = frame(self.origin, end);
        }
        hypothesis
    }

    fn remember(&mut self, span: AudioSpan) {
        self.has_source_gaps |= self
            .audio
            .last()
            .is_some_and(|previous| previous.end() != span.start);
        self.audio.push(span);
    }

    fn source_sample(&self, offset: u64, end: bool) -> u64 {
        if !self.has_source_gaps {
            return self
                .audio
                .first()
                .zip(self.audio.last())
                .map_or(0, |(first, last)| {
                    first.start.saturating_add(offset).min(last.end())
                });
        }
        let mut remaining = offset;
        for span in &self.audio {
            let len = span.samples.len() as u64;
            if remaining < len || (end && remaining == len) {
                return span.start + remaining;
            }
            remaining -= len;
        }
        self.audio.last().map_or(0, AudioSpan::end)
    }

    fn stable_cut(&mut self, limit: u64) -> Option<(u64, Hypothesis)> {
        let earliest = limit.saturating_sub(TIMED_ROLLOVER_TAIL_FRAMES * 2 / 3);
        let mut stable = self.stable.take()?;
        let (last, boundary) =
            stable
                .words
                .iter()
                .enumerate()
                .rev()
                .find_map(|(index, word)| {
                    let at = word
                        .end_sample
                        .saturating_sub(self.origin)
                        .saturating_mul(2)
                        / 3;
                    (earliest <= at && at <= limit).then_some((index, at))
                })?;
        stable.words.truncate(last + 1);
        stable.text = stable.words.iter().map(|word| word.text.as_str()).collect();
        Some((boundary, stable))
    }

    fn quiet_cut(&self, limit: u64) -> u64 {
        let earliest = limit.saturating_sub(UNTIMED_ROLLOVER_TAIL_FRAMES * 2 / 3);
        // Measure level over complete VAD windows to avoid choosing a zero crossing in loud speech.
        self.audio
            .iter()
            .filter(|span| span.start >= earliest && span.start <= limit)
            .map(|span| {
                let level = span
                    .samples
                    .iter()
                    .map(|sample| sample * sample)
                    .sum::<f32>()
                    / span.samples.len() as f32;
                (span.start, level)
            })
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map_or(limit, |(at, _)| at)
    }
}

struct Track<'model> {
    model: &'model dyn Recognizer,
    converter: RateConverter,
    gate: SpeechGate,
    origin: u64,
    next_frame: Option<u64>,
    standard_turn: Option<u64>,
    native_turn: u64,
    item: Option<Item<'model>>,
    failed: bool,
    word_timing: bool,
}

impl<'model> Track<'model> {
    fn new(model: &'model dyn Recognizer, word_timing: bool, test_gate: bool) -> Result<Self> {
        Ok(Self {
            model,
            converter: RateConverter::new()?,
            gate: SpeechGate::new(test_gate)?,
            origin: 0,
            next_frame: None,
            standard_turn: None,
            native_turn: 0,
            item: None,
            failed: false,
            word_timing,
        })
    }

    fn reset(&mut self) {
        self.item = None;
        self.converter.reset();
        self.gate.reset();
        self.next_frame = None;
        self.standard_turn = None;
        self.failed = false;
    }

    fn append(
        &mut self,
        owner: &TrackIdentity,
        start: u64,
        samples: &[i16],
        turn: u64,
    ) -> Result<()> {
        if !owner.settings.native {
            if self.standard_turn != Some(turn) {
                self.reset();
                self.standard_turn = Some(turn);
                self.item = Some(Item::new(turn, start));
            }
        } else if self.next_frame.is_some_and(|next| next != start) {
            self.complete(owner);
        }
        if self.failed || samples.is_empty() {
            return Ok(());
        }
        if self.next_frame.is_none() {
            self.origin = start;
        }
        self.next_frame = Some(
            start
                .checked_add(samples.len() as u64)
                .context("audio frame overflow")?,
        );
        let audio = self.converter.push(samples)?;
        let events = self.gate.process(audio, false);
        self.process(owner, events)
    }

    fn commit(&mut self, owner: &TrackIdentity, marker: u64) {
        if owner.settings.native || self.standard_turn == Some(marker) {
            self.complete(owner);
        }
        if owner.settings.native {
            owner.emit(committed(marker));
        }
    }

    fn gap(&mut self, owner: &TrackIdentity, marker: u64, overload: bool) {
        if overload {
            if let Some(mut item) = self.item.take()
                && item.first_owned_sample().is_some()
            {
                if let Some(hypothesis) = item
                    .stable
                    .take()
                    .filter(|hypothesis| !hypothesis.text.is_empty())
                {
                    item.emit(owner, PipelineUpdate::OverloadFinal(hypothesis));
                } else {
                    item.emit(owner, PipelineUpdate::Failed("overload".into()));
                }
            }
            self.reset();
        } else {
            self.complete(owner);
        }
        if owner.settings.native {
            owner.emit(committed(marker));
        }
    }

    fn complete(&mut self, owner: &TrackIdentity) {
        if !self.failed {
            let outcome = self
                .converter
                .finish()
                .map(|audio| self.gate.process(audio, true))
                .and_then(|events| self.process(owner, events));
            match outcome {
                Ok(()) => {
                    if let Some(item) = self.item.take() {
                        item.finish(self.model, owner);
                    }
                }
                Err(error) => self.fail(owner, error),
            }
        }
        self.reset();
    }

    fn fail(&mut self, owner: &TrackIdentity, error: anyhow::Error) {
        if let Some(item) = self.item.take() {
            item.emit(owner, PipelineUpdate::Failed(format!("{error:#}")));
        }
        if owner.settings.native {
            self.reset();
        } else {
            self.failed = true;
        }
    }

    fn process(&mut self, owner: &TrackIdentity, events: Vec<GateEvent>) -> Result<()> {
        for event in events {
            if owner.closed.load(Ordering::Acquire)
                || self.standard_turn.is_some_and(|turn| owner.cancelled(turn))
            {
                break;
            }
            match event {
                GateEvent::Pause if owner.settings.native => {
                    if let Some(item) = self.item.take() {
                        item.finish(self.model, owner);
                    }
                }
                GateEvent::Pause => {}
                GateEvent::Audio(span) if owner.settings.native => self.feed_native(owner, span)?,
                GateEvent::Audio(span) => {
                    if let Some(item) = self.item.as_mut() {
                        item.push(span, self.model, owner, true)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn new_native_item(&mut self) {
        self.native_turn += 1;
        self.item = Some(Item::new(self.native_turn, self.origin));
    }

    fn feed_native(&mut self, owner: &TrackIdentity, span: AudioSpan) -> Result<()> {
        if self.item.is_none() {
            self.new_native_item();
        }
        let item = self.item.as_mut().unwrap();
        if item.preroll.is_some()
            && item.first_owned_sample().is_none()
            && item.audio[0].end() != span.start
        {
            item.remove_preroll();
        }
        let limit = item
            .first_owned_sample()
            .unwrap_or(span.start)
            .saturating_add(MAX_NATIVE_FRAMES * 2 / 3);
        let at_limit = span.end() >= limit;
        let (head, tail) = span.split(limit);
        if let Some(head) = head {
            if self.word_timing {
                item.push(head, self.model, owner, !at_limit)?;
            } else {
                let (decode, reserve) = head.split(limit - UNTIMED_ROLLOVER_TAIL_FRAMES * 2 / 3);
                if let Some(decode) = decode {
                    item.push(decode, self.model, owner, true)?;
                }
                if let Some(reserve) = reserve {
                    item.remember(reserve);
                }
            }
        }
        if !at_limit {
            return Ok(());
        }
        let mut item = self.item.take().unwrap();
        let (boundary, stable_prefix) = if self.word_timing {
            item.stable_cut(limit)
                .map_or((limit, None), |(at, prefix)| (at, Some(prefix)))
        } else {
            (item.quiet_cut(limit), None)
        };
        let mut carry = if boundary < limit {
            item.take_tail(boundary)
        } else {
            Vec::new()
        };
        let mut preroll = None;
        if let Some(prefix) = stable_prefix {
            // The old final owns its original prefix. Only a backend that
            // explicitly supports pre-roll can give those samples to the next
            // decoder for context, without transferring word ownership.
            if let Some(audio) =
                item.preceding_audio(boundary, self.model.rollover_preroll_samples())
                && let Some(context) = RolloverContext::new(
                    frame(item.origin, boundary),
                    frame(item.origin, audio.start),
                    &prefix.words,
                )
            {
                preroll = Some((audio, context));
            }
            item.emit(owner, PipelineUpdate::Final(prefix));
            drop(item);
        } else {
            item.finish(self.model, owner);
        }
        carry.extend(tail);
        if !carry.is_empty() || preroll.is_some() {
            self.new_native_item();
            if carry.first().is_none_or(|span| span.start == boundary)
                && let Some((audio, context)) = preroll
            {
                let next = self.item.as_mut().unwrap();
                next.preroll = Some(context);
                next.remember(audio);
            }
            for audio in carry {
                self.item
                    .as_mut()
                    .unwrap()
                    .push(audio, self.model, owner, true)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn convert(input: &[i16], packet: usize) -> Vec<f32> {
        let mut converter = RateConverter::new().unwrap();
        let mut output = Vec::new();
        for samples in input.chunks(packet) {
            output.extend(converter.push(samples).unwrap());
        }
        output.extend(converter.finish().unwrap());
        output
    }

    #[test]
    fn resampler_preserves_packet_independence_length_and_clock() {
        let mut input = vec![0; 20_003];
        input[9_000] = 32_767;
        let baseline = convert(&input, 480);
        assert_eq!(baseline.len(), 13_335);
        assert_eq!(baseline, convert(&input, 2_400));
        assert_eq!(baseline, convert(&input, 293));
        let peak = baseline
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert!(peak.abs_diff(6_000) <= 1, "impulse peaked at {peak}");
    }

    #[test]
    fn resampler_reuses_storage_without_exposing_stale_samples() {
        let mut converter = RateConverter::new().unwrap();
        assert!(!converter.push(&[12_000; 2_400]).unwrap().is_empty());
        assert!(!converter.finish().unwrap().is_empty());
        assert!(converter.finish().unwrap().is_empty());
        converter.reset();
        assert!(converter.push(&[0; 3]).unwrap().is_empty());
        assert_eq!(converter.finish().unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn timed_endpoints_use_opposite_sides_of_a_suppressed_gap() {
        let mut item = Item::new(1, 24_000);
        item.remember(AudioSpan {
            start: 320,
            samples: vec![0.0; 640],
        });
        item.remember(AudioSpan {
            start: 3_200,
            samples: vec![0.0; 480],
        });
        let words = item
            .map(Hypothesis {
                text: "Hello there.".into(),
                stable: true,
                language: None,
                words: vec![
                    TimedWord {
                        text: "Hello".into(),
                        start_sample: 0,
                        end_sample: 640,
                    },
                    TimedWord {
                        text: " there.".into(),
                        start_sample: 640,
                        end_sample: 1_120,
                    },
                ],
            })
            .words;
        assert_eq!(
            (words[0].start_sample, words[0].end_sample),
            (24_480, 25_440)
        );
        assert_eq!(
            (words[1].start_sample, words[1].end_sample),
            (28_800, 29_520)
        );
    }

    #[test]
    fn real_silero_retention_is_independent_of_packets() {
        let aiff = crate::audio_convert::TempPath::new("aiff");
        let speech = std::process::Command::new("/usr/bin/say")
            .arg("-o")
            .arg(aiff.path())
            .arg("The delivery truck arrived early this morning.")
            .status();
        if !speech.is_ok_and(|status| status.success()) {
            return;
        }
        let speech = crate::decode_audio(aiff.path()).unwrap().samples;
        let mut pcm = vec![0; 12_000];
        // Place the same phrase twice with raw-clock silence between them.
        let speech_24k = (0..speech.len() * 3 / 2)
            .map(|frame| (speech[frame * 2 / 3] * 16_000.0) as i16)
            .collect::<Vec<_>>();
        pcm.extend_from_slice(&speech_24k);
        pcm.extend(std::iter::repeat_n(0, 30_000));
        pcm.extend_from_slice(&speech_24k);
        pcm.extend(std::iter::repeat_n(0, 30_000));

        fn gated(pcm: &[i16], sizes: &[usize]) -> Vec<Option<(u64, Vec<f32>)>> {
            let mut converter = RateConverter::new().unwrap();
            let mut gate = SpeechGate::new(false).unwrap();
            let mut retained = Vec::new();
            let mut offset = 0;
            for size in sizes.iter().cycle() {
                if offset == pcm.len() {
                    break;
                }
                let end = (offset + size).min(pcm.len());
                let output = converter.push(&pcm[offset..end]).unwrap();
                retained.extend(
                    gate.process(output, false)
                        .into_iter()
                        .map(|event| match event {
                            GateEvent::Audio(span) => Some((span.start, span.samples)),
                            GateEvent::Pause => None,
                        }),
                );
                offset = end;
            }
            let tail = converter.finish().unwrap();
            retained.extend(
                gate.process(tail, true)
                    .into_iter()
                    .map(|event| match event {
                        GateEvent::Audio(span) => Some((span.start, span.samples)),
                        GateEvent::Pause => None,
                    }),
            );
            retained
        }

        let baseline = gated(&pcm, &[480]);
        assert!(baseline.iter().filter(|event| event.is_none()).count() >= 2);
        assert_eq!(baseline, gated(&pcm, &[2_400]));
        assert_eq!(baseline, gated(&pcm, &[229, 51, 821, 2_095, 180]));
    }

    struct CountingRecognizer {
        word_timing: bool,
        decoded: AtomicUsize,
    }

    struct CountingStream<'a> {
        model: &'a CountingRecognizer,
        length: usize,
    }

    impl Recognizer for CountingRecognizer {
        fn start(&self, _: &RecognitionOptions) -> Result<Box<dyn RecognitionStream + '_>> {
            Ok(Box::new(CountingStream {
                model: self,
                length: 0,
            }))
        }
    }

    impl RecognitionStream for CountingStream<'_> {
        fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>> {
            let previous = self.length / 1_600;
            self.length += samples.len();
            self.model
                .decoded
                .fetch_add(samples.len(), Ordering::Relaxed);
            Ok((self.length / 1_600 > previous).then(|| {
                let mut draft = self.hypothesis();
                // Keep the most recent two seconds unconfirmed, representing
                // chunk/right-context latency and the current incomplete word.
                draft.words.truncate(draft.words.len().saturating_sub(20));
                draft
            }))
        }

        fn finish(self: Box<Self>) -> Result<Hypothesis> {
            Ok(self.hypothesis())
        }
    }

    impl CountingStream<'_> {
        fn hypothesis(&self) -> Hypothesis {
            let word_count = self.length as u64 / 1_600;
            Hypothesis {
                text: " word".repeat(word_count as usize),
                stable: true,
                language: None,
                words: if self.model.word_timing {
                    (0..word_count)
                        .map(|index| TimedWord {
                            text: " word".into(),
                            start_sample: index * 1_600,
                            end_sample: (index + 1) * 1_600,
                        })
                        .collect()
                } else {
                    Vec::new()
                },
            }
        }
    }

    struct PhraseRecognizer;

    #[derive(Default)]
    struct PhraseStream(Vec<f32>);

    impl Recognizer for PhraseRecognizer {
        fn start(&self, _: &RecognitionOptions) -> Result<Box<dyn RecognitionStream + '_>> {
            Ok(Box::new(PhraseStream::default()))
        }
    }

    impl RecognitionStream for PhraseStream {
        fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>> {
            self.0.extend_from_slice(samples);
            Ok(None)
        }

        fn finish(self: Box<Self>) -> Result<Hypothesis> {
            assert!(
                self.0.len() < 24_000,
                "the recognizer received the suppressed silence"
            );
            let words = [(1.0, "First phrase."), (-1.0, " Second phrase.")]
                .into_iter()
                .map(|(sign, phrase)| {
                    let speech = |sample: &f32| *sample * sign > 0.3;
                    TimedWord {
                        text: phrase.into(),
                        start_sample: self.0.iter().position(speech).unwrap() as u64,
                        end_sample: self.0.iter().rposition(speech).unwrap() as u64 + 1,
                    }
                })
                .collect::<Vec<_>>();
            Ok(Hypothesis {
                text: words.iter().map(|word| word.text.as_str()).collect(),
                stable: true,
                words,
                ..Hypothesis::default()
            })
        }
    }

    #[tokio::test]
    async fn explicit_commit_keeps_both_phrases_on_the_source_clock() {
        let worker = RecognitionWorker::start(|| Ok(Box::new(PhraseRecognizer)))
            .await
            .unwrap();
        let mut track = worker.track(TrackSettings {
            native: false,
            streaming: false,
            options: RecognitionOptions::default(),
        });
        let origin = 24_000_u64;
        let phrases = [(origin, origin + 9_600), (origin + 57_600, origin + 67_200)];
        let mut pcm = vec![16_384; 9_600];
        pcm.extend(std::iter::repeat_n(0, 48_000));
        pcm.extend(std::iter::repeat_n(-16_384, 9_600));
        // Multiple appends below the queue admission limit, including silence.
        let mut offset = 0;
        for chunk in pcm.chunks(12_000) {
            loop {
                match track.append(origin + offset, chunk.to_vec(), 1) {
                    Ok(()) => break,
                    Err(QueueError::Full) => tokio::task::yield_now().await,
                    Err(QueueError::Closed) => panic!("recognition worker exited"),
                }
            }
            offset += chunk.len() as u64;
        }
        track.commit(1).unwrap();
        let event = tokio::time::timeout(std::time::Duration::from_secs(5), track.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            (event.start, event.end),
            (origin, origin + pcm.len() as u64)
        );
        let PipelineUpdate::Final(finalized) = event.update else {
            panic!("expected final")
        };
        assert_eq!(finalized.text, "First phrase. Second phrase.");
        assert_eq!(finalized.words.len(), phrases.len());
        for (word, (start, end)) in finalized.words.iter().zip(phrases) {
            assert!(word.start_sample.abs_diff(start) <= 3, "{word:?}");
            assert!(word.end_sample.abs_diff(end) <= 3, "{word:?}");
        }
    }

    #[tokio::test]
    async fn clearing_a_turn_keeps_earlier_queued_commits() {
        let (worker, mut receiver) = RecognitionWorker::channel();
        let mut track = worker.track(TrackSettings {
            native: false,
            streaming: true,
            options: RecognitionOptions::default(),
        });
        track.append(0, vec![5_000; 4_800], 1).unwrap();
        track.commit(1).unwrap();
        track.append(0, vec![5_000; 4_800], 2).unwrap();
        track.clear(2).unwrap();
        track.append(0, vec![5_000; 4_800], 3).unwrap();
        track.commit(3).unwrap();

        let thread = std::thread::spawn(move || {
            let model = CountingRecognizer {
                word_timing: true,
                decoded: AtomicUsize::new(0),
            };
            run_worker(&model, &mut receiver, true, true);
        });
        let mut completed = Vec::new();
        while completed.len() < 2 {
            let event = tokio::time::timeout(std::time::Duration::from_secs(5), track.recv())
                .await
                .expect("recognition worker timed out")
                .unwrap();
            assert_ne!(event.turn, 2);
            if matches!(event.update, PipelineUpdate::Final(_)) {
                completed.push(event.turn);
            }
        }
        assert_eq!(completed, vec![1, 3]);
        drop(track);
        drop(worker);
        thread.join().unwrap();
    }

    #[tokio::test]
    async fn repeated_clears_prune_cancelled_turns() {
        let (worker, mut receiver) = RecognitionWorker::channel();
        let track = worker.track(TrackSettings {
            native: false,
            streaming: true,
            options: RecognitionOptions::default(),
        });
        // No processing occurs until all the turns have been cancelled.
        for turn in 1..=24 {
            track.append(0, vec![0; 480], turn).unwrap();
            track.clear(turn).unwrap();
        }
        let thread = std::thread::spawn(move || {
            run_worker(&PhraseRecognizer, &mut receiver, true, true);
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while worker.backlog() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(track.identity.cancelled.lock().unwrap().len() <= 1);
        drop(track);
        drop(worker);
        thread.join().unwrap();
    }

    #[test]
    fn cleanup_wakeups_leave_the_full_work_capacity_available() {
        let (worker, mut input) = RecognitionWorker::channel();
        let new_track = || {
            worker.track(TrackSettings {
                native: false,
                streaming: true,
                options: RecognitionOptions::default(),
            })
        };
        for _ in 0..QUEUE_DEPTH {
            drop(new_track());
        }
        assert_eq!(worker.backlog(), 0);
        let tracks: Vec<_> = (0..QUEUE_DEPTH.div_ceil(PER_TRACK_QUEUE_DEPTH))
            .map(|_| new_track())
            .collect();
        for packet in 0..QUEUE_DEPTH {
            let track = &tracks[packet / PER_TRACK_QUEUE_DEPTH];
            track.append(packet as u64, vec![0], 1).unwrap();
            track.clear(1).unwrap();
        }
        assert_eq!(worker.backlog(), QUEUE_DEPTH);
        assert_eq!(input.len(), QUEUE_DEPTH + 1);

        let cleanup = input.try_recv().unwrap();
        assert!(matches!(&cleanup, Command::Cleanup(_)));
        enqueue(cleanup, &mut HashMap::new(), &mut VecDeque::new());
        drop(new_track());
        assert_eq!(input.len(), QUEUE_DEPTH + 1);
    }

    #[tokio::test]
    async fn result_overflow_wakes_a_receiver_after_the_queue_is_drained() {
        let (worker, _input) = RecognitionWorker::channel();
        let mut track = worker.track(TrackSettings {
            native: true,
            streaming: true,
            options: RecognitionOptions::default(),
        });
        let identity = track.identity.clone();
        for marker in 0..RESULT_DEPTH {
            identity.emit(committed(marker as u64));
        }
        let overflow = identity.output.try_send(committed(RESULT_DEPTH as u64));
        assert!(matches!(overflow, Err(mpsc::error::TrySendError::Full(_))));
        for marker in 0..RESULT_DEPTH {
            let event = track.try_recv().unwrap();
            assert!(
                matches!(event.update, PipelineUpdate::Committed { marker: id } if id == marker as u64)
            );
        }

        // Let the reader wait in the interval between the failed send and the
        // producer reporting the overflow.
        let mut waiting = std::pin::pin!(track.recv());
        std::future::poll_fn(|cx| {
            assert!(waiting.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        identity.close_results();
        identity.emit(committed(RESULT_DEPTH as u64 + 1));
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), waiting)
                .await
                .unwrap()
                .is_none()
        );
    }

    fn timed(text: &str, start_sample: u64, end_sample: u64) -> TimedWord {
        TimedWord {
            text: text.into(),
            start_sample,
            end_sample,
        }
    }

    #[test]
    fn rollover_ownership_keeps_legitimate_repeated_words_and_exact_fragments() {
        let context =
            RolloverContext::new(600, 0, &[timed("turn", 10, 100), timed(" right", 200, 600)])
                .unwrap();
        let result = context
            .owned(Hypothesis {
                text: "Turn right, right  now unfinished".into(),
                stable: true,
                words: vec![
                    timed("Turn", 15, 105),
                    timed(" right", 210, 600),
                    timed(",", 600, 620),
                    timed(" right", 700, 900),
                    timed("  now", 920, 1_000),
                ],
                ..Hypothesis::default()
            })
            .unwrap()
            .unwrap();
        assert_eq!(result.text, "right  now");
        assert_eq!(
            result
                .words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<String>(),
            result.text
        );
        assert_eq!(
            (result.words[0].start_sample, result.words[1].end_sample),
            (700, 1_000)
        );
    }

    struct ContextRecognizer {
        conflict: bool,
    }

    struct ContextStream {
        conflict: bool,
        context: bool,
        received: usize,
    }

    impl Recognizer for ContextRecognizer {
        fn rollover_preroll_samples(&self) -> u64 {
            400
        }

        fn start(&self, _: &RecognitionOptions) -> Result<Box<dyn RecognitionStream + '_>> {
            Ok(Box::new(ContextStream {
                conflict: self.conflict,
                context: false,
                received: 0,
            }))
        }
    }

    impl RecognitionStream for ContextStream {
        fn push(&mut self, samples: &[f32]) -> Result<Option<Hypothesis>> {
            if self.received == 0 {
                self.context = samples[0] > 0.0;
            }
            self.received += samples.len();
            Ok(Some(self.hypothesis()))
        }

        fn finish(self: Box<Self>) -> Result<Hypothesis> {
            Ok(self.hypothesis())
        }
    }

    impl ContextStream {
        fn hypothesis(&self) -> Hypothesis {
            let words = if self.context {
                let mut words = vec![timed("old", 0, 100), timed(" lead", 120, 400)];
                if self.received > 500 {
                    words.push(timed(" next", if self.conflict { 390 } else { 400 }, 500));
                }
                words
            } else {
                vec![timed("next", 0, 100)]
            };
            Hypothesis {
                text: words.iter().map(|word| word.text.as_str()).collect(),
                stable: true,
                words,
                ..Hypothesis::default()
            }
        }
    }

    #[test]
    fn rollover_context_is_never_published_and_straddling_falls_back() {
        for conflict in [false, true] {
            let (output, mut results) = mpsc::channel(RESULT_DEPTH);
            let owner = TrackIdentity {
                id: 1,
                settings: TrackSettings {
                    native: true,
                    streaming: true,
                    options: RecognitionOptions::default(),
                },
                output,
                cancelled: Mutex::new(HashSet::new()),
                accepted: AtomicU64::new(0),
                queued_frames: Arc::new(AtomicUsize::new(0)),
                capacity: Arc::new(Semaphore::new(PER_TRACK_QUEUE_DEPTH)),
                closed: AtomicBool::new(false),
                closed_signal: Notify::new(),
            };
            let model = ContextRecognizer { conflict };
            let mut item = Item::new(2, 0);
            item.preroll =
                RolloverContext::new(600, 0, &[timed("old", 0, 150), timed(" lead", 180, 600)]);
            item.remember(AudioSpan {
                start: 0,
                samples: vec![1.0; 400],
            });

            item.push(
                AudioSpan {
                    start: 400,
                    samples: vec![-1.0; 100],
                },
                &model,
                &owner,
                true,
            )
            .unwrap();
            assert!(results.try_recv().is_err());
            item.push(
                AudioSpan {
                    start: 500,
                    samples: vec![-1.0; 100],
                },
                &model,
                &owner,
                true,
            )
            .unwrap();
            assert_eq!(item.preroll.is_none(), conflict);
            let event = results.try_recv().unwrap();
            assert_eq!((event.turn, event.start, event.end), (2, 600, 900));
            let PipelineUpdate::Draft(draft) = event.update else {
                panic!("expected draft")
            };
            assert_eq!(draft.text, "next");

            item.finish(&model, &owner);
            let event = results.try_recv().unwrap();
            assert_eq!((event.turn, event.start, event.end), (2, 600, 900));
            let PipelineUpdate::Final(finalized) = event.update else {
                panic!("expected final")
            };
            assert_eq!(finalized.text, "next");
            assert_eq!(finalized.words[0].start_sample, 600);
        }
    }

    #[test]
    fn rollover_gives_preceding_audio_to_only_the_new_decoder() {
        let (output, mut results) = mpsc::channel(RESULT_DEPTH);
        let owner = TrackIdentity {
            id: 1,
            settings: TrackSettings {
                native: true,
                streaming: true,
                options: RecognitionOptions::default(),
            },
            output,
            cancelled: Mutex::new(HashSet::new()),
            accepted: AtomicU64::new(0),
            queued_frames: Arc::new(AtomicUsize::new(0)),
            capacity: Arc::new(Semaphore::new(PER_TRACK_QUEUE_DEPTH)),
            closed: AtomicBool::new(false),
            closed_signal: Notify::new(),
        };
        let model = ContextRecognizer { conflict: false };
        let mut track = Track::new(&model, true, false).unwrap();
        let limit = MAX_NATIVE_FRAMES * 2 / 3;
        let cut = limit - 600;
        let context_start = cut - 400;
        track.new_native_item();
        let item = track.item.as_mut().unwrap();
        let mut samples = vec![1.0; limit as usize];
        samples[cut as usize..].fill(-1.0);
        item.remember(AudioSpan { start: 0, samples });
        item.stable = Some(Hypothesis {
            text: "old lead".into(),
            stable: true,
            words: vec![
                timed(
                    "old",
                    frame(0, context_start),
                    frame(0, context_start + 100),
                ),
                timed(" lead", frame(0, context_start + 120), frame(0, cut)),
            ],
            ..Hypothesis::default()
        });

        track
            .feed_native(
                &owner,
                AudioSpan {
                    start: limit,
                    samples: vec![-1.0; 100],
                },
            )
            .unwrap();
        let former = results.try_recv().unwrap();
        let next = results.try_recv().unwrap();
        assert_eq!(
            (former.turn, former.start, former.end),
            (1, 0, frame(0, cut))
        );
        assert!(matches!(former.update, PipelineUpdate::Final(h) if h.text == "old lead"));
        assert_eq!((next.turn, next.start), (2, frame(0, cut)));
        assert!(matches!(next.update, PipelineUpdate::Draft(h) if h.text == "next"));
        let active = track.item.as_ref().unwrap();
        assert_eq!(active.audio[0].samples.len(), 400);
        assert_eq!(active.first_owned_sample(), Some(cut));
        assert!(active.preroll.is_some());
    }

    #[test]
    fn rollover_rejects_a_lexical_match_with_conflicting_clock_anchors() {
        let context = RolloverContext::new(
            10_000,
            0,
            &[timed("old", 0, 2_000), timed(" lead", 2_400, 6_000)],
        )
        .unwrap();
        let ambiguous = Hypothesis {
            text: "old lead next".into(),
            stable: true,
            words: vec![
                timed("old", 3_000, 5_000),
                timed(" lead", 5_000, 8_000),
                timed(" next", 10_000, 12_000),
            ],
            ..Hypothesis::default()
        };
        assert!(context.owned(ambiguous).is_err());
    }

    #[test]
    fn native_rollovers_bound_items_and_cover_all_audio() {
        for word_timing in [true, false] {
            let model = CountingRecognizer {
                word_timing,
                decoded: AtomicUsize::new(0),
            };
            let mut track = Track::new(&model, word_timing, false).unwrap();
            let (output, mut results) = mpsc::channel(RESULT_DEPTH);
            let owner = TrackIdentity {
                id: 1,
                settings: TrackSettings {
                    native: true,
                    streaming: true,
                    options: RecognitionOptions::default(),
                },
                output,
                cancelled: Mutex::new(HashSet::new()),
                accepted: AtomicU64::new(0),
                queued_frames: Arc::new(AtomicUsize::new(0)),
                capacity: Arc::new(Semaphore::new(PER_TRACK_QUEUE_DEPTH)),
                closed: AtomicBool::new(false),
                closed_signal: Notify::new(),
            };
            let limit = MAX_NATIVE_FRAMES * 2 / 3;
            let sample_count = limit * 2 + 23_000;
            track.origin = u64::MAX - frame(0, sample_count);
            let mut finals = Vec::new();
            for start in (0..sample_count).step_by(WINDOW) {
                // A quieter stretch inside each reserved tail proves that an untimed decoder
                // cuts within it and still processes every sample only once.
                let level = if start % limit > limit - 8_000 && start % limit < limit - 6_000 {
                    0.01
                } else {
                    0.3
                };
                let length = (sample_count - start).min(WINDOW as u64) as usize;
                track
                    .feed_native(
                        &owner,
                        AudioSpan {
                            start,
                            samples: vec![level; length],
                        },
                    )
                    .unwrap();
                while let Ok(event) = results.try_recv() {
                    if matches!(event.update, PipelineUpdate::Final(_)) {
                        finals.push(event);
                    }
                }
            }
            track.complete(&owner);
            finals.push(results.try_recv().unwrap());
            assert_eq!(finals.len(), 3);
            let mut end = track.origin;
            for (index, event) in finals.into_iter().enumerate() {
                assert_eq!(event.turn, index as u64 + 1);
                assert_eq!(event.start, end);
                let duration = event.end - event.start;
                assert!(duration <= MAX_NATIVE_FRAMES);
                if word_timing && index < 2 {
                    assert!(
                        (MAX_NATIVE_FRAMES - TIMED_ROLLOVER_TAIL_FRAMES
                            ..=MAX_NATIVE_FRAMES - 48_000)
                            .contains(&duration)
                    );
                }
                let PipelineUpdate::Final(hypothesis) = event.update else {
                    panic!("expected final")
                };
                assert!(!hypothesis.text.is_empty());
                for word in hypothesis.words {
                    assert!(event.start <= word.start_sample && word.end_sample <= event.end);
                }
                end = event.end;
            }
            assert_eq!(end, u64::MAX);
            if !word_timing {
                assert_eq!(model.decoded.load(Ordering::Relaxed), sample_count as usize);
            }
        }
    }
}
