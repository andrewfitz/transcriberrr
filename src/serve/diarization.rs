//! Mixed-track speaker attribution from continuous, ungated PCM.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result};
use parakeet_rs::sortformer::{
    DiarizationConfig, NUM_SPEAKERS, Sortformer, SpeakerSegment, StreamingProfile,
};
use parakeet_rs::{ExecutionConfig, ExecutionProvider};
use serde::Serialize;
use tokio::sync::mpsc;

use super::pipeline::RateConverter;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(super) struct SpeakerSpan {
    pub speaker: String,
    pub start_frame: u64,
    pub end_frame: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Attribution {
    pub speaker: Option<String>,
    pub speaker_spans: Vec<SpeakerSpan>,
    pub words: Vec<Option<String>>,
    pub partial_coverage: Option<Vec<(u64, u64)>>,
}

#[derive(Default)]
struct Timeline {
    spans: Vec<SpeakerSpan>,
    coverage: Vec<(u64, u64)>,
    finished: bool,
    failed: bool,
}

impl Timeline {
    fn decided_past(&self, end: u64) -> bool {
        self.coverage.last().is_some_and(|&(_, last)| last >= end)
    }

    fn cover(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        match self.coverage.last_mut() {
            Some((_, previous_end)) if *previous_end == start => *previous_end = end,
            _ => self.coverage.push((start, end)),
        }
    }

    fn push_span(&mut self, span: SpeakerSpan) {
        match self
            .spans
            .iter_mut()
            .rev()
            .find(|previous| previous.speaker == span.speaker)
        {
            Some(previous) if previous.end_frame == span.start_frame => {
                previous.end_frame = span.end_frame;
            }
            _ => self.spans.push(span),
        }
    }

    fn retain_last(&mut self, mut frames: u64) {
        for &(start, end) in self.coverage.iter().rev() {
            if end - start >= frames {
                self.retain_since(end - frames);
                return;
            }
            frames -= end - start;
        }
    }

    fn retain_since(&mut self, oldest: u64) {
        self.spans.retain(|span| span.end_frame > oldest);
        for span in &mut self.spans {
            span.start_frame = span.start_frame.max(oldest);
        }
        self.coverage.retain(|&(_, end)| end > oldest);
        if let Some((start, _)) = self.coverage.first_mut() {
            *start = (*start).max(oldest);
        }
    }

    fn attribute(
        &self,
        start: u64,
        end: u64,
        words: &[(u64, u64)],
        allow_partial: bool,
    ) -> Option<Attribution> {
        if start >= end {
            return None;
        }
        let first = self.coverage.partition_point(|&(_, last)| last <= start);
        let intervals = &self.coverage[first..];
        let &(first_start, first_end) = intervals.first()?;
        if first_start >= end {
            return None;
        }
        let complete = first_start <= start && first_end >= end;
        if !complete && !allow_partial {
            return None;
        }
        let coverage: Vec<_> = intervals
            .iter()
            .take_while(|&&(a, _)| a < end)
            .map(|&(a, b)| (a.max(start), b.min(end)))
            .collect();
        let mut speaker_spans = Vec::new();
        for span in &self.spans {
            let first = coverage.partition_point(|&(_, b)| b <= span.start_frame);
            for &(a, b) in coverage[first..]
                .iter()
                .take_while(|&&(a, _)| a < span.end_frame)
            {
                speaker_spans.push(SpeakerSpan {
                    speaker: span.speaker.clone(),
                    start_frame: span.start_frame.max(a),
                    end_frame: span.end_frame.min(b),
                });
            }
        }
        speaker_spans.sort_by(|lhs, rhs| {
            (lhs.start_frame, lhs.end_frame, &lhs.speaker).cmp(&(
                rhs.start_frame,
                rhs.end_frame,
                &rhs.speaker,
            ))
        });
        let words: Vec<_> = words
            .iter()
            .map(|&word| word_speaker(word, &coverage, &speaker_spans))
            .collect();
        let speaker = words
            .first()
            .and_then(Option::as_ref)
            .filter(|speaker| {
                complete
                    && words.iter().all(|word| word.as_ref() == Some(*speaker))
                    && speaker_spans.iter().all(|span| &span.speaker == *speaker)
            })
            .cloned();
        Some(Attribution {
            speaker,
            speaker_spans,
            words,
            partial_coverage: (!complete).then_some(coverage),
        })
    }
}

/// A word in a pause between speaker spans takes the nearest span this close:
/// three 80 ms Sortformer output frames at the 24 kHz track rate.
const GAP_TOLERANCE: u64 = 3 * 1_920;

/// Label a word only when one speaker is in reach of it: every span the word
/// overlaps, or in a pause every span within `GAP_TOLERANCE`, names the same
/// speaker. A word two speakers touch stays unlabelled. Only decided audio names
/// a speaker, and a capture gap starts a new speaker namespace, so the word and
/// its candidate spans must lie in one coverage interval. A word without a
/// positive duration has no reliable timing.
fn word_speaker(
    (first, last): (u64, u64),
    coverage: &[(u64, u64)],
    spans: &[SpeakerSpan],
) -> Option<String> {
    if first >= last {
        return None;
    }
    let interval = coverage.partition_point(|&(_, end)| end <= first);
    let &(low, high) = coverage.get(interval)?;
    if first < low || last > high {
        return None;
    }
    // Attribution clips spans to coverage and sorts them by start. Spans
    // starting before this interval or beyond the tolerance cannot label it.
    let begin = spans.partition_point(|span| span.start_frame < low);
    let limit = last.saturating_add(GAP_TOLERANCE).min(high);
    let count = spans[begin..].partition_point(|span| span.start_frame <= limit);
    let candidates = || {
        spans[begin..begin + count]
            .iter()
            .filter(|span| span.end_frame <= high)
    };
    let distance = |span: &SpeakerSpan| {
        span.start_frame
            .saturating_sub(last)
            .max(first.saturating_sub(span.end_frame))
    };
    let overlaps = |span: &&SpeakerSpan| first < span.end_frame && last > span.start_frame;
    let spoken = candidates().any(|span| overlaps(&span));
    let mut near = candidates().filter(|span| {
        if spoken {
            overlaps(span)
        } else {
            distance(span) <= GAP_TOLERANCE
        }
    });
    let speaker = &near.next()?.speaker;
    near.all(|span| &span.speaker == speaker)
        .then(|| speaker.clone())
}

/// Recover a mutex a panicking thread left poisoned. Every critical section here
/// leaves its data consistent between statements, and a panic must not spread
/// from the diarizer to the session or from a session to the shared worker.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ReserveError {
    /// Another stream holds the single live diarizer.
    Busy,
    /// The worker thread exited or panicked; no stream can use it again.
    Stopped,
}

pub(super) struct Stream {
    timeline: Arc<Mutex<Timeline>>,
    worker: Arc<Worker>,
}

impl Drop for Stream {
    fn drop(&mut self) {
        // Release explicitly: the worker may still hold this timeline for an
        // in-flight job, which must not keep the diarizer reserved. Obsolete
        // queued PCM is discarded with the reservation.
        let mut queue = lock(&self.worker.queue);
        if queue.reserved && std::ptr::eq(queue.target.as_ptr(), Arc::as_ptr(&self.timeline)) {
            *queue = Queue::default();
        }
    }
}

impl Stream {
    /// Takes everything queued, as the worker does once it catches up.
    #[cfg(test)]
    pub(super) fn script_drain(&self) {
        let mut queue = lock(&self.worker.queue);
        while queue.pop().is_some() {}
    }

    #[cfg(test)]
    pub(super) fn script_coverage(&self, interval: (u64, u64), spans: Vec<SpeakerSpan>) {
        let mut timeline = self.timeline();
        timeline.cover(interval.0, interval.1);
        for span in spans {
            timeline.push_span(span);
        }
    }

    /// A poisoned timeline means the worker panicked mid-publish: fail the stream.
    fn timeline(&self) -> MutexGuard<'_, Timeline> {
        self.timeline.lock().unwrap_or_else(|poisoned| {
            let mut timeline = poisoned.into_inner();
            timeline.failed = true;
            timeline
        })
    }

    /// A full queue leaves the entire packet with the session for a later retry.
    /// Once the diarizer fails, ASR can continue and use the unavailable-label fallback.
    pub(super) fn try_append(&self, start: u64, samples: &[i16]) -> bool {
        if samples.is_empty() || self.worker.ready.is_closed() || self.timeline().failed {
            return true;
        }
        if !lock(&self.worker.queue).try_append(start, samples) {
            return false;
        }
        self.notify();
        true
    }

    pub(super) fn finish(&self) {
        lock(&self.worker.queue).finish();
        self.notify();
    }

    /// Whether the worker has taken everything queued. After the session drops audio
    /// for a diarizer that fell behind, it resumes here so that labelled stretches
    /// stay long instead of alternating with gaps.
    pub(super) fn caught_up(&self) -> bool {
        lock(&self.worker.queue).commands.is_empty()
    }

    /// Coverage only grows forward. Once it has passed `end`, a hole before `end` is
    /// audio the diarizer never received, and waiting cannot fill it.
    pub(super) fn decided_past(&self, end: u64) -> bool {
        self.timeline().decided_past(end)
    }

    pub(super) fn is_finished(&self) -> bool {
        let timeline = self.timeline();
        timeline.finished || timeline.failed || self.worker.ready.is_closed()
    }

    pub(super) fn attribution(
        &self,
        start: u64,
        end: u64,
        words: &[(u64, u64)],
        allow_partial: bool,
    ) -> Option<Attribution> {
        self.timeline
            .lock()
            .ok()?
            .attribute(start, end, words, allow_partial)
    }

    pub(super) fn report_timeout(&self, start: u64, end: u64) {
        if self.worker.diagnostics {
            let queued = lock(&self.worker.queue).frames;
            let timeline = self.timeline();
            let accepted_end = timeline.coverage.last().map(|&(_, end)| end);
            eprintln!(
                "live diarization deadline: start_frame={start} end_frame={end} decided_through={accepted_end:?} queued_frames={queued} failed={}",
                timeline.failed,
            );
        }
    }

    fn notify(&self) {
        // A single notification covers all queued commands. If the signal is
        // already pending, the worker will drain this append with the others.
        if let Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) =
            self.worker.ready.try_send(())
        {
            self.timeline().failed = true;
        }
    }
}

const RETAIN: u64 = 90 * 24_000;
const QUEUE_FRAMES: usize = 3 * 24_000;
const QUEUE_BATCHES: usize = 64;
const BATCH_FRAMES: usize = 24_000;
const MODEL_FRAME_SAMPLES: usize = 1_280;

enum Command {
    Append(u64, Vec<i16>),
    Finish,
}

#[derive(Default)]
struct Queue {
    /// Set while a `Stream` exists for `target`; cleared when it drops.
    reserved: bool,
    target: Weak<Mutex<Timeline>>,
    commands: VecDeque<Command>,
    frames: usize,
    finishing: bool,
}

impl Queue {
    fn try_append(&mut self, start: u64, samples: &[i16]) -> bool {
        if self.finishing || samples.len() > QUEUE_FRAMES - self.frames {
            return false;
        }
        if let Some(Command::Append(previous, pending)) = self.commands.back_mut()
            && previous.saturating_add(pending.len() as u64) == start
            && pending.len() + samples.len() <= BATCH_FRAMES
        {
            pending.extend_from_slice(samples);
        } else {
            // Each protocol packet is at most BATCH_FRAMES, including after
            // overlap removal by the track clock.
            if self.commands.len() >= QUEUE_BATCHES || samples.len() > BATCH_FRAMES {
                return false;
            }
            self.commands
                .push_back(Command::Append(start, samples.to_vec()));
        }
        self.frames += samples.len();
        true
    }

    fn finish(&mut self) {
        if !self.finishing {
            self.finishing = true;
            self.commands.push_back(Command::Finish);
        }
    }

    fn pop(&mut self) -> Option<(Weak<Mutex<Timeline>>, Command, usize)> {
        let command = self.commands.pop_front()?;
        if let Command::Append(_, samples) = &command {
            self.frames -= samples.len();
        }
        Some((self.target.clone(), command, self.frames))
    }
}

pub(super) struct Worker {
    ready: mpsc::Sender<()>,
    queue: Arc<Mutex<Queue>>,
    diagnostics: bool,
    #[cfg(test)]
    _scripted_receiver: Option<mpsc::Receiver<()>>,
}

impl Worker {
    #[cfg(test)]
    pub(super) fn scripted() -> Arc<Self> {
        let (ready, incoming) = mpsc::channel(1);
        Arc::new(Self {
            ready,
            queue: Arc::default(),
            diagnostics: false,
            _scripted_receiver: Some(incoming),
        })
    }

    /// A worker whose thread has exited.
    #[cfg(test)]
    pub(super) fn stopped() -> Arc<Self> {
        let worker = Self::scripted();
        Arc::new(Self {
            ready: worker.ready.clone(),
            queue: Arc::default(),
            diagnostics: false,
            _scripted_receiver: None,
        })
    }

    pub(super) fn start(path: PathBuf, device: crate::DeviceKind) -> Result<Arc<Self>> {
        let (ready, incoming) = mpsc::channel(1);
        let queue = Arc::new(Mutex::new(Queue::default()));
        let worker_queue = Arc::clone(&queue);
        let (startup_result, startup) = std::sync::mpsc::sync_channel(1);
        let diagnostics = std::env::var_os("TRANSCRIBERRR_DIARIZATION_DIAGNOSTICS").is_some();
        thread::Builder::new()
            .name("live-sortformer".into())
            .spawn(move || {
                let started = Instant::now();
                match open_model(path, device) {
                    Ok(model) => {
                        if diagnostics {
                            eprintln!(
                                "live diarization ready: startup_ms={}",
                                started.elapsed().as_millis()
                            );
                        }
                        if startup_result.send(Ok(())).is_ok() {
                            run(model, incoming, &worker_queue, diagnostics);
                        }
                    }
                    Err(error) => {
                        let _ = startup_result.send(Err(error));
                    }
                }
            })?;
        startup
            .recv()
            .context("live diarizer stopped during startup")??;
        Ok(Arc::new(Self {
            ready,
            queue,
            diagnostics,
            #[cfg(test)]
            _scripted_receiver: None,
        }))
    }

    /// False once the worker thread has exited or panicked. No stream can use it again.
    pub(super) fn is_running(&self) -> bool {
        !self.ready.is_closed()
    }

    #[cfg(test)]
    pub(super) fn reserve(self: &Arc<Self>) -> Option<Stream> {
        self.try_reserve().ok()
    }

    pub(super) fn try_reserve(self: &Arc<Self>) -> Result<Stream, ReserveError> {
        let mut queue = lock(&self.queue);
        if self.ready.is_closed() {
            return Err(ReserveError::Stopped);
        }
        if queue.reserved {
            return Err(ReserveError::Busy);
        }
        let timeline = Arc::new(Mutex::new(Timeline::default()));
        *queue = Queue {
            reserved: true,
            target: Arc::downgrade(&timeline),
            ..Queue::default()
        };
        Ok(Stream {
            timeline,
            worker: Arc::clone(self),
        })
    }
}

fn open_model(path: PathBuf, device: crate::DeviceKind) -> Result<Sortformer> {
    let provider = if matches!(device, crate::DeviceKind::Cpu) {
        ExecutionProvider::BurnCpu
    } else {
        ExecutionProvider::BurnWgpu
    };
    let config = ExecutionConfig::new().with_execution_provider(provider);
    let mut model = Sortformer::with_config(path, Some(config), DiarizationConfig::default())?;
    // Burn's CPU forward pass revisits the FIFO and speaker cache on every
    // window. A 1.44 s stride and bounded history keep later passes affordable.
    let profile = match device {
        crate::DeviceKind::Cpu => StreamingProfile {
            chunk_len: 18,
            right_context: 2,
            fifo_len: 36,
            spkcache_len: 96,
            spkcache_update_period: 18,
        },
        _ => StreamingProfile::low_latency(),
    };
    model.set_profile(profile)?;
    let mut probe = vec![0.; (model.chunk_len + model.right_context) * MODEL_FRAME_SAMPLES];
    for frame in probe.as_chunks_mut::<MODEL_FRAME_SAMPLES>().0 {
        frame[0] = 0.01;
    }
    model.feed(&probe)?;
    model.reset_state();
    Ok(model)
}

fn run(
    mut model: Sortformer,
    mut incoming: mpsc::Receiver<()>,
    queue: &Mutex<Queue>,
    diagnostics: bool,
) {
    let mut owner = Weak::new();
    let mut live: Option<Live> = None;
    loop {
        let pending = lock(queue).pop();
        let Some((pending, command, queued_frames)) = pending else {
            if incoming.blocking_recv().is_none() {
                break;
            }
            continue;
        };
        // The stream can end or disconnect while the worker is behind. Obsolete
        // commands no longer keep its reservation or trigger model work.
        let Some(target) = pending.upgrade() else {
            continue;
        };
        if !Weak::ptr_eq(&owner, &pending) {
            owner = pending;
            model.reset_state();
            live = None;
        }
        if lock(&target).failed {
            continue;
        }
        let started = Instant::now();
        let before = live.as_ref().map(|state| state.clock.decided);
        let result = match &command {
            Command::Append(start, audio) => (|| -> Result<()> {
                if let Some(state) = live.as_mut() {
                    if state.finished {
                        return Ok(());
                    }
                    if state.clock.next < *start {
                        // A capture gap clears the cache and starts a new speaker namespace.
                        let epoch = state.clock.epoch + NUM_SPEAKERS;
                        state.finish(&mut model, &target)?;
                        *state = Live::new(*start)?;
                        state.clock.epoch = epoch;
                        model.reset_state();
                    }
                } else {
                    live = Some(Live::new(*start)?);
                }
                let state = live.as_mut().unwrap();
                let skip = state
                    .clock
                    .next
                    .saturating_sub(*start)
                    .min(audio.len() as u64) as usize;
                state.append(&mut model, &audio[skip..], &target)
            })(),
            Command::Finish => match &mut live {
                Some(state) => state.finish(&mut model, &target),
                None => Ok(()),
            },
        };
        if matches!(command, Command::Finish) && result.is_ok() {
            lock(&target).finished = true;
        }
        match result {
            Ok(()) if diagnostics => {
                let after = live.as_ref().map(|state| state.clock.decided);
                if matches!(command, Command::Finish)
                    || before != after
                    || started.elapsed().as_millis() >= 50
                {
                    let phase = match command {
                        Command::Append(..) => "append",
                        Command::Finish => "finish",
                    };
                    let decided_through = lock(&target).coverage.last().map(|&(_, end)| end);
                    eprintln!(
                        "live diarization {phase}: elapsed_ms={} decided_through={decided_through:?} queued_frames={queued_frames}",
                        started.elapsed().as_millis(),
                    );
                }
            }
            Ok(()) => {}
            Err(error) => {
                if diagnostics {
                    eprintln!("live diarization worker error: {error:#}");
                }
                lock(&target).failed = true;
            }
        }
    }
}

struct Live {
    resampler: RateConverter,
    clock: ModelClock,
    finished: bool,
}

struct ModelClock {
    origin: u64,
    next: u64,
    epoch: usize,
    received: u64,
    decided: u64,
}

impl Live {
    fn new(origin: u64) -> Result<Self> {
        Ok(Self {
            clock: ModelClock {
                origin,
                next: origin,
                epoch: 0,
                received: 0,
                decided: 0,
            },
            finished: false,
            resampler: RateConverter::new()?,
        })
    }

    fn append(
        &mut self,
        model: &mut Sortformer,
        audio: &[i16],
        target: &Mutex<Timeline>,
    ) -> Result<()> {
        if audio.is_empty() {
            return Ok(());
        }
        self.clock.next = self.clock.next.saturating_add(audio.len() as u64);
        let converted = self.resampler.push(audio)?;
        self.clock.feed(model, converted, target)
    }

    fn finish(&mut self, model: &mut Sortformer, target: &Mutex<Timeline>) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        let converted = self.resampler.finish()?;
        self.clock.feed(model, converted, target)?;
        // The library's flush consumes the residual buffer, but its predictions
        // still exclude right context. Feed full windows to decide the actual tail.
        let step = model.chunk_len * MODEL_FRAME_SAMPLES;
        let right = model.right_context * MODEL_FRAME_SAMPLES;
        let padding = tail_padding(self.clock.received as usize, step, right);
        self.clock.feed(model, &vec![0.0; padding], target)?;
        // The input clock remains authoritative through the final sample even
        // when converting to 16 kHz rounds the resampled length down.
        self.clock.publish(target, Vec::new(), self.clock.next);
        self.finished = true;
        Ok(())
    }
}

impl ModelClock {
    fn feed(
        &mut self,
        model: &mut Sortformer,
        mut audio: &[f32],
        target: &Mutex<Timeline>,
    ) -> Result<()> {
        let step = model.chunk_len as u64 * MODEL_FRAME_SAMPLES as u64;
        let right = model.right_context as u64 * MODEL_FRAME_SAMPLES as u64;
        // One queued batch can trigger multiple costly windows. Make each
        // decision available to the session before starting the next inference.
        while !audio.is_empty() {
            let needed = self.decided + step + right - self.received;
            let (window, rest) = audio.split_at(audio.len().min(needed as usize));
            let spans = model.feed(window)?;
            self.received += window.len() as u64;
            if window.len() as u64 == needed {
                let decided = self.decided + step;
                self.publish(target, spans, self.frame(decided));
                self.decided = decided;
            }
            audio = rest;
        }
        Ok(())
    }

    fn frame(&self, position: u64) -> u64 {
        // Split the exact ratio to avoid overflowing an intermediate product.
        self.origin
            .saturating_add(position.saturating_add(position / 2))
            .min(self.next)
    }

    fn publish(&self, target: &Mutex<Timeline>, spans: Vec<SpeakerSegment>, decided: u64) {
        let mut timeline = lock(target);
        timeline.cover(self.frame(self.decided), decided);
        for segment in spans {
            let (start_frame, end_frame) = (self.frame(segment.start), self.frame(segment.end));
            if start_frame >= end_frame {
                continue;
            }
            let speaker = format!("speaker_{}", self.epoch + segment.speaker_id + 1);
            timeline.push_span(SpeakerSpan {
                speaker,
                start_frame,
                end_frame,
            });
        }
        timeline.retain_last(RETAIN);
    }
}

fn tail_padding(received: usize, step: usize, right: usize) -> usize {
    if received == 0 {
        return 0;
    }
    received.next_multiple_of(step) - received + right
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(speaker: &str, start_frame: u64, end_frame: u64) -> SpeakerSpan {
        SpeakerSpan {
            speaker: speaker.into(),
            start_frame,
            end_frame,
        }
    }

    #[test]
    fn tail_padding_decides_lookahead_as_audio_at_end() {
        assert_eq!(tail_padding(0, 11_520, 5_120), 0);
        for captured in [1, 11_519, 11_520, 11_521, 16_639, 16_640, 30_000] {
            let padded = captured + tail_padding(captured, 11_520, 5_120);
            let decided = (padded - 5_120) / 11_520 * 11_520;
            assert!(decided >= captured);
            assert!(decided - captured < 11_520);
        }
    }

    #[test]
    fn worker_batches_contiguous_packets_and_bounds_queued_audio() {
        let mut queue = Queue::default();
        for sample in 0..100 {
            assert!(queue.try_append(sample, &[1]));
        }
        assert_eq!(queue.commands.len(), 1);
        assert!(queue.try_append(100, &vec![2; BATCH_FRAMES - 100]));
        for start in (BATCH_FRAMES..QUEUE_FRAMES).step_by(BATCH_FRAMES) {
            let len = BATCH_FRAMES.min(QUEUE_FRAMES - 1 - start);
            assert!(queue.try_append(start as u64, &vec![2; len]));
        }
        assert!(!queue.try_append(QUEUE_FRAMES as u64 - 1, &[7, 8]));
        assert_eq!(queue.frames, QUEUE_FRAMES - 1);
        assert!(queue.try_append(QUEUE_FRAMES as u64 - 1, &[2]));
        assert_eq!(queue.frames, QUEUE_FRAMES);
        assert_eq!(queue.commands.len(), QUEUE_FRAMES / BATCH_FRAMES);
        assert!(!queue.try_append(QUEUE_FRAMES as u64, &[3]));

        let (_, Command::Append(0, first), _) = queue.pop().unwrap() else {
            panic!("expected first audio batch");
        };
        assert_eq!(first.len(), BATCH_FRAMES);
        assert_eq!(&first[..100], &[1; 100]);
        assert!(queue.try_append(QUEUE_FRAMES as u64, &[3]));
        queue.finish();
        queue.finish();

        for start in (BATCH_FRAMES..QUEUE_FRAMES).step_by(BATCH_FRAMES) {
            let (_, Command::Append(at, samples), queued_frames) = queue.pop().unwrap() else {
                panic!("PCM must drain before finish");
            };
            assert_eq!(at, start as u64);
            assert_eq!(samples.len(), BATCH_FRAMES);
            assert!(samples.iter().all(|&sample| sample == 2));
            assert_eq!(queued_frames, QUEUE_FRAMES - start - BATCH_FRAMES + 1);
        }
        assert!(
            matches!(queue.pop(), Some((_, Command::Append(start, retry), 0)) if start == QUEUE_FRAMES as u64 && retry == [3])
        );
        assert!(matches!(queue.pop(), Some((_, Command::Finish, 0))));
        assert!(queue.pop().is_none());
    }

    #[test]
    fn abandoned_pcm_does_not_hold_the_worker_reservation() {
        let worker = Worker::scripted();
        let original = worker.reserve().unwrap();
        assert!(original.try_append(0, &[1, 2, 3]));
        assert!(worker.reserve().is_none());
        drop(original);

        let replacement = worker.reserve().unwrap();
        assert!(replacement.try_append(40, &[4, 5]));
        let mut queue = worker.queue.lock().unwrap();
        let (target, command, _) = queue.pop().unwrap();
        assert!(Arc::ptr_eq(
            &target.upgrade().unwrap(),
            &replacement.timeline
        ));
        assert!(matches!(command, Command::Append(40, samples) if samples == [4, 5]));
        assert!(queue.pop().is_none());
    }

    #[test]
    fn released_reservation_is_free_while_the_worker_finishes_its_job() {
        let worker = Worker::scripted();
        let original = worker.reserve().unwrap();
        assert!(original.try_append(0, &[1, 2, 3]));
        // The worker upgrades the target and holds it for the whole inference.
        let (pending, _, _) = worker.queue.lock().unwrap().pop().unwrap();
        let in_flight = pending.upgrade().unwrap();
        drop(original);

        let replacement = worker.reserve().unwrap();
        assert!(worker.reserve().is_none());
        assert!(!Arc::ptr_eq(&in_flight, &replacement.timeline));
        drop(in_flight);
        drop(replacement);
        assert!(worker.reserve().is_some());
    }

    #[test]
    fn stopped_worker_is_not_reported_busy_and_finishes_open_streams() {
        let (ready, incoming) = mpsc::channel(1);
        let worker = Arc::new(Worker {
            ready,
            queue: Arc::default(),
            diagnostics: false,
            _scripted_receiver: None,
        });
        let stream = worker.try_reserve().unwrap();
        assert!(matches!(worker.try_reserve(), Err(ReserveError::Busy)));
        assert!(!stream.is_finished());
        // The worker thread owns the receiver; a panic drops it while unwinding.
        let dead = thread::spawn(move || {
            let _incoming = incoming;
            panic!("inference panicked");
        });
        assert!(dead.join().is_err());

        // Finals stop waiting for labels and ASR keeps its PCM flowing.
        assert!(stream.is_finished());
        assert!(stream.try_append(0, &[1, 2]));
        assert!(matches!(worker.try_reserve(), Err(ReserveError::Stopped)));
        drop(stream);
        assert!(matches!(worker.try_reserve(), Err(ReserveError::Stopped)));
        assert!(worker.reserve().is_none());
    }

    #[test]
    fn poisoned_timeline_fails_the_stream_without_panicking_the_session() {
        let worker = Worker::scripted();
        let stream = worker.reserve().unwrap();
        let timeline = Arc::clone(&stream.timeline);
        let poisoner = thread::spawn(move || {
            let _guard = timeline.lock().unwrap();
            panic!("worker panicked while publishing");
        });
        assert!(poisoner.join().is_err());
        assert!(stream.is_finished());
        assert!(stream.try_append(0, &[1]));
        stream.finish();
        stream.report_timeout(0, 1);
        assert!(stream.attribution(0, 1, &[], true).is_none());
        drop(stream);
        assert!(worker.reserve().is_some());
    }

    #[test]
    fn words_reaching_past_a_span_keep_its_speaker_unless_another_speaker_is_there() {
        let timeline = Timeline {
            spans: vec![span("speaker_1", 0, 500), span("speaker_2", 600, 1_000)],
            coverage: vec![(0, 1_000)],
            ..Timeline::default()
        };
        let result = timeline
            .attribute(
                0,
                1_000,
                &[(450, 560), (450, 650), (590, 700), (100, 100)],
                false,
            )
            .unwrap();
        assert_eq!(
            result.words,
            vec![
                // Ends in the pause after its only speaker.
                Some("speaker_1".into()),
                // Straddles the turn: two speakers touch it.
                None,
                Some("speaker_2".into()),
                // Zero-length timing stays unlabelled, as before.
                None,
            ]
        );

        // Simultaneous speakers leave a word unlabelled however the overlap splits.
        let overlap = Timeline {
            spans: vec![span("speaker_1", 0, 600), span("speaker_2", 400, 1_000)],
            coverage: vec![(0, 1_000)],
            ..Timeline::default()
        };
        let result = overlap
            .attribute(0, 1_000, &[(450, 550), (350, 550), (100, 300)], false)
            .unwrap();
        assert_eq!(result.words, vec![None, None, Some("speaker_1".into())]);
    }

    #[test]
    fn words_in_small_gaps_take_the_only_speaker_in_reach_within_one_coverage_interval() {
        let timeline = Timeline {
            spans: vec![
                span("speaker_1", 0, 24_000),
                span("speaker_2", 48_000, 72_000),
            ],
            coverage: vec![(0, 72_000)],
            ..Timeline::default()
        };
        let result = timeline
            .attribute(
                0,
                72_000,
                &[(25_000, 26_000), (34_000, 38_000), (44_000, 46_000)],
                false,
            )
            .unwrap();
        assert_eq!(
            result.words,
            vec![Some("speaker_1".into()), None, Some("speaker_2".into())]
        );

        // A short pause between two speakers belongs to neither.
        let handoff = Timeline {
            spans: vec![
                span("speaker_1", 0, 24_000),
                span("speaker_2", 28_000, 72_000),
            ],
            coverage: vec![(0, 72_000)],
            ..Timeline::default()
        };
        let result = handoff
            .attribute(0, 72_000, &[(25_000, 27_000)], false)
            .unwrap();
        assert_eq!(result.words, vec![None]);

        // A capture gap starts a new speaker namespace: never label across it.
        let gapped = Timeline {
            spans: vec![
                span("speaker_1", 0, 30_000),
                span("speaker_5", 40_000, 72_000),
            ],
            coverage: vec![(0, 30_000), (31_000, 72_000)],
            ..Timeline::default()
        };
        let result = gapped
            .attribute(
                0,
                72_000,
                &[(31_500, 32_000), (29_000, 32_000), (38_000, 39_000)],
                true,
            )
            .unwrap();
        assert_eq!(result.words, vec![None, None, Some("speaker_5".into())]);
    }

    #[test]
    fn joins_window_boundaries_and_preserves_discontinuities() {
        let mut timeline = Timeline::default();
        timeline.cover(0, 100);
        timeline.cover(100, 200);
        timeline.cover(250, 400);
        timeline.push_span(span("speaker_1", 0, 100));
        timeline.push_span(span("speaker_2", 70, 180));
        timeline.push_span(span("speaker_1", 100, 170));
        timeline.push_span(span("speaker_9", 250, 390));
        assert_eq!(timeline.spans[0], span("speaker_1", 0, 170));
        assert!(timeline.attribute(170, 270, &[], false).is_none());
        let recovered = timeline.attribute(255, 380, &[(300, 350)], false).unwrap();
        assert_eq!(recovered.speaker, Some("speaker_9".into()));
    }

    #[test]
    fn pruned_speaker_history_is_no_longer_reported_as_decided() {
        let mut timeline = Timeline::default();
        timeline.cover(0, 200);
        timeline.push_span(span("speaker_1", 10, 90));
        timeline.push_span(span("speaker_2", 80, 150));
        timeline.retain_last(100);
        timeline.cover(200, 230);
        assert_eq!(timeline.coverage, vec![(100, 230)]);
        assert!(timeline.attribute(90, 110, &[], false).is_none());
        assert_eq!(
            timeline
                .attribute(100, 230, &[], false)
                .unwrap()
                .speaker_spans,
            vec![span("speaker_2", 100, 150)]
        );
        assert!(timeline.attribute(120, 120, &[], false).is_none());

        timeline.cover(1_000_000, 1_000_010);
        timeline.retain_last(100);
        assert_eq!(timeline.coverage, vec![(140, 230), (1_000_000, 1_000_010)]);
        assert!(timeline.attribute(140, 150, &[], false).is_some());
    }

    #[test]
    fn a_hole_the_diarizer_has_passed_is_reported_without_waiting() {
        let mut timeline = Timeline::default();
        timeline.cover(0, 100);
        timeline.push_span(span("speaker_1", 0, 100));
        // The caption runs past everything decided so far: its labels may still come.
        assert!(!timeline.decided_past(250));
        // Audio from 100 to 200 never reached the diarizer, which has now decided
        // past the caption. Waiting out the deadline could not fill the hole.
        timeline.cover(200, 300);
        timeline.push_span(span("speaker_5", 200, 300));
        assert!(timeline.decided_past(250));
        let result = timeline.attribute(50, 250, &[(60, 90), (120, 150)], true);
        let result = result.unwrap();
        assert_eq!(result.partial_coverage, Some(vec![(50, 100), (200, 250)]));
        assert_eq!(result.words, vec![Some("speaker_1".into()), None]);
        assert_eq!(result.speaker, None);
        // A caption still ahead of the diarizer keeps waiting.
        assert!(!timeline.decided_past(350));
    }

    #[test]
    fn partial_coverage_does_not_bridge_gaps_or_label_crossing_words() {
        let timeline = Timeline {
            coverage: vec![(100, 150), (175, 230)],
            spans: vec![span("speaker_1", 100, 230), span("speaker_2", 190, 200)],
            ..Timeline::default()
        };
        let result = timeline
            .attribute(
                110,
                220,
                &[(120, 125), (140, 180), (185, 195), (201, 210)],
                true,
            )
            .unwrap();
        assert_eq!(result.partial_coverage, Some(vec![(110, 150), (175, 220)]));
        assert_eq!(result.speaker, None);
        assert_eq!(
            result.words,
            vec![
                Some("speaker_1".into()),
                None,
                None,
                Some("speaker_1".into())
            ]
        );
        assert_eq!(
            result.speaker_spans,
            vec![
                span("speaker_1", 110, 150),
                span("speaker_1", 175, 220),
                span("speaker_2", 190, 200)
            ]
        );
        assert!(timeline.attribute(155, 170, &[], true).is_none());
        assert!(timeline.attribute(110, 220, &[], false).is_none());
    }

    #[test]
    fn labels_only_unambiguous_words_and_preserves_overlap() {
        let timeline = Timeline {
            spans: vec![span("speaker_1", 0, 200), span("speaker_2", 100, 300)],
            coverage: vec![(0, 300)],
            ..Timeline::default()
        };
        let result = timeline
            .attribute(10, 280, &[(10, 30), (80, 120), (200, 250)], false)
            .unwrap();
        assert_eq!(result.speaker, None);
        assert_eq!(
            result.words,
            vec![Some("speaker_1".into()), None, Some("speaker_2".into())]
        );
        assert_eq!(
            result.speaker_spans,
            vec![span("speaker_1", 10, 200), span("speaker_2", 100, 280)]
        );
        assert_eq!(
            timeline
                .attribute(20, 90, &[(20, 90)], false)
                .unwrap()
                .speaker,
            Some("speaker_1".into())
        );
        assert_eq!(
            timeline.attribute(20, 90, &[], false).unwrap().speaker,
            None
        );
        assert!(timeline.attribute(20, 400, &[(20, 40)], false).is_none());
    }
}
