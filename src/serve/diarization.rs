//! Mixed-track speaker attribution from continuous, ungated PCM.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};
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
}

#[derive(Default)]
struct Timeline {
    spans: Vec<SpeakerSpan>,
    coverage: Vec<(u64, u64)>,
    finished: bool,
    failed: bool,
}

impl Timeline {
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

    fn attribute(&self, start: u64, end: u64, words: &[(u64, u64)]) -> Option<Attribution> {
        if start >= end
            || !self
                .coverage
                .iter()
                .any(|&(first, last)| start >= first && end <= last)
        {
            return None;
        }
        let mut speaker_spans: Vec<_> = self
            .spans
            .iter()
            .filter(|span| start < span.end_frame && end > span.start_frame)
            .map(|span| SpeakerSpan {
                speaker: span.speaker.clone(),
                start_frame: span.start_frame.max(start),
                end_frame: span.end_frame.min(end),
            })
            .collect();
        speaker_spans.sort_by(|lhs, rhs| {
            (lhs.start_frame, lhs.end_frame, &lhs.speaker).cmp(&(
                rhs.start_frame,
                rhs.end_frame,
                &rhs.speaker,
            ))
        });
        let words: Vec<_> = words
            .iter()
            .map(|&(first, last)| {
                let mut overlap = speaker_spans
                    .iter()
                    .filter(|span| first < span.end_frame && last > span.start_frame);
                let span = overlap.next()?;
                (first < last
                    && span.start_frame <= first
                    && span.end_frame >= last
                    && overlap.next().is_none())
                .then(|| span.speaker.clone())
            })
            .collect();
        let speaker = words
            .first()
            .and_then(Option::as_ref)
            .filter(|speaker| {
                words.iter().all(|word| word.as_ref() == Some(*speaker))
                    && speaker_spans.iter().all(|span| &span.speaker == *speaker)
            })
            .cloned();
        Some(Attribution {
            speaker,
            speaker_spans,
            words,
        })
    }
}

pub(super) struct Stream {
    timeline: Arc<Mutex<Timeline>>,
    worker: Arc<Worker>,
}

impl Stream {
    #[cfg(test)]
    pub(super) fn script_coverage(&self, interval: (u64, u64), spans: Vec<SpeakerSpan>) {
        let mut timeline = self.timeline.lock().unwrap();
        timeline.cover(interval.0, interval.1);
        for span in spans {
            timeline.push_span(span);
        }
    }

    /// A full queue leaves the entire packet with the session for a later retry.
    /// Once the diarizer fails, ASR can continue and use the unavailable-label fallback.
    pub(super) fn try_append(&self, start: u64, samples: &[i16]) -> bool {
        if samples.is_empty()
            || self.worker.ready.is_closed()
            || self.timeline.lock().unwrap().failed
        {
            return true;
        }
        if !self.worker.queue.lock().unwrap().try_append(start, samples) {
            return false;
        }
        self.notify();
        true
    }

    pub(super) fn finish(&self) {
        self.worker.queue.lock().unwrap().finish();
        self.notify();
    }

    pub(super) fn is_finished(&self) -> bool {
        let timeline = self.timeline.lock().unwrap();
        timeline.finished || timeline.failed || self.worker.ready.is_closed()
    }

    pub(super) fn attribution(
        &self,
        start: u64,
        end: u64,
        words: &[(u64, u64)],
    ) -> Option<Attribution> {
        self.timeline.lock().ok()?.attribute(start, end, words)
    }

    pub(super) fn report_timeout(&self, start: u64, end: u64) {
        if self.worker.diagnostics {
            let queued = self.worker.queue.lock().unwrap().frames;
            let timeline = self.timeline.lock().unwrap();
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
            self.timeline.lock().unwrap().failed = true;
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

    pub(super) fn reserve(self: &Arc<Self>) -> Option<Stream> {
        let mut queue = self.queue.lock().ok()?;
        if self.ready.is_closed() || queue.target.strong_count() > 0 {
            return None;
        }
        let timeline = Arc::new(Mutex::new(Timeline::default()));
        *queue = Queue {
            target: Arc::downgrade(&timeline),
            ..Queue::default()
        };
        Some(Stream {
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
        let pending = queue.lock().unwrap().pop();
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
        if target.lock().unwrap().failed {
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
            target.lock().unwrap().finished = true;
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
                    let decided_through =
                        target.lock().unwrap().coverage.last().map(|&(_, end)| end);
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
                target.lock().unwrap().failed = true;
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
        let mut timeline = target.lock().unwrap();
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
        assert!(timeline.attribute(170, 270, &[]).is_none());
        let recovered = timeline.attribute(255, 380, &[(300, 350)]).unwrap();
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
        assert!(timeline.attribute(90, 110, &[]).is_none());
        assert_eq!(
            timeline.attribute(100, 230, &[]).unwrap().speaker_spans,
            vec![span("speaker_2", 100, 150)]
        );
        assert!(timeline.attribute(120, 120, &[]).is_none());

        timeline.cover(1_000_000, 1_000_010);
        timeline.retain_last(100);
        assert_eq!(timeline.coverage, vec![(140, 230), (1_000_000, 1_000_010)]);
        assert!(timeline.attribute(140, 150, &[]).is_some());
    }

    #[test]
    fn labels_only_unambiguous_words_and_preserves_overlap() {
        let timeline = Timeline {
            spans: vec![span("speaker_1", 0, 200), span("speaker_2", 100, 300)],
            coverage: vec![(0, 300)],
            ..Timeline::default()
        };
        let result = timeline
            .attribute(10, 280, &[(10, 30), (80, 120), (200, 250)])
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
            timeline.attribute(20, 90, &[(20, 90)]).unwrap().speaker,
            Some("speaker_1".into())
        );
        assert_eq!(timeline.attribute(20, 90, &[]).unwrap().speaker, None);
        assert!(timeline.attribute(20, 400, &[(20, 40)]).is_none());
    }
}
