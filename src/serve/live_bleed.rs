//! Time-aligned bleed decisions on 20 ms blocks of 24 kHz PCM.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::Range;

use crate::bleed::{
    ACTIVE_DB, HOLD_FRAMES, HOLE_CONTEXT_FRAMES, HOLE_FRAMES, LOWEST_FLOOR_DB, LOWEST_THRESHOLD_DB,
    LevelMeter, MARGIN_DB, MIN_LEAD_FRAMES, RELEASE_FRAMES, RELEASE_RISE_DB, SILENCE_DB,
    TOLERANCE_DB, percentile,
};

const RATE: u32 = 24_000;
const BLOCK: u64 = (RATE / 50) as u64;
const TRAINING_BLOCKS: u64 = 30 * 50;
const RETAIN_BLOCKS: usize = 90 * 50;
// A mic is absent once another mic has delivered this much audio since it
// last delivered any. A late mic still delivers, so it stays a peer however
// far behind it runs; this only catches one that has stopped, at four times
// the final wait in `tracks.rs`.
const ABSENT_FRAMES: u64 = 5 * RATE as u64;
pub(super) const DRAFT_LOOKAHEAD: u64 = 2 * BLOCK;
pub(super) const FINAL_LOOKAHEAD: u64 =
    (HOLE_FRAMES + HOLE_CONTEXT_FRAMES + HOLD_FRAMES) as u64 * BLOCK;

#[derive(Debug, Clone, Copy)]
pub(super) enum Context {
    Draft,
    Final,
}

impl Context {
    fn index(self) -> usize {
        match self {
            Self::Draft => 0,
            Self::Final => 1,
        }
    }

    fn lookahead(self) -> u64 {
        match self {
            Self::Draft => DRAFT_LOOKAHEAD,
            Self::Final => FINAL_LOOKAHEAD,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Quality {
    pub uncalibrated: bool,
    pub missing_peer: bool,
}

impl Quality {
    pub(super) fn merge(&mut self, other: Self) {
        self.uncalibrated |= other.uncalibrated;
        self.missing_peer |= other.missing_peer;
    }
}

#[derive(Debug, Default)]
pub(super) struct Decision {
    pub muted: bool,
    pub quality: Quality,
}

#[derive(Clone, Copy)]
struct Block {
    observation: Observation,
    muted: bool,
    quality: Quality,
}

#[derive(Clone, Copy)]
struct Observation {
    start: u64,
    end: u64,
    level: f32,
    active: bool,
}

impl Observation {
    fn intersection(self, start: u64, end: u64) -> Range<u64> {
        let first = self.start.max(start);
        first..self.end.min(end).max(first)
    }
}

impl Block {
    fn speaking(self) -> bool {
        self.observation.active && !self.muted
    }
}

#[derive(Default)]
struct History {
    observations: VecDeque<(u64, f32)>,
    estimate: Option<(u64, f32)>,
    // A pair's estimates by the block each was measured at, from the one in
    // force at its first unclassified block. A final's calibration runs
    // ahead of its classification.
    estimates: VecDeque<(u64, f32)>,
}

impl History {
    // Whether this observation produced a new estimate.
    fn observe(&mut self, block: u64, value: f32, quantile: f32, values: &mut Vec<f32>) -> bool {
        self.observations.push_back((block, value));
        let update = self.observations.len() >= MIN_LEAD_FRAMES
            && self
                .estimate
                .is_none_or(|(previous, _)| block >= previous + 10);
        if update {
            values.clear();
            values.extend(self.observations.iter().map(|&(_, value)| value));
            self.estimate = Some((block, percentile(values, quantile)));
        }
        update
    }

    // The estimate measured before `block`. Without one, a final takes the
    // first one measured after it, from lead blocks that include this one.
    fn estimate_for(&self, block: u64, context: Context) -> Option<f32> {
        let preceding = self.estimates.partition_point(|&(at, _)| at < block);
        preceding
            .checked_sub(1)
            .and_then(|index| self.estimates.get(index))
            .or(match context {
                Context::Draft => None,
                Context::Final => self.estimates.front(),
            })
            .map(|&(_, estimate)| estimate)
    }

    // Blocks through `block` are classified: drop estimates no later block
    // can use.
    fn classified(&mut self, block: u64) {
        while self.estimates.get(1).is_some_and(|&(at, _)| at <= block) {
            self.estimates.pop_front();
        }
    }

    fn expire(&mut self, block: u64) {
        while self
            .observations
            .front()
            .is_some_and(|&(at, _)| at + TRAINING_BLOCKS <= block)
        {
            self.observations.pop_front();
        }
        // The last estimate outlives its observations. A speaker who was quiet
        // for the whole window keeps the measured threshold until a fresh one
        // replaces it, so their next words do not pass through as uncalibrated.
    }
}

#[derive(Default)]
struct Analysis {
    blocks: BTreeMap<u64, Block>,
    pairs: HashMap<u64, History>,
    release: Option<(u64, f32, f32)>,
    processed: Option<u64>,
    calibrated: Option<u64>,
}

struct Mic {
    start: u64,
    next: u64,
    end: Option<u64>,
    block_start: u64,
    levels: BTreeMap<u64, Option<Observation>>,
    floor: History,
    meter: LevelMeter,
    pruned_before: u64,
    // Frames of audio accepted from this mic, and each other mic's count when
    // this one last delivered audio or registered.
    received: u64,
    heard: HashMap<u64, u64>,
}

/// How a mic stands as a peer of the others.
#[derive(Clone, Copy, PartialEq)]
enum Presence {
    /// Delivering audio, however late. Waited for.
    Present,
    /// Delivered audio once, then stopped. Not waited for, but its missing
    /// blocks still count as missing: its audio may yet arrive.
    Stopped,
    /// Has sent no audio. It cannot have been the source of any bleed, so
    /// the others are decided and calibrated as if it were not there.
    Silent,
}

impl Mic {
    fn new(start: u64, heard: HashMap<u64, u64>) -> Self {
        Self {
            start,
            next: start,
            end: None,
            block_start: start,
            levels: BTreeMap::new(),
            floor: History::default(),
            meter: LevelMeter::new(RATE),
            pruned_before: 0,
            received: 0,
            heard,
        }
    }

    fn presence(&self, key: u64, mics: &HashMap<u64, Mic>) -> Presence {
        let absent = self.end.is_none()
            && mics.iter().any(|(&other, mic)| {
                other != key
                    && mic.received
                        >= self
                            .heard
                            .get(&other)
                            .map_or(0, |&heard| heard)
                            .saturating_add(ABSENT_FRAMES)
            });
        match (self.received, absent) {
            // A mic that ended before sending audio never will.
            (0, _) if absent || self.end.is_some() => Presence::Silent,
            (_, true) => Presence::Stopped,
            _ => Presence::Present,
        }
    }

    // Whether this mic, as a peer, may have audio in `start..end` to compare.
    fn involved(&self, presence: Presence, start: u64, end: u64) -> bool {
        self.active_during(start, end)
            && match presence {
                Presence::Present => true,
                Presence::Stopped => self.next > start,
                Presence::Silent => false,
            }
    }

    fn finish_block(&mut self, values: &mut Vec<f32>) {
        if let Some(level) = self.meter.finish_frame() {
            let block = self.block_start / BLOCK;
            self.floor.expire(block);
            let floor = self
                .floor
                .estimate
                .map_or(LOWEST_FLOOR_DB, |(_, estimate)| estimate)
                .max(LOWEST_FLOOR_DB);
            let observation = Observation {
                start: self.block_start,
                end: self.next,
                level,
                active: level > floor + ACTIVE_DB,
            };
            if self.next - self.block_start == BLOCK {
                self.floor.observe(block, level, 0.1, values);
            }
            self.levels
                .entry(block)
                // A gap can leave two disjoint fragments in one block.
                .and_modify(|previous| *previous = None)
                .or_insert(Some(observation));
            self.block_start = self.next;
        }
    }

    fn gap(&mut self, end: u64, values: &mut Vec<f32>) {
        if end > self.next {
            self.finish_block(values);
            self.next = end;
            self.block_start = end;
            self.meter = LevelMeter::new(RATE);
        }
    }

    fn active_during(&self, start: u64, end: u64) -> bool {
        self.start < end
            && self
                .end
                .is_none_or(|stop| stop > start && stop > self.start)
    }

    fn ready_at(&self, horizon: u64) -> bool {
        self.end.is_some() || self.next / BLOCK >= horizon.div_ceil(BLOCK)
    }
}

#[derive(Default)]
pub(super) struct LiveBleed {
    mics: HashMap<u64, Mic>,
    observed: Vec<(u64, Observation)>,
    presences: Vec<(u64, Presence)>,
    percentile_values: Vec<f32>,
    // Each mic reaches its deadlines independently. A final can also use
    // peer audio that arrived after the draft deadline.
    analysis: HashMap<u64, [Analysis; 2]>,
}

impl LiveBleed {
    pub(super) fn register(&mut self, track: u64, start_frame: u64) {
        if !self.mics.contains_key(&track) {
            let heard = self
                .mics
                .iter()
                .map(|(&other, mic)| (other, mic.received))
                .collect();
            self.mics.insert(track, Mic::new(start_frame, heard));
        }
        self.analysis.entry(track).or_default();
    }

    pub(super) fn end(&mut self, track: u64) {
        if let Some(mic) = self.mics.get_mut(&track) {
            mic.finish_block(&mut self.percentile_values);
            mic.end = Some(mic.next);
        }
    }

    pub(super) fn remove(&mut self, track: u64) {
        self.mics.remove(&track);
        self.analysis.remove(&track);
        for mic in self.mics.values_mut() {
            mic.heard.remove(&track);
        }
        for analyses in self.analysis.values_mut() {
            for analysis in analyses {
                analysis.pairs.remove(&track);
            }
        }
    }

    pub(super) fn gap(&mut self, track: u64, end: u64) {
        if let Some(mic) = self.mics.get_mut(&track) {
            mic.gap(end, &mut self.percentile_values);
        }
    }

    pub(super) fn ingest(&mut self, track: u64, start_frame: u64, samples: &[i16]) {
        let Some(mic) = self.mics.get_mut(&track) else {
            return;
        };
        mic.gap(start_frame, &mut self.percentile_values);
        let skip = mic
            .next
            .saturating_sub(start_frame)
            .min(samples.len() as u64) as usize;
        let mut remaining = &samples[skip..];
        let delivered = !remaining.is_empty();
        if delivered && mic.received == 0 {
            // A mic joins at its first audio. Before that it sent nothing,
            // so it was no one's source of bleed, even once it starts.
            mic.start = mic.start.max(mic.next);
        }
        mic.received += remaining.len() as u64;
        while !remaining.is_empty() {
            let until_boundary = (BLOCK - mic.next % BLOCK) as usize;
            let (chunk, rest) = remaining.split_at(remaining.len().min(until_boundary));
            for &sample in chunk {
                mic.meter.push(f32::from(sample) / 32768.0);
            }
            mic.next += chunk.len() as u64;
            if chunk.len() == until_boundary {
                mic.finish_block(&mut self.percentile_values);
            }
            remaining = rest;
        }
        // Bound the number of observations. A jump in the capture clock
        // cannot erase pending pre-gap decisions.
        let excess = mic.levels.len().saturating_sub(RETAIN_BLOCKS);
        if excess >= 50 {
            let before = *mic.levels.keys().nth(excess).expect("first retained block");
            mic.pruned_before = before;
            mic.levels = mic.levels.split_off(&before);
            for analysis in self.analysis.get_mut(&track).expect("registered track") {
                analysis.blocks = analysis.blocks.split_off(&before);
            }
        }
        if delivered {
            let mut heard = std::mem::take(&mut mic.heard);
            for (&other, peer) in &self.mics {
                if other != track {
                    heard.insert(other, peer.received);
                }
            }
            self.mics.get_mut(&track).expect("registered track").heard = heard;
        }
    }

    pub(super) fn ready_through(&self, track: u64, through: u64, lookahead: u64) -> bool {
        let mut horizon = through.saturating_add(lookahead);
        if let Some(end) = self.mics.get(&track).and_then(|mic| mic.end) {
            horizon = horizon.min(end);
        }
        let peer_overlaps_context = horizon > through
            && self.mics.iter().any(|(&peer, mic)| {
                track != peer && mic.involved(mic.presence(peer, &self.mics), through, horizon)
            });
        let own_ready = !peer_overlaps_context
            || self
                .mics
                .get(&track)
                .is_none_or(|mic| mic.ready_at(horizon));
        own_ready && !self.awaiting_own_block(track, 0, through) && self.peers_ready(track, horizon)
    }

    // A peer can end exactly at the requested endpoint while our open
    // capture block still needs more PCM (or track end) to compare levels.
    fn awaiting_own_block(&self, track: u64, start: u64, end: u64) -> bool {
        let Some(own) = self.mics.get(&track) else {
            return false;
        };
        if own.ready_at(end) {
            return false;
        }
        let first = start.max(own.start).max(own.next - own.next % BLOCK);
        first < end
            && self.mics.iter().any(|(&peer, mic)| {
                peer != track && mic.involved(mic.presence(peer, &self.mics), first, end)
            })
    }

    fn peers_ready(&self, track: u64, horizon: u64) -> bool {
        self.mics.iter().all(|(&peer, mic)| {
            track == peer
                || mic.start >= horizon
                || mic.ready_at(horizon)
                || mic.presence(peer, &self.mics) != Presence::Present
        })
    }

    fn horizon(&self, track: u64, through: u64, context: Context) -> u64 {
        let limit = through.saturating_add(context.lookahead());
        self.mics
            .get(&track)
            .map_or(through, |mic| mic.next.min(limit))
    }

    fn advance(&mut self, track: u64, required: u64, through: u64, context: Context) {
        let Some(mic) = self.mics.get(&track) else {
            return;
        };
        let analysis =
            &mut self.analysis.get_mut(&track).expect("registered track")[context.index()];
        // A partial observation left by a gap within this block can still
        // acquire a disjoint fragment. Only freeze closed capture blocks.
        let closed = if mic.end.is_none() {
            mic.next / BLOCK
        } else {
            mic.next.div_ceil(BLOCK)
        };
        let end = through.div_ceil(BLOCK).min(closed);
        // A final calibrates on every closed block delivered so far, so a
        // pair's first estimate can decide the lead blocks it comes from.
        let calibrate_end = match context {
            Context::Draft => end,
            Context::Final => closed,
        };
        let first = analysis.processed.map_or(0, |last| last + 1);
        let first_calibration = analysis.calibrated.map_or(0, |last| last + 1).max(first);
        if first >= end && first_calibration >= calibrate_end {
            return;
        }
        let observed = &mut self.observed;
        let presences = &mut self.presences;
        presences.clear();
        presences.extend(
            self.mics
                .iter()
                .filter(|&(&peer, _)| peer != track)
                .map(|(&peer, mic)| (peer, mic.presence(peer, &self.mics))),
        );
        for (&block, &own) in mic.levels.range(first_calibration..calibrate_end) {
            let Some(own) = own else {
                continue;
            };
            let (missing_peer, waiting_for_peer) =
                Self::compare(&self.mics, presences, block, own, observed);
            if waiting_for_peer && own.start >= required {
                break;
            }
            Self::calibrate(
                analysis,
                block,
                own,
                observed,
                missing_peer,
                &mut self.percentile_values,
            );
            analysis.calibrated = Some(block);
        }
        let end = end.min(analysis.calibrated.map_or(0, |last| last + 1));
        for (&block, &own) in mic.levels.range(first..end.max(first)) {
            let Some(own) = own else {
                continue;
            };
            let (missing_peer, waiting_for_peer) =
                Self::compare(&self.mics, presences, block, own, observed);
            // A deadline can consume its interval without a peer. Keep the
            // undecided lookahead available for when that peer catches up.
            if waiting_for_peer && own.start >= required {
                break;
            }
            // Leave a final's uncalibrated lookahead undecided: the pair's
            // first estimate may still come before a final needs it.
            if matches!(context, Context::Final)
                && own.start >= required
                && !missing_peer
                && Self::uncalibrated(analysis, block, own, observed)
            {
                break;
            }
            Self::classify(analysis, block, own, observed, missing_peer, context);
            analysis.processed = Some(block);
        }
    }

    // The peers' observations of an own block, whether one is missing, and
    // whether a present peer may still deliver it.
    fn compare(
        mics: &HashMap<u64, Mic>,
        presences: &[(u64, Presence)],
        block: u64,
        own: Observation,
        observed: &mut Vec<(u64, Observation)>,
    ) -> (bool, bool) {
        observed.clear();
        let mut missing_peer = false;
        let mut waiting_for_peer = false;
        for &(peer, presence) in presences {
            let mic = &mics[&peer];
            if !mic.active_during(own.start, own.end) {
                continue;
            }
            match mic.levels.get(&block).copied().flatten() {
                Some(other)
                    if other.start == own.start
                        && other.end == own.end
                        && mic.ready_at(own.end) =>
                {
                    observed.push((peer, other));
                }
                // Not silence at that mic, which says nothing about the
                // room, but no source of bleed and no rival either.
                _ if presence == Presence::Silent => {}
                _ => {
                    missing_peer = true;
                    waiting_for_peer |= presence == Presence::Present && !mic.ready_at(own.end);
                }
            }
        }
        (missing_peer, waiting_for_peer)
    }

    // Measure the bleed of a peer that leads this block.
    fn calibrate(
        analysis: &mut Analysis,
        block: u64,
        own: Observation,
        peers: &[(u64, Observation)],
        missing_peer: bool,
        values: &mut Vec<f32>,
    ) {
        let level = own.level;
        if missing_peer || level <= SILENCE_DB {
            return;
        }
        // A single peer can lead by the calibration margin. Include inactive
        // rivals in that comparison just as for the offline bleed estimator.
        let leader = peers
            .iter()
            .max_by(|a, b| a.1.level.total_cmp(&b.1.level))
            .filter(|&&(key, loudest)| {
                loudest.active
                    && loudest.level - level >= MARGIN_DB
                    && peers.iter().all(|&(peer, rival)| {
                        peer == key || loudest.level >= rival.level + MARGIN_DB
                    })
            });
        if let Some(&(peer, other)) = leader {
            let pair = analysis.pairs.entry(peer).or_default();
            pair.expire(block);
            if pair.observe(block, other.level - level, 0.5, values) {
                let estimate = pair.estimate.expect("new estimate");
                pair.estimates.push_back(estimate);
            }
        }
    }

    // Whether a peer leads this block by the margin without any estimate.
    fn uncalibrated(
        analysis: &Analysis,
        block: u64,
        own: Observation,
        peers: &[(u64, Observation)],
    ) -> bool {
        own.level > SILENCE_DB
            && peers.iter().any(|&(peer, other)| {
                other.active
                    && other.level - own.level >= MARGIN_DB
                    && analysis
                        .pairs
                        .get(&peer)
                        .and_then(|pair| pair.estimate_for(block, Context::Final))
                        .is_none()
            })
    }

    fn classify(
        analysis: &mut Analysis,
        block: u64,
        own: Observation,
        peers: &[(u64, Observation)],
        missing_peer: bool,
        context: Context,
    ) {
        let level = own.level;
        let mut quality = Quality {
            missing_peer,
            ..Quality::default()
        };
        let mut muted = false;
        for &(peer, other) in peers {
            let difference = other.level - level;
            if !other.active || level <= SILENCE_DB || difference <= 0.0 {
                continue;
            }
            match analysis
                .pairs
                .get(&peer)
                .and_then(|pair| pair.estimate_for(block, context))
            {
                Some(estimate) => {
                    muted |= difference >= (estimate - TOLERANCE_DB).max(LOWEST_THRESHOLD_DB);
                }
                None if difference >= MARGIN_DB => quality.uncalibrated = true,
                None => {}
            }
        }
        for pair in analysis.pairs.values_mut() {
            pair.classified(block);
        }
        if missing_peer {
            analysis.release = None;
        } else if muted {
            analysis.release = Some((block, level, level));
        } else if let Some((start, initial, previous)) = analysis.release.take() {
            muted = analysis
                .blocks
                .last_key_value()
                .is_some_and(|(_, last)| last.observation.end == own.start)
                && block - start <= RELEASE_FRAMES as u64
                && level <= initial
                && level <= previous + RELEASE_RISE_DB;
            if muted {
                analysis.release = Some((start, initial, level));
            }
        }
        analysis.blocks.insert(
            block,
            Block {
                observation: own,
                muted,
                quality,
            },
        );
    }

    fn context(track: &Analysis, at: u64, block: Block, context: Context) -> bool {
        if matches!(context, Context::Draft) {
            return block.muted
                && !track
                    .blocks
                    .range(at.saturating_sub(HOLD_FRAMES as u64)..=at + DRAFT_LOOKAHEAD / BLOCK)
                    .any(|(_, adjacent)| adjacent.speaking());
        }
        // As offline: holes fill first, then the hold keeps every muted block
        // next to kept own speech, filled holes included.
        (block.muted || Self::fills_hole(track, at))
            && !track
                .blocks
                .range(at.saturating_sub(HOLD_FRAMES as u64)..=at + HOLD_FRAMES as u64)
                .any(|(&position, adjacent)| {
                    adjacent.speaking() && !Self::fills_hole(track, position)
                })
    }

    fn fills_hole(track: &Analysis, at: u64) -> bool {
        let left = track
            .blocks
            .range(at.saturating_sub(HOLE_FRAMES as u64)..at)
            .rev()
            .find(|(_, adjacent)| adjacent.muted);
        let right = track
            .blocks
            .range(at + 1..=at + HOLE_FRAMES as u64)
            .find(|(_, adjacent)| adjacent.muted);
        let (Some((&left, first)), Some((&right, last))) = (left, right) else {
            return false;
        };
        if right - left - 1 > HOLE_FRAMES as u64
            || first.quality.missing_peer
            || last.quality.missing_peer
        {
            return false;
        }
        let hole = track.blocks.range(left + 1..right);
        let mut through = first.observation.end;
        if !hole.clone().all(|(_, adjacent)| {
            let contiguous = adjacent.observation.start == through;
            through = adjacent.observation.end;
            contiguous && !adjacent.quality.missing_peer
        }) || through != last.observation.start
        {
            return false;
        }
        let old_peak = track
            .blocks
            .range(left.saturating_sub(HOLE_CONTEXT_FRAMES as u64 - 1)..=left);
        let new_peak = track
            .blocks
            .range(right..right + HOLE_CONTEXT_FRAMES as u64);
        let bleed = old_peak
            .chain(new_peak)
            .filter(|(_, adjacent)| adjacent.muted && !adjacent.quality.missing_peer)
            .map(|(_, adjacent)| adjacent.observation.level)
            .fold(SILENCE_DB, f32::max);
        hole.map(|(_, adjacent)| adjacent.observation.level)
            .all(|level| level <= bleed)
    }

    pub(super) fn decision(
        &mut self,
        track: u64,
        start: u64,
        end: u64,
        context: Context,
    ) -> Decision {
        let mut result = Decision::default();
        if end <= start {
            return result;
        }
        let horizon = self.horizon(track, end, context);
        self.advance(track, end, horizon, context);
        let Some(mic) = self.mics.get(&track) else {
            return result;
        };
        let interval = start / BLOCK..end.div_ceil(BLOCK);
        let analysis = &self.analysis[&track][context.index()];
        result.quality.missing_peer = interval.start < mic.pruned_before
            || !self.peers_ready(track, horizon)
            || self.awaiting_own_block(track, start, end)
            || mic
                .levels
                .range(interval.clone())
                .any(|(at, level)| level.is_none() || !analysis.blocks.contains_key(at));
        let mut muted = 0;
        for (&at, &block) in analysis.blocks.range(interval) {
            let covered = block.observation.intersection(start, end);
            if covered.is_empty() {
                continue;
            }
            result.quality.merge(block.quality);
            if Self::context(analysis, at, block, context) {
                muted += covered.end - covered.start;
            }
        }
        result.muted = muted > (end - start) / 2;
        result
    }

    pub(super) fn apply(
        &mut self,
        track: u64,
        start: u64,
        samples: &mut [i16],
        context: Context,
    ) -> Quality {
        if samples.is_empty() {
            return Quality::default();
        }
        let end = start.saturating_add(samples.len() as u64);
        let horizon = self.horizon(track, end, context);
        self.advance(track, end, horizon, context);
        let mut quality = Quality {
            missing_peer: !self.peers_ready(track, horizon)
                || self.awaiting_own_block(track, start, end),
            ..Quality::default()
        };
        let Some(mic) = self.mics.get(&track) else {
            return quality;
        };
        let interval = start / BLOCK..end.div_ceil(BLOCK);
        let analysis = &self.analysis[&track][context.index()];
        quality.missing_peer |= interval.start < mic.pruned_before
            || mic
                .levels
                .range(interval.clone())
                .any(|(at, level)| level.is_none() || !analysis.blocks.contains_key(at));
        for (&at, &block) in analysis.blocks.range(interval) {
            let covered = block.observation.intersection(start, end);
            if covered.is_empty() {
                continue;
            }
            quality.merge(block.quality);
            if Self::context(analysis, at, block, context) {
                let first = (covered.start - start) as usize;
                let last = (covered.end - start) as usize;
                samples[first..last].fill(0);
            }
        }
        quality
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(amplitude: f32, block: u64) -> Vec<i16> {
        (0..BLOCK)
            .map(|sample| {
                let phase =
                    (block * BLOCK + sample) as f32 * std::f32::consts::TAU * 200.0 / 24_000.0;
                (amplitude * phase.sin()) as i16
            })
            .collect()
    }

    fn input(gate: &mut LiveBleed, block: u64, a: f32, b: Option<f32>) {
        gate.ingest(0, block * BLOCK, &tone(a, block));
        if let Some(b) = b {
            for (index, packet) in tone(b, block).chunks(97).enumerate() {
                gate.ingest(1, block * BLOCK + (97 * index) as u64, packet);
            }
        }
    }

    #[test]
    fn calibration_preserves_initial_audio_then_separates_bleed_from_overlap() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);

        for block in 0..40 {
            input(&mut gate, block, 0., Some(0.));
        }
        for block in 40..145 {
            input(&mut gate, block, 10_000.0, Some(1_000.0));
        }
        for block in 145..220 {
            input(&mut gate, block, 10_000.0, Some(7_000.0));
        }

        let initial = gate.decision(1, 45 * BLOCK, 50 * BLOCK, Context::Draft);
        assert!(initial.quality.uncalibrated);
        assert!(!initial.muted);

        let mut weak = vec![1000; 10 * BLOCK as usize];
        let quality = gate.apply(1, 115 * BLOCK, &mut weak, Context::Final);
        assert!(!quality.uncalibrated && !quality.missing_peer);
        assert!(weak.iter().all(|&sample| sample == 0));

        let simultaneous = gate.decision(1, 160 * BLOCK, 165 * BLOCK, Context::Final);
        assert!(!simultaneous.muted);
    }

    #[test]
    fn final_uses_peer_audio_that_missed_the_draft_deadline() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        for block in 0..40 {
            input(&mut gate, block, 0., Some(0.));
        }
        for block in 40..119 {
            input(&mut gate, block, 10_000.0, Some(1_000.0));
        }

        input(&mut gate, 119, 10_000.0, Some(600.0));

        gate.ingest(1, 120 * BLOCK, &tone(1_000.0, 120));
        let draft = gate.decision(1, 120 * BLOCK, 121 * BLOCK, Context::Draft);
        assert!(!draft.muted && draft.quality.missing_peer);

        for block in 120..175 {
            input(&mut gate, block, 10_000.0, Some(1_000.0));
        }
        let final_result = gate.decision(1, 120 * BLOCK, 121 * BLOCK, Context::Final);
        assert!(final_result.muted && !final_result.quality.missing_peer);
    }

    #[test]
    fn only_full_context_can_fill_a_bleed_hole() {
        let mut mic = Analysis::default();
        for at in 0..30 {
            mic.blocks.insert(
                at,
                Block {
                    observation: Observation {
                        start: at * BLOCK,
                        end: (at + 1) * BLOCK,
                        level: if (10..18).contains(&at) { -35. } else { -30. },
                        active: true,
                    },
                    muted: !(10..18).contains(&at),
                    quality: Quality::default(),
                },
            );
        }
        let center = mic.blocks[&14];
        assert!(!LiveBleed::context(&mic, 14, center, Context::Draft));
        assert!(LiveBleed::context(&mic, 14, center, Context::Final));
        assert!(LiveBleed::context(&mic, 9, mic.blocks[&9], Context::Final));
        mic.blocks.get_mut(&13).unwrap().quality.missing_peer = true;
        assert!(!LiveBleed::context(&mic, 14, center, Context::Final));
        assert!(!LiveBleed::context(&mic, 9, mic.blocks[&9], Context::Final));
        mic.blocks.remove(&13);
        assert!(!LiveBleed::context(&mic, 14, center, Context::Final));
    }

    #[test]
    fn calibration_requires_one_active_peer_to_lead_every_aligned_rival() {
        let mut analysis = Analysis::default();
        let mut scratch = Vec::new();
        let rivals = [(-30., true), (-29., true), (-20., true), (-18., false)];
        for (at, (rival_level, active)) in rivals.into_iter().enumerate() {
            let at = at as u64;
            let own = Observation {
                start: at * BLOCK,
                end: (at + 1) * BLOCK,
                level: -60.,
                active: false,
            };
            let leader = Observation {
                level: -20.,
                active: true,
                ..own
            };
            let rival = Observation {
                level: rival_level,
                active,
                ..own
            };
            LiveBleed::calibrate(
                &mut analysis,
                at,
                own,
                &[(2, rival), (1, leader)],
                false,
                &mut scratch,
            );
        }
        assert_eq!(analysis.pairs[&1].observations, [(0, 40.)]);
        assert!(!analysis.pairs.contains_key(&2));
    }

    #[test]
    fn one_tracks_deadline_does_not_commit_another_tracks_missing_pcm() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        for block in 0..40 {
            input(&mut gate, block, 0., Some(0.));
        }
        for block in 40..120 {
            input(&mut gate, block, 10_000., Some(1_000.));
        }
        for block in 120..200 {
            input(&mut gate, block, 10_000., None);
        }
        let leading = gate.decision(0, 120 * BLOCK, 121 * BLOCK, Context::Final);
        assert!(!leading.muted && leading.quality.missing_peer);

        for block in 120..200 {
            gate.ingest(1, block * BLOCK, &tone(1_000., block));
        }
        let follower = gate.decision(1, 120 * BLOCK, 121 * BLOCK, Context::Final);
        assert!(follower.muted && !follower.quality.missing_peer);
    }

    #[test]
    fn missing_peer_cannot_extend_a_calibrated_mute() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        for block in 0..40 {
            input(&mut gate, block, 0., Some(0.));
        }
        for block in 40..120 {
            input(&mut gate, block, 10_000., Some(1_000.));
        }
        for block in 120..200 {
            gate.ingest(1, block * BLOCK, &tone(900., block));
        }
        gate.gap(0, 200 * BLOCK);
        assert!(gate.ready_through(1, 121 * BLOCK, FINAL_LOOKAHEAD));
        let unavailable = gate.decision(1, 120 * BLOCK, 121 * BLOCK, Context::Final);
        assert!(!unavailable.muted && unavailable.quality.missing_peer);
    }

    #[test]
    fn partial_blocks_wait_and_disjoint_fragments_are_not_compared() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        let tone = tone(1_000., 0);
        gate.ingest(0, 0, &tone);
        gate.ingest(1, 0, &tone[..100]);
        assert!(!gate.ready_through(0, 100, 0));
        gate.gap(1, 200);
        gate.ingest(1, 200, &tone[200..]);
        assert!(gate.ready_through(0, 100, 0));
        assert!(gate.mics[&1].levels[&0].is_none());
        let missing = gate.decision(0, 0, BLOCK, Context::Final);
        assert!(!missing.muted && missing.quality.missing_peer);
    }

    #[test]
    fn a_gap_within_an_open_block_does_not_freeze_a_partial_observation() {
        let mut gate = LiveBleed::default();
        for track in 0..2 {
            gate.register(track, 0);
            gate.ingest(track, 0, &tone(1_000., 0)[..100]);
            gate.gap(track, 200);
        }
        let partial = gate.decision(0, 0, 100, Context::Draft);
        assert!(partial.quality.missing_peer);
        assert!(gate.analysis[&0][Context::Draft.index()].blocks.is_empty());

        for track in 0..2 {
            gate.ingest(track, 200, &tone(1_000., 0)[200..]);
        }
        let fragmented = gate.decision(0, 0, BLOCK, Context::Draft);
        assert!(fragmented.quality.missing_peer && !fragmented.muted);
        assert!(gate.analysis[&0][Context::Draft.index()].blocks.is_empty());
    }

    #[test]
    fn ending_track_only_waits_for_peer_audio_it_can_use() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        gate.ingest(0, 0, &tone(1_000., 0));
        gate.end(0);
        assert!(!gate.ready_through(0, BLOCK, FINAL_LOOKAHEAD));
        gate.ingest(1, 0, &tone(2_000., 0));
        assert!(gate.ready_through(0, BLOCK, FINAL_LOOKAHEAD));
        assert!(
            !gate
                .decision(0, 0, BLOCK, Context::Final)
                .quality
                .missing_peer
        );
    }

    #[test]
    fn last_peer_ending_mid_block_still_needs_an_own_observation() {
        let mut gate = LiveBleed::default();
        for track in 0..2 {
            gate.register(track, 0);
            gate.ingest(track, 0, &tone(1_000., 0)[..100]);
        }
        gate.end(1);
        assert!(!gate.ready_through(0, 100, FINAL_LOOKAHEAD));
        let unfinished = gate.decision(0, 0, 100, Context::Final);
        assert!(unfinished.quality.missing_peer && !unfinished.muted);
        assert!(gate.apply(0, 0, &mut [1; 100], Context::Final).missing_peer);
        gate.end(0);
        assert!(gate.ready_through(0, 100, FINAL_LOOKAHEAD));
        assert!(
            !gate
                .decision(0, 0, 100, Context::Final)
                .quality
                .missing_peer
        );
    }

    #[test]
    fn retransmitted_and_unaligned_packets_measure_the_same_capture_blocks() {
        let mut gate = LiveBleed::default();
        let samples: Vec<_> = (0..4).flat_map(|at| tone(2_000., at)).collect();
        let start = 37;
        for track in 0..2 {
            gate.register(track, start);
        }
        gate.ingest(0, start, &samples);
        for at in (0..samples.len()).step_by(113) {
            let first = at.saturating_sub(29);
            gate.ingest(
                1,
                start + first as u64,
                &samples[first..(at + 113).min(samples.len())],
            );
        }
        gate.end(0);
        gate.end(1);
        let reference = &gate.mics[&0].levels;
        let packetized = &gate.mics[&1].levels;
        assert_eq!(reference.len(), 5);
        assert_eq!(reference.len(), packetized.len());
        for (at, reference) in reference {
            let reference = reference.unwrap();
            let split = packetized[at].unwrap();
            assert_eq!((reference.start, reference.end), (split.start, split.end));
            assert_eq!(reference.level.to_bits(), split.level.to_bits());
        }
    }

    #[test]
    fn a_clock_jump_preserves_observations_for_the_previous_final() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        for block in 0..40 {
            input(&mut gate, block, 0., Some(0.));
        }
        for block in 40..140 {
            input(&mut gate, block, 10_000., Some(1_000.));
        }
        input(&mut gate, RETAIN_BLOCKS as u64 + 200, 0., Some(0.));
        let pre_gap = gate.decision(1, 110 * BLOCK, 111 * BLOCK, Context::Final);
        assert!(pre_gap.muted && !pre_gap.quality.missing_peer);
    }

    #[test]
    fn missing_peer_does_not_mute_and_sent_silence_counts_as_available() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        input(&mut gate, 0, 10_000.0, None);
        assert!(!gate.ready_through(0, BLOCK, 0));
        let missing = gate.decision(0, 0, BLOCK, Context::Draft);
        assert!(missing.quality.missing_peer && !missing.muted);

        input(&mut gate, 1, 10_000.0, Some(0.));
        assert!(gate.ready_through(0, 2 * BLOCK, 0));
        input(&mut gate, 2, 10_000.0, Some(0.));
        input(&mut gate, 3, 10_000.0, Some(0.));
        let silent = gate.decision(0, BLOCK, 2 * BLOCK, Context::Draft);
        assert!(!silent.quality.missing_peer && !silent.muted);

        gate.end(1);
        assert!(gate.ready_through(0, 100 * BLOCK, FINAL_LOOKAHEAD));
        input(&mut gate, 4, 10_000.0, None);
        let ended = gate.decision(0, 4 * BLOCK, 5 * BLOCK, Context::Final);
        assert!(!ended.quality.missing_peer && !ended.muted);
    }

    #[test]
    fn calibration_outlives_a_peer_that_stays_quiet_past_the_training_window() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        for block in 0..40 {
            input(&mut gate, block, 0., Some(0.));
        }
        for block in 40..145 {
            input(&mut gate, block, 10_000., Some(1_000.));
        }
        let quiet_until = 145 + TRAINING_BLOCKS + 100;
        for block in 145..quiet_until {
            input(&mut gate, block, 0., Some(0.));
        }
        // Fewer lead frames than a fresh calibration needs.
        for block in quiet_until..quiet_until + 15 {
            input(&mut gate, block, 10_000., Some(1_000.));
        }
        let returning = gate.decision(
            1,
            (quiet_until + 5) * BLOCK,
            (quiet_until + 10) * BLOCK,
            Context::Final,
        );
        assert!(returning.muted && !returning.quality.uncalibrated);
    }

    // Two mics from block 0 with each block's amplitudes for tracks 0 and 1.
    fn pair_session(blocks: u64, amplitudes: impl Fn(u64) -> (f32, f32)) -> LiveBleed {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        for block in 0..blocks {
            let (a, b) = amplitudes(block);
            input(&mut gate, block, a, Some(b));
        }
        gate
    }

    #[test]
    fn a_final_mutes_the_lead_blocks_its_first_calibration_comes_from() {
        // Track 0 talks from block 40 and leaks onto track 1 20 dB down.
        // Track 1's speaker talks over it at 60..70 and speaks quietly under
        // it at 70..80, all before the pair has a second of lead blocks.
        let mut gate = pair_session(200, |block| match block {
            0..40 | 130.. => (0., 0.),
            60..70 => (10_000., 7_000.),
            70..80 => (10_000., 3_000.),
            _ => (10_000., 1_000.),
        });
        // Drafts pass the peer's first words through, as before.
        let draft = gate.decision(1, 41 * BLOCK, 46 * BLOCK, Context::Draft);
        assert!(!draft.muted && draft.quality.uncalibrated);

        let bleed = gate.decision(1, 41 * BLOCK, 46 * BLOCK, Context::Final);
        assert!(
            bleed.muted && !bleed.quality.uncalibrated && !bleed.quality.missing_peer,
            "{bleed:?}"
        );
        for own in [62..68, 72..78] {
            let kept = gate.decision(1, own.start * BLOCK, own.end * BLOCK, Context::Final);
            assert!(
                !kept.muted && !kept.quality.uncalibrated,
                "{own:?}: {kept:?}"
            );
        }
    }

    #[test]
    fn a_first_calibration_decides_the_same_however_finals_slice_the_interval() {
        // 20 lead blocks, a pause, then 40 more: the first estimate comes
        // after the lookahead of the first blocks' own final.
        let amplitudes = |block| match block {
            40..60 | 120..160 => (10_000., 1_000.),
            _ => (0., 0.),
        };
        let mut whole = pair_session(240, amplitudes);
        let mut sliced = pair_session(240, amplitudes);
        let interval = 41..59;
        let together = whole.decision(
            1,
            interval.start * BLOCK,
            interval.end * BLOCK,
            Context::Final,
        );
        assert!(together.muted && !together.quality.uncalibrated);
        for block in interval {
            let alone = sliced.decision(1, block * BLOCK, (block + 1) * BLOCK, Context::Final);
            assert!(alone.muted && !alone.quality.uncalibrated, "block {block}");
        }
    }

    #[test]
    fn a_final_leaves_a_quick_replys_uncalibrated_bleed_for_its_own_final() {
        // Track 0 talks, then track 1 answers 0.2 s later. Track 0's final
        // comes as soon as its lookahead is in, before a second of track 1's
        // lead exists, and that lookahead covers track 1's first words.
        let turns = |block| match block {
            40..100 => (10_000., 1_000.),
            110..200 => (1_000., 10_000.),
            _ => (0., 0.),
        };
        let mut gate = pair_session(152, turns);
        let own = gate.decision(0, 40 * BLOCK, 100 * BLOCK, Context::Final);
        assert!(!own.muted && !own.quality.missing_peer);
        let tail = gate.decision(0, 95 * BLOCK, 100 * BLOCK, Context::Final);
        assert!(!tail.muted);
        for block in 152..260 {
            let (a, b) = turns(block);
            input(&mut gate, block, a, Some(b));
        }
        let reply = gate.decision(0, 112 * BLOCK, 150 * BLOCK, Context::Final);
        assert!(reply.muted && !reply.quality.uncalibrated, "{reply:?}");
    }

    #[test]
    fn a_final_verdict_reported_before_calibration_stands() {
        let lead = |block| match block {
            40.. => (10_000., 1_000.),
            _ => (0., 0.),
        };
        let mut gate = pair_session(70, lead);
        let early = gate.decision(1, 41 * BLOCK, 46 * BLOCK, Context::Final);
        assert!(!early.muted && early.quality.uncalibrated);
        for block in 70..200 {
            let (a, b) = lead(block);
            input(&mut gate, block, a, Some(b));
        }
        let again = gate.decision(1, 41 * BLOCK, 46 * BLOCK, Context::Final);
        assert!(!again.muted && again.quality.uncalibrated);
        // Blocks first decided after the estimate exists use it.
        let later = gate.decision(1, 75 * BLOCK, 80 * BLOCK, Context::Final);
        assert!(later.muted && !later.quality.uncalibrated);
    }

    #[test]
    fn hold_keeps_a_filled_hole_next_to_own_speech() {
        let mut mic = Analysis::default();
        for at in 0..50 {
            let (level, muted) = match at {
                8..14 => (-35., false),
                16.. => (-10., false),
                _ => (-30., true),
            };
            mic.blocks.insert(
                at,
                Block {
                    observation: Observation {
                        start: at * BLOCK,
                        end: (at + 1) * BLOCK,
                        level,
                        active: true,
                    },
                    muted,
                    quality: Quality::default(),
                },
            );
        }
        assert!(LiveBleed::fills_hole(&mic, 10));
        // Speech starts at block 16; the hold reaches back to block 10.
        assert!(LiveBleed::context(&mic, 9, mic.blocks[&9], Context::Final));
        assert!(!LiveBleed::context(
            &mic,
            10,
            mic.blocks[&10],
            Context::Final
        ));
        assert!(!LiveBleed::context(
            &mic,
            15,
            mic.blocks[&15],
            Context::Final
        ));
    }

    // One block for each listed track, with digital silence every fifth block
    // so each mic keeps a low noise floor. Then the decisions `tracks.rs`
    // asks for as audio arrives: a draft for the new block and a final 60
    // blocks behind it.
    fn stream(gate: &mut LiveBleed, block: u64, tracks: &[(u64, f32)]) {
        let pause = block.is_multiple_of(5);
        for &(track, amplitude) in tracks {
            let amplitude = if pause { 0. } else { amplitude };
            gate.ingest(track, block * BLOCK, &tone(amplitude, block));
        }
        for &(track, _) in tracks {
            gate.decision(track, block * BLOCK, (block + 1) * BLOCK, Context::Draft);
            if block >= 60 {
                let final_block = (block - 60) * BLOCK;
                gate.decision(track, final_block, final_block + BLOCK, Context::Final);
            }
        }
    }

    fn three_mics() -> LiveBleed {
        let mut gate = LiveBleed::default();
        for track in 0..3 {
            gate.register(track, 0);
        }
        gate
    }

    #[test]
    fn a_silent_registered_peer_stops_delaying_verdicts() {
        let mut gate = three_mics();
        for block in 0..400 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.)]);
            if block == 100 {
                // Within the grace a peer that has sent nothing may be late.
                assert!(!gate.ready_through(1, 98 * BLOCK, DRAFT_LOOKAHEAD));
            }
        }
        for track in 0..2 {
            assert!(gate.ready_through(track, 398 * BLOCK, DRAFT_LOOKAHEAD));
            assert!(gate.ready_through(track, 340 * BLOCK, FINAL_LOOKAHEAD));
            let draft = gate.decision(track, 397 * BLOCK, 398 * BLOCK, Context::Draft);
            assert!(!draft.quality.missing_peer);
            let final_result = gate.decision(track, 340 * BLOCK, 341 * BLOCK, Context::Final);
            assert!(!final_result.quality.missing_peer);
        }
    }

    #[test]
    fn a_silent_registered_peer_does_not_block_calibration() {
        let mut gate = three_mics();
        for block in 0..400 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.)]);
        }
        assert!(
            gate.analysis[&1][Context::Final.index()].pairs[&0]
                .estimate
                .is_some()
        );
        let bleed = gate.decision(1, 351 * BLOCK, 355 * BLOCK, Context::Final);
        assert!(bleed.muted && !bleed.quality.uncalibrated && !bleed.quality.missing_peer);
        let mut weak = vec![1_000; 4 * BLOCK as usize];
        let quality = gate.apply(1, 351 * BLOCK, &mut weak, Context::Final);
        assert!(!quality.uncalibrated && !quality.missing_peer);
        assert!(weak.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn a_jitter_late_peer_is_waited_for_and_vetoes_calibration() {
        let mut gate = three_mics();
        // Track 2 is as loud as track 0, so track 0 never leads it by the
        // calibration margin. Its audio arrives five blocks late.
        for block in 0..400 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.)]);
            if let Some(late) = block.checked_sub(5) {
                stream_one(&mut gate, late, 2, 10_000.);
            }
            if block >= 10 {
                assert!(!gate.ready_through(1, (block - 1) * BLOCK, DRAFT_LOOKAHEAD));
            }
        }
        for late in 395..400 {
            stream_one(&mut gate, late, 2, 10_000.);
        }
        assert!(gate.ready_through(1, 398 * BLOCK, DRAFT_LOOKAHEAD));
        let pairs = &gate.analysis[&1][Context::Final.index()].pairs;
        assert!(pairs.values().all(|pair| pair.observations.is_empty()));
        let vetoed = gate.decision(1, 351 * BLOCK, 355 * BLOCK, Context::Final);
        assert!(!vetoed.muted && vetoed.quality.uncalibrated && !vetoed.quality.missing_peer);
    }

    fn stream_one(gate: &mut LiveBleed, block: u64, track: u64, amplitude: f32) {
        let amplitude = if block.is_multiple_of(5) {
            0.
        } else {
            amplitude
        };
        gate.ingest(track, block * BLOCK, &tone(amplitude, block));
    }

    #[test]
    fn a_streaming_peer_is_waited_for_however_far_behind() {
        let mut gate = three_mics();
        for block in 0..700 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.)]);
            if let Some(late) = block.checked_sub(300) {
                stream_one(&mut gate, late, 2, 1_000.);
            }
        }
        assert!(!gate.ready_through(1, 600 * BLOCK, DRAFT_LOOKAHEAD));
        assert!(
            gate.decision(1, 600 * BLOCK, 601 * BLOCK, Context::Final)
                .quality
                .missing_peer
        );
    }

    #[test]
    fn a_stopped_peer_is_not_waited_for_but_still_counts_as_missing() {
        let mut gate = three_mics();
        for block in 0..100 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.), (2, 1_000.)]);
        }
        for block in 100..400 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.)]);
            if block == 200 {
                assert!(!gate.ready_through(1, 198 * BLOCK, DRAFT_LOOKAHEAD));
            }
        }
        // Its audio may still arrive, so results say they were decided
        // without it, and it still vetoes calibration.
        assert!(gate.ready_through(1, 398 * BLOCK, DRAFT_LOOKAHEAD));
        let stopped = gate.decision(1, 380 * BLOCK, 381 * BLOCK, Context::Draft);
        assert!(stopped.quality.missing_peer);
        let pairs = &gate.analysis[&1][Context::Final.index()].pairs;
        assert!(pairs[&0].observations.iter().all(|&(at, _)| at < 100));
        // Once it delivers again it is waited for like any late peer.
        gate.gap(2, 400 * BLOCK);
        for block in 400..406 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.)]);
            if block < 402 {
                stream_one(&mut gate, block, 2, 1_000.);
            }
        }
        assert!(!gate.ready_through(1, 403 * BLOCK, DRAFT_LOOKAHEAD));
        for block in 402..406 {
            stream_one(&mut gate, block, 2, 1_000.);
        }
        assert!(gate.ready_through(1, 403 * BLOCK, DRAFT_LOOKAHEAD));
    }

    #[test]
    fn a_silent_peer_that_starts_behind_is_waited_for_again() {
        let mut gate = three_mics();
        for block in 0..400 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.)]);
        }
        let earlier = gate.decision(1, 330 * BLOCK, 331 * BLOCK, Context::Final);
        // Its first audio is from its registered start: delayed, not skipped.
        for block in 0..380 {
            stream_one(&mut gate, block, 2, 1_000.);
        }
        // The verdict stands. Until the peer catches up through the final
        // lookahead, the interval is reported as pending on it again.
        let again = gate.decision(1, 330 * BLOCK, 331 * BLOCK, Context::Final);
        assert_eq!(again.muted, earlier.muted);
        assert!(again.quality.missing_peer && !gate.ready_through(1, 331 * BLOCK, FINAL_LOOKAHEAD));
        assert!(!gate.ready_through(1, 398 * BLOCK, DRAFT_LOOKAHEAD));
        let own = gate.decision(2, 100 * BLOCK, 101 * BLOCK, Context::Final);
        assert!(!own.quality.missing_peer);
        for block in 380..400 {
            stream_one(&mut gate, block, 2, 1_000.);
        }
        assert!(gate.ready_through(1, 398 * BLOCK, DRAFT_LOOKAHEAD));
        let settled = gate.decision(1, 330 * BLOCK, 331 * BLOCK, Context::Final);
        assert_eq!(
            (settled.muted, settled.quality.missing_peer),
            (earlier.muted, earlier.quality.missing_peer)
        );
    }

    #[test]
    fn a_peer_that_ends_without_sending_is_no_peer() {
        let mut gate = LiveBleed::default();
        gate.register(0, 0);
        gate.register(1, 0);
        gate.gap(1, 10 * BLOCK);
        gate.end(1);
        gate.ingest(0, 0, &tone(1_000., 0));
        assert!(gate.ready_through(0, BLOCK, FINAL_LOOKAHEAD));
        assert!(
            !gate
                .decision(0, 0, BLOCK, Context::Final)
                .quality
                .missing_peer
        );
    }

    #[test]
    fn a_silent_peer_that_starts_sending_rejoins_without_gap_errors() {
        let mut gate = three_mics();
        for block in 0..400 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.)]);
        }
        let earlier = gate.decision(1, 330 * BLOCK, 331 * BLOCK, Context::Final);
        assert!(!earlier.quality.missing_peer);
        // Track 2 starts capturing at block 400. `tracks.rs` reports the
        // skipped range as a gap before the first samples.
        gate.gap(2, 400 * BLOCK);
        for block in 400..403 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.), (2, 1_000.)]);
        }
        // Blocks before its first audio, decided after it started, did not
        // miss it: it sent nothing there.
        let before = gate.decision(1, 393 * BLOCK, 394 * BLOCK, Context::Final);
        assert!(!before.quality.missing_peer && before.muted);
        let again = gate.decision(1, 330 * BLOCK, 331 * BLOCK, Context::Final);
        assert_eq!(
            (again.muted, again.quality.missing_peer),
            (earlier.muted, earlier.quality.missing_peer)
        );
        // From its first audio on it is a peer like any other: waited for.
        for block in 403..410 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.)]);
        }
        assert!(!gate.ready_through(1, 405 * BLOCK, DRAFT_LOOKAHEAD));
        let waiting = gate.decision(1, 405 * BLOCK, 406 * BLOCK, Context::Draft);
        assert!(waiting.quality.missing_peer);
        for block in 403..410 {
            stream_one(&mut gate, block, 2, 1_000.);
        }
        assert!(gate.ready_through(1, 405 * BLOCK, DRAFT_LOOKAHEAD));
        // Then it calibrates and mutes in both directions.
        for block in 410..550 {
            stream(&mut gate, block, &[(0, 10_000.), (1, 1_000.), (2, 1_000.)]);
        }
        for block in 550..700 {
            stream(&mut gate, block, &[(0, 1_000.), (1, 1_000.), (2, 10_000.)]);
        }
        for (track, at) in [(2, 520), (0, 630), (1, 630)] {
            let bleed = gate.decision(track, (at + 1) * BLOCK, (at + 5) * BLOCK, Context::Final);
            assert!(
                bleed.muted && !bleed.quality.uncalibrated && !bleed.quality.missing_peer,
                "track {track} at block {at}: {bleed:?}"
            );
        }
        let own = gate.decision(2, 631 * BLOCK, 635 * BLOCK, Context::Final);
        assert!(!own.muted);
    }
}
