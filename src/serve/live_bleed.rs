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
}

impl History {
    fn observe(&mut self, block: u64, value: f32, quantile: f32, values: &mut Vec<f32>) {
        self.observations.push_back((block, value));
        if self.observations.len() >= MIN_LEAD_FRAMES
            && self
                .estimate
                .is_none_or(|(previous, _)| block >= previous + 10)
        {
            values.clear();
            values.extend(self.observations.iter().map(|&(_, value)| value));
            self.estimate = Some((block, percentile(values, quantile)));
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
        if self.observations.is_empty() {
            self.estimate = None;
        }
    }
}

#[derive(Default)]
struct Analysis {
    blocks: BTreeMap<u64, Block>,
    pairs: HashMap<u64, History>,
    release: Option<(u64, f32, f32)>,
    processed: Option<u64>,
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
}

impl Mic {
    fn new(start: u64) -> Self {
        Self {
            start,
            next: start,
            end: None,
            block_start: start,
            levels: BTreeMap::new(),
            floor: History::default(),
            meter: LevelMeter::new(RATE),
            pruned_before: 0,
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
    percentile_values: Vec<f32>,
    // Each mic reaches its deadlines independently. A final can also use
    // peer audio that arrived after the draft deadline.
    analysis: HashMap<u64, [Analysis; 2]>,
}

impl LiveBleed {
    pub(super) fn register(&mut self, track: u64, start_frame: u64) {
        self.mics
            .entry(track)
            .or_insert_with(|| Mic::new(start_frame));
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
    }

    pub(super) fn ready_through(&self, track: u64, through: u64, lookahead: u64) -> bool {
        let mut horizon = through.saturating_add(lookahead);
        if let Some(end) = self.mics.get(&track).and_then(|mic| mic.end) {
            horizon = horizon.min(end);
        }
        let peer_overlaps_context = horizon > through
            && self
                .mics
                .iter()
                .any(|(&peer, mic)| track != peer && mic.active_during(through, horizon));
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
            && self
                .mics
                .iter()
                .any(|(&peer, mic)| peer != track && mic.active_during(first, end))
    }

    fn peers_ready(&self, track: u64, horizon: u64) -> bool {
        self.mics
            .iter()
            .all(|(&peer, mic)| track == peer || mic.start >= horizon || mic.ready_at(horizon))
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
        let mut end = through.div_ceil(BLOCK);
        if mic.end.is_none() {
            end = end.min(mic.next / BLOCK);
        }
        let first = analysis.processed.map_or(0, |last| last + 1);
        if first >= end {
            return;
        }
        let observed = &mut self.observed;
        for (&block, &own) in mic.levels.range(first..end) {
            let Some(own) = own else {
                continue;
            };
            observed.clear();
            let mut missing_peer = false;
            let mut waiting_for_peer = false;
            for (&peer, mic) in &self.mics {
                if peer == track || !mic.active_during(own.start, own.end) {
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
                    _ => {
                        missing_peer = true;
                        waiting_for_peer |= !mic.ready_at(own.end);
                    }
                }
            }
            // A deadline can consume its interval without a peer. Keep the
            // undecided lookahead available for when that peer catches up.
            if waiting_for_peer && own.start >= required {
                break;
            }
            Self::classify(
                analysis,
                block,
                own,
                observed,
                missing_peer,
                &mut self.percentile_values,
            );
            analysis.processed = Some(block);
        }
    }

    fn classify(
        analysis: &mut Analysis,
        block: u64,
        own: Observation,
        peers: &[(u64, Observation)],
        missing_peer: bool,
        values: &mut Vec<f32>,
    ) {
        let level = own.level;
        let mut quality = Quality {
            missing_peer,
            ..Quality::default()
        };
        let mut muted = false;
        // A single peer can lead by the calibration margin. Include inactive
        // rivals in that comparison just as for the offline bleed estimator.
        let leader = if missing_peer || level <= SILENCE_DB {
            None
        } else {
            peers
                .iter()
                .max_by(|a, b| a.1.level.total_cmp(&b.1.level))
                .filter(|&&(key, loudest)| {
                    loudest.active
                        && loudest.level - level >= MARGIN_DB
                        && peers.iter().all(|&(peer, rival)| {
                            peer == key || loudest.level >= rival.level + MARGIN_DB
                        })
                })
                .map(|&(peer, _)| peer)
        };
        for &(peer, other) in peers {
            let difference = other.level - level;
            if !other.active || level <= SILENCE_DB || difference <= 0.0 {
                continue;
            }
            let pair = analysis.pairs.entry(peer).or_default();
            pair.expire(block);
            if pair.observations.len() >= MIN_LEAD_FRAMES {
                if let Some((_, estimate)) = pair.estimate {
                    muted |= difference >= (estimate - TOLERANCE_DB).max(LOWEST_THRESHOLD_DB);
                }
            } else if difference >= MARGIN_DB {
                quality.uncalibrated = true;
            }
            if leader == Some(peer) {
                pair.observe(block, difference, 0.5, values);
            }
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
        if block.muted {
            return !track
                .blocks
                .range(at.saturating_sub(HOLD_FRAMES as u64)..=at + HOLD_FRAMES as u64)
                .any(|(&position, adjacent)| {
                    adjacent.speaking() && !Self::fills_hole(track, position)
                });
        }
        Self::fills_hole(track, at)
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
            LiveBleed::classify(
                &mut analysis,
                at,
                own,
                &[(2, rival), (1, leader)],
                false,
                &mut scratch,
            );
        }
        assert_eq!(analysis.pairs[&1].observations, [(0, 40.)]);
        assert!(analysis.pairs[&2].observations.is_empty());
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
}
