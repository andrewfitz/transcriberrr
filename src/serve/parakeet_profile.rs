//! Parakeet streaming configuration shared by the listener and inference benchmark.

use parakeet_rs::UnifiedStreamingConfig;

pub(super) const STREAMING_CONFIG: UnifiedStreamingConfig = UnifiedStreamingConfig {
    left_context_secs: 0.64,
    chunk_secs: 1.44,
    right_context_secs: 0.16,
};

pub(super) fn warmup_samples(config: UnifiedStreamingConfig) -> usize {
    // Exercise every regular encoder window, including the one with full left context.
    let windows = 1 + config
        .left_context_samples()
        .div_ceil(config.chunk_samples());
    windows * config.chunk_samples() + config.right_context_samples()
}

pub(super) fn final_padding_samples(received: usize, config: UnifiedStreamingConfig) -> usize {
    let frame_samples = config.chunk_samples() / config.chunk_encoder_frames();
    received.next_multiple_of(frame_samples) - received
}
