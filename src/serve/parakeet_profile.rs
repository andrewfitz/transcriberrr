//! Parakeet streaming configuration shared by the listener and inference benchmark.

use parakeet_rs::UnifiedStreamingConfig;

pub(super) const STREAMING_CONFIG: UnifiedStreamingConfig = UnifiedStreamingConfig {
    left_context_secs: 0.64,
    chunk_secs: 0.96,
    right_context_secs: 0.16,
};

/// One encoder frame: eight 10 ms feature frames.
const ENCODER_FRAME_SECS: f32 = 0.08;

/// The longest encoder window of `profiles`, in encoder frames. Windows are whole numbers
/// of encoder frames, so every shorter count is a possible window as well.
pub(super) fn warmup_window_frames(profiles: &[UnifiedStreamingConfig]) -> usize {
    profiles
        .iter()
        .map(|config| {
            config.total_window_frames() * config.chunk_encoder_frames() / config.chunk_frames()
        })
        .max()
        .unwrap_or(0)
}

/// A profile whose only window is exactly `encoder_frames` long.
pub(super) fn single_window(encoder_frames: usize) -> UnifiedStreamingConfig {
    UnifiedStreamingConfig {
        left_context_secs: 0.0,
        chunk_secs: encoder_frames as f32 * ENCODER_FRAME_SECS,
        right_context_secs: 0.0,
    }
}

pub(super) fn final_padding_samples(received: usize, config: UnifiedStreamingConfig) -> usize {
    let frame_samples = config.chunk_samples() / config.chunk_encoder_frames();
    received.next_multiple_of(frame_samples) - received
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_up_covers_every_window_a_profile_can_produce() {
        let standard = UnifiedStreamingConfig {
            chunk_secs: 1.44,
            ..STREAMING_CONFIG
        };
        // The longest window: 0.64 s of left context, the hop and 0.16 s of right context.
        assert_eq!(warmup_window_frames(&[STREAMING_CONFIG]), 22);
        assert_eq!(warmup_window_frames(&[STREAMING_CONFIG, standard]), 28);
        let three_hops = UnifiedStreamingConfig {
            chunk_secs: 2.88,
            ..STREAMING_CONFIG
        };
        assert_eq!(warmup_window_frames(&[standard, three_hops]), 46);

        let window = single_window(9);
        assert_eq!(window.chunk_samples(), 9 * 1_280);
        assert_eq!(window.total_window_samples(), window.chunk_samples());
        assert!(window.validate().is_ok());
    }
}
