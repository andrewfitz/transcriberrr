//! Manual Parakeet encoder-window benchmark. Pass the Unified model directory and
//! optionally `cpu`. Prints one JSON line per window length with the time of the first
//! call on that shape and of later calls, so shape-dependent startup cost is visible.

#[cfg(feature = "parakeet")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use parakeet_rs::{
        ExecutionConfig, ExecutionProvider, ParakeetUnified, ParakeetUnifiedHandle,
        UnifiedStreamingConfig,
    };
    use std::time::Instant;

    let mut args = std::env::args_os().skip(1);
    let model = args.next().context("pass a Unified model path")?;
    let provider = match args.next() {
        Some(device) if device == "cpu" => ExecutionProvider::BurnCpu,
        _ => ExecutionProvider::BurnWgpu,
    };
    let execution = ExecutionConfig::new().with_execution_provider(provider);
    let shared = ParakeetUnifiedHandle::from_pretrained(model, Some(execution))?;
    // One window of exactly `frames` 80 ms encoder frames: no context on either side.
    let run = |frames: usize| -> anyhow::Result<f64> {
        let config = UnifiedStreamingConfig {
            left_context_secs: 0.0,
            chunk_secs: frames as f32 * 0.08,
            right_context_secs: 0.0,
        };
        let mut decoder = ParakeetUnified::from_shared_with_streaming_config(&shared, config)?;
        let samples = vec![0.0; config.chunk_samples()];
        let started = Instant::now();
        decoder.transcribe_chunk(&samples)?;
        Ok(started.elapsed().as_secs_f64() * 1e3)
    };
    // `idle`: time a regular first window (1.12 s) after sleeping, with and without a
    // small priming call just before it.
    let mode = args.next().map(|mode| mode.to_string_lossy().into_owned());
    if mode.as_deref() == Some("idle") {
        for frames in [4, 14, 22] {
            run(frames)?;
            run(frames)?;
        }
        for round in 0..4 {
            for idle in [0.2, 1.0, 2.0, 2.5, 3.0, 4.0, 8.0, 15.0] {
                for prime in [false, true] {
                    std::thread::sleep(std::time::Duration::from_secs_f64(idle));
                    let primed = if prime { Some(run(4)?) } else { None };
                    let first = run(14)?;
                    let second = run(22)?;
                    println!(
                        "{}",
                        serde_json::json!({
                            "round": round, "idle_s": idle, "prime_ms": primed,
                            "first_ms": first, "second_ms": second,
                        })
                    );
                }
            }
        }
        return Ok(());
    }
    // `ascending`: the first call on each length from one encoder frame up, as a startup
    // warm-up would run them.
    if mode.as_deref() == Some("ascending") {
        for frames in 1..=58 {
            println!(
                "{}",
                serde_json::json!({"frames": frames, "first_ms": run(frames)?, "ms": run(frames)?})
            );
        }
        return Ok(());
    }
    // Visit lengths out of order so neighbouring shapes do not warm each other in sequence.
    let lengths: Vec<usize> = (0..48).map(|index| 4 + (index * 29) % 48).collect();
    let mut first = std::collections::BTreeMap::new();
    for &frames in &lengths {
        first.insert(frames, run(frames)?);
    }
    for pass in 0..3 {
        for &frames in &lengths {
            let again = run(frames)?;
            println!(
                "{}",
                serde_json::json!({
                    "pass": pass, "frames": frames, "seconds": frames as f64 * 0.08,
                    "first_ms": first[&frames], "ms": again,
                })
            );
        }
    }
    Ok(())
}

#[cfg(not(feature = "parakeet"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("Build this benchmark with --features parakeet");
}
