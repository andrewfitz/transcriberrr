//! Timing trace for the recognition path, enabled by `TRANSCRIBERRR_TRACE_RECOGNITION`.
//! One JSON object per line on stderr, stamped on the monotonic clock in nanoseconds.

use std::io::Write;
use std::sync::OnceLock;

use serde_json::{Map, Value, json};

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("TRANSCRIBERRR_TRACE_RECOGNITION").is_some())
}

/// The clock that `time.monotonic_ns()` and `Instant` read on macOS.
fn monotonic_ns() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `time` is a valid timespec and the clock id is a constant.
    unsafe { libc::clock_gettime(libc::CLOCK_UPTIME_RAW, &mut time) };
    time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
}

/// Writes one record. Callers check `enabled` before building `fields`.
pub(super) fn emit(event: &str, fields: Value) {
    let mut record = Map::new();
    record.insert("trace".into(), json!(event));
    record.insert("time_ns".into(), json!(monotonic_ns()));
    if let Value::Object(fields) = fields {
        record.extend(fields);
    }
    let mut line = Value::Object(record).to_string();
    line.push('\n');
    // One write per record keeps lines whole when several threads trace.
    let _ = std::io::stderr().lock().write_all(line.as_bytes());
}
