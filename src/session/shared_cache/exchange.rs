//! Version 2 of the recognition file exchanged with MPCut. Version 1 is JSON holding
//! base64 of each JSON document. This is the same two documents, byte for byte, behind
//! the same digest:
//!
//! ```text
//!  0   4  "MPCX"
//!  4   1  version, 2
//!  5   1  flags: bit 0 is set when levels follow the result, and the others are zero
//!  6   2  zero
//!  8  32  SHA-256 over the result, a zero byte and the levels
//! 40      the result, then the levels when flagged, each as
//!         1  method   4  length of the document   4  stored length   the stored bytes
//! ```
//!
//! Both lengths are little-endian. Method 0 stores the document, 1 is its raw DEFLATE
//! stream, and 2 and 3 are the raw DEFLATE stream of the levels packing that MPCut's
//! analysis sidecar uses (`SidecarActivityPacking`) for MPCut's layout and for this
//! program's. Nothing follows the last document. Anything else is not an exchange.
use anyhow::{Context, Result, anyhow, bail, ensure};
use ring::digest::{Context as DigestContext, SHA256};
use std::path::Path;

/// The highest version this build reads. It reads every earlier one.
pub(super) const VERSION: u8 = 2;
pub(super) const EXTENSION: &str = "mpx";
/// MPCut writes the highest version it reads to this file in the shared folder. The
/// session manifest rejects unknown fields, so it cannot carry a field that an earlier
/// build of this program has not heard of.
const READER: &str = "exchange-version";

const MAGIC: &[u8] = b"MPCX";
const HEADER: usize = 40;
const LIMIT: usize = i32::MAX as usize;
const STORED: u8 = 0;
const DEFLATED: u8 = 1;
const PACKED: u8 = 2;
const PACKED_PROVIDER_LAYOUT: u8 = 3;

pub(super) fn digest(result: &[u8], activity: Option<&[u8]>) -> [u8; 32] {
    let mut hash = DigestContext::new(&SHA256);
    hash.update(result);
    hash.update(&[0]);
    if let Some(activity) = activity {
        hash.update(activity);
    }
    hash.finish()
        .as_ref()
        .try_into()
        .expect("SHA-256 is 32 bytes")
}

/// The version that the MPCut using this folder reads. One that says nothing reads 1.
pub(super) fn reader_version(directory: &Path) -> u8 {
    std::fs::read_to_string(directory.join(READER))
        .ok()
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or(1)
}

pub(super) fn encode(result: &[u8], activity: Option<&[u8]>) -> Result<Vec<u8>> {
    let mut file = MAGIC.to_vec();
    file.extend_from_slice(&[VERSION, u8::from(activity.is_some()), 0, 0]);
    file.extend_from_slice(&digest(result, activity));
    for document in std::iter::once(result).chain(activity) {
        ensure!(
            document.len() <= LIMIT,
            "recognition document is too large to share"
        );
        let (method, stored) = payload(document);
        file.push(method);
        file.extend_from_slice(&(document.len() as u32).to_le_bytes());
        file.extend_from_slice(&(stored.len() as u32).to_le_bytes());
        file.extend_from_slice(&stored);
    }
    Ok(file)
}

pub(super) fn decode(file: &[u8]) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    ensure!(
        file.len() >= HEADER
            && file.starts_with(MAGIC)
            && file[4] == VERSION
            && file[5] <= 1
            && file[6..8] == [0, 0],
        "not a version {VERSION} recognition exchange"
    );
    let mut rest = &file[HEADER..];
    let result = document(&mut rest)?;
    let activity = (file[5] == 1).then(|| document(&mut rest)).transpose()?;
    ensure!(
        rest.is_empty() && file[8..HEADER] == digest(&result, activity.as_deref()),
        "recognition exchange changed"
    );
    Ok((result, activity))
}

fn document(rest: &mut &[u8]) -> Result<Vec<u8>> {
    ensure!(rest.len() >= 9, "recognition exchange is truncated");
    let length = |offset: usize| {
        u32::from_le_bytes(rest[offset..offset + 4].try_into().expect("four bytes")) as usize
    };
    let (method, size, stored_size) = (rest[0], length(1), length(5));
    ensure!(
        size <= LIMIT && stored_size <= rest.len() - 9,
        "recognition exchange is truncated"
    );
    let (stored, following) = rest[9..].split_at(stored_size);
    *rest = following;
    // A packed document is smaller than the document, so one limit serves both.
    let inflate = || {
        miniz_oxide::inflate::decompress_to_vec_with_limit(stored, size)
            .map_err(|_| anyhow!("recognition exchange is damaged"))
    };
    let value = match method {
        STORED => stored.to_vec(),
        DEFLATED => inflate()?,
        PACKED => unpack(&inflate()?, size).context("packed levels are damaged")?,
        PACKED_PROVIDER_LAYOUT => {
            unpack_provider_layout(&inflate()?, size).context("packed levels are damaged")?
        }
        _ => bail!("unknown recognition exchange method {method}"),
    };
    ensure!(value.len() == size, "recognition exchange is damaged");
    Ok(value)
}

/// Packed levels are a small fraction of their document, so they are tried first. A
/// document is stored in a form only after that form has reproduced it.
fn payload(document: &[u8]) -> (u8, Vec<u8>) {
    let packed = match pack(document) {
        Some(packed) => (unpack(&packed, document.len()).as_deref() == Some(document))
            .then_some((PACKED, packed)),
        None => pack_provider_layout(document)
            .filter(|packed| {
                unpack_provider_layout(packed, document.len()).as_deref() == Some(document)
            })
            .map(|packed| (PACKED_PROVIDER_LAYOUT, packed)),
    };
    if let Some((method, packed)) = packed
        && let Some(stored) = deflate(&packed)
    {
        return (method, stored);
    }
    match deflate(document) {
        Some(stored) => (DEFLATED, stored),
        None => (STORED, document.to_vec()),
    }
}

/// The raw DEFLATE stream of the input, when that is smaller than the input.
fn deflate(input: &[u8]) -> Option<Vec<u8>> {
    let output = miniz_oxide::deflate::compress_to_vec(input, 6);
    (output.len() < input.len()).then_some(output)
}

// The rest follows MPCut's `SidecarActivityPacking` statement for statement. A 16 kHz
// levels document has a regular 320-sample clock, so a frame is stored as the decimal
// spelling of its level, two symbols to a byte, behind the final sample. Everything
// outside the frames is kept verbatim.

const WINDOW: i64 = 320;
const ALPHABET: &[u8; 15] = b"0123456789.-+eE";
const PREFIX: &[u8] = br#"{"activity":["#;
const FRAME: &[u8] = br#"{"end_sample":"#;
const LEVEL: &[u8] = br#","level":"#;
const START: &[u8] = br#","start_sample":"#;
const PROVIDER_OPENING: &[u8] = br#""activity":["#;
const PROVIDER_FRAME: &[u8] = br#"{"start_sample":"#;
const PROVIDER_END: &[u8] = br#","end_sample":"#;
const PROVIDER_HEADER_LIMIT: usize = 65536;

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Cursor<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn consume(&mut self, token: &[u8]) -> bool {
        let found = self.bytes[self.position..].starts_with(token);
        if found {
            self.position += token.len();
        }
        found
    }

    fn sample(&mut self) -> Option<i64> {
        let first = self.position;
        let mut value: i64 = 0;
        while let Some(digit @ b'0'..=b'9') = self.peek() {
            value = value
                .checked_mul(10)?
                .checked_add(i64::from(digit - b'0'))?;
            self.position += 1;
        }
        (self.position > first && (self.position == first + 1 || self.bytes[first] != b'0'))
            .then_some(value)
    }

    fn level(&mut self, packed: &mut Vec<u8>) -> bool {
        let first = self.position;
        let mut high = None;
        while let Some(code) = self
            .peek()
            .and_then(|byte| ALPHABET.iter().position(|&symbol| symbol == byte))
        {
            match high.take() {
                Some(top) => packed.push(top << 4 | code as u8),
                None => high = Some(code as u8),
            }
            self.position += 1;
        }
        packed.push(high.unwrap_or(15) << 4 | 15);
        self.position > first
    }

    /// Packs every frame and stops on the bracket that closes them. Returns the final sample.
    fn frames(&mut self, packed: &mut Vec<u8>, provider: bool) -> Option<i64> {
        let mut previous = 0;
        loop {
            let end;
            if provider {
                if !(self.consume(PROVIDER_FRAME)
                    && self.sample() == Some(previous)
                    && self.consume(PROVIDER_END))
                {
                    return None;
                }
                end = self.sample()?;
                if !(end > previous
                    && end - previous <= WINDOW
                    && self.consume(LEVEL)
                    && self.level(packed))
                {
                    return None;
                }
            } else {
                if !self.consume(FRAME) {
                    return None;
                }
                end = self.sample()?;
                if !(end > previous
                    && end - previous <= WINDOW
                    && self.consume(LEVEL)
                    && self.level(packed)
                    && self.consume(START)
                    && self.sample() == Some(previous))
                {
                    return None;
                }
            }
            if self.peek() != Some(b'}') {
                return None;
            }
            self.position += 1;
            previous = end;
            match self.peek()? {
                b']' => return Some(previous),
                b',' if end % WINDOW == 0 => self.position += 1,
                _ => return None,
            }
        }
    }
}

fn pack(document: &[u8]) -> Option<Vec<u8>> {
    let mut cursor = Cursor {
        bytes: document,
        position: 0,
    };
    if !cursor.consume(PREFIX) {
        return None;
    }
    let mut packed = vec![0; 8];
    let duration = cursor.frames(&mut packed, false)?;
    packed[..8].copy_from_slice(&duration.to_le_bytes());
    packed.extend_from_slice(&document[cursor.position..]);
    Some(packed)
}

fn pack_provider_layout(document: &[u8]) -> Option<Vec<u8>> {
    // The header holds provider evidence, which may quote the array's own key.
    // Only an opening that is followed by a frame starts the levels.
    let reach = document
        .len()
        .min(PROVIDER_HEADER_LIMIT - PROVIDER_OPENING.len());
    let header = (0..reach).find_map(|candidate| {
        document[candidate..]
            .strip_prefix(PROVIDER_OPENING)
            .filter(|frames| frames.starts_with(PROVIDER_FRAME))
            .map(|_| candidate + PROVIDER_OPENING.len())
    })?;
    let mut cursor = Cursor {
        bytes: document,
        position: header,
    };
    let mut packed = vec![0; 12];
    packed.extend_from_slice(&document[..header]);
    let duration = cursor.frames(&mut packed, true)?;
    packed[..8].copy_from_slice(&duration.to_le_bytes());
    packed[8..12].copy_from_slice(&(header as u32).to_le_bytes());
    packed.extend_from_slice(&document[cursor.position..]);
    Some(packed)
}

fn unpack(packed: &[u8], size: usize) -> Option<Vec<u8>> {
    if packed.len() <= 8 {
        return None;
    }
    let duration = i64::from_le_bytes(packed[..8].try_into().ok()?);
    if duration <= 0 || (duration - 1) / WINDOW >= (packed.len() - 8) as i64 {
        return None;
    }
    unpack_frames(packed, 8, duration, size, false, PREFIX.to_vec())
}

fn unpack_provider_layout(packed: &[u8], size: usize) -> Option<Vec<u8>> {
    if packed.len() <= 12 {
        return None;
    }
    let duration = i64::from_le_bytes(packed[..8].try_into().ok()?);
    let header = u32::from_le_bytes(packed[8..12].try_into().ok()?) as usize;
    if header > PROVIDER_HEADER_LIMIT.min(size) || header >= packed.len() - 12 {
        return None;
    }
    let position = 12 + header;
    if duration <= 0 || (duration - 1) / WINDOW >= (packed.len() - position) as i64 {
        return None;
    }
    let document = packed[12..position].to_vec();
    unpack_frames(packed, position, duration, size, true, document)
}

fn unpack_frames(
    packed: &[u8],
    mut position: usize,
    duration: i64,
    size: usize,
    provider: bool,
    mut document: Vec<u8>,
) -> Option<Vec<u8>> {
    document.reserve(size.min(8 << 20).saturating_sub(document.len()));
    let mut cursor = 0;
    let mut start = String::from("0");
    while cursor < duration {
        let end = cursor + WINDOW.min(duration - cursor);
        let end_text = end.to_string();
        if cursor > 0 {
            document.push(b',');
        }
        if provider {
            document.extend_from_slice(PROVIDER_FRAME);
            document.extend_from_slice(start.as_bytes());
            document.extend_from_slice(PROVIDER_END);
            document.extend_from_slice(end_text.as_bytes());
        } else {
            document.extend_from_slice(FRAME);
            document.extend_from_slice(end_text.as_bytes());
        }
        document.extend_from_slice(LEVEL);
        loop {
            if document.len() >= size {
                return None;
            }
            let byte = *packed.get(position)?;
            position += 1;
            if byte >> 4 < 15 {
                document.push(ALPHABET[usize::from(byte >> 4)]);
            }
            if byte & 15 == 15 {
                break;
            }
            if byte >> 4 == 15 {
                return None;
            }
            document.push(ALPHABET[usize::from(byte & 15)]);
        }
        if !provider {
            document.extend_from_slice(START);
            document.extend_from_slice(start.as_bytes());
        }
        document.push(b'}');
        start = end_text;
        cursor = end;
    }
    let following = &packed[position..];
    if following.first() != Some(&b']') || document.len() + following.len() != size {
        return None;
    }
    document.extend_from_slice(following);
    Some(document)
}

/// The documents and files here are repeated in MPCut's
/// `Tests/MPCutCoreTests/TranscriptRecognitionExchangeTests.swift`. Each program reads
/// what the other wrote.
#[cfg(test)]
mod tests {
    use super::super::{Bundle, artifacts_digest, hex, share, shared};
    use super::*;
    use crate::session::{Activity, ActivityArtifact, LocalWord, Precision, Track, TrackArtifact};
    use base64::{Engine, engine::general_purpose::STANDARD};
    use std::fs;
    use std::path::PathBuf;

    const FRAMES: i64 = 48;
    const DURATION: i64 = 47 * 320 + 100;

    /// This program's levels: header first, each start before its end, a short last frame.
    fn provider_frames() -> String {
        let spellings = [
            "0.0",
            "0.012345679",
            "1.0",
            "0.5",
            "3.0517578e-5",
            "0.25",
            "0.0009765625",
        ];
        let frame = |index: i64| {
            format!(
                r#"{{"start_sample":{},"end_sample":{},"level":{}}}"#,
                index * 320,
                (index * 320 + 320).min(DURATION),
                spellings[index as usize % spellings.len()]
            )
        };
        (0..FRAMES).map(frame).collect::<Vec<_>>().join(",")
    }

    fn provider_levels() -> Vec<u8> {
        format!(
            r#"{{"version":1,"sample_rate":16000,"stream_evidence":"fixture","duration_samples":{DURATION},"activity":[{}]}}"#,
            provider_frames()
        )
        .into_bytes()
    }

    /// MPCut's levels: sorted keys, and no fraction on a whole number.
    fn own_levels() -> Vec<u8> {
        let spellings = ["0", "0.012345679", "1", "0.5", "3.0517578e-05", "0.25"];
        let frame = |index: i64| {
            format!(
                r#"{{"end_sample":{},"level":{},"start_sample":{}}}"#,
                (index * 320 + 320).min(DURATION),
                spellings[index as usize % spellings.len()],
                index * 320
            )
        };
        format!(
            r#"{{"activity":[{}],"duration_samples":{DURATION},"sample_rate":16000,"stream_evidence":"fixture","version":1}}"#,
            (0..FRAMES).map(frame).collect::<Vec<_>>().join(",")
        )
        .into_bytes()
    }

    fn result(key: &str, activity: &str) -> Vec<u8> {
        format!(
            r#"{{"version":1,"sample_rate":16000,"recognition_key":"{}","stream_evidence":"fixture","duration_samples":{DURATION},"words":[{{"id":"w1","utterance_id":"u1","text":"hello","start_sample":320,"end_sample":3200,"speaker_id":null,"precision":"word"}}],"activity":[{activity}],"warning":null}}"#,
            key.repeat(64)
        )
        .into_bytes()
    }

    struct Sample {
        name: &'static str,
        result: Vec<u8>,
        activity: Option<Vec<u8>>,
        digest: &'static str,
        /// The method of each stored document, which shows that the file exercises it.
        methods: &'static [u8],
        from_mpcut: &'static str,
        from_transcriberrr: &'static str,
    }

    fn samples() -> [Sample; 3] {
        [
            Sample {
                name: "provider-levels",
                result: result("a", ""),
                activity: Some(provider_levels()),
                digest: "80225a610ac8bd6f95bd9bae9247ef62461d04d779f6faf993382a285cf2bff4",
                methods: &[1, 3],
                from_mpcut: "TVBDWAIBAACAIlphCsi9b5W9m66SR+9iRh0E13n2+vmTOCooXPK/9AFLAQAAtQAAAKWQvQ7CMAyE38Vzh5a/oa+CUGU1BixCWrlOW4R4d46WgZ1M8cW+7+InjWKDdonqqqCB732UxtgF9aEsy4JM2u6S1NHT3ORBNfGfhwByE743MmqQ1AJGZ509m+AtZPA/tDXNgCT7aockU2cB1fFJGjAxVWjO7mIMi2bR8kdzmR33q8TYLSw2/5pRvd3ASVL4FaAMvfBNbHFJOcaCenxc180sZHqdCuLWdVTHFo6oJrak6bJOvN4D7woAAIEAAABTsWYAgyQgrlYqSy0qzszPU7Iy1FEqTswtyEmNL0osSQXyzQwMDIBiJUWpibnxqWWZKal5yUBxpbTMipLSolQlHaWUUqBSoOZ4iMZioCZTQxOgpsTkksyyzJJKJatoLn4uRmXX9PlS/FzxVqzi4b1b/3Op/udi4CwLA1JDQzK2FgA=",
                from_transcriberrr: "TVBDWAIBAACAIlphCsi9b5W9m66SR+9iRh0E13n2+vmTOCooXPK/9AFLAQAAtQAAAKWQvQ7CMAyE38Vzh5a/oa+CUGU1BixCWrlOC0K8O9eWgZ1M8cW+7+IXjWKDdonqqqCB732UxtgF9aEsy4JM2u6S1NHT3ORJNfGfhwByE743MmqQ1AJGZ314NsFbyODPtDXNgCT7aockU2cB1fFFGjAxVWjO7mIMi2bR8qy5PBz3q8TYLSw2/5pRvd3ASVL4FaAMvfBNbHFJOcaCenxc180sZHqfCuLWdVTHFo6oJrak6bJOvD8D7woAAI0AAADViz0KwkAQhWNh4TYpBctRuxQbUIu19gpaiCxrMsqAibI7WRTxCJ7BU3i+OJhTZGB4vJ9vtk7+d5R/QkQf6FqDyTMIrrpd0HrHKH6ltZaMPbrKYqQS60JyONGdG4+QQdnIVGDbgUGgZb4QyBVMkfgBZq9SNZhuzp9JqqwZjnfvb6vmrUpGcSvSj/Lw+gE=",
            },
            Sample {
                name: "own-levels",
                result: b"{}".to_vec(),
                activity: Some(own_levels()),
                digest: "fdbe01857120da5330faaa5eafa91a25f1266e6747bfe9a51f01fe67aea9c57e",
                methods: &[0, 2],
                from_mpcut: "TVBDWAIBAAD9vgGFcSDaUzD6ql6vqRol8SZuZ0e/6aUfAf5nrqnFfgACAAAAAgAAAHt9ArMKAABpAAAAU7FmAAN+LkZl1/T58lzxVqzi4b0b4rlU/w8msVgdpZTSosSSzPy8+OLE3IKc1GIlK0NTQxMDHSUIPx4omwoUMzMwAImVFKUm5sanlmWmpOYlA8WV0jIrSkqLUpV0lMpSi4qB5gDV1gIA",
                from_transcriberrr: "TVBDWAIBAAD9vgGFcSDaUzD6ql6vqRol8SZuZ0e/6aUfAf5nrqnFfgACAAAAAgAAAHt9ArMKAAB6AAAAzcw9DoJAEAVgLWy2sbQ0GbXbYknUAmvPYGUmGxjIJvxldyE0HIEzcAVuCANcgunmey/v/jmsdxbH2zcdrgLD0+XXjyge057sLyGurfamLNDpvMrIQRi8gqeSsP3IKbG9lVrMW9I5UmNiKiJ2SEzra0sgoSHreIe73Qw=",
            },
            Sample {
                name: "embedded-levels",
                result: result("a", &provider_frames()),
                activity: None,
                digest: "8ce67567a3e6e9b4441b9a67235774645e97ce711a52ac56397dcb7dd202cfaf",
                methods: &[3],
                from_mpcut: "TVBDWAIAAACM5nVno+bptEQbmmcjV3RkXpfOcRpSrFY5fct90gLPrwPWCwAA7wAAANVQO1LEMAzNFhS42ZIZSgFdioRfYWquAMUO4/HEInjW62QUOVlmZ4/AGTgF51u0DgU9FWpsPUnvPenyocihF0WxgxFp8F0EXZcw2E0f0JBllPy+qqoSCJuujZ6lx6zxHTTYPwaIEBPajcHRO4yNiMGr33IilJpLon9Um90M4uSuvhUnU0dOstUOvJOJqZbmxIxkhcJkLB0xxi3L/w1D6LKWJf4hA31zLUwY3W9AkKFHu0bKLDGFUEIvi/v5MlkZ9i8l2Ib96FmusFJLtbh4bD/Pl8rok7Pnj6+Dujqo4nR8kud/FGWjyVL0sZ233n8D",
                from_transcriberrr: "TVBDWAIAAACM5nVno+bptEQbmmcjV3RkXpfOcRpSrFY5fct90gLPrwPWCwAA7wAAANVQO1LEMAzNFhS42ZIZSgFdioRfYWquAMUO4/HEInjW62QUOVlmZ4/AGTgF51u0DgU9FWpsPUnv6enyocihF0WxgxFp8F0EXZcw2E0f0JBllPy+qqoSCJuujZ6lx6zxHTTYPwaIEBPajcHRO4yNiMGr33IilJpLon9Um7cZZJO7+lY2mTpykq124J1MTLU0J2YkKxQmY+mIMW5Z/m8YQpe1LPEPGeiba2HC6H4Dggw92jVSZokphBJ6Me7ny2Rl2L+UYBv2o2e5wkot1eLisf08XyqjT86eP74O6uqgitPxSZ7/URRHk6XoYzu73n8D",
            },
        ]
    }

    fn methods(file: &[u8]) -> Vec<u8> {
        let (mut found, mut position) = (Vec::new(), HEADER);
        while position + 9 <= file.len() {
            found.push(file[position]);
            let stored = u32::from_le_bytes(file[position + 5..position + 9].try_into().unwrap());
            position += 9 + stored as usize;
        }
        found
    }

    #[test]
    fn each_program_reads_the_files_the_other_wrote() {
        for sample in samples() {
            let activity = sample.activity.as_deref();
            assert_eq!(
                hex(&digest(&sample.result, activity)),
                sample.digest,
                "{}",
                sample.name
            );
            for (writer, text) in [
                ("MPCut", sample.from_mpcut),
                ("Transcriberrr", sample.from_transcriberrr),
            ] {
                let file = STANDARD.decode(text).unwrap();
                assert_eq!(
                    hex(&file[8..HEADER]),
                    sample.digest,
                    "{} from {writer}",
                    sample.name
                );
                assert_eq!(
                    methods(&file),
                    sample.methods,
                    "{} from {writer}",
                    sample.name
                );
                let (result, levels) = decode(&file).unwrap();
                assert_eq!(result, sample.result, "{} from {writer}", sample.name);
                assert_eq!(levels, sample.activity, "{} from {writer}", sample.name);
            }
            // This build still writes what the checked-in file holds.
            let written = encode(&sample.result, activity).unwrap();
            assert_eq!(methods(&written), sample.methods, "{}", sample.name);
            let (result, levels) = decode(&written).unwrap();
            assert_eq!(result, sample.result, "{}", sample.name);
            assert_eq!(levels, sample.activity, "{}", sample.name);
        }
    }

    #[test]
    fn every_document_reads_back_as_its_own_bytes() {
        let round_trip = |result: &[u8], activity: Option<&[u8]>, expected: &[u8]| {
            let file = encode(result, activity).unwrap();
            assert_eq!(methods(&file), expected);
            let (read, levels) = decode(&file).unwrap();
            assert_eq!(read, result);
            assert_eq!(levels.as_deref(), activity);
        };
        let result = result("a", "");
        round_trip(&result, Some(&provider_levels()), &[1, 3]);
        round_trip(&result, Some(&own_levels()), &[1, 2]);
        // Levels that are present but empty differ from no levels, under the same digest.
        round_trip(&result, Some(&[]), &[1, 0]);
        round_trip(&result, None, &[1]);
        assert_eq!(digest(&result, Some(&[])), digest(&result, None));
        // A layout that neither packing describes is compressed as it is.
        let spaced = String::from_utf8(provider_levels())
            .unwrap()
            .replace(r#""start_sample":0,"#, r#""start_sample": 0,"#);
        round_trip(&result, Some(spaced.as_bytes()), &[1, 1]);
        // One frame packs to less than DEFLATE can shorten, so its document is compressed whole.
        let single = br#"{"version":1,"sample_rate":16000,"stream_evidence":"fixture","duration_samples":7,"activity":[{"start_sample":0,"end_sample":7,"level":0.5}]}"#;
        let file = encode(&result, Some(single)).unwrap();
        assert_eq!(decode(&file).unwrap().1.as_deref(), Some(&single[..]));
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let noise: Vec<u8> = (0..5000)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 56) as u8
            })
            .collect();
        round_trip(&noise, Some(&noise), &[0, 0]);
        // Evidence may quote the array's key before the levels begin.
        let quoting = String::from_utf8(provider_levels()).unwrap().replace(
            r#""fixture""#,
            r#"{"note":"\"activity\":[","activity":[1]}"#,
        );
        round_trip(&result, Some(quoting.as_bytes()), &[1, 3]);
    }

    #[test]
    fn damaged_or_unknown_files_are_not_exchanges() {
        let file = encode(&result("a", ""), Some(&provider_levels())).unwrap();
        let reads = |change: &dyn Fn(&mut Vec<u8>)| {
            let mut copy = file.clone();
            change(&mut copy);
            decode(&copy).is_ok()
        };
        assert!(reads(&|_| ()));
        assert!(!reads(&|file| file[0] = b'N'));
        assert!(!reads(&|file| file[4] = 3));
        assert!(!reads(&|file| file[5] = 3));
        assert!(!reads(&|file| file[5] = 0));
        assert!(!reads(&|file| file[6] = 1));
        assert!(!reads(&|file| file[8] ^= 1));
        assert!(!reads(&|file| file[40] = 9));
        assert!(!reads(&|file| file[41] ^= 1));
        assert!(!reads(&|file| file[44] = 0x7F));
        assert!(!reads(&|file| {
            let middle = file.len() - 20;
            file[middle] ^= 0x10
        }));
        assert!(!reads(&|file| file.push(0)));
        assert!(!reads(&|file| file.truncate(file.len() - 1)));
        for length in [0, 3, 39, 40, 48] {
            assert!(decode(&file[..length]).is_err());
        }
        assert!(decode(br#"{"result":"","digest":""}"#).is_err());
    }

    fn folder(name: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!(
            "transcriberrr-exchange-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&folder);
        fs::create_dir_all(&folder).unwrap();
        folder
    }

    #[test]
    fn shares_the_version_mpcut_reads_and_reads_both() {
        let folder = folder("share");
        let track: Track = serde_json::from_value(serde_json::json!({
            "track_id": "t", "source_id": "s", "stream_evidence": "fixture",
            "binding": {"stream_index": 0, "origin": {"numerator": 0, "denominator": 1},
                "session_start": {"numerator": 0, "denominator": 1}},
            "channel": null, "name": "Track", "role": "isolated_mic", "speaker": null,
            "recognize": true, "start_frame": 0, "audio": null,
            "result": folder.join("result.json"), "activity": folder.join("activity.json"),
            "recognition_key": "a".repeat(64), "unavailable": null, "shared_cache": null,
        }))
        .unwrap();
        let legacy = folder.join(format!("{}.json", "c".repeat(64)));
        let current = legacy.with_extension(EXTENSION);
        let (first, levels) = (result("a", ""), provider_levels());
        let second = String::from_utf8(first.clone())
            .unwrap()
            .replace("hello", "again")
            .into_bytes();
        let stored = |path: &Path| {
            let bundle: Bundle = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
            let result = STANDARD.decode(&bundle.result).unwrap();
            let activity = bundle.activity.map(|text| STANDARD.decode(text).unwrap());
            assert_eq!(
                bundle.digest,
                artifacts_digest(&result, activity.as_deref())
            );
            (result, activity)
        };

        // An MPCut that has not said it reads version 2 is written version 1, as before.
        assert_eq!(reader_version(&folder), 1);
        assert_eq!(shared(&track, &legacy), None);
        share(&track, &legacy, 1, first.clone(), Some(levels.clone())).unwrap();
        assert!(!current.exists());
        assert_eq!(stored(&legacy), (first.clone(), Some(levels.clone())));
        assert_eq!(
            shared(&track, &legacy),
            Some((first.clone(), Some(levels.clone())))
        );
        let legacy_size = fs::metadata(&legacy).unwrap().len();
        // The result already shared stays, whatever version MPCut reads now.
        share(&track, &legacy, 2, second.clone(), Some(levels.clone())).unwrap();
        assert!(!current.exists());
        assert_eq!(stored(&legacy).0, first);

        // One that has is written version 2, and a stale version 1 file goes.
        fs::write(&legacy, b"stale").unwrap();
        for (text, version) in [("", 1), ("soon\n", 1), ("1\n", 1), ("2\n", 2), ("3", 3)] {
            fs::write(folder.join(READER), text).unwrap();
            assert_eq!(reader_version(&folder), version);
        }
        share(&track, &legacy, 2, first.clone(), Some(levels.clone())).unwrap();
        assert!(!legacy.exists());
        let file = fs::read(&current).unwrap();
        assert_eq!(
            decode(&file).unwrap(),
            (first.clone(), Some(levels.clone()))
        );
        assert_eq!(
            hex(&file[8..HEADER]),
            artifacts_digest(&first, Some(&levels))
        );
        assert!(file.len() as u64 * 4 < legacy_size);
        assert_eq!(
            shared(&track, &legacy),
            Some((first.clone(), Some(levels.clone())))
        );
        share(&track, &legacy, 2, second.clone(), Some(levels.clone())).unwrap();
        assert_eq!(decode(&fs::read(&current).unwrap()).unwrap().0, first);

        // For an MPCut that reads only version 1, the shared result is written again as that.
        share(&track, &legacy, 1, second.clone(), Some(levels.clone())).unwrap();
        assert!(!current.exists());
        assert_eq!(stored(&legacy), (first.clone(), Some(levels.clone())));

        // A current file that is damaged, or holds another key's result, leaves version 1 to answer.
        fs::write(&current, &file[..file.len() - 7]).unwrap();
        assert_eq!(shared(&track, &legacy).unwrap().0, first);
        let other = encode(&result("b", ""), Some(&levels)).unwrap();
        fs::write(&current, &other).unwrap();
        assert_eq!(shared(&track, &legacy).unwrap().0, first);
        fs::remove_file(&legacy).unwrap();
        assert_eq!(shared(&track, &legacy), None);
        // MPCut's own layout of the levels is as good as this program's.
        fs::write(&current, encode(&first, Some(&own_levels())).unwrap()).unwrap();
        assert_eq!(shared(&track, &legacy), Some((first, Some(own_levels()))));
        fs::remove_dir_all(&folder).unwrap();
    }

    /// With `MPCUT_TEST_EXCHANGE_FOLDER` set, prints each sample for MPCut's tests to read
    /// and writes ten minutes and three hours of synthetic recognition there. For every
    /// `<name>.result.json` in the folder (with `<name>.activity.json`), reports both
    /// versions' sizes, writes `<name>.transcriberrr.mpx`, and reads `<name>.mpcut.mpx`.
    #[test]
    fn exchanges_files_with_mpcut_in_a_folder() {
        let Some(folder) = std::env::var_os("MPCUT_TEST_EXCHANGE_FOLDER").map(PathBuf::from) else {
            return;
        };
        for sample in samples() {
            let file = encode(&sample.result, sample.activity.as_deref()).unwrap();
            println!(
                "sample {} digest={} methods={:?} transcriberrr={}",
                sample.name,
                hex(&file[8..HEADER]),
                methods(&file),
                STANDARD.encode(&file)
            );
        }
        let vocabulary = [
            "so",
            "the",
            "recording",
            "was",
            "about",
            "three",
            "hours",
            "long",
        ];
        for (name, count) in [("provider-10min", 30_000_i64), ("provider-3h", 540_000)] {
            let mut state = 0x9E37_79B9_7F4A_7C15_u64;
            let (mut speech, mut remaining, mut utterance) = (false, 0, 0);
            let (mut activity, mut words) = (Vec::new(), Vec::<LocalWord>::new());
            let duration = count * 320 - 60;
            for index in 0..count {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (state >> 40) as f32 / (1_u32 << 24) as f32;
                if remaining == 0 {
                    speech = !speech;
                    utterance += usize::from(speech);
                    remaining = 25 + (state >> 33) as usize % if speech { 200 } else { 100 };
                }
                remaining -= 1;
                let level = if speech {
                    0.02 + 0.28 * unit * (index as f32 * 0.21).sin().abs()
                } else {
                    0.0002 + 0.0008 * unit
                };
                let start_sample = index * 320;
                if speech && index % 12 == 0 && index + 10 < count {
                    words.push(LocalWord {
                        id: format!("w{}", words.len()),
                        utterance_id: format!("u{utterance}"),
                        text: vocabulary[words.len() % vocabulary.len()].into(),
                        start_sample,
                        end_sample: start_sample + 9 * 320,
                        speaker_id: None,
                        precision: Precision::Word,
                    });
                }
                activity.push(Activity {
                    start_sample,
                    end_sample: (start_sample + 320).min(duration),
                    level,
                });
            }
            let levels = ActivityArtifact {
                version: 1,
                sample_rate: 16_000,
                stream_evidence: Some("measured".into()),
                duration_samples: duration,
                activity,
            };
            let recognition = TrackArtifact {
                version: 1,
                sample_rate: 16_000,
                recognition_key: Some("b".repeat(64)),
                stream_evidence: Some("measured".into()),
                duration_samples: duration,
                words,
                activity: Vec::new(),
                warning: None,
            };
            let write = |suffix: &str, document: Vec<u8>| {
                fs::write(folder.join(format!("{name}{suffix}")), document).unwrap()
            };
            write(".result.json", serde_json::to_vec(&recognition).unwrap());
            write(".activity.json", serde_json::to_vec(&levels).unwrap());
        }
        let mut names: Vec<String> = fs::read_dir(&folder)
            .unwrap()
            .filter_map(|entry| entry.unwrap().file_name().into_string().ok())
            .filter_map(|name| name.strip_suffix(".result.json").map(str::to_owned))
            .collect();
        names.sort();
        for name in names {
            let path = |suffix: &str| folder.join(format!("{name}{suffix}"));
            let result = fs::read(path(".result.json")).unwrap();
            let activity = fs::read(path(".activity.json")).ok();
            let legacy = serde_json::to_vec(&Bundle {
                result: STANDARD.encode(&result),
                activity: activity.as_ref().map(|levels| STANDARD.encode(levels)),
                digest: artifacts_digest(&result, activity.as_deref()),
            })
            .unwrap();
            let started = std::time::Instant::now();
            let file = encode(&result, activity.as_deref()).unwrap();
            let encoded = started.elapsed();
            assert_eq!(
                decode(&file).unwrap(),
                (result.clone(), activity.clone()),
                "{name}"
            );
            let decoded = started.elapsed() - encoded;
            fs::write(path(".transcriberrr.mpx"), &file).unwrap();
            let compressed = |document: &[u8]| {
                9 + deflate(document).map_or(document.len(), |stored| stored.len())
            };
            let unpacked = HEADER + compressed(&result) + activity.as_deref().map_or(0, compressed);
            let mut line = format!(
                "exchange {name}: documents={} v1={} v2={} compression_only={unpacked} methods={:?} digest={} encode={:.2}s decode={:.2}s",
                result.len() + activity.as_ref().map_or(0, Vec::len),
                legacy.len(),
                file.len(),
                methods(&file),
                hex(&file[8..HEADER]),
                encoded.as_secs_f32(),
                decoded.as_secs_f32()
            );
            if let Ok(other) = fs::read(path(".mpcut.mpx")) {
                assert_eq!(decode(&other).unwrap(), (result, activity), "{name}");
                assert_eq!(other[8..HEADER], file[8..HEADER], "{name}");
                line += &format!(" mpcut_v2={} read_identically=true", other.len());
            }
            println!("{line}");
        }
    }
}
