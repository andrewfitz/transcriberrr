//! Format fallbacks for audio that Symphonia or AVFoundation cannot read directly.
//!
//! Symphonia 0.5 has no Opus decoder, and AVFoundation cannot open WebM at all. macOS's
//! `afconvert` (Core Audio) decodes Ogg Opus, so Opus in Matroska/WebM — what browsers record —
//! is rewrapped as Ogg Opus without re-encoding, then decoded by Core Audio.

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use symphonia::core::checksum::Crc32;
use symphonia::core::codecs::{CODEC_TYPE_AAC, CODEC_TYPE_NULL, CODEC_TYPE_OPUS};
use symphonia::core::errors::Error as FormatError;
use symphonia::core::formats::{FormatOptions, FormatReader, Packet};
use symphonia::core::io::{MediaSourceStream, Monitor};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::get_probe;

const AFCONVERT: &str = "/usr/bin/afconvert";

/// A temporary file removed on drop, including when a conversion fails part-way.
pub(crate) struct TempPath(PathBuf);

impl TempPath {
    fn next_path(extension: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let name = format!(
            "transcriberrr-{}-{}.{extension}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        std::env::temp_dir().join(name)
    }

    /// Reserve the temporary pathname before handing it to an encoder. Only remove files
    /// created by this process, including when a PID from an earlier run has been reused.
    pub(crate) fn create(extension: &str) -> Result<Self> {
        loop {
            let path = Self::next_path(extension);
            match File::create_new(&path) {
                Ok(_) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| format!("creating {}", path.display()));
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn new(extension: &str) -> Self {
        Self(Self::next_path(extension))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Remove temporary files left by CLI processes that were force-killed (SIGKILL skips `Drop`).
/// Names carry the owning PID; files whose process no longer exists are stale.
pub(crate) fn remove_stale_temp_files() {
    remove_stale_files(&std::env::temp_dir(), |name| {
        let (pid, tail) = name.strip_prefix("transcriberrr-")?.split_once('-')?;
        let (serial, extension) = tail.split_once('.')?;
        if !matches!(extension, "caf" | "ogg" | "wav") || serial.parse::<u32>().is_err() {
            return None;
        }
        pid.parse().ok()
    });
}

/// Remove only files the caller recognizes as temporary and whose owner process is gone.
pub(crate) fn remove_stale_files(folder: &Path, owner_pid: impl Fn(&str) -> Option<libc::pid_t>) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(&owner_pid).filter(|&pid| pid > 0) else {
            continue;
        };
        // kill(pid, 0) checks for existence without sending a signal.
        let alive = unsafe { libc::kill(pid, 0) } == 0
            || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH);
        if !alive {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Decode `input` with Core Audio into a 16 kHz float CAF that Symphonia can read.
pub(crate) fn convert_with_core_audio(input: &Path) -> Result<TempPath> {
    // Core Audio reads Ogg Opus but no Matroska/WebM, so rewrap Opus packets first.
    let remuxed = if is_ogg(input)? {
        None
    } else {
        remux_opus_to_ogg(input)?
    };
    let source = remuxed.as_ref().map_or(input, TempPath::path);
    // CAF has 64-bit sizes; WAV caps at 4 GB, which 48 kHz stereo float passes after ~3 h.
    // Converting to the model's 16 kHz here also cuts the temporary file to a third.
    let converted = TempPath::create("caf")?;
    let output = Command::new(AFCONVERT)
        .args(["-f", "caff", "-d", "LEF32@16000"])
        .arg(source)
        .arg(converted.path())
        .output()
        .with_context(|| format!("launching {AFCONVERT}"))?;
    if !output.status.success() {
        bail!(
            "Core Audio could not decode {}: {}",
            input.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(converted)
}

fn is_ogg(path: &Path) -> Result<bool> {
    let mut magic = [0u8; 4];
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(file.read_exact(&mut magic).is_ok() && &magic == b"OggS")
}

/// Zoom saves the same AAC audio in the meeting video and its audio-only export.
/// Compare the complete stream so a renamed video with different audio stays in the queue.
pub(crate) fn same_aac_audio(left: &Path, right: &Path) -> Result<bool> {
    fn probe(path: &Path) -> Result<Box<dyn FormatReader>> {
        let stream = MediaSourceStream::new(Box::new(File::open(path)?), Default::default());
        Ok(get_probe()
            .format(
                &Hint::new(),
                stream,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )?
            .format)
    }
    fn aac_track(reader: &dyn FormatReader) -> Option<&symphonia::core::formats::Track> {
        let mut tracks = reader.tracks().iter().filter(|track| {
            track.codec_params.sample_rate.is_some() && track.codec_params.codec != CODEC_TYPE_NULL
        });
        let track = tracks.next()?;
        (track.codec_params.codec == CODEC_TYPE_AAC && tracks.next().is_none()).then_some(track)
    }
    fn next_audio(reader: &mut dyn FormatReader, id: u32) -> Result<Option<Packet>> {
        loop {
            match reader.next_packet() {
                Ok(packet) if packet.track_id() == id => return Ok(Some(packet)),
                Ok(_) => {}
                Err(FormatError::IoError(error))
                    if error.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None);
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    let (mut left, mut right) = (probe(left)?, probe(right)?);
    let (Some(a), Some(b)) = (aac_track(left.as_ref()), aac_track(right.as_ref())) else {
        return Ok(false);
    };
    let (a_id, b_id) = (a.id, b.id);
    let (a, b) = (&a.codec_params, &b.codec_params);
    let Some(expected_frames) = a.n_frames else {
        return Ok(false);
    };
    if expected_frames == 0
        || b.n_frames != Some(expected_frames)
        || a.time_base.map(|base| (base.numer, base.denom))
            != b.time_base.map(|base| (base.numer, base.denom))
        || a.sample_rate != b.sample_rate
        || a.channels != b.channels
        || a.extra_data != b.extra_data
    {
        return Ok(false);
    }
    let mut frames = 0u64;
    loop {
        match (
            next_audio(left.as_mut(), a_id)?,
            next_audio(right.as_mut(), b_id)?,
        ) {
            (Some(a), Some(b)) if a.ts == b.ts && a.dur == b.dur && a.data == b.data => {
                frames = frames.saturating_add(a.dur);
                if frames > expected_frames {
                    return Ok(false);
                }
            }
            (None, None) => return Ok(frames == expected_frames),
            _ => return Ok(false),
        }
    }
}

/// Rewrap the default Opus track of a Matroska/WebM file as Ogg Opus. Returns `None` when the
/// file has no Opus track (Core Audio then gets the original file).
fn remux_opus_to_ogg(input: &Path) -> Result<Option<TempPath>> {
    let file = File::open(input).with_context(|| format!("opening {}", input.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = input.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let Ok(probed) = get_probe().format(
        &hint,
        stream,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    ) else {
        return Ok(None);
    };
    let mut format = probed.format;
    // The Opus track, not `default_track()`: screen recordings list video first.
    let Some(track) = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec == CODEC_TYPE_OPUS)
    else {
        return Ok(None);
    };
    let track_id = track.id;
    let channels = track
        .codec_params
        .channels
        .map_or(1, |channels| channels.count()) as u8;
    let head = match track.codec_params.extra_data.as_deref() {
        Some(head) if head.starts_with(b"OpusHead") => head.to_vec(),
        _ => opus_head(channels.max(1)),
    };
    let output = TempPath::create("ogg")?;
    let mut writer = OggWriter::new(BufWriter::new(
        File::create(output.path())
            .with_context(|| format!("creating {}", output.path().display()))?,
    ));
    writer.page(&head, 0, PAGE_FIRST)?;
    writer.page(&opus_tags(), 0, 0)?;
    let mut granule = 0u64;
    let mut pending: Option<Box<[u8]>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) if packet.track_id() == track_id => packet,
            Ok(_) => continue,
            Err(symphonia::core::errors::Error::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            // A truncated recording (a browser upload cut short) ends in a damaged element;
            // keep what came before it.
            Err(error) if pending.is_some() => {
                eprintln!(
                    "warning: {}: stopped reading at {:.1} s: {error}",
                    input.display(),
                    granule as f64 / 48_000.0
                );
                break;
            }
            Err(error) => return Err(error.into()),
        };
        // Hold one packet back so the final page can carry the end-of-stream flag.
        if let Some(previous) = pending.replace(packet.data) {
            granule += opus_packet_samples(&previous);
            writer.page(&previous, granule, 0)?;
        }
    }
    let Some(last) = pending else {
        bail!("{} contains no Opus audio packets", input.display());
    };
    granule += opus_packet_samples(&last);
    writer.page(&last, granule, PAGE_LAST)?;
    writer.finish()?;
    Ok(Some(output))
}

/// Minimal `OpusHead` for streams without codec private data (RFC 7845 §5.1).
fn opus_head(channels: u8) -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.push(1); // version
    head.push(channels);
    head.extend(0u16.to_le_bytes()); // pre-skip
    head.extend(48_000u32.to_le_bytes()); // original input rate (informational)
    head.extend(0i16.to_le_bytes()); // output gain
    head.push(0); // mapping family: mono/stereo
    head
}

fn opus_tags() -> Vec<u8> {
    let vendor = b"transcriberrr";
    let mut tags = b"OpusTags".to_vec();
    tags.extend((vendor.len() as u32).to_le_bytes());
    tags.extend(vendor);
    tags.extend(0u32.to_le_bytes()); // no user comments
    tags
}

/// Decoded duration of one Opus packet at 48 kHz, from its TOC byte (RFC 6716 §3.1).
fn opus_packet_samples(packet: &[u8]) -> u64 {
    let Some(&toc) = packet.first() else {
        return 0;
    };
    let config = (toc >> 3) as usize;
    let frame = match config {
        0..=11 => [480, 960, 1920, 2880][config % 4],
        12..=15 => [480, 960][config % 2],
        _ => [120, 240, 480, 960][config % 4],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => packet.get(1).map_or(0, |count| u64::from(count & 0x3F)),
    };
    frame * frames
}

const PAGE_FIRST: u8 = 0x02;
const PAGE_LAST: u8 = 0x04;

/// Writes one packet per Ogg page (RFC 3533). Opus packets are far below the 65,025-byte
/// single-page limit, so continuation pages are unnecessary.
struct OggWriter<W: Write> {
    out: W,
    sequence: u32,
    page: Vec<u8>,
}

impl<W: Write> OggWriter<W> {
    fn new(out: W) -> Self {
        Self {
            out,
            sequence: 0,
            page: Vec::new(),
        }
    }

    fn page(&mut self, packet: &[u8], granule: u64, flags: u8) -> Result<()> {
        if packet.len() >= 255 * 255 {
            bail!(
                "Opus packet of {} bytes is too large to rewrap",
                packet.len()
            );
        }
        let full_segments = packet.len() / 255;
        let page = &mut self.page;
        page.clear();
        page.extend(b"OggS");
        page.push(0); // stream structure version
        page.push(flags);
        page.extend(granule.to_le_bytes());
        page.extend(1u32.to_le_bytes()); // bitstream serial number
        page.extend(self.sequence.to_le_bytes());
        page.extend([0u8; 4]); // CRC, filled below
        page.push((full_segments + 1) as u8);
        page.resize(27 + full_segments, 255);
        page.push((packet.len() % 255) as u8);
        page.extend(packet);
        // Ogg uses an initial CRC state of zero and includes the zeroed checksum field.
        let mut crc = Crc32::new(0);
        crc.process_buf_bytes(page);
        page[22..26].copy_from_slice(&crc.crc().to_le_bytes());
        self.out.write_all(page)?;
        self.sequence += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.out.flush()?;
        Ok(())
    }
}

/// Write 32-bit float mono WAV, used to hand already-decoded audio to the Apple helper.
pub(crate) fn write_float_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    // RIFF sizes are u32 and include the 36 header bytes after the size field.
    let data_len = u32::try_from(samples.len() * 4)
        .ok()
        .filter(|length| length.checked_add(36).is_some())
        .context("audio too long for WAV")?;
    let mut out =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    out.write_all(b"RIFF")?;
    out.write_all(&(36 + data_len).to_le_bytes())?;
    out.write_all(b"WAVEfmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&3u16.to_le_bytes())?; // IEEE float
    out.write_all(&1u16.to_le_bytes())?; // mono
    out.write_all(&sample_rate.to_le_bytes())?;
    out.write_all(&(sample_rate * 4).to_le_bytes())?; // byte rate
    out.write_all(&4u16.to_le_bytes())?; // block align
    out.write_all(&32u16.to_le_bytes())?; // bits per sample
    out.write_all(b"data")?;
    out.write_all(&data_len.to_le_bytes())?;
    for sample in samples {
        out.write_all(&sample.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symphonia::core::codecs::CODEC_TYPE_OPUS;

    #[test]
    fn opus_packet_durations_follow_the_toc_byte() {
        assert_eq!(opus_packet_samples(&[31 << 3]), 960); // CELT 20 ms, one frame
        assert_eq!(opus_packet_samples(&[(1 << 3) | 1]), 1920); // SILK 20 ms x2
        assert_eq!(opus_packet_samples(&[(16 << 3) | 3, 0x05]), 600); // CELT 2.5 ms x5
        assert_eq!(opus_packet_samples(&[]), 0);
    }

    #[test]
    fn rewrapped_ogg_pages_parse_back_as_opus() {
        let path = TempPath::new("ogg");
        let mut writer = OggWriter::new(File::create(path.path()).unwrap());
        writer.page(&opus_head(1), 0, PAGE_FIRST).unwrap();
        writer.page(&opus_tags(), 0, 0).unwrap();
        // Include a packet whose length is exactly one lacing segment.
        let packets = [255, 300, 120, 510, 1]
            .into_iter()
            .enumerate()
            .map(|(index, size)| {
                let mut packet = vec![index as u8; size];
                packet[0] = 31 << 3;
                packet
            })
            .collect::<Vec<_>>();
        for (index, packet) in packets.iter().enumerate() {
            let flags = if index + 1 == packets.len() {
                PAGE_LAST
            } else {
                0
            };
            writer
                .page(packet, 960 * (index as u64 + 1), flags)
                .unwrap();
        }
        writer.finish().unwrap();
        assert!(is_ogg(path.path()).unwrap());

        // Symphonia's demuxer verifies each page CRC; a wrong CRC drops the pages.
        let stream = MediaSourceStream::new(
            Box::new(File::open(path.path()).unwrap()),
            Default::default(),
        );
        let mut format = get_probe()
            .format(
                &Hint::new(),
                stream,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .unwrap()
            .format;
        assert_eq!(
            format.default_track().unwrap().codec_params.codec,
            CODEC_TYPE_OPUS
        );
        let mut read = Vec::new();
        while let Ok(packet) = format.next_packet() {
            read.push(packet.data.to_vec());
        }
        assert_eq!(read, packets);
    }

    #[test]
    fn float_wav_has_a_canonical_header() {
        let path = TempPath::new("wav");
        write_float_wav(path.path(), &[0.0, 0.5, -0.5, 0.25], 16_000).unwrap();
        let bytes = std::fs::read(path.path()).unwrap();
        assert_eq!(bytes.len(), 44 + 16);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(f32::from_le_bytes(bytes[48..52].try_into().unwrap()), 0.5);
    }
}
