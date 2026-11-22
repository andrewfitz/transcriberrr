#!/usr/bin/env python3
"""Stream 24 kHz mono PCM16 WAV files. Requires websockets==15.0.1.

After sending native session.end, wait up to 30 seconds between transcript or
track progress events while the server drains buffered audio.
"""

import argparse
import asyncio
import base64
import json
import math
import os
import sys
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

from websockets.asyncio.client import connect

CLIENT_VERSION = "1"
SAMPLE_RATE = 24_000
PACKET_FRAMES = SAMPLE_RATE // 10
NATIVE = "transcriberrr."
NATIVE_DRAIN_IDLE_SECONDS = 30
NATIVE_DRAIN_PROGRESS = {
    NATIVE + "transcript.draft",
    NATIVE + "transcript.final",
    NATIVE + "transcript.retracted",
    NATIVE + "track.registered",
    NATIVE + "track.committed",
    NATIVE + "track.gap",
    NATIVE + "track.ended",
}
STANDARD_FINAL = "conversation.item.input_audio_transcription.completed"
STANDARD_FAILED = "conversation.item.input_audio_transcription.failed"


def endpoint(url, native, reconnecting=False):
    parts = urlsplit(url)
    query = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if name not in {"intent", "transcriberrr", "resume"}
    ]
    query.append(("intent", "transcription"))
    if native:
        query.append(("transcriberrr", "1"))
        if reconnecting:
            query.append(("resume", "1"))
    return parts._replace(path="/v1/realtime", query=urlencode(query)).geturl()


def open_socket(args, native, reconnecting=False):
    key = os.environ.get("TRANSCRIBERRR_API_KEY")
    protocols = ["realtime"]
    headers = {}
    if key and args.browser_auth:
        protocols.append(f"openai-insecure-api-key.{key}")
    elif key:
        headers["Authorization"] = f"Bearer {key}"
    return connect(
        endpoint(args.url, native, reconnecting),
        additional_headers=headers,
        subprotocols=protocols,
        origin=args.origin,
        # Native flow control pauses socket reads, including WebSocket pings.
        # Audio and drain progress keep this connection active through playback.
        ping_interval=None if native else 20,
    )


async def send(socket, kind, **fields):
    await socket.send(json.dumps({"type": kind, **fields}, separators=(",", ":")))


def encoded(data):
    return base64.b64encode(data).decode("ascii")


def read_pcm(path):
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
            SAMPLE_RATE,
            1,
            2,
        ):
            raise ValueError(f"Expected 24 kHz mono PCM16 WAV: {path}")
        frames = source.getnframes()
        pcm = source.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError(f"Incomplete PCM frames in WAV: {path}")
        return pcm


class Events:
    def __init__(self, declared_gaps=None):
        self.cursor = 0
        self.open_items = set()
        self.closed = set()
        self.revisions = {}
        self.declared_gaps = declared_gaps or {}
        self.unexpected_gap_frames = Counter()
        self.lost_events = 0
        self.max_reported_backlog_frames = 0

    def accept(self, event):
        kind = event["type"]
        if kind == NATIVE + "session.resumed":
            earliest = event["earliest_replayable_sequence"]
            lost = event.get("lost_range")
            expected_loss = (
                [self.cursor + 1, earliest - 1] if earliest > self.cursor + 1 else None
            )
            if lost != expected_loss:
                raise RuntimeError(f"Invalid lost sequence range: {lost}")
            if lost:
                self.cursor = lost[1]
                self.lost_events += lost[1] - lost[0] + 1
                # A lost terminal event might have resolved any prior draft.
                self.open_items.clear()
        sequence = event.get("sequence")
        if sequence is not None:
            if sequence <= self.cursor:
                return False
            if sequence != self.cursor + 1:
                raise RuntimeError(
                    f"Expected sequence {self.cursor + 1}, received {sequence}"
                )
            self.cursor = sequence
        if kind == "error":
            raise RuntimeError(json.dumps(event["error"]))
        if kind == NATIVE + "track.backlog":
            self.max_reported_backlog_frames = max(
                self.max_reported_backlog_frames, event["queued_frames"]
            )
        if kind == NATIVE + "track.gap":
            track = event["track_id"]
            start, end = event["start_frame"], event["end_frame"]
            if not 0 <= start < end:
                raise RuntimeError(f"Invalid gap for {track}: [{start}, {end})")
            unexpected = end - start
            if event["reason"] == "client_gap" and track in self.declared_gaps:
                left, right = self.declared_gaps[track]
                unexpected -= max(0, min(end, right) - max(start, left))
            if unexpected:
                self.unexpected_gap_frames[track] += unexpected
        item = event.get("item_id")
        if kind == NATIVE + "transcript.draft":
            revision = event["revision"]
            if item in self.closed or revision <= self.revisions.get(item, -1):
                raise RuntimeError(f"Invalid draft revision for {item}")
            self.open_items.add(item)
            self.revisions[item] = revision
        if kind in (NATIVE + "transcript.final", NATIVE + "transcript.retracted"):
            if item in self.closed:
                raise RuntimeError(f"Repeated terminal outcome for {item}")
            self.closed.add(item)
            self.open_items.discard(item)
            self.revisions.pop(item, None)
        if kind == STANDARD_FAILED:
            raise RuntimeError(json.dumps(event))
        redacted = {k: v for k, v in event.items() if k != "resume_credential"}
        print(json.dumps(redacted, ensure_ascii=False), flush=True)
        return True

    async def next(self, socket):
        while True:
            event = json.loads(await socket.recv())
            if self.accept(event):
                return event

    async def until(self, socket, kind):
        async with asyncio.timeout(30):
            while (event := await self.next(socket))["type"] != kind:
                pass
            return event


async def play_standard(args, audio):
    events = Events()
    async with open_socket(args, False) as socket:
        await events.until(socket, "session.created")
        await send(
            socket,
            "session.update",
            session={
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "transcription": {"model": args.model},
                        "turn_detection": None,
                    }
                },
            },
        )
        await events.until(socket, "session.updated")
        for _ in range(args.repeat):
            async with asyncio.TaskGroup() as tasks:
                reader = tasks.create_task(read_standard_turn(events, socket))
                clock = CaptureClock()
                for frame in range(0, len(audio) // 2, PACKET_FRAMES):
                    packet = audio[frame * 2 : (frame + PACKET_FRAMES) * 2]
                    await clock.capture_through(frame + len(packet) // 2)
                    await send(
                        socket, "input_audio_buffer.append", audio=encoded(packet)
                    )
                await send(socket, "input_audio_buffer.commit")
                async with asyncio.timeout(30):
                    await reader


async def read_standard_turn(events, socket):
    item = None
    partial_item = None
    while True:
        event = await events.next(socket)
        kind = event["type"]
        if kind == "conversation.item.input_audio_transcription.delta":
            if partial_item is not None and partial_item != event["item_id"]:
                raise RuntimeError("A delta changed item IDs during one turn")
            partial_item = event["item_id"]
        elif kind == "input_audio_buffer.committed":
            if item is not None:
                raise RuntimeError("Received multiple commits for one turn")
            item = event["item_id"]
        elif kind == STANDARD_FINAL:
            if (
                item is None
                or item != event["item_id"]
                or partial_item not in (None, item)
            ):
                raise RuntimeError("Delta, commit and completion disagree on item ID")
            return


class CaptureClock:
    def __init__(self, frame=0):
        self.frame = frame
        self.deadline = asyncio.get_running_loop().time()

    async def capture_through(self, frame):
        loop = asyncio.get_running_loop()
        interval = (frame - self.frame) / SAMPLE_RATE
        # Keep the capture cadence through normal timer jitter. Rebase after a
        # stalled write or a late wake so two packet groups never catch up at once.
        jitter = min(0.001, interval / 10)
        deadline = max(self.deadline, loop.time() - jitter) + interval
        await asyncio.sleep(max(0, deadline - loop.time()))
        self.deadline = max(deadline, loop.time() - jitter)
        self.frame = frame


async def collect_native(events, socket, drain_timeout):
    while (event := await events.next(socket))["type"] != NATIVE + "session.ended":
        if (
            drain_timeout.when() is not None
            and "sequence" in event
            and event["type"] in NATIVE_DRAIN_PROGRESS
        ):
            drain_timeout.reschedule(
                asyncio.get_running_loop().time() + NATIVE_DRAIN_IDLE_SECONDS
            )


async def register(socket, track, frame):
    await send(
        socket,
        NATIVE + "track.register",
        track_id=track,
        role="isolated_mic",
        speaker=track,
        start_frame=frame,
    )


async def append(socket, track, frame, packet):
    await send(
        socket,
        NATIVE + "track.append",
        track_id=track,
        start_frame=frame,
        audio=encoded(packet),
    )


async def resume(socket, greeting, events, accept_loss):
    fields = {
        "session_id": greeting["session_id"],
        "resume_credential": greeting["resume_credential"],
        "last_sequence": events.cursor,
    }
    await send(socket, NATIVE + "session.resume", **fields)
    loss_accepted = False
    async with asyncio.timeout(30):
        result = json.loads(await socket.recv())
        if result.get("error", {}).get("code") == "replay_unavailable" and accept_loss:
            await send(socket, NATIVE + "session.resume", **fields, accept_loss=True)
            loss_accepted = True
            result = json.loads(await socket.recv())
    if result["type"] == "error":
        events.accept(result)
    if (
        result["type"] != NATIVE + "session.resumed"
        or result["session_id"] != greeting["session_id"]
    ):
        raise RuntimeError(
            f"Expected resume acknowledgement, received {result['type']}"
        )
    if result.get("lost_range") is not None and not loss_accepted:
        raise RuntimeError("Server discarded replay events without explicit acceptance")
    events.accept(result)
    return result["next_expected_frames"]


@dataclass(frozen=True)
class Track:
    audio: bytes
    start: int
    gap: tuple[int, int] | None = None

    def omits(self, frame):
        return self.gap is not None and self.gap[0] <= frame < self.gap[1]


def packets(start, stop, loop_frames, boundaries):
    while start < stop:
        loop_end = (start // loop_frames + 1) * loop_frames
        boundary = next((frame for frame in boundaries if frame > start), stop)
        end = min(start + PACKET_FRAMES, stop, loop_end, boundary)
        yield start, end
        start = end


async def send_capture(socket, track_id, track, start, end, loop_frames):
    if track.omits(start):
        if end == track.gap[1]:
            await send(socket, NATIVE + "track.gap", track_id=track_id, end_frame=end)
        return
    offset = (start % loop_frames) * 2
    size = (end - start) * 2
    packet = track.audio[offset : offset + size].ljust(size, b"\0")
    await append(socket, track_id, start, packet)


async def restore_capture(socket, tracks, positions, through, loop_frames, boundaries):
    active = {
        track_id: track for track_id, track in tracks.items() if track.start < through
    }
    if positions.keys() - active.keys():
        raise RuntimeError("Resume returned an unregistered track")
    expected = {}
    for track_id, track in active.items():
        if track_id not in positions:
            await register(socket, track_id, track.start)
        position = positions.get(track_id, track.start)
        if not track.start <= position <= through:
            raise RuntimeError(f"Invalid resume frame for {track_id}: {position}")
        expected[track_id] = position

    first = min(expected.values(), default=through)
    clock = CaptureClock()
    replayed_frames = 0
    # The original clips are still available. Regenerate every missing sample on the
    # shared clock, keeping the intentional gap and any track's late start intact.
    for start, end in packets(first, through, loop_frames, boundaries):
        starts = {
            track_id: max(start, position)
            for track_id, position in expected.items()
            if position < end
        }
        pcm_start = min(
            (
                frame
                for track_id, frame in starts.items()
                if not active[track_id].omits(frame)
            ),
            default=end,
        )
        if pcm_start < end:
            replayed_frames += end - pcm_start
            await clock.capture_through(replayed_frames)
        for track_id, start_for_track in starts.items():
            await send_capture(
                socket, track_id, active[track_id], start_for_track, end, loop_frames
            )


async def play_native(args, audio):
    loop_frames = max(len(clip) for clip in audio) // 2
    duration_frames = loop_frames * args.repeat
    duration = duration_frames / SAMPLE_RATE
    fault_times = (args.reconnect_at, args.late_join, args.gap_at)
    if any(time is not None and time >= duration for time in fault_times):
        raise ValueError("Fault-injection times must be within the audio duration")
    reconnect_at, late_join, gap_start = (
        min(round(time * SAMPLE_RATE), duration_frames - 1)
        if time is not None
        else None
        for time in fault_times
    )
    boundaries = {
        frame for frame in (reconnect_at, late_join, gap_start) if frame is not None
    }
    gap = None
    if gap_start is not None:
        gap_frames = max(1, round(min(args.gap_seconds, duration) * SAMPLE_RATE))
        gap_end = min(duration_frames, gap_start + gap_frames)
        gap = (gap_start, gap_end)
        boundaries.add(gap_end)
    boundaries = sorted(boundaries)
    tracks = {
        f"track-{index + 1}": Track(
            clip, (late_join or 0) if index else 0, gap if index == 1 else None
        )
        for index, clip in enumerate(audio)
    }
    events = Events(
        {
            track_id: (max(track.start, track.gap[0]), track.gap[1])
            for track_id, track in tracks.items()
            if track.gap and track.start < track.gap[1]
        }
    )
    async with open_socket(args, True) as original:
        greeting = await events.until(original, NATIVE + "session.created")
        socket = original
        async with asyncio.timeout(None) as drain_timeout, asyncio.TaskGroup() as tasks:
            receiver = tasks.create_task(collect_native(events, socket, drain_timeout))
            try:
                clock = CaptureClock()
                loop = asyncio.get_running_loop()
                started = loop.time()
                max_send_time = 0.0
                for frame, end in packets(0, duration_frames, loop_frames, boundaries):
                    if frame == reconnect_at:
                        receiver.cancel()
                        await asyncio.gather(receiver, return_exceptions=True)
                        await socket.close()
                        socket = await open_socket(args, True, reconnecting=True)
                        expected_frames = await resume(
                            socket, greeting, events, args.accept_replay_loss
                        )
                        receiver = tasks.create_task(
                            collect_native(events, socket, drain_timeout)
                        )
                        await restore_capture(
                            socket,
                            tracks,
                            expected_frames,
                            frame,
                            loop_frames,
                            boundaries,
                        )
                        clock = CaptureClock(frame)
                    for track_id, track in tracks.items():
                        if track.start == frame:
                            await register(socket, track_id, frame)
                    await clock.capture_through(end)
                    sending = loop.time()
                    for track_id, track in tracks.items():
                        if track.start <= frame:
                            await send_capture(
                                socket, track_id, track, frame, end, loop_frames
                            )
                    max_send_time = max(max_send_time, loop.time() - sending)
                capture_ended = loop.time()
                await send(socket, NATIVE + "session.end")
                drain_timeout.reschedule(loop.time() + NATIVE_DRAIN_IDLE_SECONDS)
                await receiver
                drain_timeout.reschedule(None)
                print(
                    json.dumps(
                        {
                            "source_seconds": duration,
                            "capture_wall_seconds": capture_ended - started,
                            "drain_seconds": loop.time() - capture_ended,
                            "max_live_packet_group_send_ms": max_send_time * 1000,
                            "max_reported_backlog_frames": events.max_reported_backlog_frames,
                            "unobserved_replay_events": events.lost_events,
                            "unexpected_gap_frames": events.unexpected_gap_frames,
                        }
                    ),
                    file=sys.stderr,
                )
                if events.open_items:
                    raise RuntimeError(
                        f"Items remained open at session end: {events.open_items}"
                    )
            finally:
                if socket is not original:
                    await socket.close()
    if events.unexpected_gap_frames:
        raise RuntimeError(
            f"Server reported unexpected audio loss (frames per track): "
            f"{dict(events.unexpected_gap_frames)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {CLIENT_VERSION}"
    )
    parser.add_argument("audio", type=Path, nargs="+", help="One WAV per track")
    parser.add_argument("--url", default="ws://127.0.0.1:8080")
    parser.add_argument(
        "--native", action="store_true", help="Use the multi-track extension"
    )
    parser.add_argument(
        "--model", default="gpt-transcribe", help="OpenAI session model alias"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat the input this many times"
    )
    parser.add_argument(
        "--reconnect-at", type=float, help="Native reconnect time, in seconds"
    )
    parser.add_argument(
        "--late-join", type=float, help="Native join time for additional tracks"
    )
    parser.add_argument(
        "--gap-at", type=float, help="Time to start dropping audio for the second track"
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=1,
        help="Duration of the second track's gap",
    )
    parser.add_argument(
        "--accept-replay-loss",
        action="store_true",
        help="Resume from retained events if the server has discarded earlier events",
    )
    parser.add_argument(
        "--origin", help="Origin to send during the WebSocket handshake"
    )
    parser.add_argument(
        "--browser-auth",
        action="store_true",
        help="Offer the key as an OpenAI browser subprotocol",
    )
    args = parser.parse_args()
    native_options = (args.reconnect_at, args.late_join, args.gap_at)
    if not args.native and (
        len(args.audio) != 1 or any(opt is not None for opt in native_options)
    ):
        parser.error("Multiple files and fault-injection options require --native")
    if (
        any(
            opt is not None and (not math.isfinite(opt) or opt < 0)
            for opt in native_options
        )
        or not math.isfinite(args.gap_seconds)
        or args.gap_seconds <= 0
    ):
        parser.error(
            "Fault-injection times must be nonnegative and the gap duration positive"
        )
    if (args.late_join is not None or args.gap_at is not None) and len(args.audio) < 2:
        parser.error("--late-join and --gap-at require at least two files")
    if args.accept_replay_loss and args.reconnect_at is None:
        parser.error("--accept-replay-loss requires --reconnect-at")
    if args.repeat < 1:
        parser.error("--repeat requires a positive count")
    return args


async def main():
    args = parse_args()
    audio = [read_pcm(path) for path in args.audio]
    if not all(audio):
        raise ValueError("Every WAV file must contain at least one frame")
    if args.native:
        await play_native(args, audio)
    else:
        await play_standard(args, audio[0])


if __name__ == "__main__":
    asyncio.run(main())
