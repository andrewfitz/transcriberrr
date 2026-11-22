"""Exercise the reference client against a scripted WebSocket peer."""

import argparse
import asyncio
import base64
import contextlib
import importlib.util
import io
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import serve

CLIENT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "realtime-client.py"
SPEC = importlib.util.spec_from_file_location("realtime_client", CLIENT_PATH)
CLIENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLIENT
SPEC.loader.exec_module(CLIENT)
NATIVE = CLIENT.NATIVE


async def recv(socket):
    return json.loads(await socket.recv())


def pcm(frames, period):
    return b"".join((frame % period).to_bytes(2, "little") for frame in range(frames))


def drain_metrics(stderr):
    # asyncio debug mode can also write slow-callback diagnostics under load.
    (line,) = (line for line in stderr.getvalue().splitlines() if line.startswith("{"))
    return json.loads(line)


class ReferenceClient(unittest.IsolatedAsyncioTestCase):
    def args(self, **kwargs):
        return argparse.Namespace(
            repeat=1,
            origin=None,
            browser_auth=False,
            model="gpt-live-transcribe",
            **kwargs,
        )

    async def run_peer(self, peer, run_client, args, audio):
        failures = []

        async def handler(socket):
            try:
                await peer(socket)
            except Exception as error:
                failures.append(error)
                raise

        output = io.StringIO()
        async with (
            asyncio.timeout(5),
            serve(handler, "127.0.0.1", 0, subprotocols=["realtime"]) as server,
        ):
            args.url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}?probe=client&probe="
            try:
                with contextlib.redirect_stdout(output):
                    await run_client(args, audio)
            finally:
                if failures:
                    raise failures[0]
        return [json.loads(line) for line in output.getvalue().splitlines()]

    async def test_resume_recovers_older_audio_and_lost_registration(self):
        first_audio, second_audio = pcm(6600, 31), pcm(5600, 29)
        reconnect, join, gap = 5760, 1032, 5880
        first_position = 1111
        connections = 0
        replayed = {"track-1": bytearray(), "track-2": bytearray()}
        expected = {"track-1": first_position, "track-2": join}

        async def peer(socket):
            nonlocal connections
            connections += 1
            query = {
                "intent": ["transcription"],
                "transcriberrr": ["1"],
                "probe": ["client", ""],
            }
            if connections > 1:
                query["resume"] = ["1"]
            self.assertEqual(
                parse_qs(urlsplit(socket.request.path).query, keep_blank_values=True),
                query,
            )
            if connections == 1:
                await CLIENT.send(
                    socket,
                    NATIVE + "session.created",
                    session_id="session",
                    resume_credential="hidden",
                )
                await CLIENT.send(
                    socket,
                    NATIVE + "transcript.draft",
                    item_id="old",
                    revision=4,
                    sequence=1,
                )
                async for message in socket:
                    self.assertIn(
                        json.loads(message)["type"],
                        (NATIVE + "track.register", NATIVE + "track.append"),
                    )
                return

            request = await recv(socket)
            self.assertEqual(
                request,
                {
                    "type": NATIVE + "session.resume",
                    "session_id": "session",
                    "resume_credential": "hidden",
                    "last_sequence": 1,
                },
            )
            await CLIENT.send(socket, "error", error={"code": "replay_unavailable"})
            self.assertEqual(await recv(socket), {**request, "accept_loss": True})
            await CLIENT.send(
                socket,
                NATIVE + "session.resumed",
                session_id="session",
                earliest_replayable_sequence=4,
                next_expected_frames={"track-1": first_position},
                lost_range=[2, 3],
            )
            await CLIENT.send(
                socket,
                NATIVE + "transcript.draft",
                item_id="next",
                revision=1,
                sequence=4,
            )
            registration = await recv(socket)
            self.assertEqual(registration["type"], NATIVE + "track.register")
            self.assertEqual(
                (registration["track_id"], registration["start_frame"]),
                ("track-2", join),
            )
            while (message := await recv(socket))["type"] != NATIVE + "session.end":
                track = message["track_id"]
                if message["type"] == NATIVE + "track.append":
                    self.assertEqual(message["start_frame"], expected[track])
                    chunk = base64.b64decode(message["audio"])
                    replayed[track].extend(chunk)
                    expected[track] += len(chunk) // 2
                else:
                    self.assertEqual(message["type"], NATIVE + "track.gap")
                    self.assertEqual(
                        (track, expected[track], message["end_frame"]),
                        ("track-2", gap, 6600),
                    )
                    expected[track] = message["end_frame"]
            await CLIENT.send(
                socket, NATIVE + "transcript.final", item_id="next", sequence=5
            )
            await CLIENT.send(
                socket,
                NATIVE + "track.gap",
                track_id="track-2",
                start_frame=gap,
                end_frame=6600,
                reason="client_gap",
                sequence=6,
            )
            await CLIENT.send(socket, NATIVE + "session.ended", sequence=7)

        args = self.args(
            reconnect_at=reconnect / CLIENT.SAMPLE_RATE,
            late_join=join / CLIENT.SAMPLE_RATE,
            gap_at=gap / CLIENT.SAMPLE_RATE,
            gap_seconds=1,
            accept_replay_loss=True,
        )
        summary = io.StringIO()
        with contextlib.redirect_stderr(summary):
            events = await self.run_peer(
                peer, CLIENT.play_native, args, [first_audio, second_audio]
            )
        self.assertEqual(expected, {"track-1": 6600, "track-2": 6600})
        self.assertEqual(replayed["track-1"], first_audio[first_position * 2 :])
        self.assertEqual(
            replayed["track-2"],
            second_audio[join * 2 : gap * 2].ljust((gap - join) * 2, b"\0"),
        )
        self.assertEqual(
            [event["sequence"] for event in events if "sequence" in event],
            [1, 4, 5, 6, 7],
        )
        self.assertNotIn("resume_credential", events[0])
        metrics = drain_metrics(summary)
        self.assertEqual(metrics["unexpected_gap_frames"], {})
        self.assertEqual(metrics["unobserved_replay_events"], 2)

    async def test_native_finishes_trace_and_reports_unexpected_loss(self):
        async def peer(socket):
            await CLIENT.send(socket, NATIVE + "session.created")
            while (await recv(socket))["type"] != NATIVE + "session.end":
                pass
            await CLIENT.send(
                socket,
                NATIVE + "track.backlog",
                track_id="track-1",
                queued_frames=3200,
                sequence=1,
            )
            gap = {
                "track_id": "track-1",
                "start_frame": 40,
                "end_frame": 200,
                "reason": "overload",
            }
            await CLIENT.send(socket, NATIVE + "track.gap", **gap, sequence=2)
            await CLIENT.send(socket, NATIVE + "track.gap", **gap, sequence=2)
            await CLIENT.send(socket, NATIVE + "session.ended", sequence=3)

        args = self.args(
            reconnect_at=None, late_join=None, gap_at=None, accept_replay_loss=False
        )
        summary = io.StringIO()
        with (
            contextlib.redirect_stderr(summary),
            self.assertRaisesRegex(RuntimeError, "unexpected audio loss"),
        ):
            await self.run_peer(peer, CLIENT.play_native, args, [pcm(240, 7)])
        metrics = drain_metrics(summary)
        self.assertEqual(metrics["unexpected_gap_frames"], {"track-1": 160})
        self.assertEqual(metrics["max_reported_backlog_frames"], 3200)
        self.assertEqual(metrics["source_seconds"], 0.01)
        self.assertGreaterEqual(metrics["capture_wall_seconds"], 0.009)

    async def test_native_drain_extends_deadline_only_after_capture(self):
        async def peer(socket):
            await CLIENT.send(socket, NATIVE + "session.created")
            while (await recv(socket))["type"] != NATIVE + "session.end":
                pass
            # Silence during capture exceeded the idle deadline. Draining also
            # takes longer than that deadline, with progress on every interval.
            for sequence in range(1, 6):
                await asyncio.sleep(0.1)
                await CLIENT.send(
                    socket,
                    NATIVE + "transcript.final",
                    item_id=f"turn-{sequence}",
                    sequence=sequence,
                )
            await CLIENT.send(socket, NATIVE + "session.ended", sequence=6)

        args = self.args(
            reconnect_at=None, late_join=None, gap_at=None, accept_replay_loss=False
        )
        with (
            patch.object(CLIENT, "NATIVE_DRAIN_IDLE_SECONDS", 0.4),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            events = await self.run_peer(
                peer, CLIENT.play_native, args, [pcm(12_000, 7)]
            )
        self.assertEqual(
            [event["sequence"] for event in events[1:]], [1, 2, 3, 4, 5, 6]
        )

    async def test_paused_native_socket_reads_preserve_audio_and_connection(self):
        audio = pcm(6000, 17)

        async def peer(socket):
            await CLIENT.send(socket, NATIVE + "session.created")
            self.assertEqual((await recv(socket))["type"], NATIVE + "track.register")
            socket.transport.pause_reading()
            await asyncio.sleep(0.2)
            socket.transport.resume_reading()
            received = bytearray()
            while (event := await recv(socket))["type"] != NATIVE + "session.end":
                self.assertEqual(event["start_frame"], len(received) // 2)
                received.extend(base64.b64decode(event["audio"]))
            self.assertEqual(received, audio)
            await CLIENT.send(socket, NATIVE + "session.ended", sequence=1)

        connect = CLIENT.connect

        def short_heartbeat(url, **options):
            # Accelerate the library defaults so a paused peer would time out.
            for key in ("ping_interval", "ping_timeout"):
                if options.get(key, 20) is not None:
                    options[key] = 0.02
            return connect(url, **options)

        args = self.args(
            reconnect_at=None, late_join=None, gap_at=None, accept_replay_loss=False
        )
        with (
            patch.object(CLIENT, "connect", short_heartbeat),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            events = await self.run_peer(peer, CLIENT.play_native, args, [audio])
        self.assertEqual(events[-1]["type"], NATIVE + "session.ended")

    async def test_native_drain_times_out_despite_backlog_and_replayed_final(self):
        async def peer(socket):
            await CLIENT.send(socket, NATIVE + "session.created")
            while (await recv(socket))["type"] != NATIVE + "session.end":
                pass
            await CLIENT.send(
                socket, NATIVE + "transcript.final", sequence=1, item_id="a"
            )
            for sequence in range(2, 8):
                try:
                    await asyncio.wait_for(socket.wait_closed(), 0.06)
                    return
                except TimeoutError:
                    pass
                await CLIENT.send(
                    socket, NATIVE + "transcript.final", sequence=1, item_id="a"
                )
                await CLIENT.send(
                    socket,
                    NATIVE + "track.backlog",
                    sequence=sequence,
                    queued_frames=5000,
                )
            await CLIENT.send(socket, NATIVE + "session.ended", sequence=8)

        args = self.args(
            reconnect_at=None, late_join=None, gap_at=None, accept_replay_loss=False
        )
        with (
            patch.object(CLIENT, "NATIVE_DRAIN_IDLE_SECONDS", 0.2),
            self.assertRaises(TimeoutError),
        ):
            await self.run_peer(peer, CLIENT.play_native, args, [pcm(240, 7)])

    async def test_stalled_socket_write_keeps_next_capture_paced(self):
        clock = CLIENT.CaptureClock()
        await clock.capture_through(CLIENT.PACKET_FRAMES)
        await asyncio.sleep(0.15)
        previous_write_finished = asyncio.get_running_loop().time()
        await clock.capture_through(2 * CLIENT.PACKET_FRAMES)
        self.assertGreaterEqual(
            asyncio.get_running_loop().time() - previous_write_finished, 0.08
        )

    async def test_resume_rejects_loss_without_explicit_acceptance(self):
        async def peer(socket):
            self.assertEqual((await recv(socket))["type"], NATIVE + "session.resume")
            await CLIENT.send(
                socket,
                NATIVE + "session.resumed",
                session_id="session",
                earliest_replayable_sequence=2,
                lost_range=[1, 1],
                next_expected_frames={},
            )

        async def resume_only(args, _audio):
            async with CLIENT.open_socket(args, True, reconnecting=True) as socket:
                await CLIENT.resume(
                    socket,
                    {"session_id": "session", "resume_credential": "hidden"},
                    CLIENT.Events(),
                    accept_loss=False,
                )

        with self.assertRaisesRegex(RuntimeError, "without explicit acceptance"):
            await self.run_peer(peer, resume_only, self.args(), [])

    async def test_standard_captures_before_sending_and_does_not_burst(self):
        audio = pcm(3 * CLIENT.PACKET_FRAMES + 1200, 11)
        loop = asyncio.get_running_loop()
        times = []

        async def peer(socket):
            await CLIENT.send(socket, "session.created")
            self.assertEqual((await recv(socket))["type"], "session.update")
            times.append(loop.time())
            await CLIENT.send(socket, "session.updated")
            for index, frame in enumerate(
                range(0, len(audio) // 2, CLIENT.PACKET_FRAMES)
            ):
                message = await recv(socket)
                times.append(loop.time())
                self.assertEqual(message["type"], "input_audio_buffer.append")
                self.assertEqual(
                    base64.b64decode(message["audio"]),
                    audio[frame * 2 : (frame + CLIENT.PACKET_FRAMES) * 2],
                )
                if index == 0:
                    # Deschedule sender and receiver past two packet deadlines.
                    loop.call_soon(time.sleep, 0.25)
                    await CLIENT.send(
                        socket,
                        "conversation.item.input_audio_transcription.delta",
                        item_id="turn",
                        delta="a",
                    )
            self.assertEqual((await recv(socket))["type"], "input_audio_buffer.commit")
            await CLIENT.send(socket, "input_audio_buffer.committed", item_id="turn")
            await CLIENT.send(socket, CLIENT.STANDARD_FINAL, item_id="turn")

        await self.run_peer(peer, CLIENT.play_standard, self.args(), audio)
        self.assertGreaterEqual(times[1] - times[0], 0.08)
        self.assertGreaterEqual(times[3] - times[2], 0.08)
        self.assertGreaterEqual(times[4] - times[3], 0.04)


if __name__ == "__main__":
    unittest.main()
