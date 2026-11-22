"""Exercise two real turns against a running server using the official Python SDK."""

import argparse
import asyncio
import base64
import json
import os
import time
import wave

import openai


async def transcribe(connection, pcm, require_delta, expected):
    started = time.monotonic()
    commit = first_delta = delta_item = committed_item = None

    async def send():
        nonlocal commit
        for offset in range(0, len(pcm), 4800):
            packet = pcm[offset : offset + 4800]
            packet_start = time.monotonic()
            await connection.input_audio_buffer.append(
                audio=base64.b64encode(packet).decode()
            )
            await asyncio.sleep(
                max(0, packet_start + len(packet) / 48_000 - time.monotonic())
            )
        commit = time.monotonic()
        await connection.input_audio_buffer.commit()

    producer = asyncio.create_task(send())
    try:
        while True:
            event = await connection.recv()
            if (
                event.type == "conversation.item.input_audio_transcription.delta"
                and event.delta
            ):
                assert delta_item in (None, event.item_id), event
                delta_item = event.item_id
                first_delta = first_delta or time.monotonic()
            elif event.type == "input_audio_buffer.committed":
                committed_item = event.item_id
            elif event.type == "conversation.item.input_audio_transcription.completed":
                assert commit is not None and committed_item == event.item_id, event
                assert delta_item in (None, committed_item), event
                assert not require_delta or (
                    first_delta is not None and first_delta < commit
                ), event
                assert all(
                    word.casefold() in event.transcript.casefold() for word in expected
                ), event
                return {
                    "item": event.item_id,
                    "transcript": event.transcript,
                    "first_delta_s": round(first_delta - started, 3)
                    if first_delta
                    else None,
                    "commit_to_final_s": round(time.monotonic() - commit, 3),
                }
            elif event.type == "error" or event.type.endswith(".failed"):
                raise RuntimeError(str(event))
    finally:
        if not producer.done():
            producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


async def main(args):
    with wave.open(args.audio) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            24_000,
            1,
            2,
        ):
            raise ValueError("Expected a 24 kHz mono PCM16 WAV")
        pcm = wav.readframes(wav.getnframes())
    model = "gpt-live-transcribe" if args.stable else "gpt-transcribe"
    key = os.getenv("TRANSCRIBERRR_API_KEY", "local")
    async with (
        openai.AsyncOpenAI(base_url=args.base_url, api_key=key) as client,
        client.realtime.connect(
            extra_query={"intent": "transcription"}, max_retries=0
        ) as ws,
    ):
        assert (await ws.recv()).type == "session.created"
        await ws.session.update(
            session={
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24_000},
                        "transcription": {"model": model},
                        "turn_detection": None,
                    }
                },
            }
        )
        assert (await ws.recv()).type == "session.updated"
        turns = [
            await transcribe(ws, pcm, args.stable, args.contains) for _ in range(2)
        ]
        assert turns[0]["item"] != turns[1]["item"]
        print(
            json.dumps(
                {"sdk": openai.__version__, "model": model, "turns": turns},
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="24 kHz mono PCM16 WAV containing speech")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument(
        "--contains",
        action="append",
        required=True,
        help="Expected transcript substring",
    )
    parser.add_argument(
        "--stable", action="store_true", help="Require a nonempty delta before commit"
    )
    asyncio.run(asyncio.wait_for(main(parser.parse_args()), timeout=120))
