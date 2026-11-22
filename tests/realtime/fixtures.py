"""Generate two annotated speech tracks and mix 2.5% crosstalk into each."""

import json
import math
import subprocess
import wave
from argparse import ArgumentParser
from array import array
from pathlib import Path

parser = ArgumentParser(description=__doc__)
parser.add_argument(
    "output", type=Path, help="Directory for synthetic PCM WAVs and cue annotations"
)
root = parser.parse_args().output
root.mkdir(parents=True, exist_ok=True)
rate = 24000
duration_s = 18
frame_count = duration_s * rate
gain = 0.025
layout = [
    (
        "alice",
        "Samantha",
        [
            (0.35, "The cedar garden grows behind the library."),
            (4.85, "Tomorrow we will bring a basket of apples for the children."),
            (9.25, "The blue bicycle is outside the front gate."),
        ],
    ),
    (
        "bob",
        "Daniel",
        [
            (4.10, "A red canoe floats under the old wooden bridge."),
            (8.85, "Keep looking for the white lighthouse on the edge of the island."),
            (13.10, "Our yellow umbrella is behind the kitchen door."),
        ],
    ),
]


def speech_bounds(samples):
    count = rate // 100
    rms = [
        math.sqrt(
            sum(x * x for x in samples[i : i + count]) / len(samples[i : i + count])
        )
        for i in range(0, len(samples), count)
    ]
    start = next(i for i in range(len(rms) - 2) if min(rms[i : i + 3]) >= 328) * count
    end = (
        next(i for i in range(len(rms) - 1, 1, -1) if min(rms[i - 2 : i + 1]) >= 328)
        + 1
    ) * count
    return start, end


def wav(path, samples):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", ""))
        output.writeframes(samples.tobytes())


tracks = []
cues = {}
for index, (name, voice, placements) in enumerate(layout, 1):
    track = array("h", [0]) * frame_count
    cues[f"track-{index}"] = []
    previous_end = 0
    for number, (start, text) in enumerate(placements, 1):
        path = root / f"{name}-{number}.aiff"
        subprocess.run(
            ["say", "-v", voice, "-r", "190", "-o", str(path), text], check=True
        )
        samples = array("h")
        samples.frombytes(
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    str(path),
                    "-ar",
                    str(rate),
                    "-ac",
                    "1",
                    "-f",
                    "s16le",
                    "-",
                ]
            )
        )
        start_frame = round(start * rate)
        end_frame = start_frame + len(samples)
        if not previous_end <= start_frame <= end_frame <= frame_count:
            raise ValueError(
                f"{name} phrase {number} does not fit through {end_frame / rate:.2f}s"
            )
        track[start_frame:end_frame] = samples
        previous_end = end_frame
        first, last = speech_bounds(samples)
        cues[f"track-{index}"].append(
            {
                "text": text,
                "source_start_frame": start_frame,
                "reference_onset_frame": start_frame + first,
                "reference_end_frame": start_frame + last,
            }
        )
    tracks.append(track)
    wav(root / f"{name}-clean.wav", track)

for index, (name, _, _) in enumerate(layout):
    own, other = tracks[index], tracks[1 - index]
    mixed = array(
        "h", (min(32767, max(-32768, round(a + gain * b))) for a, b in zip(own, other))
    )
    wav(root / f"{name}-bleed.wav", mixed)

with (root / "fixtures.json").open("w") as output:
    print(
        json.dumps(
            {
                "cycle_seconds": duration_s,
                "sample_rate": rate,
                "crosstalk_gain": gain,
                "cues": cues,
            },
            indent=2,
        ),
        file=output,
    )
