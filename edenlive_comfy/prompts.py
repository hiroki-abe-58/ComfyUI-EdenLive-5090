"""Timed prompts for EdenLive's chunk-autoregressive model.

EdenLive describes a recording's text as a static prompt plus timed events (`prompts.json`: `staticPrompt` and `dynamicPrompts`
with `startMs`/`endMs`/`text`). For the few-step model each chunk of 3 latent frames gets one prompt: the static prompt followed by
the text of every event that overlaps the chunk's frames. That rule is EdenLive's `prompt_schedule` (edenlive/data/
recording_manifest.py, VAST-AI-Research/EdenLive, Apache-2.0); `chunk_prompts` follows it. LightX2V takes the result as
`prompt_travel`: each text with the first frame it applies to.
"""

from __future__ import annotations

import json
import re

FPS = 24
CHUNK = 3  # latent frames per chunk
STRIDE = 4  # pixel frames per latent frame after the first

_LINE = re.compile(r"^\s*(?P<time>\d+(?:\.\d+)?)\s*(?P<unit>s|ms)?\s*:\s*(?P<text>.*\S)\s*$")


def chunk_prompts(metadata, num_frames, fps=FPS, chunk_size=CHUNK):
    """The prompt of every chunk of a clip of num_frames frames (EdenLive's prompt_schedule)."""
    prompts = []
    latent_frames = (num_frames - 1) // STRIDE + 1
    for start in range(0, latent_frames, chunk_size):
        end = min(start + chunk_size, latent_frames)
        # latent 0 decodes to one frame, every later latent to four
        start_ms, end_ms = max(0, STRIDE * start - 3) * 1000 / fps, (STRIDE * end - 3) * 1000 / fps
        texts = dict.fromkeys(e["text"] for e in metadata.get("dynamicPrompts", []) if e["startMs"] < end_ms and e["endMs"] > start_ms)
        prompts.append(" ".join([metadata["staticPrompt"], *texts]))
    return prompts


def prompt_travel(prompts, chunk_size=CHUNK):
    """LightX2V's prompt_travel for per-chunk prompts: a text wherever it changes, with the first pixel frame of that chunk."""
    texts, starts = [], []
    for index, text in enumerate(prompts):
        if not texts or texts[-1] != text:
            texts.append(text)
            starts.append(index * chunk_size * STRIDE)
    return {"prompt_travel_text": texts, "prompt_travel_start_frames": starts}


def parse(schedule: str, static_prompt: str):
    """EdenLive metadata from the node's schedule text: the content of a prompts.json, or one event per line,
    `<start>: <text>` with the start in frames (24 fps), seconds (`2.5s`) or milliseconds (`2500ms`); a line's event lasts until
    the next line's start. Returns None for an empty schedule. A prompts.json without staticPrompt uses static_prompt."""
    schedule = schedule.strip()
    if not schedule:
        return None
    if schedule.startswith("{"):
        metadata = json.loads(schedule)
        events = metadata.get("dynamicPrompts", [])
        for e in events:
            if not {"startMs", "endMs", "text"} <= set(e):
                raise ValueError(f"dynamicPrompts entries need startMs, endMs and text: {e}")
        return {"staticPrompt": metadata.get("staticPrompt", static_prompt).strip(), "dynamicPrompts": events}
    starts = []
    for number, line in enumerate(schedule.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            raise ValueError(f"prompt_schedule line {number}: expected '<frame>: text', '<seconds>s: text' or '<ms>ms: text', got {line!r}")
        value, unit = float(match["time"]), match["unit"]
        start_ms = value * 1000 if unit == "s" else value if unit == "ms" else value * 1000 / FPS
        starts.append((start_ms, match["text"]))
    if not starts:
        return None
    if any(b[0] <= a[0] for a, b in zip(starts, starts[1:])):
        raise ValueError("prompt_schedule: start times must increase")
    events = [{"startMs": s, "endMs": starts[i + 1][0] if i + 1 < len(starts) else float("inf"), "text": t} for i, (s, t) in enumerate(starts)]
    return {"staticPrompt": static_prompt.strip(), "dynamicPrompts": events}


def describe(prompts, fps=FPS, chunk_size=CHUNK):
    """One line per chunk: the frames it covers and its prompt."""
    lines = []
    for index, text in enumerate(prompts):
        first = max(0, index * chunk_size * STRIDE - 3)
        last = (index + 1) * chunk_size * STRIDE - 4
        lines.append(f"chunk {index} (frames {first}-{last}, {first / fps:.2f}-{(last + 1) / fps:.2f} s): {text}")
    return "\n".join(lines)
