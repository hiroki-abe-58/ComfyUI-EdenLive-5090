"""Timed prompts. EDENLIVE_UPSTREAM_DIR (a VAST-AI-Research/EdenLive checkout) enables the comparison with EdenLive's own
prompt_schedule on its two shipped samples; CI sets it."""

import bisect
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from edenlive_comfy import prompts

UPSTREAM = os.environ.get("EDENLIVE_UPSTREAM_DIR")


def official_prompt_schedule():
    if not UPSTREAM:
        if os.environ.get("EDENLIVE_REQUIRE_REFERENCES") == "1":
            pytest.fail("EDENLIVE_UPSTREAM_DIR is required")
        pytest.skip("set EDENLIVE_UPSTREAM_DIR to an EdenLive checkout")
    # recording_manifest.py only needs ManifestDataset from the package at import time
    stub = types.ModuleType("edenlive.data.dataset_manifest")
    stub.ManifestDataset = object
    sys.modules.setdefault("edenlive.data.dataset_manifest", stub)
    spec = importlib.util.spec_from_file_location("edenlive_recording_manifest", Path(UPSTREAM) / "edenlive" / "data" / "recording_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prompt_schedule


@pytest.mark.parametrize("sample", ["cryo-crossing", "pulse-skirmish"])
def test_chunk_prompts_match_edenlive(sample):
    prompt_schedule = official_prompt_schedule()
    metadata = json.loads((Path(UPSTREAM) / "examples" / "samples" / sample / "prompts.json").read_text(encoding="utf-8"))
    for num_frames in (1, 9, 13, 121, 717, round(metadata["durationMs"] * 24 / 1000)):
        want = [row["prompt"] for row in prompt_schedule(metadata, num_frames, fps=24, chunk_size=3)]
        assert prompts.chunk_prompts(metadata, num_frames) == want


def lightx2v_frame_prompts(travel, latent_frames, chunk_size=3, temporal_stride=4):
    """LightX2V's frame_prompts (lightx2v/models/input_encoders/edenlive/recording.py), for the round trip below."""
    texts, starts = travel["prompt_travel_text"], travel["prompt_travel_start_frames"]
    start_latents = [frame // temporal_stride for frame in starts]
    assert starts[0] == 0 and all(a < b for a, b in zip(start_latents, start_latents[1:]))
    return [texts[bisect.bisect_right(start_latents, start) - 1] for start in range(0, latent_frames, chunk_size)]


def test_prompt_travel_gives_every_chunk_its_prompt():
    metadata = {"staticPrompt": "Scene.", "dynamicPrompts": [
        {"startMs": 0, "endMs": 400, "text": "A"}, {"startMs": 400, "endMs": 1500, "text": "B"}, {"startMs": 1500, "endMs": 5000, "text": "C"}]}
    chunk_texts = prompts.chunk_prompts(metadata, 121)
    assert chunk_texts[0] == "Scene. A"  # chunk 0 is frames 0-8, i.e. up to 375 ms; B starts at 400 ms
    assert chunk_texts[1] == "Scene. A B"  # chunk 1 (frames 9-20, 375-875 ms) overlaps both
    travel = prompts.prompt_travel(chunk_texts)
    assert travel["prompt_travel_start_frames"][0] == 0
    assert lightx2v_frame_prompts(travel, (121 - 1) // 4 + 1) == chunk_texts


def test_first_chunk_boundary():
    # chunk 0 covers frames 0..8, i.e. [0, 9/24 s) = [0, 375 ms)
    assert prompts.chunk_prompts({"staticPrompt": "S", "dynamicPrompts": [{"startMs": 374, "endMs": 900, "text": "X"}]}, 9) == ["S X"]
    assert prompts.chunk_prompts({"staticPrompt": "S", "dynamicPrompts": [{"startMs": 375, "endMs": 900, "text": "X"}]}, 9) == ["S"]


def test_parse_lines_and_json():
    metadata = prompts.parse("0: walk in\n# comment\n\n2s: turn around\n96: leave", "Static.")
    assert metadata["staticPrompt"] == "Static."
    assert [(e["startMs"], e["text"]) for e in metadata["dynamicPrompts"]] == [(0.0, "walk in"), (2000.0, "turn around"), (4000.0, "leave")]
    assert metadata["dynamicPrompts"][0]["endMs"] == 2000.0 and metadata["dynamicPrompts"][-1]["endMs"] == float("inf")
    assert prompts.parse("2500ms: x", "S")["dynamicPrompts"][0]["startMs"] == 2500.0
    from_json = prompts.parse(json.dumps({"staticPrompt": "J", "dynamicPrompts": [{"startMs": 0, "endMs": 10, "text": "t"}]}), "ignored")
    assert from_json["staticPrompt"] == "J"
    assert prompts.parse(json.dumps({"dynamicPrompts": []}), "fallback")["staticPrompt"] == "fallback"
    assert prompts.parse("   ", "S") is None
    with pytest.raises(ValueError):
        prompts.parse("1s: a\n0.5s: b", "S")
    with pytest.raises(ValueError):
        prompts.parse("soon: a", "S")
    with pytest.raises(ValueError):
        prompts.parse(json.dumps({"dynamicPrompts": [{"startMs": 0, "text": "no end"}]}), "S")


def test_describe_lists_frames_and_seconds():
    text = prompts.describe(["a", "b"])
    assert text.splitlines() == ["chunk 0 (frames 0-8, 0.00-0.38 s): a", "chunk 1 (frames 9-20, 0.38-0.88 s): b"]
