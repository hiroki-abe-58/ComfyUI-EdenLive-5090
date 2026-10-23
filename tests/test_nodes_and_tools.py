import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prepare_model  # noqa: E402


def test_manifest_is_the_pinned_5b_subset():
    revision, files = prepare_model.manifest()
    assert revision == "de40b28f3f6eb6a0f12bb3d3967ca35578273713"
    assert "dit-5b/diffusion_pytorch_model.safetensors" in files and "vision-encoder/projection.safetensors" in files
    assert not any(name.startswith(("dit-14b", "dit-5b-bidir", "clip/", "flux-c2f/")) for name in files)
    assert round(sum(size for _, size in files.values()) / 2**30, 2) == 34.21


def test_git_blob_digest_matches_git(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"EdenLive\n")
    expected = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    assert prepare_model.file_digest(path, "git") == expected
    assert prepare_model.file_digest(path, "sha256") == hashlib.sha256(b"EdenLive\n").hexdigest()


def test_node_schemas(comfy):
    from edenlive_comfy import nodes

    assert nodes.EdenLiveLoader.define_schema().node_id == "EdenLiveLoader"
    render = nodes.EdenLiveRender.define_schema()
    assert render.node_id == "EdenLiveRender"
    # v0.1.0 workflows keep working: the new inputs are optional and come after the old ones, frames stays output 0
    assert [i.id for i in render.inputs] == ["edenlive", "proxy", "first_frame", "prompt", "seed", "prompt_schedule", "preview"]
    assert [i.optional for i in render.inputs[5:]] == [True, True]
    assert [o.display_name for o in render.outputs] == ["frames", "schedule"]


def test_render_rejects_an_empty_prompt_before_loading(comfy):
    from edenlive_comfy import nodes

    with pytest.raises(ValueError, match="prompt is empty"):
        nodes.EdenLiveRender.execute({"model_dir": "x", "profile": "bf16 + sage", "t5_on_cpu": True}, None, None, "  ", 1)
    with pytest.raises(ValueError, match="start times must increase"):
        nodes.EdenLiveRender.execute({"model_dir": "x", "profile": "bf16 + sage", "t5_on_cpu": True}, None, None, "p", 1, "2s: a\n1s: b")


def test_render_sets_prompt_travel_for_one_request(comfy, monkeypatch):
    import torch

    from edenlive_comfy import runtime

    class Config(dict):
        locked = True

        def __setitem__(self, key, value):
            assert not self.locked
            super().__setitem__(key, value)

        def pop(self, *args):
            assert not self.locked
            return super().pop(*args)

        def temporarily_unlocked(self):
            config = self

            class Unlocked:
                def __enter__(self):
                    config.locked = False

                def __exit__(self, *exc):
                    config.locked = True

            return Unlocked()

    seen = {}

    class Runner:
        config = Config()

        def prepare_request(self, request):
            seen["travel"] = dict(self.config).get("prompt_travel")
            return request

        def set_chunk_callback(self, callback):
            pass

        def run_request(self, info):
            return {"video": torch.zeros(1, 2, 2, 3)}

    monkeypatch.setattr(runtime, "_lightx2v", lambda: (lambda: type("P", (), {"parse_args": lambda self, a: a})(), None, lambda args: (None, args)))
    travel = {"prompt_travel_text": ["a", "b"], "prompt_travel_start_frames": [0, 12]}
    runtime.render(Runner(), "m", "c.json", "p.mp4", "f.png", "a", 1, prompt_travel=travel)
    assert seen["travel"] == travel and "prompt_travel" not in Runner.config
    runtime.render(Runner(), "m", "c.json", "p.mp4", "f.png", "a", 1)
    assert seen["travel"] is None
