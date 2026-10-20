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
    assert [i.id for i in render.inputs] == ["edenlive", "proxy", "first_frame", "prompt", "seed"]


def test_render_rejects_an_empty_prompt_before_loading(comfy):
    from edenlive_comfy import nodes

    with pytest.raises(ValueError, match="prompt is empty"):
        nodes.EdenLiveRender.execute({"model_dir": "x", "profile": "bf16 + sage", "t5_on_cpu": True}, None, None, "  ", 1)
