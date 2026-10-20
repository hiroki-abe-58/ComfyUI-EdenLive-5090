"""EDENLIVE_COMFYUI_DIR: a ComfyUI checkout (for comfy_api and folder_paths). Tests run on the CPU and need no model or LightX2V."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COMFYUI = os.environ.get("EDENLIVE_COMFYUI_DIR")
if COMFYUI:
    sys.path.insert(0, COMFYUI)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args
    import torch

    comfy.cli_args.args.cpu = not torch.cuda.is_available()


@pytest.fixture
def comfy():
    if not COMFYUI:
        pytest.skip("set EDENLIVE_COMFYUI_DIR to a ComfyUI checkout")
    return COMFYUI


def fake_model(root: Path, int8=False):
    from edenlive_comfy import paths

    for name in paths.REQUIRED + (paths.INT8 if int8 else ()):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"x")
    return root
