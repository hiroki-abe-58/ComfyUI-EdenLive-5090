"""EdenLive through LightX2V's own runner, in the ComfyUI process.

The runner is built and called exactly as `python -m lightx2v.infer --model_cls wan22_edenlive_ar --task i2v` does (build_cli_inputs,
build_runner, prepare_request, run_request), so a seed gives the frames the command line would write. One runner stays loaded
and serves later requests; choosing another model, profile or text-encoder device releases it before the next one loads.
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
import os
import sys
import tempfile
import weakref
from pathlib import Path

import numpy as np
import torch

from . import profiles

MODEL_CLS = "wan22_edenlive_ar"
# LightX2V reads its platform from PLATFORM; the MSVC environment (vcvars, Developer prompts) sets Platform=x64, and Windows
# environment names are case-insensitive.
MSVC_PLATFORMS = {"x64", "x86", "win32", "arm64", "arm"}

_state: dict = {"key": None, "runner": None, "config_path": None}


class LightX2VMissing(RuntimeError):
    pass


def _prepare_environment():
    platform = os.environ.get("PLATFORM")
    if platform is not None and platform.lower() in MSVC_PLATFORMS:
        logging.warning("[EdenLive] PLATFORM=%s comes from the MSVC environment; using PLATFORM=cuda for LightX2V", platform)
        os.environ["PLATFORM"] = "cuda"
    # as LightX2V's EdenLive launchers set them
    os.environ.setdefault("DTYPE", "BF16")
    os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "BF16")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _lightx2v():
    _prepare_environment()
    try:
        from lightx2v.infer import build_parser
        from lightx2v.models.runners.runner_factory import RUNNER_MODULES, build_runner
        from lightx2v.utils.set_config import build_cli_inputs

        # here rather than on first use inside build_runner(), whose frame would then be kept (see _drop_import_tracebacks)
        importlib.import_module(RUNNER_MODULES[MODEL_CLS])
    except ImportError as error:
        raise LightX2VMissing(
            "LightX2V (or a package its EdenLive runner imports) is missing in ComfyUI's Python. Install LightX2V from a source "
            "checkout at the tested commit d2d7424 with `pip install --no-deps -e LightX2V`, then the packages in "
            f"requirements-lightx2v.txt (see the README). Import error: {error}"
        ) from error
    _drop_import_tracebacks()
    return build_parser, build_runner, build_cli_inputs


def _drop_import_tracebacks():
    """Keep LightX2V's stored optional-import errors, without their tracebacks.

    A module that keeps the error of a failed optional import (lightx2v_kernel's MXFP8 GEMM, which has no Windows build) also keeps
    its traceback, and with it the frames of whatever imported the module and everything those frames hold: in LightX2V d2d7424
    that is build_runner() and the first runner it builds, which could then never be freed (ModelTC/LightX2V#1640).
    """
    for name, module in list(sys.modules.items()):
        if module is None or name.partition(".")[0] != "lightx2v":
            continue
        for value in list(getattr(module, "__dict__", {}).values()):
            if isinstance(value, BaseException) and (value.__traceback__ is not None or value.__context__ is not None):
                value.__context__ = value.__cause__ = None
                value.with_traceback(None)


def _cli_args(build_parser, model_dir, config_path, prompt="", image="", video="", seed=0):
    return build_parser().parse_args([
        "--model_cls", MODEL_CLS, "--task", "i2v", "--model_path", model_dir, "--config_json", config_path,
        "--prompt", prompt, "--image_path", image, "--video_path", video, "--seed", str(seed), "--return_result_tensor",
    ])


def _describe(key) -> str:
    return f"{os.path.basename(key[0])}, {key[1]}, t5_on_cpu={key[2]}"


def release():
    """Drop the loaded runner and give its GPU memory back. If it cannot be freed, it stays loaded and a restart is needed."""
    if _state["runner"] is None:
        return
    key, config_path = _state["key"], _state["config_path"]
    probe = weakref.ref(_state["runner"])
    _state.update(key=None, runner=None, config_path=None)
    # After its first request LightX2V moves the whole object graph into the GC's permanent generation (gc.freeze()), where the
    # runner's reference cycles are never collected.
    gc.unfreeze()
    gc.collect()
    runner = probe()
    if runner is not None:
        _state.update(key=key, runner=runner, config_path=config_path)
        raise RuntimeError(f"EdenLive ({_describe(key)}) could not be released in-process (something still refers to it), so a second "
                           "model would not fit next to it. Restart ComfyUI to use another model, profile or text-encoder device.")
    import comfy.model_management

    comfy.model_management.soft_empty_cache()
    logging.info("[EdenLive] released %s", _describe(key))


def load(model_dir: str, profile: str, t5_on_cpu: bool):
    """The runner for this model folder and profile. A loaded runner with other settings is released first."""
    key = (os.path.abspath(model_dir), profile, bool(t5_on_cpu))
    if _state["key"] == key:
        return _state["runner"], _state["config_path"]
    missing = profiles.missing_modules(profile)
    if missing:
        raise RuntimeError(f"profile {profile!r} needs {', '.join(missing)}; install it or choose 'bf16 + pytorch kernels'")
    if profiles.PROFILES[profile]["int8"]:
        from .paths import has_int8

        if not has_int8(model_dir):
            raise FileNotFoundError(f"profile {profile!r} needs the INT8 ConvRot files in {model_dir}/quantized (tools/prepare_model.py --int8)")
    build_parser, build_runner, build_cli_inputs = _lightx2v()
    release()
    import comfy.model_management

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    config_dir = Path(tempfile.gettempdir()) / "comfyui_edenlive"
    config_dir.mkdir(exist_ok=True)
    config_path = str(config_dir / f"profile_{abs(hash(key)):x}.json")
    Path(config_path).write_text(json.dumps(profiles.build(profile, model_dir, t5_on_cpu), indent=1), encoding="utf-8")
    startup, _ = build_cli_inputs(_cli_args(build_parser, model_dir, config_path))
    runner = build_runner(startup)
    _state.update(key=key, runner=runner, config_path=config_path)
    return runner, config_path


def proxy_file(video, tmp: str) -> str:
    """A file LightX2V's recording reader can open for a ComfyUI VIDEO.

    A video loaded from a file is passed as that file (or its bytes), so the reader decodes it exactly as the command line would;
    ComfyUI's own decode applies colour management and gives other RGB values. A video made from frames is written losslessly.
    """
    from comfy_api.latest import InputImpl

    path = os.path.join(tmp, "proxy.mp4")
    if isinstance(video, InputImpl.VideoFromFile):
        source = video.get_stream_source()
        if isinstance(source, str):
            return source
        source.seek(0)
        Path(path).write_bytes(source.read())
        return path
    # other videos (e.g. Create Video) would encode their frames lossily in get_stream_source()
    components = video.get_components()
    write_proxy(components.images, float(components.frame_rate), path)
    return path


def write_proxy(frames: torch.Tensor, fps: float, path: str):
    """IMAGE frames ([N, H, W, 3], 0..1) as a lossless RGB H.264 file, which LightX2V's recording reader decodes."""
    import av

    pixels = (frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    height, width = pixels.shape[1] // 2 * 2, pixels.shape[2] // 2 * 2
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264rgb", rate=_rate(fps))
        stream.width, stream.height, stream.pix_fmt = width, height, "rgb24"
        stream.options = {"crf": "0", "preset": "veryfast"}
        for frame in pixels:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame[:height, :width]), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _rate(fps: float):
    from fractions import Fraction

    return Fraction(fps).limit_denominator(1001)


def write_image(image: torch.Tensor, path: str):
    from PIL import Image

    Image.fromarray((image[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()).save(path)


def render(runner, model_dir, config_path, proxy_path, cover_path, prompt, seed, on_chunk=None) -> torch.Tensor:
    """Frames [T, H, W, 3] float32 in 0..1 on the CPU; the same request as the command line with these files."""
    build_parser, _, build_cli_inputs = _lightx2v()
    _, request = build_cli_inputs(_cli_args(build_parser, model_dir, config_path, prompt, cover_path, proxy_path, seed))
    info = runner.prepare_request(request)
    runner.set_chunk_callback(on_chunk)
    try:
        video = runner.run_request(info)["video"]
    finally:
        runner.set_chunk_callback(None)
    return video.to(torch.float32)
