import gc
import os
import sys
import types
import weakref

import av
import numpy as np
import pytest
import torch
from conftest import fake_model

from edenlive_comfy import paths, runtime


def decode_rgb(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(s)]), float(s.average_rate)


def test_write_proxy_is_lossless(tmp_path):
    torch.manual_seed(0)
    frames = torch.randint(0, 256, (9, 64, 96, 3), dtype=torch.uint8).float() / 255
    path = str(tmp_path / "p.mp4")
    runtime.write_proxy(frames, 24.0, path)
    rgb, fps = decode_rgb(path)
    assert fps == 24.0 and rgb.shape == (9, 64, 96, 3)
    assert np.array_equal(rgb, (frames * 255).round().to(torch.uint8).numpy())


def test_write_proxy_keeps_ntsc_rates_and_even_sizes(tmp_path):
    path = str(tmp_path / "p.mp4")
    runtime.write_proxy(torch.zeros(3, 65, 97, 3), 30000 / 1001, path)
    rgb, fps = decode_rgb(path)
    assert rgb.shape == (3, 64, 96, 3) and abs(fps - 29.97002997) < 1e-6


def test_write_image(tmp_path):
    from PIL import Image

    image = torch.rand(1, 8, 12, 3)
    runtime.write_image(image, str(tmp_path / "c.png"))
    assert np.array_equal(np.asarray(Image.open(tmp_path / "c.png")), (image[0] * 255).round().to(torch.uint8).numpy())


def test_msvc_platform_is_replaced(monkeypatch):
    monkeypatch.setenv("PLATFORM", "x64")
    runtime._prepare_environment()
    assert os.environ["PLATFORM"] == "cuda"
    monkeypatch.setenv("PLATFORM", "ascend_npu")
    runtime._prepare_environment()
    assert os.environ["PLATFORM"] == "ascend_npu"


class FakeRunner:
    pass


def test_release_frees_a_runner_frozen_in_a_cycle(comfy, monkeypatch):
    runner = FakeRunner()
    runner.self_ref = runner  # a reference cycle, only collectable by the GC
    probe = weakref.ref(runner)
    monkeypatch.setattr(runtime, "_state", {"key": ("A", "bf16 + sage", True), "runner": runner, "config_path": "c.json"})
    del runner
    gc.collect()
    gc.freeze()  # as LightX2V does after the first request
    try:
        runtime.release()
    finally:
        gc.unfreeze()
    assert probe() is None and runtime._state == {"key": None, "runner": None, "config_path": None}


def test_a_runner_that_cannot_be_freed_stays_loaded(monkeypatch):
    runner = FakeRunner()
    state = {"key": ("A", "bf16 + sage", True), "runner": runner, "config_path": "c.json"}
    monkeypatch.setattr(runtime, "_state", state)
    with pytest.raises(RuntimeError, match="Restart ComfyUI"):
        runtime.release()  # `runner` above still refers to it
    assert state == {"key": ("A", "bf16 + sage", True), "runner": runner, "config_path": "c.json"}


def test_stored_import_tracebacks_are_dropped(monkeypatch):
    class Weights:
        pass

    def import_fails():
        weights = Weights()
        try:
            try:
                raise ModuleNotFoundError("first")
            except ModuleNotFoundError:
                raise ModuleNotFoundError("second")  # chained, like mxfp8_fuse's nested import
        except ModuleNotFoundError as error:
            module.STORED_ERROR = error
        return weakref.ref(weights)

    module = types.ModuleType("lightx2v._edenlive_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    probe = import_fails()
    gc.collect()
    assert probe() is not None  # the traceback keeps import_fails' frame
    runtime._drop_import_tracebacks()
    gc.collect()
    assert probe() is None and str(module.STORED_ERROR) == "second" and module.STORED_ERROR.__traceback__ is None


def test_int8_profile_needs_the_quantized_files(monkeypatch, tmp_path):
    monkeypatch.setitem(runtime._state, "key", None)
    monkeypatch.setattr(runtime.profiles, "missing_modules", lambda profile: [])
    with pytest.raises(FileNotFoundError, match="--int8"):
        runtime.load(str(fake_model(tmp_path / "m")), "int8 + sage (fast)", True)


def test_model_folder_detection(tmp_path):
    assert not paths.is_model_dir(str(tmp_path))
    fake_model(tmp_path / "m")
    assert paths.is_model_dir(str(tmp_path / "m")) and not paths.has_int8(str(tmp_path / "m"))
    fake_model(tmp_path / "q", int8=True)
    assert paths.has_int8(str(tmp_path / "q"))


def test_proxy_file_uses_a_loaded_file_as_is(comfy, tmp_path):
    from comfy_api.latest import InputImpl, Types

    source = str(tmp_path / "src.mp4")
    runtime.write_proxy(torch.rand(5, 32, 48, 3), 24.0, source)
    assert runtime.proxy_file(InputImpl.VideoFromFile(source), str(tmp_path)) == source
    with open(source, "rb") as f:
        import io

        from_bytes = runtime.proxy_file(InputImpl.VideoFromFile(io.BytesIO(f.read())), str(tmp_path))
    assert open(from_bytes, "rb").read() == open(source, "rb").read()
    frames = torch.randint(0, 256, (5, 32, 48, 3)).float() / 255
    built = runtime.proxy_file(InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=None, frame_rate=24)), str(tmp_path))
    rgb, _ = decode_rgb(built)
    assert np.array_equal(rgb, (frames * 255).round().to(torch.uint8).numpy())
