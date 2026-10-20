"""ComfyUI nodes for EdenLive 5B: render a proxy video (greybox, game capture, any reference) into the look of a first frame and a
prompt, through LightX2V's EdenLive runner."""

from __future__ import annotations

import os
import tempfile

from comfy_api.latest import ComfyExtension, io

from . import paths, profiles, runtime

EdenLiveModelType = io.Custom("EDENLIVE_MODEL")


class EdenLiveLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EdenLiveLoader",
            display_name="EdenLive Loader",
            category="loaders/edenlive",
            description=("Loads EdenLive 5B (few-step, chunk-autoregressive) from models/edenlive/<name>/, converted to LightX2V's layout "
                         "by tools/prepare_model.py. It stays loaded for the rest of the ComfyUI session (LightX2V cannot release it in-process); "
                         "ComfyUI's own models are unloaded first."),
            inputs=[
                io.Combo.Input("model", options=paths.model_names()),
                io.Combo.Input("profile", options=list(profiles.PROFILES), default="int8 + sage (fast)",
                               tooltip=("int8 + sage: INT8 ConvRot weights and SageAttention 2 (fastest, needs the --int8 files). bf16 + sage: "
                                        "the BF16 weights. bf16 + pytorch kernels: no SageAttention or Triton needed")),
                io.Boolean.Input("t5_on_cpu", default=True,
                                 tooltip="run the 10.6 GB UMT5 text encoder on the CPU (about 0.7 s per prompt) to leave the GPU memory to the DiT"),
            ],
            outputs=[EdenLiveModelType.Output(display_name="edenlive")],
        )

    @classmethod
    def execute(cls, model, profile, t5_on_cpu) -> io.NodeOutput:
        if profile not in profiles.PROFILES:
            raise ValueError(f"unknown profile {profile!r}")
        return io.NodeOutput({"model_dir": paths.model_dir(model), "profile": profile, "t5_on_cpu": bool(t5_on_cpu)})


class EdenLiveRender(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EdenLiveRender",
            display_name="EdenLive Render",
            category="video/edenlive",
            description=("Renders the proxy frames into detailed video with the look of first_frame and the prompt: 1280x704 at 24 fps, "
                         "in chunks of 12 frames. The output has 4n+1 frames (up to 3 fewer than the proxy at 24 fps)."),
            inputs=[
                EdenLiveModelType.Input("edenlive"),
                io.Video.Input("proxy", tooltip=("the proxy video (camera, layout and timing come from here), e.g. from Load Video. A loaded file is "
                                                 "read by LightX2V exactly as its command line reads it; a video made from frames (Create Video) is "
                                                 "passed losslessly")),
                io.Image.Input("first_frame", tooltip="the target look of the proxy's first frame (resized to 1280x704)"),
                io.String.Input("prompt", multiline=True, default="Cinematic photorealism."),
                io.Int.Input("seed", default=2026092901, min=0, max=0xffffffff, control_after_generate=True),
            ],
            outputs=[io.Image.Output(display_name="frames")],
        )

    @classmethod
    def execute(cls, edenlive, proxy, first_frame, prompt, seed) -> io.NodeOutput:
        import comfy.model_management
        import comfy.utils

        if not prompt.strip():
            raise ValueError("prompt is empty")
        runner, config_path = runtime.load(edenlive["model_dir"], edenlive["profile"], edenlive["t5_on_cpu"])
        bar = comfy.utils.ProgressBar(max(1, (int(proxy.get_duration() * 24) + 11) // 12))

        def on_chunk(pixels, index, is_last):
            bar.update_absolute(index + 1)
            comfy.model_management.throw_exception_if_processing_interrupted()

        with tempfile.TemporaryDirectory(prefix="edenlive_") as tmp:
            proxy_path, cover_path = runtime.proxy_file(proxy, tmp), os.path.join(tmp, "first_frame.png")
            runtime.write_image(first_frame, cover_path)
            frames = runtime.render(runner, edenlive["model_dir"], config_path, proxy_path, cover_path, prompt, seed, on_chunk)
        return io.NodeOutput(frames)


class EdenLiveExtension(ComfyExtension):
    async def get_node_list(self):
        return [EdenLiveLoader, EdenLiveRender]


async def comfy_entrypoint() -> EdenLiveExtension:
    paths.register()
    return EdenLiveExtension()
