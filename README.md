# EdenLive 5B for ComfyUI

[EdenLive](https://github.com/VAST-AI-Research/EdenLive) (VAST) renders a reference video (a greybox, a game capture, any rough
render) into detailed video, guided by a target first frame and a text prompt. The reference video sets the layout, camera motion
and timing; the first frame and the prompt set the look. These nodes run the 5B few-step (chunk-autoregressive) model at
1280×704 and 24 fps through [LightX2V](https://github.com/ModelTC/LightX2V)'s EdenLive runner, called the same way as its
command line, so a seed gives the frames `python -m lightx2v.infer --model_cls wan22_edenlive_ar` writes.

This is an unofficial integration, not affiliated with VAST, ModelTC or Comfy Org.

## Nodes

| Node | What it does |
| --- | --- |
| **EdenLive Loader** | Picks a model folder in `models/edenlive/`, a kernel profile and where the text encoder runs. |
| **EdenLive Render** | `proxy` (a VIDEO, e.g. from Load Video), `first_frame` (IMAGE), `prompt`, `seed` → the rendered frames (IMAGE). |

Save the frames with Create Video (fps 24) and Save Video, or Save Image.

## Status

Work in progress: measured on one Windows 11 machine with an RTX 5090. See below for what was checked.
