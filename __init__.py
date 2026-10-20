"""EdenLive 5B for ComfyUI (unofficial integration of https://github.com/VAST-AI-Research/EdenLive through LightX2V)."""

import logging

try:
    from .edenlive_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # a ComfyUI without the V3 node API
    logging.error("[EdenLive] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
