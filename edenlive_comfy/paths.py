"""Model folders: models/edenlive/<name>/ holds an EdenLive 5B release converted to LightX2V's layout (tools/prepare_model.py)."""

from __future__ import annotations

import os

FOLDER = "edenlive"

REQUIRED = (
    "dit-5b/config.json",
    "dit-5b/diffusion_pytorch_model.safetensors",
    "vae-5b/Wan2.2_VAE.pth",
    "decoder-5b/lighttaew2_2.pth",
    "umt5/text_encoder/model.safetensors",
    "umt5/tokenizer/tokenizer.json",
    "vision-encoder/backbone/model.safetensors",
    "vision-encoder/projection.safetensors",
)
INT8 = ("quantized/dit-5b-int8-convrot.safetensors", "quantized/proxy-dino-int8-convrot.safetensors")


def register():
    import folder_paths

    if FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=True)


def _roots():
    try:
        import folder_paths

        return folder_paths.get_folder_paths(FOLDER)
    except Exception:  # noqa: BLE001 - outside ComfyUI (tests)
        return []


def is_model_dir(path: str) -> bool:
    return all(os.path.isfile(os.path.join(path, name)) for name in REQUIRED)


def has_int8(path: str) -> bool:
    return all(os.path.isfile(os.path.join(path, name)) for name in INT8)


def model_names() -> list[str]:
    names = []
    for root in _roots():
        if os.path.isdir(root):
            names += sorted(entry.name for entry in os.scandir(root) if entry.is_dir() and is_model_dir(entry.path))
    return names or ["(none: run tools/prepare_model.py)"]


def model_dir(name: str) -> str:
    for root in _roots():
        path = os.path.join(root, name)
        if is_model_dir(path):
            return path
    raise FileNotFoundError(f"no EdenLive model folder {name!r} in models/{FOLDER}/ (see tools/prepare_model.py)")
