"""Download EdenLive 5B (few-step, chunk-autoregressive) and convert it to the folder layout the nodes load.

    python tools/prepare_model.py --models-dir ComfyUI/models            # BF16 profiles
    python tools/prepare_model.py --models-dir ComfyUI/models --int8     # also the INT8 ConvRot files of the "int8 + sage" profile

Only the 5B few-step files are fetched (34.2 GiB), from VAST-AI/EdenLive at a pinned revision, and each is checked against
tools/model_files.txt. The 14B and bidirectional DiTs, CLIP and the FLUX workbench model are not downloaded. The conversion uses
LightX2V's own functions (tools/convert in a LightX2V source checkout, which must be importable as `lightx2v`); --int8 runs
LightX2V's quantizer on the GPU, which compiles a small C++ extension (on Windows: run it from an x64 Native Tools prompt, with
PLATFORM=cuda).

The vision encoder (vision-encoder/) is DINOv3 by Meta and is distributed under the DINOv3 License, not Apache-2.0; using it
means accepting that license. See the README.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = "VAST-AI/EdenLive"
HERE = Path(__file__).resolve().parent


def manifest():
    lines = (HERE / "model_files.txt").read_text(encoding="utf-8").splitlines()
    revision = lines[0].split(" at ")[1].split(":")[0]
    files = {}
    for line in lines[1:]:
        digest, size, name = line.split(" ", 2)
        files[name] = (digest, int(size))
    return revision, files


def file_digest(path: Path, kind: str) -> str:
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def download(release: Path):
    from huggingface_hub import hf_hub_download

    revision, files = manifest()
    for name, (digest, size) in files.items():
        kind, expected = digest.split(":")
        path = release / name
        if not (path.is_file() and path.stat().st_size == size and file_digest(path, kind) == expected):
            path = Path(hf_hub_download(REPO, name, revision=revision, local_dir=release))
            actual = file_digest(path, kind)
            if path.stat().st_size != size or actual != expected:
                raise SystemExit(f"{name}: checksum mismatch ({kind} {actual})")
        print("ok", name, flush=True)


def lightx2v_root() -> Path:
    try:
        import lightx2v
    except ImportError as error:
        raise SystemExit(f"LightX2V is not importable ({error}); see the README for the pinned install") from error
    root = Path(lightx2v.__file__).resolve().parents[1]
    if not (root / "tools" / "convert" / "convert_edenlive.py").is_file():
        raise SystemExit(f"{root} has no tools/convert: install LightX2V from a source checkout (pip install -e)")
    return root


def convert(release: Path, output: Path, root: Path):
    sys.path.insert(0, str(root))
    from tools.convert.convert_edenlive import convert_dit, convert_encoder, copy_tree

    output.mkdir(parents=True, exist_ok=True)
    convert_dit(release / "dit-5b", output / "dit-5b", overwrite=False)
    convert_encoder(release / "umt5/text_encoder", output / "umt5/text_encoder", "t5", overwrite=False)
    for name in ("vae-5b", "decoder-5b", "scheduler-5b", "vision-encoder", "umt5/tokenizer"):
        copy_tree(release / name, output / name, overwrite=False)
    for name in ("LICENSE", "README.md"):
        (output / name).write_bytes((release / name).read_bytes())


def quantize(output: Path, root: Path):
    import importlib.util

    missing = [name for name in ("qtorch", "ninja") if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"--int8 runs LightX2V's quantizer, which needs {' and '.join(missing)}: pip install qtorch ninja")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])))
    quantized = output / "quantized"
    if not (quantized / "dit-5b-int8-convrot.safetensors").is_file():
        subprocess.run([sys.executable, str(root / "tools/convert/convert_edenlive.py"), "--source", str(output / "dit-5b/diffusion_pytorch_model.safetensors"),
                        "--output", str(quantized), "--output_name", "dit-5b-int8-convrot", "--quantize", "int8-convrot", "--device", "cuda:0"],
                       check=True, cwd=root, env=env)
    if not (quantized / "proxy-dino-int8-convrot.safetensors").is_file():
        subprocess.run([sys.executable, str(root / "tools/convert/convert_edenlive_proxy.py"),
                        "--source", str(output / "vision-encoder/projection.safetensors"),
                        "--output", str(quantized / "proxy-dino-int8-convrot.safetensors"), "--device", "cuda:0", "--group-size", "256"],
                       check=True, cwd=root, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models folder")
    parser.add_argument("--name", default="EdenLive-5B", help="folder name under models/edenlive/")
    parser.add_argument("--int8", action="store_true", help="also write the INT8 ConvRot DiT and proxy encoder (GPU)")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    release = args.models_dir / "edenlive" / "_release" / f"{REPO.replace('/', '--')}"
    output = args.models_dir / "edenlive" / args.name
    download(release)
    if args.download_only:
        return
    root = lightx2v_root()
    convert(release, output, root)
    if args.int8:
        quantize(output, root)
    print("ready:", output)


if __name__ == "__main__":
    main()
