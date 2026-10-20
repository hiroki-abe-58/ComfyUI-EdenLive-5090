"""LightX2V configs for EdenLive 5B few-step (chunk-autoregressive), as measured on an RTX 5090 under Windows.

All profiles follow LightX2V's configs/edenlive/wan22_5b_few_step.json and the RTX 5090 preset
(configs/edenlive/realtime/wan22_5b_ar_int8_convrot_5090.json): 1280x704, 24 fps, 4 steps, 3 latent frames per chunk, an
attention window of 18 latent frames with 6 sink frames, and the LightTAE streaming decoder. They differ only in kernels.
"""

from __future__ import annotations

import os

BASE = {
    "size": [704, 1280],
    "fps": 24,
    "num_frames": None,
    "infer_steps": 4,
    "sample_shift": 5.0,
    "sample_guide_scale": 1.0,
    "enable_cfg": False,
    "use_image_encoder": False,
    "proxy_resize_mode": "bilinear",
    "proxy_condition_rmsnorm": False,
    "proxy_unconditional_control": "same",
    "cpu_offload": False,
    "mxfp8_fuse_enable": False,
    "ar_config": {
        "num_frame_per_chunk": 3,
        "local_attn_size": 18,
        "sink_size": 6,
        "kv_cache_scheme": "fp",
        "step_kv_cache": False,
        "denoising_step_list": [1000, 937.5, 833.3333333333, 625],
    },
    "use_tae": True,
    "need_scaled": True,
    "streaming_vae": True,
    # flashinfer RoPE and sgl-kernel RMSNorm have no Windows builds; these are LightX2V's PyTorch versions of the same ops
    "rope_type": "torch_real_rope",
    "rms_norm_type": "torch",
}

PROFILES = {
    "int8 + sage (fast)": {
        "kernels": {"self_attn_1_type": "sage_attn2", "cross_attn_1_type": "sage_attn2", "cross_attn_2_type": "sage_attn2",
                    "layer_norm_type": "Triton", "modulate_type": "triton"},
        "int8": True,
        "needs": ("sageattention", "triton"),
    },
    "bf16 + sage": {
        "kernels": {"self_attn_1_type": "sage_attn2", "cross_attn_1_type": "sage_attn2", "cross_attn_2_type": "sage_attn2",
                    "layer_norm_type": "Triton", "modulate_type": "triton"},
        "int8": False,
        "needs": ("sageattention", "triton"),
    },
    "bf16 + pytorch kernels": {
        "kernels": {"self_attn_1_type": "torch_sdpa", "cross_attn_1_type": "torch_sdpa", "cross_attn_2_type": "torch_sdpa",
                    "layer_norm_type": "torch", "modulate_type": "torch"},
        "int8": False,
        "needs": (),
    },
}


def build(profile: str, model_dir: str, t5_on_cpu: bool) -> dict:
    """The LightX2V config dict for a profile and a converted model folder."""
    spec = PROFILES[profile]
    config = dict(BASE, **spec["kernels"])
    config["ar_config"] = dict(BASE["ar_config"])
    config["tae_path"] = os.path.join(model_dir, "decoder-5b", "lighttaew2_2.pth")
    config["t5_cpu_offload"] = bool(t5_on_cpu)
    if spec["int8"]:
        config.update(
            dit_quantized=True,
            dit_quant_scheme="int8-convrot",
            dit_quantized_ckpt=os.path.join(model_dir, "quantized", "dit-5b-int8-convrot.safetensors"),
            proxy_quantized_ckpt=os.path.join(model_dir, "quantized", "proxy-dino-int8-convrot.safetensors"),
        )
    return config


def missing_modules(profile: str) -> list[str]:
    import importlib.util

    return [name for name in PROFILES[profile]["needs"] if importlib.util.find_spec(name) is None]
