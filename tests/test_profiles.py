import os

import pytest

from edenlive_comfy import profiles


@pytest.mark.parametrize("name", list(profiles.PROFILES))
def test_every_profile_is_the_5b_few_step_setup(name):
    config = profiles.build(name, "M", t5_on_cpu=True)
    assert config["size"] == [704, 1280] and config["fps"] == 24 and config["infer_steps"] == 4
    assert config["ar_config"] == {"num_frame_per_chunk": 3, "local_attn_size": 18, "sink_size": 6, "kv_cache_scheme": "fp", "step_kv_cache": False,
                                   "denoising_step_list": [1000, 937.5, 833.3333333333, 625]}
    assert config["use_tae"] and config["streaming_vae"] and config["tae_path"] == os.path.join("M", "decoder-5b", "lighttaew2_2.pth")
    assert config["t5_cpu_offload"] is True
    assert config["rope_type"] == "torch_real_rope" and config["rms_norm_type"] == "torch"


def test_int8_profile_points_at_the_quantized_files():
    config = profiles.build("int8 + sage (fast)", "M", t5_on_cpu=False)
    assert config["dit_quantized"] and config["dit_quant_scheme"] == "int8-convrot"
    assert config["dit_quantized_ckpt"] == os.path.join("M", "quantized", "dit-5b-int8-convrot.safetensors")
    assert config["proxy_quantized_ckpt"] == os.path.join("M", "quantized", "proxy-dino-int8-convrot.safetensors")
    assert config["t5_cpu_offload"] is False
    assert "dit_quantized" not in profiles.build("bf16 + sage", "M", True)


def test_kernels():
    assert profiles.build("bf16 + pytorch kernels", "M", True)["self_attn_1_type"] == "torch_sdpa"
    assert profiles.build("bf16 + sage", "M", True)["self_attn_1_type"] == "sage_attn2"
    assert profiles.missing_modules("bf16 + pytorch kernels") == []


def test_build_does_not_share_the_ar_config():
    a = profiles.build("bf16 + sage", "M", True)
    a["ar_config"]["sink_size"] = 0
    assert profiles.build("bf16 + sage", "M", True)["ar_config"]["sink_size"] == 6
