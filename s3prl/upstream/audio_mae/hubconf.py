import os
import torch.nn as nn
from functools import partial

# from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert

def audio_mae(*args, **kwargs):
    """
    vit_small_patch16
    """
    kwargs.update({
        "ckpt": os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/audio_mae/pretrained.pth"),
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    })
    assert os.path.isfile(kwargs["ckpt"])
    return _UpstreamExpert(*args, **kwargs)