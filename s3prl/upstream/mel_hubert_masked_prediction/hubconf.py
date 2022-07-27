"""
    Hubconf for Mel HuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
"""

import os
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def mel_hubert_masked_prediction_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def chimera_melhubert(*args, **kwargs):

    ckpt = "~/.cache/torch/hub/s3prl_cache/chimera_v1/states-40000.ckpt"
    kwargs["model_config"] = "../../pretrain/mel_hubert_masked_prediction/pretraining-config/chimera_v1/config_model.yaml"
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)