# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wavlm/hubconf.py ]
#   Synopsis     [ the WavLM torch hubconf ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os

# -------------#
from .expert import UpstreamExpert as _UpstreamExpert

user = os.path.expanduser('~')
pth_path = os.path.join(
        user, ".cache/torch/hub/s3prl_cache/wav2vec_u/joint_wav2vec_u_model.pth")

def wav2vec_u2_pt5_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)

def wav2vec_u2_pt5_base(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = pth_path
    return wav2vec_u2_pt5_local(*args, ppg = True, hidden = True, **kwargs)


def wav2vec_u2_pt5_base_ppg(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = pth_path
    return wav2vec_u2_pt5_local(*args, ppg = True, **kwargs)

def wav2vec_u2_pt5_base_hidden(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = pth_path
    return wav2vec_u2_pt5_local(*args, hidden = True, **kwargs)

def wav2vec_u2_pt5_base_ppg_token(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = pth_path
    return wav2vec_u2_pt5_local(*args, use_tokenizer=True, ppg = True, **kwargs)

def wav2vec_u2_pt5_base_hidden_token(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = pth_path
    return wav2vec_u2_pt5_local(*args, use_tokenizer=True, hidden = True, **kwargs)

def wav2vec_u2_pt5_base_token(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = pth_path
    return wav2vec_u2_pt5_local(*args, use_tokenizer=True, ppg = True, hidden = True, **kwargs)