"""
    hubconf for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import os
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def distiller_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def distiller_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from url
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return distiller_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def distilhubert(refresh=False, *args, **kwargs):
    """
    DistilHuBERT
    """
    return distilhubert_base(refresh=refresh, *args, **kwargs)


def distilhubert_base(refresh=False, *args, **kwargs):
    """
    DistilHuBERT Base
    Default model in https://arxiv.org/abs/2110.01900
    """
    kwargs[
        "ckpt"
    ] = "https://www.dropbox.com/s/hcfczqo5ao8tul3/disilhubert_ls960_4-8-12.ckpt?dl=0"
    return distiller_url(refresh=refresh, *args, **kwargs)

def distilhubert_base_robust_mgwham_rbp(refresh=False, *args, **kwargs):
    """
    DistilHuBERT Base with robust training
    Default model please ask https://huggingface.co/kphuang68
    """
    kwargs[
        "ckpt"
    ] = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/setup2_2-dis_t_cont_cp/dev-dis-best.ckpt")
    return distiller_local(refresh=refresh, *args, **kwargs)

def distilhubert_base_robust_mgwham_rbp_paper(refresh=False, *args, **kwargs):
    """
    DistilHuBERT Base with robust training
    Default model please ask https://huggingface.co/kphuang68
    """
    kwargs[
        "ckpt"
    ] = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/setup2_2-dis_t_cont_cp/dev-dis-best.ckpt")
    return distiller_local(refresh=refresh, feature_selection="paper", no_pred=True, *args, **kwargs)

def chimera_melhubert(*args, **kwargs):

    kwargs["ckpt"] = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/chimera_v1/states-40000.ckpt")
    kwargs["config"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "pretrain/mel_hubert_masked_prediction/pretraining-config/chimera_v1/config_model.yaml"
    )
    assert os.path.isfile(kwargs["ckpt"])
    return _UpstreamExpert(*args, **kwargs)

def distilmelhubert_t_cont_init_mel_tr3(*args, **kwargs):

    kwargs["ckpt"] = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/distilmelhubert_t_cont_init_mel_tr3/dev-dis-best.ckpt")
    kwargs["config"] = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/distilmelhubert_t_cont_init_mel_tr3/config_model.yaml")
    assert os.path.isfile(kwargs["ckpt"])
    return _UpstreamExpert(*args, **kwargs)

def distilmelhubert_t_cont_init_teacher_tr3(*args, **kwargs):

    kwargs["ckpt"] = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/distilmelhubert_t_cont_init_teacher_tr3/dev-dis-best.ckpt")
    kwargs["config"] = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/distilmelhubert_t_cont_init_teacher_tr3/config_model.yaml")
    assert os.path.isfile(kwargs["ckpt"])
    return _UpstreamExpert(*args, **kwargs)

def distilw2v2(*args, **kwargs):

    kwargs["ckpt"] = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/distilw2v2/dev-dis-best.ckpt")
    assert os.path.isfile(kwargs["ckpt"])
    return _UpstreamExpert(*args, **kwargs)

def distilwavlm_base_plus_init_enc_f_student_hubert(*args, **kwargs):

    kwargs["ckpt"] = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/distilwavlm_base_plus_init_enc_f_student_hubert/dev-dis-best.ckpt")
    assert os.path.isfile(kwargs["ckpt"])
    return _UpstreamExpert(*args, **kwargs)