"""
    hubconf for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import os
from .expert import UpstreamExpert as _UpstreamExpert


def distiller_ensemble(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): <ckpt1>,,<ckpt2>,,<ckpt3>
        e.g. ''                  -> default (distilhubert)
             'distilw2v2.ckpt,,' -> distilw2v2 + default
    """
    return _UpstreamExpert(ckpt, *args, **kwargs)

def distiller_ensemble_hubert_rhubert_wavlmb_plus(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): <ckpt1>,,<ckpt2>,,<ckpt3>
        e.g. ''                  -> default (distilhubert)
             'distilw2v2.ckpt,,' -> distilw2v2 + default
    """
    rhubert = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/distilhubertmgwhamrbp/dev-dis-best-new.ckpt")
    wavlmb_plus = os.path.join(os.path.expanduser('~'), ".cache/torch/hub/s3prl_cache/distilwavlm_base_plus_init_enc_f_student_hubert/dev-dis-best.ckpt")
    ckpt = ",,".join([rhubert, wavlmb_plus, ""])
    
    return _UpstreamExpert(ckpt, *args, **kwargs)