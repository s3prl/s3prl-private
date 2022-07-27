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
