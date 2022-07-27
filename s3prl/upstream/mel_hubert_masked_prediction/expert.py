"""
    UpstreamExpert of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/tree/master/s3prl/upstream/distiller)
    Reference author: Heng-Jui Chang (https://github.com/vectominist)
"""

import yaml
import torch
from ..interfaces import UpstreamBase
from .builder import PretrainedMelHuBERT


class UpstreamExpert(UpstreamBase):
    """
    The Mel Hubert wrapper
    """

    def __init__(self, ckpt, model_config=None, **kwargs):
        super().__init__(**kwargs)
        if model_config is not None:
            print(
                "[UpstreamExpert] - Using upstream expert config file from:",
                model_config,
            )
            with open(model_config, "r") as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        else:
            print("[UpstreamExpert] - Using the default upstream expert config")
            options = {
                "load_pretrain": "True",
                "no_grad": "False",
                "permute_input": "False",
            }

        options["ckpt_file"] = ckpt

        self.model = PretrainedMelHuBERT(options)
        
    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs, no_pred=True):
        hidden, _, _, _, _, layer_hiddens, pre_feat = self.model(
            wavs, get_hidden=True, no_pred=no_pred
        )

        hidden_states = [pre_feat] + layer_hiddens

        states = {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden
        }

        return states
