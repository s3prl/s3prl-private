"""
    Upstream expert for Audio-MAE
    Model Owner: Meta AI
"""

import yaml
import torch
from .model import VisionTransformer
from ..interfaces import UpstreamBase

class UpstreamExpert(UpstreamBase):
    """
    The Audio-MAE wrapper
    """

    def __init__(self, **kwargs):
        ckpt = torch.load(kwargs.pop("ckpt"))
        
        super().__init__(**kwargs)
        
        kwargs["use_custom_patch"] = ckpt["args"].use_custom_patch
        self.feature_selection = kwargs.pop("upstream_feature_selection", None)
        self.model = VisionTransformer(**kwargs)
        self.model.load_state_dict(ckpt["model"], strict=False)

    def get_downsample_rates(self, key: str) -> int:
        return 320 * 2

    def forward(self, wavs):
        states = self.model(wavs)
        if self.feature_selection:
            return {
                "hidden_states": states.get(self.feature_selection, states["hidden_states"]), "feature_lengths": states["feature_lengths"]
            }

        return states