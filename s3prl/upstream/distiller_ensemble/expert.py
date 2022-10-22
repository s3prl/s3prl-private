"""
    Upstream expert for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import torch
import yaml

import torch.nn as nn
from ..interfaces import UpstreamBase
from ..distiller.builder import PretrainedDistiller
from s3prl.utility.download import _urls_to_filepaths


class UpstreamExpert(UpstreamBase):
    """
    Ensemble Distillers
    """

    def __init__(self, ckpt, model_config=None, **kwargs):
        super().__init__(**kwargs)

        '''
        ckpt        : <ckpt1>,,<ckpt2>,,<ckpt3>
        model_config: <model_config1>,,<model_config2>,,<model_config3>
        '''
        default_ckpt = _urls_to_filepaths(
            "https://www.dropbox.com/s/hcfczqo5ao8tul3/disilhubert_ls960_4-8-12.ckpt?dl=1"
        )
        ckpts = ckpt.split(',,')
        ckpts = [_ckpt.strip(' \n') for _ckpt in ckpts]
        ckpts = [_ckpt if _ckpt else default_ckpt for _ckpt in ckpts]

        if model_config:
            model_configs = model_config.split(',,')

        self.models = nn.ModuleList()
        for i, _ckpt in enumerate(ckpts):
            if model_config:
                print(
                    "[UpstreamExpert] - Using upstream expert config file from:",
                    model_configs[i],
                )
                with open(model_configs[i], "r") as file:
                    options = yaml.load(file, Loader=yaml.FullLoader)
            else:
                print("[UpstreamExpert] - Using the default upstream expert config")
                options = {
                    "load_pretrain": "True",
                    "no_grad": "False" if kwargs.get("trainable", False) else "True",
                    "permute_input": "False",
                }
            
            options["ckpt_file"] = _ckpt
            self.models.append(PretrainedDistiller(options))
        self.ensemble_num = len(self.models) # for featurizer.tolist assertion

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs, no_pred=True):
        feat_finals = []
        preds = []
        pad_masks = []
        layer_hiddens = []
        for i, model in enumerate(self.models):
            feat_final, pred, pad_mask, layer_hidden = model(
                wavs, get_hidden=True, no_pred=no_pred
            )[1:5]

            feat_finals.append(feat_final)
            preds.append(pred)
            pad_masks.append(pad_mask)
            layer_hiddens.append(layer_hidden)

        all_states = []
        for feat_final, pred, pad_mask, layer_hidden in \
            zip(feat_finals, preds, pad_masks, layer_hiddens):
                if not no_pred:
                    if type(pred) is list:
                        pred = sum(pred) / len(pred)
                    hidden_feats = pred.transpose(0, 1).split(1, 0)
                    hidden_feats = [hid.squeeze(0) for hid in hidden_feats]
                else:
                    hidden_feats = []
                hidden_feats = [feat_final] + layer_hidden + hidden_feats

                states = {
                    "hidden_states": hidden_feats,
                }
                all_states.append(states)
        
        '''
        1. concatenate          : all concatenate         -> 1 feature
        2. layerwise_concatenate: concatenate layerwisely -> #L features
        3. no_concatenate       : without concatenation   -> #distiller * #L features
        '''

        # concatenate
        all_concatenate_feat = torch.cat(
            sum([states["hidden_states"] for states in all_states], []), dim=1
        )

        # layerwise_concatenate
        for i in range(1, self.ensemble_num):
            assert len(all_states[i]["hidden_states"]) == len(all_states[i - 1]["hidden_states"])
        n_hidden_feats = len(all_states[0]["hidden_states"])
        layerwise_concatenate_feats = [
             torch.cat(
                [states["hidden_states"][i] for states in all_states], dim=1
             )
             for i in range(n_hidden_feats)
        ]

        # no_concatenate
        no_concatenate_feats = sum(
            [states["hidden_states"] for states in all_states], []
        )

        # # layerwise_mean
        # layerwise_mean_feats = [
        #     torch.mean(
        #         [states["hidden_states"][i] for states in all_states], dim=1
        #      )
        #      for i in range(n_hidden_feats)
        # ]

        results = {
            "concatenate": all_concatenate_feat,
            "layerwise_concatenate": layerwise_concatenate_feats,
            "last_layer_concatenate": layerwise_concatenate_feats[-1],
            "no_concatenate": no_concatenate_feats,
            # "layerwise_mean": layerwise_mean_feats,
            "hidden_states": no_concatenate_feats # same with no_concatenate
        }

        # Use only for "concatenate" in interfaces.py
        self.n_hidden_feats = n_hidden_feats
        return results
