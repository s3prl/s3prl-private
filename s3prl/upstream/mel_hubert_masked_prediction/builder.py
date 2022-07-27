"""
    Builder of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/tree/master/s3prl/upstream/distiller)
    Reference author: Heng-Jui Chang (https://github.com/vectominist)
"""

import sys
from distutils.util import strtobool
import yaml
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from .model import MelHuBERTConfig, MelHuBERTModel
from .audio import create_transform
import s3prl.optimizers
from s3prl.utility import prune
 
class MelHuBERTBuilder(nn.Module):

    def __init__(self, options, config, verbose=False):
        super().__init__()

        # read config
        if config is not None:
            self.config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        else:
            # Since some old checkpoints contained pickled scheduler which needs 'optimizers'
            # module which is now moved into s3prl package.
            original_optimizer = sys.modules.get("optimizers")
            sys.modules["optimizers"] = s3prl.optimizers

            self.all_states = torch.load(options["ckpt_file"], map_location="cpu")
        
            self.config = self.all_states["Upstream_Config"]
           
            del sys.modules["optimizers"]
            if original_optimizer is not None:
                sys.modules["optimizers"] = original_optimizer

        # parse the options dict
        self.load = bool(strtobool(options["load_pretrain"]))
        self.no_grad = bool(strtobool(options["no_grad"]))
        self.permute_input = bool(strtobool(options["permute_input"]))

        # Set model config
        self.model_config = MelHuBERTConfig(self.config['hubert'])
        self.data_config = self.config["data"]["audio"]
        self.hidden_size = self.model_config.encoder_embed_dim
        self.max_input_length = 0

        if self.max_input_length > 0 and verbose:
            print("[MelHuBERTBuilder] - Maximum input length: ", self.max_input_length)

    def load_model(self, model, state_dict, verbose=False):
        try:
            tmp = "".join(state_dict.keys())
            if "_orig" in tmp and "_mask" in tmp:
                params_to_prune, _ = model.get_params_to_prune()
                prune.global_unstructured(
                    params_to_prune,
                    pruning_method=prune.Identity,
                )
                model.load_state_dict(state_dict)
                for module, name in params_to_prune:
                    prune.remove(module, name)
            else:
                model.load_state_dict(state_dict)
            if verbose:
                print("[MelHuBERTBuilder] - Pre-trained weights loaded!")
            return model
        except Exception as e:
            raise RuntimeError("[MelHuBERTBuilder] - Pre-trained weights NOT loaded!\n" + str(e))

    def process_input_data(self, waves):
        """Process input data for the model"""
        # add arbitary batch axis B if input `wave` has shape of T
        processed_wav = []
        for wav in waves:
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            elif wav.dim() > 2:
                raise ValueError
            processed_wav.append(wav<<15)
        return processed_wav

class PretrainedMelHuBERT(MelHuBERTBuilder):

    def __init__(self, options, config=None, verbose=False):
        super().__init__(options, config, verbose)

        # Build model
        self.model = MelHuBERTModel(self.model_config)
        
        self.model.eval() if self.no_grad else self.model.train()
    
        self.out_dim = self.hidden_size

        # Load from a PyTorch state_dict
        if self.load:
            self.model = self.load_model(
                self.model, self.all_states["model"], verbose
            )

            if verbose:
                print(
                    "[PretrainedMelHuBERT] - Number of parameters: "
                    + str(
                        sum(
                            p.numel()
                            for p in self.model.parameters()
                            if p.requires_grad
                        )
                    )
                )
    
        self.preprocessor, feat_dim = create_transform(self.data_config)

    def forward(self, wave_inputs, get_hidden=False, no_pred=True):
        wave_inputs = self.process_input_data(wave_inputs)
        features = [self.preprocessor(wav) for wav in wave_inputs]
        feat_lengths = [len(feat) for feat in features]
        feat_pad_batch = pad_sequence(features, batch_first=True) # (B x S x D)

        pad_mask = torch.ones(feat_pad_batch.shape[:-1])  # (B x S)
        # zero vectors for padding dimension
        for idx in range(feat_pad_batch.shape[0]):
            pad_mask[idx, feat_lengths[idx]:] = 0

        feat = feat_pad_batch.to(dtype=torch.float32)
        pad_mask = torch.FloatTensor(pad_mask).to( 
            device=feat.device, dtype=torch.float32
        )  # (batch_size, seq_len)

        if self.no_grad:
            with torch.no_grad():
                x = self.model(feat, pad_mask, get_hidden=get_hidden, no_pred=no_pred, mask=False)
        else:
            x = self.model(feat, pad_mask, get_hidden=get_hidden, no_pred=no_pred, mask=False)
        return x
