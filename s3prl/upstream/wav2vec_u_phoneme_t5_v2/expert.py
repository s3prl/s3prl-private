# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec-u/expert.py ]
#   Synopsis     [ the wav2vec-u wrapper ]
#   Author       [ Jiatong Shi ]
#   Copyright    [ Copyleft(c), Carnegie Mellon Uiversity ]
"""*********************************************************************************************"""


from packaging import version

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from s3prl.utility.helper import zero_mean_unit_var_norm

# Wav2vec-u inference interface
from .wav2vec_u_t5 import Wav2VecUInfer, SimpleDict, PhonemeT5
from ..interfaces import UpstreamBase

# Fairseq-related information
import fairseq
from fairseq.file_io import PathManager


def load_wav2vec_u(filename, dict_file, downsample_rate, strict=True):

    # Load state dict
    state_dict = torch.load(filename, map_location="cpu")
    if not PathManager.exists(filename):
        raise IOError("Model file not found: {}".format(filename))

    if "cfg" in state_dict and state_dict["cfg"] is not None:
        cfg = state_dict["cfg"]
    else:
        raise RuntimeError(
            f"Neither args nor cfg exist in state_dict keys = {state_dict.keys()}"
        )

    model_cfg = cfg["model"]

    # Load dictionary file
    dict_state = SimpleDict(dict_file).pack_to_dict_info()

    model = Wav2VecUInfer(model_cfg, dict_state, downsample_rate)

    model.load_state_dict(
        state_dict["model"], strict=strict
    )
    return model


class UpstreamExpert(UpstreamBase):
    def __init__(self, joint_dict, ppg=False, hidden=False, use_tokenizer=True, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."

        assert ppg or hidden
        joint_dict = torch.load(joint_dict)
        ssl_ckpt = joint_dict["ssl"]
        u_model_ckpt = joint_dict["u_model"]
        u_model_dict = joint_dict["u_dict"]

        # interface information load
        self.interface_mode = joint_dict["interface"]["mode"] # mode for interface prediction
        self.interface_value = joint_dict["interface"]["value"] # layer number for prediction
        self.u_downsample_rate = joint_dict["interface"]["u_downsample_rate"] # unsupervised downsample rate

        ssl_model, ssl_cfg, ssl_task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_ckpt])
        self.model = ssl_model[0]
        self.wav_normalize = ssl_cfg.task.normalize

        # Loading unsupervised ASR model
        self.u_model = load_wav2vec_u(u_model_ckpt, u_model_dict, self.u_downsample_rate)

        # Loading phoneme T5 model
        self.T5 = PhonemeT5(self.u_downsample_rate, use_tokenizer=use_tokenizer)

        # These options are only used for aligning representations between s3prl and huggingface
        # See utility/compare_wav2vec2.py
        self.apply_padding_mask = True
        self.numpy_wav_normalize = False

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            if hidden:
                self.add_hook("self.u_model", lambda input, output: output["inter_x"])
            if ppg:
                self.add_hook("self.u_model", lambda input, output: output["upsampled_x"])
            
            self.add_hook("self.T5", lambda input, output: output)

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))
            self.hook_postprocess = postprocess
        
        self.feature_num = 25 if not hidden else 26
        self.feature_groups = {"hidden_states": [[i for i in range(self.feature_num)]]}
        if ppg:
            self.feature_groups["hidden_states"].append([self.feature_num])
            self.feature_num += 1
        self.feature_groups["hidden_states"].append([self.feature_num])

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        if self.wav_normalize:
            if self.numpy_wav_normalize:
                wavs = zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
                wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
            else:
                wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model.extract_features(
            padded_wav,
            wav_padding_mask if self.apply_padding_mask else None,
            layer=24
        )

        if self.interface_mode == "single_layer":
            assert len(results["layer_results"]) > self.interface_value
            u_feature = results["layer_results"][self.interface_value][0]
        else:
            raise RuntimeError("Do not support interface mode than single_layer")
        
        padding_mask = results["padding_mask"]
        if padding_mask is None:
            padding_mask = torch.BoolTensor(u_feature.shape[:2]).fill_(False).permute(1, 0)
        

        final_result = self.u_model(u_feature.permute(1, 0, 2), padding_mask)

        results = self.T5(final_result)

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
        return None
