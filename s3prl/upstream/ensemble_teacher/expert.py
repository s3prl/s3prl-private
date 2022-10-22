"""
    Upstream expert for ensemble hubert and WavLM
"""

import torch

from ..interfaces import UpstreamBase
from ..wavlm.WavLM import WavLM,WavLMConfig
from ...utility.download import _urls_to_filepaths
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import s3prl.hub as hub


# +
class UpstreamExpert(UpstreamBase):

    def __init__(self, ckpt=None, model_config=None, **kwargs):
        super().__init__(**kwargs)
        device='cuda'
        
        self.model = nn.ModuleList()
# ==============
        if ckpt==None:
            self.model.append(hub.wavlm_base_plus().model)
            self.model.append(hub.hubert().model)
            self.model.append(hub.hubert_base_robust_mgwham_rbp().model)
        else:
            ckpts=ckpt.split(',,')
            for i in range(len(ckpts)):
                ckpts[i]=ckpts[i].lower()
            print(ckpts)
            if "wavlm" in ckpts:
                WavLM_url="https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base_plus.pt"
                WavLM_ckpt=_urls_to_filepaths(WavLM_url, refresh=False)
                checkpoint = torch.load(WavLM_ckpt)
                cfg = WavLMConfig(checkpoint['cfg'])
                WavLM_model = WavLM(cfg)
                WavLM_model.load_state_dict(checkpoint['model'])
                self.model.append(WavLM_model.to(device))
            if "hubert" in ckpts:
                hubert_url="https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_base_ls960.pt"
                hubert_ckpt=_urls_to_filepaths(hubert_url,refresh=False)
                hubert_model,task_cfg = load_converted_model(hubert_ckpt)
                self.model.append(hubert_model.to(device))
            if 'hubertrobust' in ckpts:
                hubertrobust_url="https://huggingface.co/s3prl/converted_ckpts/resolve/main/HuBERT_base_robust_mgr_best_loss_2.7821.pt"
                hubertrobust_ckpt=_urls_to_filepaths(hubertrobust_url,refresh=False)
                hubertrobust_model,_ = load_converted_model(hubertrobust_ckpt)
                self.model.append(hubertrobust_model.to(device))
                

                
            
    
        
# ========================
        self.ensemble_num = len(self.model) # for featurizer.tolist assertion
        for i in range(self.ensemble_num):
            self.model[i].feature_grad_mult = 0.0
            self.model[i].encoder.layerdrop = 0.0
        #return values of model(from hubert,WavLM expert)
        if len(self.hooks) == 0:
            for model_id in range(self.ensemble_num):
                module_name = f"self.model[{model_id}].encoder.layers"
                for module_id in range(len(eval(module_name))):
                    self.add_hook(
                        f"{module_name}[{module_id}]",
                        lambda input, output: input[0].transpose(0, 1),
                    )
                self.add_hook(f"self.model[{model_id}].encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                """hiddens is a list of 26 features,first 13 are from hubert,remaining belongs to WavLM"""
                sum_hiddens=[]
                mean_hiddens=[]
                concate_hiddens=[]
                for i in range(int(len(hiddens)/self.ensemble_num)):
                    hidden=[]
                    for j in range(self.ensemble_num):
                        hidden.append(hiddens[(int(len(hiddens)/self.ensemble_num)-1)*j+i])
                    sum_hiddens.append(sum(hidden))
                    mean_hiddens.append(sum(hidden)/self.ensemble_num)
                    concate_hiddens.append(torch.cat(hidden,dim=-1))            
                return {'sum':sum_hiddens,'mean':mean_hiddens,'concate':concate_hiddens,'hidden_states':hiddens}

            self.hook_postprocess = postprocess

    def get_downsample_rates(self, key: str) -> int:
        return 320
    def forward(self, wavs):
        #from hubert expert


        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)
        for i,model in enumerate(self.model):
            features, feat_padding_mask = model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )


# -


