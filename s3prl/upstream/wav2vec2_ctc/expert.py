from collections import OrderedDict
from typing import List, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

HIDDEN_DIM = 8


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(ckpt)
        self.model = Wav2Vec2ForCTC.from_pretrained(ckpt)

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        device = wavs[0].device

        wavs = [wav.cpu().numpy() for wav in wavs]
        processor_outputs = self.processor(wavs, return_tensors="pt", padding="longest", sampling_rate=16000)
        attention_mask = processor_outputs.get("attention_mask", None)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device)

        model_outputs = self.model(
            processor_outputs.input_values.to(device),
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # predicted_ids = torch.argmax(model_outputs.logits, dim=-1)
        # transcription = self.processor.batch_decode(predicted_ids)
        # print(transcription)

        return {
            "hidden_states": model_outputs.hidden_states,
            "logits": model_outputs.logits,
            "probs": model_outputs.logits.softmax(dim=-1),
        }
