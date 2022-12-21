# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/whisper/expert.py ]
#   Synopsis     [ the whisper wrapper ]
#   Author       [ OpenAI ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############

import torch
import whisper
import importlib

from ..interfaces import UpstreamBase

############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)

        spam_spec = importlib.util.find_spec("whisper")
        assert spam_spec, "Please install the whisper package first: pip install git+https://github.com/openai/whisper.git"

        self.model = whisper.load_model(name, device="cpu")
        del self.model.decoder

        if len(self.hooks) == 0:
            for module_id in range(len(self.model.encoder.blocks)):
                self.add_hook(
                    f"self.model.encoder.blocks[{module_id}]",
                    lambda input, output: input[0],
                )
            self.add_hook("self.model.encoder", lambda input, output: output)

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        # chunk mel into 30 sec block
        mel_blocks = []
        mel_block_idx = [0]
        feat_lengths = []
        for wav in wavs:
            mel_block_idx.append(mel_block_idx[-1])
            # mel: 80 x seq_len
            mel = whisper.log_mel_spectrogram(wav)
            feat_lengths.append(mel.size(1)>>1)
            for m in torch.split(mel, 3000, -1):
                m = whisper.pad_or_trim(m, 3000)
                mel_blocks.append(m)
                mel_block_idx[-1] += 1
        inputs = torch.stack(mel_blocks)
        # forward
        self.model.encoder(inputs)
        # reshape hidden states
        for i in range(len(self._hook_hiddens)):
            name, tensor = self._hook_hiddens[i]
            # cat blocks from same wav and truncate the padding
            tensor = [
                torch.cat(tuple(tensor[s:e]))[:l]
                for l, s, e in 
                zip(feat_lengths, mel_block_idx[:-1], mel_block_idx[1:])
            ]
            # padding the tensors to the maximum of length
            tensor = torch.nn.utils.rnn.pad_sequence(tensor, batch_first=True)
            self._hook_hiddens[i] = (name, tensor)