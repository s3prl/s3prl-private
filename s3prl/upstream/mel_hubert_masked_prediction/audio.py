"""
The feature extractor for downstream evaluation. 
Author: Tzu-Quan Lin (https://github.com/nervjack2)
Reference: (https://github.com/s3prl/s3prl/blob/master/s3prl/upstream/apc/audio.py)
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np 
from torchaudio.compliance import kaldi


############
# CONSTANT #
############
WINDOW_TYPE = 'hamming'
SAMPLE_RATE = 16000


class FeatureExtractor(nn.Module):
    ''' Feature extractor, transforming file path to Mel spectrogram '''
    def __init__(self, mean_std_npy_path, mode="fbank", num_mel_bins=80, decode_wav=False, **kwargs):
        super(FeatureExtractor, self).__init__()
        # ToDo: Other surface representation
        assert mode=="fbank", "Only Mel-spectrogram implemented"
        self.mode = mode
        self.extract_fn = kaldi.fbank

        self.num_mel_bins = num_mel_bins
        self.kwargs = kwargs
        self.decode_wav = decode_wav
        if self.decode_wav:
            # HACK: sox cannot deal with wav with incorrect file length
            torchaudio.set_audio_backend('soundfile')
        try:
            mean_std = np.load(mean_std_npy_path)
        except:
            raise FileNotFoundError(f'Can not find the numpy array file which stores mean and std using in pretraining phase: {mean_std_npy_path}')
        self.mean = torch.Tensor(mean_std[0].reshape(-1))
        self.std = torch.Tensor(mean_std[1].reshape(-1))

    def _load_file(self, filepath):
        if self.decode_wav:
            waveform, sample_rate = torchaudio.load_wav(filepath)
        else:
            waveform, sample_rate = torchaudio.load(filepath)
        return waveform, sample_rate

    def forward(self, waveform):
        y = self.extract_fn(waveform,
                            num_mel_bins=self.num_mel_bins,
                            sample_frequency=SAMPLE_RATE,
                            window_type = WINDOW_TYPE,
                            **self.kwargs)
        # Normalize 
        mean = self.mean.to(y.device, dtype=torch.float32)
        std = self.std.to(y.device, dtype=torch.float32)
        y = (y-mean)/std
        return y

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)

    def create_msg(self):
        ''' List msg for verbose function '''
        msg = 'Audio spec.| Audio feat. = {}\t\t| feat. dim = {}\t| CMVN = {}'\
                        .format(self.mode, self.num_mel_bins, self.apply_cmvn)
        return [msg]


def create_transform(audio_config):
    mean_std_npy_path = audio_config.pop("mean_std_npy_path")
    feat_type = audio_config.pop("feat_type")
    feat_dim = audio_config.pop("feat_dim")
    decode_wav = audio_config.pop("decode_wav",False)
    cmvn = audio_config.pop("cmvn", False)
    transforms = FeatureExtractor(mean_std_npy_path, feat_type, feat_dim,
                            decode_wav, **audio_config)
    return transforms, feat_dim