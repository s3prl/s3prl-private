# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ LibriMix speech enhancement and separation dataset ]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import yaml
import pickle
import numpy as np
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

import s3prl
import librosa

s3prl_path = Path(s3prl.__path__[0])

class SeparationDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split=None,
        rate=16000,
        src=['mix_clean'],
        tgt=['s1', 's2'],
        n_fft=512,
        hop_length=320,
        win_length=512,
        window='hann', 
        center=True,
        use_cache=True,
        task_name="",
        sr=16000,
        num_spks=1,
        addition_cond=[],
        **kwargs,
    ):
        super(SeparationDataset, self).__init__()
        """
        Args:
            data_dir (str):
                prepared data directory

            split (str):
                dataset split

            rate (int):
                audio sample rate

            src and tgt (list(str)):
                the input and desired output.
                LibriMix offeres different options for the users. For
                clean source separation, src=['mix_clean'] and tgt=['s1', 's2'].
                Please see https://github.com/JorisCos/LibriMix for details

            n_fft (int):
                length of the windowed signal after padding with zeros.

            hop_length (int):
                number of audio samples between adjacent STFT columns.

            win_length (int):
                length of window for each frame

            window (str):
                type of window function, only support Hann window now

            center (bool):
                whether to pad input on both sides so that the
                t-th frame is centered at time t * hop_length

            The STFT related parameters are the same as librosa.
        """
        self.data_dir = data_dir
        self.split = split
        self.rate = rate
        self.sr = sr
        self.src = src
        self.tgt = tgt
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.n_srcs = len(self.tgt)
        self.use_cache = use_cache

        assert len(self.src) == 1 and len(self.tgt) == num_spks

        # mix_clean (utterances only) mix_both (utterances + noise) mix_single (1 utterance + noise)
        cond_list = ["s1", "s2", "noise", "mix_clean", "mix_both", "mix_single"]
        cond_list.extend(addition_cond)
        
        # create the mapping from utterances to the audio paths
        # reco2path[utt][cond] is the path for utterance utt with condition cond
        reco2path = {}
        for cond in src + tgt:
            assert cond in cond_list
            assert os.path.exists("{}/{}/wav.scp".format(self.data_dir, cond)), f"{self.data_dir}/{cond}/wav.scp"
            with open("{}/{}/wav.scp".format(self.data_dir, cond), 'r') as fh:
                content = fh.readlines()
            for line in content:
                line = line.strip('\n')
                uttname, path = line.split()
                if uttname not in reco2path:
                    reco2path[uttname] = {}
                reco2path[uttname][cond] = path
        self.reco2path = reco2path

        self.recolist = sorted(self.reco2path.keys())
        
        cache_path = s3prl_path / f"data/{task_name}" / str(hop_length)
        self.cache_path = cache_path
        if os.path.exists(cache_path / "0.pkl"):
            with open(cache_path / "datarc.yaml", 'r') as f:
                datarc = yaml.load(f, Loader=yaml.FullLoader)
                assert (
                    datarc['rate'] == rate and
                    datarc['src'] == src and
                    datarc['tgt'] == tgt and
                    datarc['n_fft'] == n_fft and
                    datarc['win_length'] == win_length and
                    datarc['window'] == window and
                    datarc['center'] == center
                ), "config differs with preprocessed data, please preprocess again."
            
            

    def __len__(self):
        return len(self.recolist)

    def __getitem__(self, i):
        if self.use_cache:
            with open(self.cache_path / self.split / f"{i}.pkl", 'rb') as f:
                return pickle.load(f)
        
        reco = self.recolist[i]
        src_path = self.reco2path[reco][self.src[0]]
        src_samp, rate = librosa.load(src_path, sr=self.sr)
        assert rate == self.rate
        src_feat = np.transpose(librosa.stft(src_samp, 
            n_fft=self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = self.window,
            center = self.center))

        tgt_samp_list, tgt_feat_list = [], []
        for j in range(self.n_srcs):
            tgt_path = self.reco2path[reco][self.tgt[j]]
            tgt_samp, rate = librosa.load(tgt_path, sr=self.sr)
            assert rate == self.rate
            tgt_feat = np.transpose(librosa.stft(tgt_samp, 
                n_fft=self.n_fft,
                hop_length = self.hop_length,
                win_length = self.win_length,
                window = self.window,
                center = self.center))
            tgt_samp_list.append(tgt_samp)
            tgt_feat_list.append(tgt_feat)
        """
        reco (str):
            name of the utterance

        src_samp (ndarray):
            audio samples for the source [T, ]

        src_feat (ndarray):
            the STFT feature map for the source with shape [T1, D]

        tgt_samp_list (list(ndarray)):
            list of audio samples for the targets

        tgt_feat_list (list(ndarray)):
            list of STFT feature map for the targets
        """
        return reco, src_samp, src_feat, tgt_samp_list, tgt_feat_list


    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        uttname_list = [sample[0] for sample in sorted_batch]

        if self.use_cache:
            mix_stft_list = [sample[2] for sample in sorted_batch]
            source_wav_list = [sample[1] for sample in sorted_batch]
            target_wav_list = [pad_sequence([sample[3][j] for sample in sorted_batch], batch_first=True) for j in range(self.n_srcs)]
        else:
            mix_stft_list = [torch.from_numpy(sample[2]) for sample in sorted_batch]
            source_wav_list = [torch.from_numpy(sample[1]) for sample in sorted_batch]
            target_wav_list = [pad_sequence([torch.from_numpy(sample[3][j]) for sample in sorted_batch], batch_first=True) for j in range(self.n_srcs)]
        source_stft = pad_sequence(mix_stft_list, batch_first=True)
        source_wav = pad_sequence(source_wav_list, batch_first=True)

        target_attr = {
            "magnitude": [],
            "phase": [],
        }
        for j in range(self.n_srcs):
            if self.use_cache:
                target_stft_list = [sample[4][j] for sample in sorted_batch]
            else:
                target_stft_list = [torch.from_numpy(sample[4][j]) for sample in sorted_batch]
            target_stft = pad_sequence(target_stft_list, batch_first=True)
            target_attr["magnitude"].append(target_stft.abs())
            target_attr["phase"].append(target_stft.angle())

        wav_length = torch.IntTensor([len(sample[1]) for sample in sorted_batch])
        feat_length = torch.IntTensor([stft.size(0) for stft in mix_stft_list])
        """
        source_wav_list (list(tensor)):
            list of audio samples for the source

        uttname_list (list(str)):
            list of utterance names

        source_stft (tensor):
            sources stft

        source_wav (tensor):
            padded version of source_wav_list, with size [bs, max_T]

        target_attr (dict):
            dictionary containing magnitude and phase information for the targets

        feat_length (tensor):
            length of the STFT feature for each utterance

        wav_length (tensor):
            number of samples in each utterance
        """
        return source_wav_list, uttname_list, source_stft, source_wav, target_attr, target_wav_list, feat_length, wav_length
