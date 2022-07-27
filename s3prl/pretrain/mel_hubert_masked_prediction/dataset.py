"""
Pretraining feature dataset of MelHuBERT.
Author: Tzu-Quan Lin (https://github.com/nervjack2)
"""
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from pretrain.bucket_dataset import FeatLabelDataset


class MelFeatDataset(FeatLabelDataset):
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, **kwargs):
        super(MelFeatDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets, 
                                                   max_timestep, libri_root, **kwargs)

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        else:
            raise Exception('Do not support on-the-fly clustering id extraction!')

    def _load_label(self, label_path):
        if self.libri_root is None:
            return torch.LongTensor(np.load(os.path.join(self.root, label_path)))
        else:
            raise Exception('Do not support on-the-fly clustering id extraction!')
            
    def __getitem__(self, index):
        # Load acoustic feature, label and pad
        x_batch, y_batch = [], []
        for x_file, y_file in zip(self.X[index], self.Y[index]):
            feat = self._load_feat(x_file)
            label = self._load_label(y_file)
            x, y = self._sample(feat, label)
            x_batch.append(x)
            y_batch.append(y)

        x_len = [len(x_b) for x_b in x_batch]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        y_pad_batch = pad_sequence(y_batch, batch_first=True, padding_value=-100) # Pad -100 for ignore index

        pad_mask = torch.ones(x_pad_batch.shape[:-1])  # (batch_size, seq_len)
        # zero vectors for padding dimension
        for idx in range(x_pad_batch.shape[0]):
            pad_mask[idx, x_len[idx]:] = 0

        return x_pad_batch, y_pad_batch, pad_mask, x_len