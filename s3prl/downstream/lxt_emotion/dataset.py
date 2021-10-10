# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import pandas
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

ALL_LABELS = ["neutral", "joy", "anger", "sadness"]


class LxtEmotionDataset(Dataset):
    def __init__(self, split, root, **kwargs):
        self.data_dir = Path(root)
        assert self.data_dir.is_dir()

        split_file = kwargs[split]
        self.pairs = []
        with open(split_file, "r") as file:
            for line in file.readlines():
                uid, emotion = line.split(maxsplit=1)
                uid, emotion = uid.strip(), emotion.strip()
                if emotion not in ALL_LABELS:
                    continue
                self.pairs.append((uid, ALL_LABELS.index(emotion)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        uid, emotion = self.pairs[index]
        wav, sr = torchaudio.load(str(self.data_dir / f"{uid}.wav"))
        return wav.view(-1), emotion

    @staticmethod
    def collate_fn(samples):
        return zip(*samples)
    
    @staticmethod
    def get_class_num():
        return len(ALL_LABELS)
