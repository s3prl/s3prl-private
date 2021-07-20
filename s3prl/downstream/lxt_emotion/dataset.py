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

SAMPLE_RATE = 16000
LXT_SAMPLE_RATE = 44100
ALL_LABELS = ["neutral", "joy", "anger", "sadness"]


class LxtEmotionDataset(Dataset):
    def __init__(self, root, csv, **kwargs):
        data_dir = Path(root)
        assert data_dir.is_dir()

        self.resampler = torchaudio.transforms.Resample(LXT_SAMPLE_RATE, SAMPLE_RATE)
        self.audio_paths = []
        self.labels = []

        table = pandas.read_csv(csv)
        for _, row in table.iterrows():
            emotion = row["emotion"]
            if emotion not in ALL_LABELS:
                continue
            
            audio_path = data_dir / (row["utterance_id"] + ".wav")
            if not audio_path.is_file():
                print(f"[lxt_emotion] - {audio_path} not exists.")
                continue
            
            self.audio_paths.append(str(audio_path))
            self.labels.append(ALL_LABELS.index(emotion))

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        wav, sr = torchaudio.load(self.audio_paths[index])
        assert sr == LXT_SAMPLE_RATE
        wav = self.resampler(wav).mean(dim=0)
        return wav, self.labels[index]

    @staticmethod
    def collate_fn(samples):
        return zip(*samples)
    
    @staticmethod
    def get_class_num():
        return len(ALL_LABELS)
