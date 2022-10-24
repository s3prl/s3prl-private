# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset
from torchaudio.sox_effects import apply_effects_file

ALL_LABELS = ["neutral", "joy", "anger", "sadness"]
EFFECTS = [
["channels", "1"],
["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

class LxtEmotionDataset(Dataset):
    def __init__(self, split, root, **kwargs):
        self.data_dir = Path(root)
        assert self.data_dir.is_dir()

        split_file = kwargs[split]
        self.utt_ids = []
        self.labels = []
        class_count = defaultdict(lambda: 0)
        with open(split_file, "r") as file:
            for line in file.readlines():
                uid, emotion = line.split(maxsplit=1)
                uid, emotion = uid.strip(), emotion.strip()
                if emotion not in ALL_LABELS:
                    continue
                self.utt_ids.append(uid)

                emotion_id = ALL_LABELS.index(emotion)
                self.labels.append(emotion_id)
                class_count[emotion_id] += 1

        for label in ALL_LABELS:
            print(f"{label} has {self.labels.count(ALL_LABELS.index(label))} samples")

        self.weights = []
        for index in range(len(self.labels)):
            weight = len(self.labels) / class_count[self.labels[index]]
            self.weights.append(weight)

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        uid = self.utt_ids[index]
        emotion = self.labels[index]
        wav, sr = apply_effects_file(str(self.data_dir / f"{uid}.wav"), EFFECTS)
        return wav.view(-1), emotion

    @staticmethod
    def collate_fn(samples):
        return zip(*samples)
    
    @staticmethod
    def get_class_num():
        return len(ALL_LABELS)
