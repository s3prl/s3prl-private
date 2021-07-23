# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the phone dataset ]
#   Author       [ S3PRL, Xuankai Chang ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import logging
import os
import random
import numpy as np
#-------------#
import pandas as pd
from tqdm import tqdm
from pathlib import Path
#-------------#
import torch
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000


class LxtAsrDataset(Dataset):
    def __init__(self, split, dictionary, lxt_audio, lxt_text, split_ratio=None, **kwargs):
        super().__init__()
        self.dictionary = dictionary
        self.audio_root = Path(lxt_audio)
        self.text_root = Path(lxt_text)

        pairs = []
        with self.text_root.open() as file:
            for line in tqdm(file.readlines()):
                utterance_id, transcript = line.strip().split(",", maxsplit=1)
                audio_path = self.audio_root / f"{utterance_id}.wav"
                if not audio_path.is_file():
                    print(f"[lxt_asr] - {audio_path} not found.")
                    continue
                audio_info = torchaudio.info(audio_path)
                audio_seconds = audio_info.num_frames / audio_info.sample_rate
                pairs.append((audio_path, self.encode_transcript(transcript), audio_seconds))

        rand_indices = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(2147483647))
        pairs = [pairs[index] for index in rand_indices]

        train_end = round(split_ratio[0] * len(pairs))
        dev_end = round(split_ratio[1] * len(pairs)) + train_end
        train_pairs, dev_pairs, test_pairs = pairs[:train_end], pairs[train_end:dev_end], pairs[dev_end:]
        self.pairs = eval(f"{split}_pairs")

    def get_frames(self, index):
        return round(self.pairs[index][2] * SAMPLE_RATE)

    def load_audio(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        assert sr == SAMPLE_RATE
        wav = wav.view(-1).numpy()
        return wav

    def encode_transcript(self, transcript):
        transcript = transcript.upper()
        transcript = " ".join(list("|".join(transcript.split()))) + " |"
        label = self.dictionary.encode_line(
            transcript,
            line_tokenizer=lambda x: x.split(),
            add_if_not_exist=False,
            append_eos=False,
        )
        assert (label == self.dictionary.unk_index).sum() == 0
        return label

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        audio_path, label, _ = self.pairs[index]
        wav = self.load_audio(audio_path)
        return wav, label, audio_path.stem

    def collate_fn(self, items):
        return zip(*items)
