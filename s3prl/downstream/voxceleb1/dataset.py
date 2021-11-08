import torch
from torch._C import set_anomaly_enabled
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from librosa.util import find_files
from torchaudio import load
from torch import nn
import os 
import re
import random
import pickle
import torchaudio
import sys
import time
import glob
import tqdm
from pathlib import Path
from torchaudio.sox_effects import apply_effects_file
from collections import defaultdict

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')


# Voxceleb 1 Speaker Identification
class SpeakerClassifiDataset(Dataset):
    def __init__(self, mode, file_path, meta_data, max_timestep=None, train_utt=1):

        self.mode = mode
        self.root = file_path
        self.meta_data = meta_data
        self.max_timestep = max_timestep
        self.train_utt = train_utt
        self.usage_list = open(self.meta_data, "r").readlines()

        cache_path = os.path.join(CACHE_PATH, f'{mode}-{"all" if train_utt is None else train_utt}.pkl')
        if os.path.isfile(cache_path):
            print(f'[SpeakerClassifiDataset] - Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                dataset = pickle.load(cache)
        else:
            dataset = eval("self.{}".format(mode))()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump(dataset, cache)
        print(f'[SpeakerClassifiDataset] - there are {len(dataset)} files found')

        random.seed(0)
        all_spks = sorted(list(set([Path(path).parts[-3] for path in dataset])))
        self.chosen_spks = random.sample(all_spks, k=60)
        print(f"Chosen {len(self.chosen_spks)} speakers: {self.chosen_spks}")
        self.dataset = [path for path in dataset if Path(path).parts[-3] in self.chosen_spks]
        self.label = [self.chosen_spks.index(Path(path).parts[-3]) for path in self.dataset]

    @property
    def speaker_num(self):
        return len(self.chosen_spks)

    def label2speaker(self, labels):
        return [self.chosen_spks[label] for label in labels]

    def train(self):

        spk2paths = defaultdict(list)
        print("search specified wav name for training set")
        for string in tqdm.tqdm(self.usage_list):
            pair = string.split()
            index = pair[0]
            if int(index) == 1:
                x = list(self.root.glob("*/wav/" + pair[1]))
                spk = Path(pair[1]).parts[0]
                spk2paths[spk].append(str(x[0]))
        print("finish searching training set wav")

        random.seed(0)
        dataset = []
        for spk, paths in spk2paths.items():
            if self.train_utt is not None:
                paths = random.sample(paths, k=self.train_utt)
            dataset.extend(paths)
        return dataset
        
    def dev(self):

        dataset = []
        print("search specified wav name for dev set")
        for string in tqdm.tqdm(self.usage_list):
            pair = string.split()
            index = pair[0]
            if int(index) == 2:
                x = list(self.root.glob("*/wav/" + pair[1]))
                dataset.append(str(x[0])) 
        print("finish searching dev set wav")

        return dataset       

    def test(self):

        dataset = []
        print("search specified wav name for test set")
        for string in tqdm.tqdm(self.usage_list):
            pair = string.split()
            index = pair[0]
            if int(index) == 3:
                x = list(self.root.glob("*/wav/" + pair[1]))
                dataset.append(str(x[0])) 
        print("finish searching test set wav")

        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        effects = [
            ["channels", "1"],
            ["gain", "-3.0"],
            ["silence", "1", "0.1", "0.2%", "-1", "0.1", "0.2%"],
        ]
        wav, _ = apply_effects_file(self.dataset[idx], effects)
        wav = wav.squeeze(0)
        length = wav.shape[0]

        if self.max_timestep !=None:
            if length > self.max_timestep:
                start = random.randint(0, int(length-self.max_timestep))
                wav = wav[start:start+self.max_timestep]
                length = self.max_timestep

        def path2name(path):
            return Path("-".join((Path(path).parts)[-3:])).stem

        path = self.dataset[idx]
        return wav.numpy(), self.label[idx], path2name(path)
        
    def collate_fn(self, samples):
        return zip(*samples)
