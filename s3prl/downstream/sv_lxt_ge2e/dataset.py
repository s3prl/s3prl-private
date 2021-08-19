from collections import defaultdict
import os 
import random
from hashlib import sha256

import tqdm
import torch
from pathlib import Path
from torch.utils.data import Dataset
from joblib.parallel import Parallel, delayed
from torchaudio.sox_effects import apply_effects_file

HIDDEN_SAMPLE_RATE = 44100

EFFECTS = [
["channels", "1"],
["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]


class LxtSvTrain(Dataset):
    def __init__(self, vad_config, lxt_audio, lxt_train, max_timestep=None, n_jobs=12, preload=False, n_uttr=5, n_spkr=10, **kwargs):
        self.max_timestep = max_timestep
        self.vad_c = vad_config
        self.preload = preload
        self.n_uttr = n_uttr
        self.n_spkr = n_spkr

        with Path(lxt_train).open() as train_file:
            def extract_uttr_spkr(line):
                uttr, spkr = line.strip().split(maxsplit=1)
                return uttr.strip(), spkr.strip()

            train_utterances = [extract_uttr_spkr(line) for line in train_file.readlines()]  # (utterance_id, speaker)
            utterance_ids, spkrs = zip(*train_utterances)

        self.all_speakers = sorted(list(set(spkrs)))
        self.speaker_num = len(self.all_speakers)

        hashed = sha256(str.encode(" ".join(utterance_ids))).hexdigest()
        cache_path = Path(os.path.dirname(__file__)) / '.wav_lengths' / f'{hashed}_length.pt'
        cache_path.parent.mkdir(exist_ok=True)
        wav_paths = [Path(lxt_audio) / f"{uid}.wav" for uid in utterance_ids]

        if not cache_path.is_file():
            def trimmed_length(path):
                wav_sample, _ = apply_effects_file(str(path), EFFECTS)
                wav_sample = wav_sample.squeeze(0)
                length = wav_sample.shape[0]
                return length

            wav_lengths = Parallel(n_jobs=n_jobs)(delayed(trimmed_length)(path) for path in tqdm.tqdm(wav_paths, desc="Preprocessing"))
            wav_tags = utterance_ids
            torch.save([wav_tags, wav_lengths], str(cache_path))
        else:
            wav_tags, wav_lengths = torch.load(str(cache_path))
            assert wav_tags == utterance_ids

        self.dataset = defaultdict(list)
        for path, length, spkr in tqdm.tqdm(zip(wav_paths, wav_lengths, spkrs), total=len(wav_paths)):
            if length > self.vad_c['min_sec']:
                if self.preload:
                    wav = self.load_audio(path)
                    self.dataset[spkr].append((str(path), wav))
                else:
                    self.dataset[spkr].append(str(path))

    def load_audio(self, path):
        wav, _ = apply_effects_file(str(path), EFFECTS)
        wav = wav.squeeze(0)
        return wav

    def __len__(self):
        return 100000

    def __getitem__(self, index):
        spkrs = random.sample(self.all_speakers, k=self.n_spkr)
        all_wavs, all_paths = [], []
        for spkr in spkrs:
            uttrs = self.dataset[spkr]
            uttrs = random.sample(uttrs, k=self.n_uttr)
            if not self.preload:
                uttrs = [(str(uttr), self.load_audio(uttr)) for uttr in uttrs]

            wavs, paths = [], []
            for path, wav in uttrs:
                length = wav.shape[0]
                if self.max_timestep != None:
                    if length > self.max_timestep:
                        start = random.randint(0, int(length - self.max_timestep))
                        wav = wav[start : start + self.max_timestep]
                wavs.append(wav.numpy())
                paths.append(path)
            
            all_wavs += wavs
            all_paths += paths

        return all_wavs, all_paths

    def collate_fn(self, samples):
        return samples[0]


class LxtSvEval(Dataset):
    def __init__(self, split, vad_config, lxt_audio, **kwargs):
        self.root = Path(lxt_audio)
        self.meta_data = kwargs[split]

        self.pairs = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
            for pair in usage_list:
                match, uid1, uid2 = pair.split()
                one_pair = [int(match), uid1.strip(), uid2.strip()]
                self.pairs.append(one_pair)
        
        random.seed(0)
        random.shuffle(self.pairs)

        self.vad_c = vad_config 

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        y_label, uid1, uid2 = self.pairs[idx]

        wav1, _ = apply_effects_file(str(self.root / f"{uid1}.wav"), EFFECTS)
        wav2, _ = apply_effects_file(str(self.root / f"{uid2}.wav"), EFFECTS)

        wav1 = wav1.squeeze(0).numpy()
        wav2 = wav2.squeeze(0).numpy()

        return wav1, wav2, uid1, uid2, y_label
    
    def collate_fn(self, data_sample):
        wav1s, wav2s, uid1s, uid2s, y_labels = zip(*data_sample)
        all_wavs = wav1s + wav2s
        all_names = uid1s + uid2s
        return all_wavs, all_names, y_labels