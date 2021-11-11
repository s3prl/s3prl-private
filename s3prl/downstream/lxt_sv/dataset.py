import os 
import random
import torchaudio

import tqdm
import torch
from pathlib import Path
from librosa.util import find_files
from torch.utils.data import Dataset
from joblib.parallel import Parallel, delayed
from torchaudio.sox_effects import apply_effects_file

HIDDEN_SAMPLE_RATE = 44100
SAMPLE_RATE = 16000
LONG_ENOUGH_SECS = 4
TRIM_START_SECS = 2
OPTIMIZE_STEPS = 5000
BATCH_SIZE = 32

EFFECTS = [
["channels", "1"],
["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

class LxtSvTrain(Dataset):
    def __init__(self, lxt_audio, lxt_train, min_secs=1, max_secs=2, seed=0, **kwargs):
        random.seed(seed)

        with Path(lxt_train).open() as train_file:
            def extract_uttr_spkr(line):
                uttr, spkr = line.strip().split(maxsplit=1)
                return uttr.strip(), spkr.strip()

            train_utterances = [extract_uttr_spkr(line) for line in train_file.readlines()]  # (utterance_id, speaker)
            utterance_ids, spkrs = zip(*train_utterances)

        self.dataset = []
        self.all_speakers = sorted(list(set(spkrs)))
        self.speaker_num = len(self.all_speakers)

        for uid, spkr in tqdm.tqdm(zip(utterance_ids, spkrs), desc="Loading wavs", total=len(spkrs)):
            path = Path(lxt_audio) / f"{uid}.wav"
            wav, _ = torchaudio.load(str(path))
            wav = wav.squeeze(0)

            if min_secs < 0 or max_secs < 0:
                self.dataset.append((wav, spkr, f"{uid}_0_-1"))
            else:
                start = 0
                if (len(wav) / SAMPLE_RATE) > LONG_ENOUGH_SECS:
                    # prevent microphone noises
                    start = TRIM_START_SECS * SAMPLE_RATE
                frames = len(wav)
                while (frames - start) / SAMPLE_RATE > min_secs:
                    interval = random.randint(min_secs * SAMPLE_RATE, max_secs * SAMPLE_RATE)
                    end = start + interval
                    self.dataset.append((wav[start : end], spkr, f"{uid}_{start}_{end}"))
                    start = end

        total_items = OPTIMIZE_STEPS * BATCH_SIZE
        self.dataset = self.dataset * (total_items // len(self.dataset))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        wav, spkr, uid = self.dataset[idx]
        label = self.all_speakers.index(spkr)
        return wav.numpy(), uid, label

    def collate_fn(self, samples):
        return zip(*samples)


class LxtSvEval(Dataset):
    def __init__(self, split, lxt_audio, seed=0, **kwargs):
        random.seed(seed)

        self.root = Path(lxt_audio)
        self.meta_data = kwargs[split]
        self.max_timestep = kwargs.get("max_eval_timestep")

        self.pairs = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
            for pair in tqdm.tqdm(usage_list, desc=f"Initializing {split}"):
                match, uid1, uid2 = pair.split()

                wav1, _ = torchaudio.load(str(self.root / f"{uid1}.wav"))
                wav2, _ = torchaudio.load(str(self.root / f"{uid2}.wav"))

                one_pair = [wav1.view(-1), wav2.view(-1), uid1.strip(), uid2.strip(), int(match)]
                self.pairs.append(one_pair)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
    
    def collate_fn(self, data_sample):
        wav1s, wav2s, uid1s, uid2s, y_labels = zip(*data_sample)
        all_wavs = wav1s + wav2s
        all_names = uid1s + uid2s
        return all_wavs, all_names, y_labels


class LxtSvSegEval(Dataset):
    def __init__(self, split, lxt_audio, seed=0, **kwargs):
        random.seed(seed)

        self.root = Path(lxt_audio)
        self.meta_data = kwargs[split]

        self.pairs = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
            for pair in tqdm.tqdm(usage_list, desc=f"Initializing {split}"):
                match, uid1, uid2 = pair.split()
                uid1 = uid1.strip()
                uid2 = uid2.strip()

                one_pair = [uid1, uid2, int(match)]
                self.pairs.append(one_pair)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        uid1, uid2, match = self.pairs[idx]

        def parse_segment(uid):
            fields = uid.split("_")
            start = fields[-2]
            end = fields[-1]
            uid = "_".join(fields[:-2])
            return uid, start, end

        def get_segment(uid):
            uid, start, end = parse_segment(uid)
            wav, _ = torchaudio.load(str(self.root / f"{uid}.wav"))
            wav = wav.squeeze(0).numpy()
            wav_segment = wav[int(start) : int(end)]
            return wav_segment

        seg1 = get_segment(uid1)
        seg2 = get_segment(uid2)

        return seg1, seg2, uid1, uid2, match

    def collate_fn(self, data_sample):
        wav1s, wav2s, uid1s, uid2s, y_labels = zip(*data_sample)
        all_wavs = wav1s + wav2s
        all_names = uid1s + uid2s
        return all_wavs, all_names, y_labels


class VoxCeleb1Train(Dataset):
    def __init__(self, vad_config, voxceleb1_path, voxceleb1_spkr=-1, max_timestep=None, n_jobs=12, **kwargs):
        self.vad_c = vad_config 
        self.max_timestep = max_timestep
        self.dataset = []
        self.all_speakers = []

        cache_path = Path(os.path.dirname(__file__)) / '.wav_lengths' / f'voxceleb1_length.pt'
        cache_path.parent.mkdir(exist_ok=True)
        root = Path(voxceleb1_path) / "dev" / "wav"

        if not cache_path.is_file():
            def trimmed_length(path):
                wav_sample, _ = apply_effects_file(path, EFFECTS)
                wav_sample = wav_sample.squeeze(0)
                length = wav_sample.shape[0]
                return length

            wav_paths = find_files(root)
            wav_lengths = Parallel(n_jobs=n_jobs)(delayed(trimmed_length)(path) for path in tqdm.tqdm(wav_paths, desc="Preprocessing"))
            wav_tags = [Path(path).parts[-3:] for path in wav_paths]
            torch.save([wav_tags, wav_lengths], str(cache_path))
        else:
            wav_tags, wav_lengths = torch.load(str(cache_path))
            wav_paths = [root.joinpath(*tag) for tag in wav_tags]

        speaker_dirs = ([f.stem for f in root.iterdir() if f.is_dir()])
        self.all_speakers.extend(speaker_dirs)
        for path, length in zip(wav_paths, wav_lengths):
            if length > self.vad_c['min_sec']:
                self.dataset.append(path)

        random.seed(0)
        self.all_speakers.sort()
        if voxceleb1_spkr >= 0:
            self.all_speakers = random.sample(self.all_speakers, k=voxceleb1_spkr)
        self.speaker_num = len(self.all_speakers)
        self.dataset = [path for path in self.dataset if self.path2spkr(path) in self.all_speakers]
        print(f"[VoxCeleb1] - {self.speaker_num} speakers with {len(self.dataset)} utterances")

    @staticmethod
    def path2spkr(path):
        return Path(path).parts[-3]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        path = self.dataset[idx]
        wav, _ = apply_effects_file(str(path), EFFECTS)
        wav = wav.squeeze(0)
        length = wav.shape[0]
        
        if self.max_timestep != None:
            if length > self.max_timestep:
                start = random.randint(0, int(length - self.max_timestep))
                wav = wav[start : start + self.max_timestep]

        tags = Path(path).parts[-3:]
        utterance_id = "-".join(tags).replace(".wav", "")
        label = self.all_speakers.index(tags[0])
        return wav.numpy(), utterance_id, label
        
    def collate_fn(self, samples):
        return zip(*samples)
