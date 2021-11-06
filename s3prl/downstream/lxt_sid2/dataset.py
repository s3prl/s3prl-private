import random
import torchaudio
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torchaudio.sox_effects import apply_effects_file
from collections import defaultdict

SAMPLE_RATE = 16000

class LxtSid(Dataset):
    def __init__(self, split, lxt_audio, utts, min_secs=2, max_secs=4, vad=True, seed=0, n_seg=5, **kwargs) -> None:
        super().__init__()
        random.seed(seed)

        self.lxt_audio = Path(lxt_audio)
        with open(utts) as file:
            def process_line(line):
                uid, spkr = line.strip().split(maxsplit=1)
                path = self.lxt_audio / f"{uid.strip()}.wav"

                effects = [
                    ["channels", "1"],
                    ["gain", "-3.0"],
                ]
                if vad:
                    effects.append(["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"])

                wav, sr = apply_effects_file(str(path), effects)
                wav = wav.squeeze(0)
                assert sr == SAMPLE_RATE

                if min_secs < 0 or max_secs < 0:
                    yield wav, spkr.strip(), uid.strip()
                else:
                    frames = len(wav)
                    start = 0
                    while (frames - start) / SAMPLE_RATE > min_secs:
                        interval = random.randint(min_secs * SAMPLE_RATE, max_secs * SAMPLE_RATE)
                        end = start + interval
                        yield wav[start : end], spkr.strip(), f"{uid.strip()}_{start}_{end}"
                        start = end

            spk2utt = defaultdict(list)
            wavs = {}
            for line in tqdm(file.readlines()):
                for wav, spk, uid in list(process_line(line)):
                    spk2utt[spk].append(uid)
                    wavs[uid] = wav

            self.pairs = []
            for spk in spk2utt.keys():
                utts = spk2utt[spk]
                random.shuffle(utts)
                eval_num = len(utts) - n_seg
                if split == "train":
                    chosen_utts = utts[:n_seg]
                elif split == "dev":
                    chosen_utts = utts[n_seg:n_seg + eval_num // 2]
                elif split == "test":
                    chosen_utts = utts[n_seg + eval_num // 2:]
                else:
                    raise ValueError
                chosen_wavs = [wavs[utt] for utt in chosen_utts]
                self.pairs.extend(list(zip(chosen_wavs, [spk] * len(chosen_wavs), chosen_utts)))

        self.spkrs = sorted(list(spk2utt.keys()))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        wav, spkr, uids = self.pairs[index]
        label = self.spkrs.index(spkr)
        return wav.numpy(), label, uids

    @property
    def speaker_num(self):
        return len(self.spkrs)

    @staticmethod
    def collate_fn(items):
        return zip(*items)
