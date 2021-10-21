import random
import torchaudio
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torchaudio.sox_effects import apply_effects_file

SAMPLE_RATE = 16000

class LxtSid(Dataset):
    def __init__(self, split, lxt_audio, split_dir, min_secs=2, max_secs=4, vad=True, seed=0, **kwargs) -> None:
        super().__init__()
        random.seed(seed)

        self.lxt_audio = Path(lxt_audio)
        split_path = Path(split_dir) / f"{split}.txt"
        with Path(split_path).open() as split_file:
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

                frames = len(wav)
                start = 0
                while (frames - start) / SAMPLE_RATE > min_secs:
                    interval = random.randint(min_secs * SAMPLE_RATE, max_secs * SAMPLE_RATE)
                    end = start + interval
                    yield wav[start : end], spkr.strip(), f"{uid.strip()}_{start}_{end}"
                    start = end

            self.pairs = []
            for line in tqdm(split_file.readlines()):
                pairs = list(iter(process_line(line)))
                self.pairs.extend(pairs)

        self.spkrs = sorted(list(set(list(zip(*self.pairs))[1])))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        wav, spkr, uids = self.pairs[index]
        label = self.spkrs.index(spkr)
        return wav, label, uids

    @property
    def speaker_num(self):
        return len(self.spkrs)

    @staticmethod
    def collate_fn(items):
        return zip(*items)
