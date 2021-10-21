import random
import torchaudio
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

SAMPLE_RATE = 16000

class LxtSid(Dataset):
    def __init__(self, split, lxt_audio, split_dir, min_secs=2, max_secs=4, seed=0, **kwargs) -> None:
        super().__init__()
        random.seed(seed)
        self.lxt_audio = Path(lxt_audio)
        split_path = Path(split_dir) / f"{split}.txt"
        with Path(split_path).open() as split_file:
            def process_line(line):
                uid, spkr = line.strip().split(maxsplit=1)
                path = self.lxt_audio / f"{uid}.wav"
                frames = torchaudio.info(str(path)).num_frames
                start = 0
                while (frames - start) / SAMPLE_RATE > min_secs:
                    interval = random.randint(min_secs * SAMPLE_RATE, min_secs * SAMPLE_RATE)
                    end = start + interval
                    yield uid.strip(), spkr.strip(), start, end
                    start = end

            self.pairs = []
            for line in tqdm(split_file.readlines()):
                pairs = list(iter(process_line(line)))
                self.pairs.extend(pairs)
        self.spkrs = sorted(list(set(list(zip(*self.pairs))[1])))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        uid, spkr, start, end = self.pairs[index]
        audio_path = self.lxt_audio / f"{uid}.wav"
        wav, sr = torchaudio.load(audio_path)
        assert sr == SAMPLE_RATE
        label = self.spkrs.index(spkr)
        return wav.view(-1)[start : end], label, uid

    @property
    def speaker_num(self):
        return len(self.spkrs)

    @staticmethod
    def collate_fn(items):
        return zip(*items)
