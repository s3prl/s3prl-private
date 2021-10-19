import torchaudio
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

SAMPLE_RATE = 16000

class LxtSid(Dataset):
    def __init__(self, split, lxt_audio, split_dir, **kwargs) -> None:
        super().__init__()
        self.lxt_audio = Path(lxt_audio)
        split_path = Path(split_dir) / f"{split}.txt"
        with Path(split_path).open() as split_file:
            def process_line(line):
                uid, spkr = line.strip().split(maxsplit=1)
                path = self.lxt_audio / f"{uid}.wav"
                info = torchaudio.info(str(path))
                return uid.strip(), spkr.strip(), info.num_frames

            self.pairs = [process_line(line) for line in tqdm(split_file.readlines())]
        self.spkrs = sorted(list(set(list(zip(*self.pairs))[1])))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        uid, spkr, _ = self.pairs[index]
        audio_path = self.lxt_audio / f"{uid}.wav"
        wav, sr = torchaudio.load(audio_path)
        assert sr == SAMPLE_RATE
        label = self.spkrs.index(spkr)
        return wav.view(-1), label, uid

    def get_frames(self, index):
        return self.pairs[index][2]

    @property
    def speaker_num(self):
        return len(self.spkrs)

    @staticmethod
    def collate_fn(items):
        return zip(*items)
