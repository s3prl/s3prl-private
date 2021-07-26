import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

SAMPLE_RATE = 16000

class LxtSid(Dataset):
    def __init__(self, split, lxt_audio, **kwargs) -> None:
        super().__init__()
        self.lxt_audio = Path(lxt_audio)
        split_path = kwargs[split]
        with Path(split_path).open() as split_file:
            def process_line(line):
                uid, spkr = line.strip().split(maxsplit=1)
                return uid.strip(), spkr.strip()
            
            self.pairs = [process_line(line) for line in split_file.readlines()]
        self.spkrs = sorted(list(set(list(zip(*self.pairs))[1])))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        uid, spkr = self.pairs[index]
        audio_path = self.lxt_audio / f"{uid}.wav"
        wav, sr = torchaudio.load(audio_path)
        assert sr == SAMPLE_RATE
        label = self.spkrs.index(spkr)
        return wav.view(-1), label, uid

    def get_frames(self, index):
        uid, _ = self.pairs[index]
        audio_path = self.lxt_audio / f"{uid}.wav"
        info = torchaudio.info(str(audio_path))
        return info.num_frames

    @property
    def speaker_num(self):
        return len(self.spkrs)

    @staticmethod
    def collate_fn(items):
        return zip(*items)