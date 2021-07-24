from tqdm import tqdm
from pathlib import Path
from librosa.util import find_files

import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16000


class LxtAsrDataset(Dataset):
    def __init__(self, split, dictionary, lxt_audio, lxt_text, **kwargs):
        super().__init__()
        self.dictionary = dictionary
        self.audio_root = Path(lxt_audio)
        self.text_root = Path(lxt_text)
        
        with Path(kwargs[split]).open() as split_file:
            whitelist = [line.strip() for line in split_file.readlines()]

        pairs = []
        with self.text_root.open() as file:
            for line in tqdm(file.readlines()):
                utterance_id, transcript = line.strip().split(",", maxsplit=1)
                if utterance_id not in whitelist:
                    continue

                audio_path = self.audio_root / f"{utterance_id}.wav"
                if not audio_path.is_file():
                    print(f"[lxt_asr] - {audio_path} not found.")
                    continue

                audio_info = torchaudio.info(audio_path)
                audio_seconds = audio_info.num_frames / audio_info.sample_rate
                pairs.append((audio_path, self.encode_transcript(transcript), audio_seconds))
        self.pairs = pairs

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


class LibriAsrDataset(Dataset):
    def __init__(self, split, dictionary, libri_root, **kwargs):
        super().__init__()
        self.dictionary = dictionary
        audio_roots = [Path(libri_root).joinpath(relative_path) for relative_path in kwargs[split]]

        def load_transcripts(audio_path):
            audio_path = Path(audio_path)
            text_filename = f"{'-'.join(audio_path.stem.split('-')[:2])}.trans.txt"
            with audio_path.with_name(text_filename).open() as file:
                lines = [line.strip().split(maxsplit=1) for line in file.readlines()]
                mapping = {uid: transcript for uid, transcript in lines}
            return mapping[audio_path.stem]

        pairs = []
        for audio_root in audio_roots:
            audios = find_files(audio_root)
            for audio in tqdm(audios):
                audio_path = Path(audio)
                audio_info = torchaudio.info(audio_path)
                audio_seconds = audio_info.num_frames / audio_info.sample_rate
                transcript = load_transcripts(audio_path)
                pairs.append((audio_path, self.encode_transcript(transcript), audio_seconds))

        self.pairs = pairs

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