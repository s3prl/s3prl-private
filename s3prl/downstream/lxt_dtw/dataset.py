import re
from pathlib import Path
from torch.utils.data import dataset

from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file


class QUESST14Dataset(Dataset):
    def __init__(self, split, query_audio, query_lst, doc_audio, doc_lst, **kwargs):
        doc_paths = english_audio_paths(doc_audio, doc_lst)
        query_paths = english_audio_paths(query_audio, query_lst)

        self.n_queries = len(query_paths)
        self.n_docs = len(doc_paths)
        self.data = query_paths + doc_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]
        wav, _ = apply_effects_file(
            str(audio_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["gain", "-3.0"],
                ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
            ],
        )
        wav = wav.squeeze(0)
        return wav.numpy(), audio_path.with_suffix("").name

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        wavs, audio_names = zip(*samples)
        return wavs, audio_names


def english_audio_paths(dataset_root_path, lst):
    """Extract English audio paths."""
    dataset_root_path = Path(dataset_root_path)
    audio_paths = [dataset_root_path / (line.strip() + ".wav") for line in Path(lst).open().readlines()]
    return audio_paths
