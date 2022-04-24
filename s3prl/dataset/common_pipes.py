from dataclasses import dataclass

import torch
import torchaudio
from ..encoder.category import CategoryEncoder
from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class SetOutputKeys(DataPipe):
    output_keys: dict = None

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        dataset.set_output_keys(self.output_keys)
        return dataset


@dataclass
class LoadPseudoAudio(DataPipe):
    def load_wav(self, wav_path):
        wav = torch.randn(16000 * 10)
        return wav, len(wav)

    def load_metadatas(self, wav_path):
        return dict(
            sample_rate=16000,
            num_frames=500,
            num_channels=1,
        )

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        dataset.add_dynamic_item(
            self.load_wav, takes="wav_path", provides=["wav", "wav_len"]
        )
        dataset.add_dynamic_item(
            self.load_metadata, takes="wav_path", provides="wav_metadata"
        )
        return dataset


@dataclass
class LoadAudio(DataPipe):
    audio_sample_rate: int = 16000
    audio_channel_reduction: str = "first"
    wav_path_name: str = "wav_path"
    wav_name: str = "wav"

    def load_audio(self, wav_path, metadata: bool = False):
        if not metadata:
            torchaudio.set_audio_backend("sox_io")
            wav, sr = torchaudio.load(wav_path)
            if sr != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
                wav = resampler(wav)

            if self.audio_channel_reduction == "first":
                wav = wav[0]
            elif self.audio_channel_reduction == "mean":
                wav = wav.mean(dim=0)

            wav = wav.view(-1, 1)
            return wav
        else:
            torchaudio.set_audio_backend("sox_io")
            info = torchaudio.info(wav_path)
            ratio = self.audio_sample_rate / info.sample_rate
            return dict(
                sample_rate=self.audio_sample_rate,
                num_frames=round(info.num_frames * ratio),
                num_channels=1,
            )

    def compute_length(self, wav):
        return len(wav)

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        dataset.add_dynamic_item_and_metadata(
            self.load_audio, takes=self.wav_path_name, provide=self.wav_name
        )
        dataset.add_dynamic_item(
            self.compute_length,
            takes=self.wav_name,
            provides=f"{self.wav_name}_len",
        )
        return dataset


@dataclass
class EncodeCategory(DataPipe):
    train_category_encoder: bool = False
    label_name: str = "label"
    category_encoder_name: str = "category"
    encoded_target_name: str = "class_id"

    def prepare_category(self, labels):
        return CategoryEncoder(sorted(list(set(labels))))

    def encode_label(self, category, label):
        return category.encode(label)

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        if self.train_category_encoder:
            with dataset.output_keys_as([self.label_name]):
                labels = [item[self.label_name] for item in dataset]
            category = self.prepare_category(labels)
            dataset.add_tool(self.category_encoder_name, category)
            dataset.add_tool("output_size", len(category))

        dataset.add_dynamic_item(
            self.encode_label,
            takes=[self.category_encoder_name, self.label_name],
            provides=self.encoded_target_name,
        )
        return dataset


@dataclass
class EncodeMultipleCategory(EncodeCategory):
    train_category_encoder: bool = False
    label_name: str = "labels"
    category_encoder_name: str = "categories"
    encoded_target_name: str = "class_ids"

    def encode_label(self, categories, labels):
        return torch.LongTensor(
            [category.encode(label) for category, label in zip(categories, labels)]
        )

    def __call__(self, dataset: AugmentedDynamicItemDataset):
        if self.train_category_encoder:
            with dataset.output_keys_as([self.label_name]):
                labels = [item[self.label_name] for item in dataset]
            label_types = list(zip(*labels))
            categories = [
                self.prepare_category(label_type) for label_type in label_types
            ]
            dataset.add_tool(self.category_encoder_name, categories)
            dataset.add_tool("output_size", sum([len(c) for c in categories]))

        dataset.add_dynamic_item(
            self.encode_label,
            takes=[self.category_encoder_name, self.label_name],
            provides=self.encoded_target_name,
        )
        return dataset