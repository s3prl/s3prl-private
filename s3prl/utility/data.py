import os
import math
from typing import Optional, Iterator, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Dataset
from torch.distributed import is_initialized
from torch.utils.data import Dataset, DistributedSampler

T_co = TypeVar("T_co", covariant=True)


def get_extracted_dataset(dataset_cls, extract_to_single_file=False):
        
    class ExtractedDataset(dataset_cls):
        def __init__(self, *args, **kwargs):
            self._split = kwargs.pop("split_name")
            self._extracted_path = kwargs.pop("extracted_path")
            self._use_single_file = extract_to_single_file

            if self._use_single_file:
                self._all_data = torch.load(os.path.join(self._extracted_path, "extracted_feats/", self._split, "all_data.ckpt"), map_location="cpu")
            super().__init__(*args, **kwargs)
        def __getitem__(self, index):
            if self._use_single_file:
                # copy to avoid memory leakage
                feature = self._all_data[index][0]
                if isinstance(feature, (list, tuple)):
                    feature = tuple(f.clone() for f in feature)
                elif isinstance(feature, torch.Tensor):
                    feature = feature.clone()
                return feature, *self._all_data[index][1:]
            else:
                return torch.load(os.path.join(self._extracted_path, "extracted_feats/", self._split, f"{index}.ckpt"), map_location="cpu")
    
    return ExtractedDataset


def get_ddp_sampler(dataset: Dataset, epoch: int):
    """
    This function will create a DistributedSampler if DDP is initialized,
    and will just return None if DDP is not initialized.
    """
    if is_initialized():
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(epoch)
    else:
        sampler = None
    return sampler


class DistributedMaxFramesBatchSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 2147483647,
        drop_last: bool = False,
        max_frames: int = 16000 * 160,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank() if dist.is_initialized() else 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # for constraining max frames in a batch
        assert isinstance(dataset.get_frames(0), int)
        self.max_frames = max_frames
        self.set_epoch(0)

    def __iter__(self) -> Iterator[T_co]:
        indices = list(range(len(self.batches)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter([self.batches[index] for index in indices])

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            # or else different nodes might not produce the same shuffled indices
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        self.batches = []
        batch = []
        batch_lengths = []
        for index in indices:
            new_length = self.dataset.get_frames(index)
            assert (
                new_length <= self.max_frames
            ), f"A single instacne has length greater than max_frames: {new_length} > {self.max_frames}. Please increase max_frames for DistributedMaxFramesBatchSampler."

            if max([new_length, *batch_lengths]) * (len(batch) + 1) > self.max_frames:
                self.batches.append(batch)
                batch = []
                batch_lengths = []
            batch.append(index)
            batch_lengths.append(new_length)

        if len(batch) > 0:
            self.batches.append(batch)

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.batches) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.batches) - self.num_replicas)
                / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.batches) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
