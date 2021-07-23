import math
from typing import Optional, Iterator, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Dataset

T_co = TypeVar('T_co', covariant=True)

class DistributedMaxFramesBatchSampler(Sampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 2147483647, drop_last: bool = False,
                 max_frames: int = 16000 * 160) -> None:
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
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.max_frames = max_frames
        self.set_epoch(0)

        # for constraining max frames in a batch
        assert isinstance(dataset.get_frames(0), int)
        self.max_frames = max_frames

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.batches) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.batches) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.batches) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        indices = list(range(len(self.batches)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
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
            assert new_length <= self.max_frames, new_length

            if max([new_length, *batch_lengths]) * (len(batch) + 1) > self.max_frames:
                self.batches.append(batch)
                batch = []
                batch_lengths = []
            batch.append(index)
            batch_lengths.append(new_length)

        if len(batch) > 0:
            self.batches.append(batch)
