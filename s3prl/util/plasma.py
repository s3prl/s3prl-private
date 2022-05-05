import pathlib
import logging
import subprocess
from typing import Any
import pyarrow.plasma as plasma

import torch

from s3prl import Container, Output
from s3prl.dataset.base import default_collate_fn

logger = logging.getLogger(__name__)

GB = 1024 ** 3
_plasma_path = str(pathlib.Path.home() / ".plasma")


def set_default_plasma_path(plasma_path: str):
    global _plasma_path
    _plasma_path = plasma_path


def get_plasma_client(plasma_path: str = _plasma_path):
    try:
        client = plasma.connect(plasma_path)
    except OSError:
        logger.error(
            f"Cannot connect to the plasma store at {plasma_path}. "
            "Please start the plasma store with the plasma_store() "
            "context manager."
        )
        raise
    return client


class plasma_store:
    def __init__(self, path: str = _plasma_path, gigabyte: float = 1.0):
        self.path = path
        self.nbytes = round(gigabyte * GB)

    def __enter__(self):
        if not self.connectable(self.path):
            logger.info(f"Starting a plasma server at {self.path}")
            self.server = self.start(self.path, self.nbytes)
        else:
            logger.info(f"Found a connectable plasma server at {self.path}")
            self.server = None
            client = self.connect(self.path)
            outer_bytes = client.store_capacity()
            inner_bytes = self.nbytes
            assert outer_bytes >= inner_bytes, (
                f"Inner plasma store requests nbyte {inner_bytes} larger than outer plasma store {outer_bytes}, "
                f"under the same plasma store file path {self.path}. Please increase the size of the outer "
                "plasma store or set different plasma store file paths between inner/outer plasma store."
            )
        return self.connect(self.path)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.server is not None:
            self.server.kill()

    @classmethod
    def start(cls, path: str, nbytes: int = 1 * GB) -> subprocess.Popen:
        _server = subprocess.Popen(["plasma_store", "-m", str(nbytes), "-s", path])
        assert cls.connectable(path)
        return _server

    @classmethod
    def connectable(cls, path, num_retries: int = 20):
        try:
            plasma.connect(path, num_retries=num_retries)
        except OSError:
            return False
        else:
            return True

    @classmethod
    def connect(cls, path, num_retries: int = 20):
        client = plasma.connect(path, num_retries)
        return client


def plasma_collate_fn(samples, padding_value: int = 0, plasma_path: str = _plasma_path):
    """
    Don't serialize into the disk upon pickling for torch.Tensor.
    Instead, put torch.Tensor into the plasma store
    """
    output: Output = default_collate_fn(samples, padding_value)
    return PlasmaOutput(plasma_path=plasma_path, **output)


class PlasmaOutput(Container):
    def __init__(self, *args, plasma_path: str = _plasma_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_setattr("_plasma_path", plasma_path)

    def __reduce__(self):
        reduce = super().__reduce__()

        iterator = reduce[4]
        # key-value pairs of a dictionary subclass is pickled via the iterator protocol
        # See: https://docs.python.org/3/library/pickle.html

        state = reduce[2]
        key_value_pairs = []
        plasma_client = plasma.connect(self._plasma_path, num_retries=20)
        for key, value in iterator:
            if isinstance(value, torch.Tensor):
                value = PlasmaTensorView(value, self._plasma_path, plasma_client)
            key_value_pairs.append((key, value))
        state["_key_value_pairs"] = key_value_pairs
        plasma_client.disconnect()

        return (reduce[0], (), state)

    def __setstate__(self, state):
        self.__dict__.update(state)

        plasma_client = plasma.connect(self._plasma_path, num_retries=20)
        key_value_pairs = state.pop("_key_value_pairs")
        for key, value in key_value_pairs:
            if isinstance(value, PlasmaTensorView):
                value = value.to_tensor(plasma_client)
            self[key] = value
        plasma_client.disconnect()


class PlasmaTensorView:
    def __init__(
        self,
        tensor: torch.Tensor,
        plasma_path: str,
        plasma_client: plasma.PlasmaClient = None,
    ) -> None:
        if plasma_client is None:
            plasma_client = plasma.connect(plasma_path)
        self._object_id = plasma_client.put(tensor.numpy())
        self._plasma_path = plasma_path

    def to_tensor(self, plasma_client: plasma.PlasmaClient = None):
        current_plasma_client = plasma_client or plasma.connect(self._plasma_path)

        array = current_plasma_client.get(self._object_id)
        current_plasma_client.delete([self._object_id])

        if plasma_client is None:
            current_plasma_client.disconnect()

        return torch.from_numpy(array)
