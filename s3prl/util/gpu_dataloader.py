import logging
from typing import List, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.connection import Connection

from s3prl import Container
from s3prl.util.benchmark import benchmark

BLOCK_SECS = 5

logger = logging.getLogger(__name__)


class MultipleGPUDataLoader:
    def __init__(
        self,
        dataloader,
        preprocessor_cls,
        preprocessor_init_args: Tuple,
        main_device_id: int,
        worker_device_ids: List[int],
        start_method: str = "forkserver",
    ) -> None:
        self.dataloader = dataloader
        self.main_device_id = main_device_id
        self.worker_device_ids = worker_device_ids

        self.ctx = mp.get_context(start_method)
        self.dones: List[mp.Event] = []
        self.parent_conns: List[Connection] = []
        self.processes: List[mp.Process] = []

        for device_id in worker_device_ids:
            done = self.ctx.Event()
            parent_conn, child_conn = self.ctx.Pipe()
            process = self.ctx.Process(
                target=self.preprocess,
                args=(
                    preprocessor_cls,
                    preprocessor_init_args,
                    device_id,
                    child_conn,
                    done,
                ),
            )
            process.start()

            self.dones.append(done)
            self.parent_conns.append(parent_conn)
            self.processes.append(process)

    def __del__(self):
        for process in self.processes:
            result = process.join(BLOCK_SECS)
            if result is None:
                process.terminate()

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        num_batches = len(self.dataloader)
        iterator = iter(self.dataloader)

        for parent_id in range(len(self.parent_conns)):
            parent_conn = self.parent_conns[parent_id]
            new_batch = next(iterator)
            new_batch = deepcopy(new_batch)
            parent_conn.send(new_batch)
            del new_batch

        for batch_id in range(num_batches):
            parent_id = batch_id % len(self.parent_conns)
            parent_conn = self.parent_conns[parent_id]

            received_batch = parent_conn.recv()
            preprocessed_batch = received_batch.to(self.main_device_id)
            del received_batch
            yield preprocessed_batch
            del preprocessed_batch

            if batch_id + len(self.parent_conns) >= num_batches:
                parent_conn.send(None)
            else:
                new_batch = next(iterator)
                new_batch = deepcopy(new_batch)
                parent_conn.send(new_batch)
                del new_batch

        for done in self.dones:
            done.set()
        for process in self.processes:
            process.join()

    @staticmethod
    def preprocess(
        preprocessor_cls,
        preprocessor_init_args: Tuple,
        device_id: int,
        child_conn: Connection,
        done: mp.Event,
    ):
        preprocessor = preprocessor_cls(*preprocessor_init_args).to(f"cuda:{device_id}")
        preprocessor.eval()
        with torch.no_grad():
            while True:
                batch = child_conn.recv()
                if batch is None:
                    del batch
                    break
                else:
                    assert isinstance(batch, Container)
                    batch_cuda = batch.to(f"cuda:{device_id}")
                    del batch
                    preprocessed_batch = preprocessor(**batch_cuda)
                    child_conn.send(preprocessed_batch)
                    del preprocessed_batch

        child_conn.close()
        done.wait()
