import queue
import logging
import threading
from typing import Iterable, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing.connection import Connection

from s3prl import Container
from s3prl.util.benchmark import benchmark

BLOCK_SECS = 5

logger = logging.getLogger(__name__)

# Usually we won't get benefit from increasing this constant
# Since the GPU computation is the bottleneck
CUDA_QUEUE_SIZE = 1


@dataclass
class WorkerContext:
    sender2worker: mp.Queue
    worker2main: mp.Queue
    sender2worker_done: mp.Event
    worker2main_done: mp.Event
    process: mp.Process


class MultipleGPUDataLoader:
    def __init__(
        self,
        dataloader,
        preprocessor_cls,
        preprocessor_init_args: Tuple,
        main_device_id: int,
        worker_device_ids: List[int],
        start_method: str = "forkserver",
        data_queue_size: int = 10,
        device_thread_num: int = 4,
        premoved_cuda_limit: int = 10,
    ) -> None:
        self.dataloader = dataloader
        self.preprocessor_cls = preprocessor_cls
        self.preprocessor_init_args = preprocessor_init_args
        self.main_device_id = main_device_id
        self.worker_device_ids = worker_device_ids
        self.start_method = start_method
        self.data_queue_size = data_queue_size
        self.device_thread_num = device_thread_num
        self.premoved_cuda_limit = premoved_cuda_limit

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        preprocessor = self.preprocessor_cls(*self.preprocessor_init_args).cuda(
            self.main_device_id
        )
        preprocessor.eval()

        ctx = mp.get_context(self.start_method)

        worker_contexts: List[WorkerContext] = []
        for device_id in self.worker_device_ids:
            sender2worker = ctx.Queue(maxsize=self.data_queue_size)
            worker2main = ctx.Queue(maxsize=CUDA_QUEUE_SIZE)
            sender2worker_done = ctx.Event()
            worker2main_done = ctx.Event()
            process = ctx.Process(
                target=self.preprocess,
                args=(
                    self.preprocessor_cls,
                    self.preprocessor_init_args,
                    device_id,
                    self.main_device_id,
                    sender2worker,
                    worker2main,
                    sender2worker_done,
                    worker2main_done,
                ),
            )
            process.start()
            worker_contexts.append(
                WorkerContext(
                    sender2worker,
                    worker2main,
                    sender2worker_done,
                    worker2main_done,
                    process,
                )
            )

        sender2main = ctx.Queue(self.data_queue_size)
        sender2main_done = ctx.Event()

        sender2workers = [sender2main] + [c.sender2worker for c in worker_contexts]
        sender2workers_done = [sender2main_done] + [
            c.sender2worker_done for c in worker_contexts
        ]
        sender = ctx.Process(
            target=self.sender,
            args=(self.dataloader, sender2workers, sender2workers_done),
        )
        sender.start()

        raw_batch_cuda_semaphore = threading.BoundedSemaphore(self.premoved_cuda_limit)
        raw_batch_cuda_queue = queue.Queue()
        raw_batch_cudar = threading.Thread(
            target=self.device_mover,
            args=(
                sender2main,
                sender2main_done,
                raw_batch_cuda_queue,
                raw_batch_cuda_semaphore,
                self.main_device_id,
            ),
        )
        raw_batch_cudar.start()

        preprocessed_batch_cuda_semaphore = threading.BoundedSemaphore(
            self.premoved_cuda_limit
        )
        preprocessed_batch_cuda_queue = queue.Queue()
        preprocessed_batch_cudars = []
        for worker_context in worker_contexts:
            for i in range(self.device_thread_num):
                thread = threading.Thread(
                    target=self.device_mover,
                    args=(
                        worker_context.worker2main,
                        worker_context.worker2main_done,
                        preprocessed_batch_cuda_queue,
                        preprocessed_batch_cuda_semaphore,
                        self.main_device_id,
                    ),
                )
                thread.start()
                preprocessed_batch_cudars.append(thread)

        worker_processes = [c.process for c in worker_contexts]
        with torch.no_grad():
            while not self.all_processes_end([sender, *worker_processes]):
                if sender.is_alive():
                    try:
                        batch = raw_batch_cuda_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        raw_batch_cuda_semaphore.release()
                        preprocessed_batch = preprocessor(**batch)
                        del batch
                        yield preprocessed_batch
                        del preprocessed_batch

                if not self.all_processes_end(worker_processes):
                    try:
                        preprocessed_batch = preprocessed_batch_cuda_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        preprocessed_batch_cuda_semaphore.release()
                        yield preprocessed_batch
                        del preprocessed_batch

        raw_batch_cudar.join()
        for cudar in preprocessed_batch_cudars:
            cudar.join()

        sender.join()
        for context in worker_contexts:
            context.process.join()

    @staticmethod
    def all_processes_end(processes: List[mp.Process]):
        for p in processes:
            if p.is_alive():
                return False
        return True

    @staticmethod
    def device_mover(
        data_queue: mp.Queue,
        data_queue_done: mp.Event,
        result_queue: mp.Queue,
        premoved_cuda_semaphore: threading.BoundedSemaphore,
        device_id: int,
    ):
        while True:
            try:
                premoved_cuda_semaphore.acquire()
                batch = data_queue.get()
            except (ValueError, OSError):
                # the data_queue is closed
                break
            else:
                if batch is None:
                    break
                else:
                    batch_on_device = batch.to(f"cuda:{device_id}")
                    result_queue.put(batch_on_device)
                    del batch
                    del batch_on_device
        data_queue_done.set()

    @staticmethod
    def sender(
        dataloader: Iterable,
        sender2workers: List[mp.Queue],
        sender2workers_done: List[mp.Event],
    ):
        data_iter = iter(dataloader)
        batch = next(data_iter)
        unfinished_sender2workers = sender2workers.copy()
        while len(unfinished_sender2workers) > 0:
            for sender2worker in unfinished_sender2workers.copy():
                try:
                    sender2worker.put_nowait(batch)
                except queue.Full:
                    continue
                else:
                    if batch is None:
                        unfinished_sender2workers.remove(sender2worker)

                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        batch = None

        for sender2worker_done in sender2workers_done:
            sender2worker_done.wait()

    @staticmethod
    def preprocess(
        preprocessor_cls,
        preprocessor_init_args,
        device_id: int,
        main_device_id: int,
        sender2worker: mp.Queue,
        worker2main: mp.Queue,
        sender2worker_done: mp.Event,
        worker2main_done: mp.Event,
    ):
        preprocessor = preprocessor_cls(*preprocessor_init_args).to(f"cuda:{device_id}")
        preprocessor.eval()

        with torch.no_grad():
            while True:
                batch = sender2worker.get()
                if batch is None:
                    sender2worker_done.set()
                    worker2main.put(None)
                    break
                else:
                    batch_cuda = batch.to(f"cuda:{device_id}")
                    del batch
                    preprocessed_batch = preprocessor(**batch_cuda)
                    worker2main.put(preprocessed_batch)
                    del preprocessed_batch

        worker2main_done.wait()
