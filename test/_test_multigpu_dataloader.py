import queue
import pyarrow as pa
from time import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from s3prl import hub
from s3prl.base.output import Output
from s3prl.util.gpu_dataloader import MultipleGPUDataLoader
from s3prl.util.benchmark import benchmark

BATCH_SIZE = 8


class Project(nn.Module):
    def __init__(self):
        super().__init__()
        self.upstream = getattr(hub, "wav2vec2_large_ll60k")()

    def forward(self, wav):
        hidden_states = self.upstream(wav)["hidden_states"]
        hidden_states = torch.stack(hidden_states, dim=2)
        return hidden_states


class simple_iterator:
    def __init__(self, secs: int = 10, batch_size: int = 8) -> None:
        self.num = 1000
        self.secs = secs
        self.batch_size = batch_size

    def __len__(self):
        return self.num

    def __iter__(self):
        for i in range(self.num):
            yield Output(
                wav=[torch.randn(16000 * self.secs) for i in range(self.batch_size)]
            )


def test_multigpu_dataloader():
    dataloader = MultipleGPUDataLoader(simple_iterator(), Project, (), 1, [2, 0])
    for batch_id, batch in enumerate(tqdm(dataloader)):
        pass


def test_singlegpu_dataloader():
    device_id = 1
    model = Project().cuda(device_id)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(simple_iterator()):
            batch.to(f"cuda:{device_id}")
            model(**batch)


def test_speed():
    model = Project().cuda(0)
    model.eval()
    with torch.no_grad():
        for batch in simple_iterator():

            with benchmark("move wavs", device=f"cuda:0", freq=1):
                batch = batch.to(f"cuda:0")

            with benchmark("forward", device=f"cuda:0", freq=1):
                result = model(**batch)

            with benchmark("move to cpu", device="cuda:0", freq=1):
                result: torch.Tensor = result.cpu()
                result = result.share_memory_()

            with benchmark("move to cuda", device="cuda:0", freq=1):
                result = result.to(f"cuda:1")

            print(
                pa.serialize(result.detach().cpu().numpy()).to_buffer().size
                / (1024**3),
                "GB",
            )


def produce_tensor(q, e):
    model = Project().cuda(0)
    model.eval()
    with torch.no_grad():
        for batch in simple_iterator(13, 12):
            batch = batch.to(f"cuda:0")
            result = model(**batch)
            result = result.to("cuda:1")
            q.put(result)

    q.put(None)
    e.wait()


def test_queue():
    ctx = mp.get_context("forkserver")
    q = ctx.Queue(maxsize=1)
    e = ctx.Event()

    p = ctx.Process(target=produce_tensor, args=(q, e))
    p.start()

    while True:
        start = time()
        try:
            result = q.get_nowait()
        except queue.Empty:
            continue
        else:
            if result is None:
                break
            else:
                print("get from queue", time() - start)
                del result

    e.set()
    p.join()


if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("function")
    args = parser.parse_args()

    eval(args.function)()
