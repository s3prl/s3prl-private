import time
import queue
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import List

import torch
import torch.multiprocessing as mp

from s3prl import hub
from s3prl.util.benchmark import benchmark

logger = logging.getLogger(__name__)


def f(q, done, device_id: int, per_gpu_num: int, mode: str, upstream: str, batch_size: int, secs: int, use_tqdm: bool = False):
    device = f"cuda:{device_id}"
    model = getattr(hub, upstream)().to(device)
    model.eval()

    if use_tqdm:
        pbar = tqdm(total=per_gpu_num)
    step = 0
    repre = None
    with torch.no_grad():
        while step < per_gpu_num:
            repre = torch.stack(
                model(
                    [torch.randn(16000 * secs).to(device) for i in range(batch_size)]
                )["hidden_states"],
                dim=2
            )
            step += 1
            if use_tqdm:
                pbar.update()

            if mode == "cuda":
                repre = repre
            elif mode == "tensor":
                repre = repre.detach().cpu()
            elif mode == "numpy":
                repre = repre.detach().cpu().numpy()
            else:
                raise NotImplementedError

            q.put(repre)
            del repre

    q.put(None)
    done.wait()


def all_alive(ps: List[mp.Process]):
    for p in ps:
        if not p.is_alive():
            return False
    return True


def all_end(ps: List[mp.Process]):
    for p in ps:
        if p.is_alive():
            return False
    return True


class PseudoQueue:
    def put(self, x):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", default="wav2vec2_large_ll60k")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--total_num", type=int, default=100)
    parser.add_argument("--mode", default="cuda")
    parser.add_argument("--secs", type=int, default=14)
    parser.add_argument("--queue_size", type=int, default=1)
    parser.add_argument("--single_gpu", action="store_true")
    args = parser.parse_args()

    ctx = mp.get_context("forkserver")
    main_gpu = 1
    worker_gpus = [1, 2]

    if args.single_gpu:
        q = PseudoQueue()
        done = ctx.Event()
        f(q, done, main_gpu, args.total_num, args.mode, args.upstream, args.batch_size, args.secs, use_tqdm=True)
        done.set()
        exit(0)

    queues = []
    dones = []
    processes = []
    for i in worker_gpus:
        q = ctx.Queue(maxsize=args.queue_size)
        done = ctx.Event()
        queues.append(q)
        dones.append(done)

        p = ctx.Process(
            target=f, args=(q, done, i, int(args.total_num / len(worker_gpus)), args.mode, args.upstream, args.batch_size, args.secs)
        )
        processes.append(p)

    for p in processes:
        p.start()

    pbar = tqdm(total=args.total_num, dynamic_ncols=True)
    while not all_end(processes):
        for q, done in zip(queues, dones):
            try:
                recv = q.get_nowait()
            except queue.Empty:
                continue

            if recv is None:
                done.set()
            elif isinstance(recv, torch.Tensor):
                # do not move to cpu (and then to gpu)
                # moving from cuda to cpu is a serious I/O bottleneck
                with benchmark(f"move from {recv.device} to cuda:{main_gpu}", freq=1, device=f"cuda:{main_gpu}"):
                    recv_cuda = recv.to(f"cuda:{main_gpu}")
                del recv
            pbar.update()

    logger.info("start joining process")
    for p in processes:
        p.join()
