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

logger = logging.getLogger(__name__)


def f(q, done, device_id: int, per_gpu_num: int, mode: str):
    device = f"cuda:{device_id}"
    model = getattr(hub, "wav2vec2")().to(device)
    model.eval()

    step = 0
    secs = random.randint(20, 40)
    with torch.no_grad():
        while step < per_gpu_num:
            try:
                wav = torch.randn(16000 * secs).to(device)
                repre = torch.stack(model([wav])["hidden_states"], dim=2)
            except RuntimeError:
                torch.cuda.empty_cache()
                continue
            else:
                step += 1

            if mode == "cuda":
                repre = repre
            elif mode == "tensor":
                repre = repre.detach().cpu()
            elif mode == "numpy":
                repre = repre.detach().cpu().numpy()
            else:
                raise NotImplementedError
            q.put(repre)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_num", type=int, default=100)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--mode", default="cuda")
    parser.add_argument("--queue_size", type=int, default=1)
    args = parser.parse_args()

    ctx = mp.get_context("forkserver")
    queues = []
    dones = []
    processes = []
    for i in range(args.gpu_num):
        q = ctx.Queue(maxsize=args.queue_size)
        done = ctx.Event()
        queues.append(q)
        dones.append(done)

        p = ctx.Process(
            target=f, args=(q, done, i, int(args.total_num / args.gpu_num), args.mode)
        )
        processes.append(p)

    for p in processes:
        p.start()

    pbar = tqdm(total=args.total_num, dynamic_ncols=True)
    while not all_end(processes):
        for q, done in zip(queues, dones):
            recv = q.get()
            if recv is None:
                done.set()
            elif isinstance(recv, torch.Tensor):
                # do not move to cpu (and then to cpu)
                # moving from cuda to cpu is a serious I/O bottleneck
                recv_cuda = recv.to(f"cuda:{args.gpu_num}")
                del recv
            pbar.update()

    logger.info("start joining process")
    for p in processes:
        p.join()
