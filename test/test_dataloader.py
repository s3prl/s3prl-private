import time
import queue
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import List

import torch
import multiprocessing as mp

from s3prl import hub

logger = logging.getLogger(__name__)


def f(q, device_id: int, per_gpu_num: int):
    device = f"cuda:{device_id}"
    model = getattr(hub, "wav2vec2")().to(device)
    model.eval()

    secs = random.randint(20, 40)
    with torch.no_grad():
        for i in range(per_gpu_num):
            wav = torch.randn(16000 * secs).to(device)
            repre = torch.stack(model([wav])["hidden_states"], dim=2)
            q.put(repre)


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
    args = parser.parse_args()

    ctx = mp.get_context("forkserver")
    queues = []
    processes = []
    for i in range(args.gpu_num):
        q = ctx.Queue()
        queues.append(q)

        p = ctx.Process(target=f, args=(q, i, int(args.total_num / args.gpu_num)))
        processes.append(p)

    for p in processes:
        p.start()

    pbar = tqdm(total=args.total_num, dynamic_ncols=True)
    while not all_end(processes) or pbar.n < args.total_num:
        for q in queues:
            if not q.empty():
                tmp = q.get()
                tmp = tmp.cpu()
                pbar.update()

    logger.info("start joining process")
    for p in processes:
        p.join()
