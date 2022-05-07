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


def f(device_id: int, per_gpu_num: int, mode: str, upstream: str, batch_size: int):
    device = f"cuda:{device_id}"
    model = getattr(hub, upstream)().to(device)
    model.eval()

    pbar = tqdm(total=per_gpu_num)
    repre = None
    with torch.no_grad():
        while pbar.n < per_gpu_num:
            repre = torch.stack(
                model(
                    [torch.randn(16000 * 14).to(device) for i in range(batch_size)]
                )["hidden_states"],
                dim=2
            )
            pbar.update()

            if mode == "cuda":
                repre = repre
            elif mode == "tensor":
                repre = repre.detach().cpu()
            elif mode == "numpy":
                repre = repre.detach().cpu().numpy()
            else:
                raise NotImplementedError

            del repre


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
    parser.add_argument("upstream")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--total_num", type=int, default=100)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--mode", default="cuda")
    parser.add_argument("--queue_size", type=int, default=1)
    args = parser.parse_args()

    f(0, args.total_num, args.mode, args.upstream, args.batch_size)

