import re
import os
import yaml
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import s3prl
from separation import SeparationDataset

s3prl_path = Path(os.path.dirname(s3prl.__file__))

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices=["enhancement_stft", "separation_stft"], help="SE or SS")
    parser.add_argument('-s', '--split', help='The split you want to process (sperated by space)', default="all")
    parser.add_argument('-c', '--config', help='The path of config file.')
    parser.add_argument('-d', '--upstream_downsample_rate', help="The downsample rate of your upstream (used as hop_length)", default=320, type=int)
    parser.add_argument('-n', '--num_workers', help="Number of workers", default=12, type=int)
    parser.add_argument('-p', '--output_path', help="Where to store the output files", default=s3prl_path / "data/")
    
    args = parser.parse_args()
    # split
    if args.split == "all":
        args.split = ['train', 'dev', 'test']
    else:
        args.split = re.split(r"[, ]+", args.split)
    # config
    if args.config is None:
        if args.task == "enhancement_stft":
            args.num_spks = 1
        elif args.task == "separation_stft":
            args.num_spks = 2
        else:
            raise
        args.config = s3prl_path / f"downstream/{args.task}/configs/cfg_LXT{args.num_spks}mix.yaml"
    with open(args.config, 'r') as f:
        args.config = yaml.load(f, Loader=yaml.FullLoader)["downstream_expert"]
    return args

def preprocess(
    split,
    data_dir,
    hop_length,
    datarc,
    
):
    path = s3prl_path / f"data/{args.task}" / str(hop_length)
    os.makedirs(path, exist_ok=True)
    with open(path / "datarc.yaml", 'w') as f:
        yaml.dump(datarc, f, Dumper=yaml.Dumper)
        
    path /= split
    os.makedirs(path, exist_ok=True)
    
    dataset = SeparationDataset(
        data_dir=data_dir,
        hop_length=hop_length,
        split=split,
        split_name=split,
        addition_cond=["noisy", "clean"],
        task_name=args.task,
        num_spks=args.num_spks,
        use_cache=False,
        **datarc
    )
    loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        collate_fn=list
    )
    
    for i, data in enumerate(tqdm(loader)):
        data = data[0]
        with open(path / f"{i}.pkl", "wb") as f:
            pickle.dump((data[0], *map(torch.from_numpy, data[1:3]), *map(lambda d: list(map(torch.from_numpy, d)), data[3:])), f)
    

if __name__ == "__main__":
    args = getargs()
    
    for split in args.split:
        preprocess(
            split=split,
            data_dir=args.config["loaderrc"][f"{split}_dir"],
            hop_length=args.upstream_downsample_rate,
            datarc=args.config['datarc'],
        )

