import random
import argparse
from pathlib import Path
from collections import defaultdict

import pandas
import torchaudio
from s3prl.lxt.dataset import SpkrInfo, DatasetInfo
torchaudio.set_audio_backend("sox_io")

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--utt2spk", required=True)
parser.add_argument("--train", type=int, default=5)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
random.seed(args.seed)

lxt = Path(args.lxt)
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

spk2utt = defaultdict(list)
with open(args.utt2spk, "r") as file:
    for line in file.readlines():
        utt, spk = line.split(" ", maxsplit=1)
        spk2utt[spk.strip()].append(utt.strip())

train, dev, test = [], [], []
for spk in spk2utt.keys():
    utts = spk2utt[spk]
    random.shuffle(utts)
    eval_num = len(utts) - args.train

    def wrap_with_spk(utts: list):
        return [(utt, spk) for utt in utts]

    train.extend(wrap_with_spk(utts[ : args.train]))
    dev.extend(wrap_with_spk(utts[ : args.train + eval_num // 2]))
    test.extend(wrap_with_spk(utts[args.train + eval_num // 2 : ]))

for split in ["train", "dev", "test"]:
    uttr_with_spkrs = eval(split)
    with (output_dir / f"{split}.txt").open("w") as split_file:
        for uttr, spkr in uttr_with_spkrs:
            print(f"{uttr} {spkr}", file=split_file)
