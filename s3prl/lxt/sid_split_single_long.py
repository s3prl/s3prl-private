import random
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torchaudio
torchaudio.set_audio_backend("sox_io")

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--utt2spk", required=True)
parser.add_argument("--csv", required=True)
parser.add_argument("--argmax", type=int, default=0)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
random.seed(args.seed)
csv = pd.read_csv(args.csv)
whitelist = csv["utterance_id"].tolist()

lxt = Path(args.lxt)
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

spk2utt = defaultdict(list)
with open(args.utt2spk, "r") as file:
    for line in file.readlines():
        utt, spk = line.split(" ", maxsplit=1)
        if utt not in whitelist:
            continue
        spk2utt[spk.strip()].append(utt.strip())

train, dev, test = [], [], []
for spk in spk2utt.keys():
    def get_length(path):
        info = torchaudio.info(str(path))
        return info.num_frames

    utts = spk2utt[spk]
    utts = sorted(utts, key=lambda x: get_length(lxt / f"{x}.wav"), reverse=True)
    train_utt = utts.pop(args.argmax)
    eval_num = len(utts)

    def wrap_with_spk(utts: list):
        return [(utt, spk) for utt in utts]

    train.extend(wrap_with_spk([train_utt]))
    dev.extend(wrap_with_spk(utts[: eval_num // 2]))
    test.extend(wrap_with_spk(utts[eval_num // 2 : ]))

for split in ["train", "dev", "test"]:
    uttr_with_spkrs = eval(split)
    with (output_dir / f"{split}.txt").open("w") as split_file:
        for uttr, spkr in uttr_with_spkrs:
            print(f"{uttr} {spkr}", file=split_file)
