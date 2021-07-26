import random
import argparse
from pathlib import Path

import pandas
import torchaudio
from s3prl.lxt.dataset import SpkrInfo, DatasetInfo
torchaudio.set_audio_backend("sox_io")

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--csv", required=True)
parser.add_argument("--dev", type=float, default=0.25)
parser.add_argument("--test", type=float, default=0.25)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
random.seed(args.seed)

lxt = Path(args.lxt)
csv = pandas.read_csv(args.csv)
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

spkrs = sorted(list(set(csv["speaker (gender/id)"].tolist())))
spkrs = [SpkrInfo(spkr, csv, lxt) for spkr in spkrs]

def random_split(items, ratios):
    assert sum(ratios) == 1
    items = random.sample(items, k=len(items))
    start = 0
    for ratio in ratios:
        length = round(len(items) * ratio)
        yield items[start : start + length]
        start = start + length

train, dev, test = [], [], []
ratios = [args.test, args.dev, 1 - args.test - args.dev]
for spkr in spkrs:
    uttr_with_spkr = list(zip(spkr.uttrs, [spkr] * len(spkr.uttrs)))
    test_1spkr, dev_1spkr, train_1spkr = random_split(uttr_with_spkr, ratios)
    train.extend(train_1spkr)
    dev.extend(dev_1spkr)
    test.extend(test_1spkr)

for split in ["train", "dev", "test"]:
    uttr_with_spkrs = eval(split)
    with (output_dir / f"{split}.txt").open("w") as split_file:
        for uttr, spkr in uttr_with_spkrs:
            print(f"{uttr.id} {spkr.name}", file=split_file)
