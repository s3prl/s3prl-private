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
full_dataset = DatasetInfo(spkrs, "full")

def random_split(items, ratios):
    assert sum(ratios) == 1
    items = random.sample(items, k=len(items))
    start = 0
    for ratio in ratios:
        length = round(len(items) * ratio)
        yield items[start : start + length]
        start = start + length

ratios = [args.test, args.dev, 1 - args.test - args.dev]
male_spkrs_test, male_spkrs_dev, male_spkrs_train = list(
    random_split(full_dataset.male, ratios)
)
female_spkrs_test, female_spkrs_dev, female_spkrs_train = list(
    random_split(full_dataset.female, ratios)
)

train_dataset = DatasetInfo([*male_spkrs_train, *female_spkrs_train], "train")
dev_dataset = DatasetInfo([*male_spkrs_dev, *female_spkrs_dev], "dev")
test_dataset = DatasetInfo([*male_spkrs_test, *female_spkrs_test], "test")

for dataset in [
    full_dataset,
    full_dataset.male_dataset,
    full_dataset.female_dataset,
    train_dataset,
    dev_dataset,
    test_dataset,
]:
    print(f"{dataset.name.upper()}:")
    print(dataset)
    dataset.save(output_dir / f"{dataset.name}.pkl")

for split in ["train", "dev", "test"]:
    dataset: DatasetInfo = eval(f"{split}_dataset")
    with (output_dir / f"{split}.txt").open("w") as split_file:
        for uid in dataset.uttr_ids:
            print(uid, file=split_file)
