import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--split_dir", required=True)
args = parser.parse_args()

csv = pd.read_csv(args.csv)
def get_info(split):
    return [(csv[csv.utterance_id == line.strip()]["speaker (gender/id)"].values[0], csv[csv.utterance_id == line.strip()]["utterance_text"].values[0]) for line in (Path(args.split_dir) / f"{split}.txt").open().readlines()]

train_spkrs, train_content = zip(*get_info("train"))
dev_spkrs, dev_content = zip(*get_info("dev"))
test_spkrs, test_content = zip(*get_info("test"))

content = train_content + dev_content + test_content
spkrs = list(set(train_spkrs)) + list(set(dev_spkrs)) + list(set(test_spkrs))

def compare_list(lst1, lst2):
    for item1, item2 in zip(lst1, lst2):
        assert item1 == item2, f"{item1}, {item2}"

compare_list(sorted(content), sorted(list(set(content))))
compare_list(sorted(spkrs), sorted(list(set(spkrs))))
