import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import product
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("clean_eer")
parser.add_argument("noisy_eer")
parser.add_argument("outdir")
parser.add_argument("--train", type=int, default=15)
parser.add_argument("--dev", type=int, default=7)
parser.add_argument("--test", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iter", type=int, default=100000)
args = parser.parse_args()
random.seed(args.seed)

def get_pair_eer(filepath):
    spk_pair_eer = {}
    with open(filepath) as file:
        for line in file.readlines():
            line = line.strip()
            spk1, spk2, eer = line.split(",")
            spk_pair_eer[(spk1.strip(), spk2.strip())] = float(eer)
    return spk_pair_eer

clean_eers = get_pair_eer(args.clean_eer)
noisy_eers = get_pair_eer(args.noisy_eer)

def get_unordered_pairs(lst1, lst2):
    ordered_pairs = list(product(lst1, lst2))
    pair_existed = defaultdict(lambda: False)
    unordered_pairs = []
    for pair in ordered_pairs:
        pair = sorted(pair)
        pair_tag = " ".join(pair)
        if not pair_existed[pair_tag]:
            pair_existed[pair_tag] = True
            unordered_pairs.append(pair)
    return unordered_pairs

male_spks = []
female_spks = []
for idx in range(1, 31):
    male_spks.append(f"male {idx}")
    female_spks.append(f"female {idx}")

def get_spk_pair_score(spk_lst):
    spk_pairs = [(spk1, spk2) for spk1, spk2 in get_unordered_pairs(spk_lst, spk_lst) if spk1 != spk2]
    clean = []
    noisy = []
    for spk1, spk2 in spk_pairs:
        clean.append(clean_eers[(spk1, spk2)])
        noisy.append(noisy_eers[(spk1, spk2)])
    return clean, noisy

def find_best(lst):
    best_split = None
    best_score = 100
    for _ in tqdm(range(args.iter)):
        random.shuffle(lst)
        train = lst[ : args.train]
        dev = lst[args.train : args.train + args.dev]
        test = lst[args.train + args.dev : ]

        dev_clean, dev_noisy = get_spk_pair_score(dev)
        test_clean, test_noisy = get_spk_pair_score(test)
        score = abs(sum(dev_clean) - sum(test_clean)) + abs(sum(dev_noisy) - sum(test_noisy))
        score += sum(dev_clean) + sum(test_clean) + sum(dev_noisy) + sum(test_noisy)
        if score < best_score:
            best_split = (train, dev, test)
            best_scores = (dev_clean, dev_noisy, test_clean, test_noisy)
            best_score = score
    return best_split, best_scores, best_score

male_split, male_scores, male_score = find_best(male_spks)
female_split, female_scores, female_score = find_best(female_spks)

dev_clean_score = np.array(male_scores[0] + female_scores[0]).mean()
dev_noisy_score = np.array(male_scores[1] + female_scores[1]).mean()
test_clean_score = np.array(male_scores[2] + female_scores[2]).mean()
test_noisy_score = np.array(male_scores[3] + female_scores[3]).mean()

print(dev_clean_score)
print(dev_noisy_score)
print(test_clean_score)
print(test_noisy_score)

train = male_split[0] + female_split[0]
dev = male_split[1] + female_split[1]
test = male_split[2] + female_split[2]

outdir = Path(args.outdir)
os.makedirs(outdir, exist_ok=True)

with open(outdir / "train.spk", "w") as file:
    for spk in train:
        print(spk, file=file)

with open(outdir / "dev.spk", "w") as file:
    for spk in dev:
        print(spk, file=file)

with open(outdir / "test.spk", "w") as file:
    for spk in test:
        print(spk, file=file)
