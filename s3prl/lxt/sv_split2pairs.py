import math
import torch
import random
import argparse
import torchaudio
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
from collections import defaultdict

torchaudio.set_audio_backend("sox_io")

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--split_file", required=True)
parser.add_argument("--output_list", required=True)
parser.add_argument("--pair_num", type=int, default=2000)
parser.add_argument("--min_seconds", type=float, default=2)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
random.seed(args.seed)

lxt = Path(args.lxt)
with Path(args.split_file).open() as split_file:
    def extract_uttr_spkr(line):
        uttr, spkr = line.strip().split(maxsplit=1)
        return uttr.strip(), spkr.strip()

    def get_seconds(uttr):
        path = lxt / f"{uttr}.wav"
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate

    uttr_spkrs = [extract_uttr_spkr(line) for line in split_file.readlines()]
    uttr_spkrs = [uttr_spkr for uttr_spkr in uttr_spkrs if get_seconds(uttr_spkr[0]) >= args.min_seconds]

spkr2uttr = defaultdict(list)
uttr2spkr = {}
for uttr,spkr in uttr_spkrs:
    spkr2uttr[spkr].append(uttr)
    uttr2spkr[uttr] = spkr
spkrs = sorted(list(set(spkr2uttr.keys())))

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

# SAME SPKR
final_same_pairs = []
same_pair_num = args.pair_num // 2
pair_per_spkr = math.ceil(same_pair_num / len(spkrs))
for spkr in spkrs:
    uttr_pairs = [(uttr1, uttr2) for uttr1, uttr2 in get_unordered_pairs(spkr2uttr[spkr], spkr2uttr[spkr]) if uttr1 != uttr2]
    sampled_pairs = random.sample(uttr_pairs, k=pair_per_spkr)
    final_same_pairs += sampled_pairs

# DIFF SPKR
final_diff_pairs = []
diff_pair_num = args.pair_num - same_pair_num
diff_spkr_pairs = [(spkr1, spkr2) for spkr1, spkr2 in get_unordered_pairs(spkrs, spkrs) if spkr1 != spkr2]
uttr_pair_per_spkr_pair = math.ceil(diff_pair_num / len(diff_spkr_pairs))
for (spkr1, spkr2) in diff_spkr_pairs:
    uttr_pairs = get_unordered_pairs(spkr2uttr[spkr1], spkr2uttr[spkr2])
    sampled_pairs = random.sample(uttr_pairs, k=uttr_pair_per_spkr_pair)
    final_diff_pairs += sampled_pairs

final_pairs = []
pivots = [final_same_pairs, final_diff_pairs] * 1000000
while len(final_same_pairs) > 0 or len(final_diff_pairs) > 0:
    pivot = pivots.pop(0)
    if len(pivot) > 0:
        final_pairs.append(pivot.pop(0))

matrix = torch.zeros((len(spkrs), len(spkrs)))
with Path(args.output_list).open("w") as output:
    for uttr1, uttr2 in final_pairs:
        spkr1 = uttr2spkr[uttr1]
        spkr2 = uttr2spkr[uttr2]
        print(int(spkr1 == spkr2), uttr1, uttr2, file=output)

        if spkr1 != spkr2:
            matrix[spkrs.index(spkr1), spkrs.index(spkr2)] += 1
            matrix[spkrs.index(spkr2), spkrs.index(spkr1)] += 1
        else:
            matrix[spkrs.index(spkr1), spkrs.index(spkr2)] += 1

plt.imshow(matrix)
plt.colorbar()
plt.savefig("pair-stats.png")