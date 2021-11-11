import random
import argparse
from itertools import product
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("utt2spk")
parser.add_argument("src_spk")
parser.add_argument("tgt_spk")
parser.add_argument("--pairs", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
random.seed(args.seed)

utt2spk = {}
spk2utt = defaultdict(list)
for line in open(args.utt2spk).readlines():
    line = line.strip()
    utt, spk = line.split(maxsplit=1)
    utt2spk[utt] = spk
    spk2utt[spk].append(utt)

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

src_same_pairs = [(utt1, utt2) for utt1, utt2 in get_unordered_pairs(spk2utt[args.src_spk], spk2utt[args.src_spk]) if utt1 != utt2]
tgt_same_pairs = [(utt1, utt2) for utt1, utt2 in get_unordered_pairs(spk2utt[args.tgt_spk], spk2utt[args.tgt_spk]) if utt1 != utt2]
all_same_pairs = src_same_pairs + tgt_same_pairs
all_diff_pairs = get_unordered_pairs(spk2utt[args.src_spk], spk2utt[args.tgt_spk])
same_pairs = random.sample(all_same_pairs, k=args.pairs)
for utt1, utt2 in same_pairs:
    print("1", f"{utt1}.wav", f"{utt2}.wav")

diff_pairs = random.sample(all_diff_pairs, k=args.pairs)
for utt1, utt2 in diff_pairs:
    print("0", f"{utt1}.wav", f"{utt2}.wav")
