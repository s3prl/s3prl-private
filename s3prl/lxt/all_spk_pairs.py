from itertools import product
from collections import defaultdict

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

spks = []
for i in range(1, 31):
    spks.append(f"male_{i}")
    spks.append(f"female_{i}")

for spk1, spk2 in get_unordered_pairs(spks, spks):
    if spk1 == spk2:
        continue
    print(spk1, spk2, sep="_")
