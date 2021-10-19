import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--utt2spk", required=True)
parser.add_argument("--spk2utt", required=True)
args = parser.parse_args()

spk2utt = defaultdict(list)
for line in open(args.utt2spk).readlines():
    utt, spk = line.split(maxsplit=1)
    spk2utt[spk.strip()].append(utt.strip())

with open(args.spk2utt, "w") as file:
    for spk, utts in spk2utt.items():
        print(spk, " ".join(utts), file=file)
