import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--utt2spk", required=True)
parser.add_argument("--utt2length", required=True)
args = parser.parse_args()

spk2utts = defaultdict(list)
for line in open(args.utt2spk).readlines():
    line = line.strip()
    utt, spk = line.split(maxsplit=1)
    spk2utts[spk].append(utt)

utt2length = {}
for line in open(args.utt2length).readlines():
    line = line.strip()
    utt, length = line.split(maxsplit=1)
    utt2length[utt] = int(length)

spk2secs = {}
for spk, utts in spk2utts.items():
    lengths = [utt2length[utt] for utt in utts]
    total_length = sum(lengths)
    total_mins = total_length / 16000 / 60
    spk2secs[spk] = total_mins
    print(spk, total_mins)