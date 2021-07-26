import random
import argparse
import torchaudio
from pathlib import Path
from itertools import product

torchaudio.set_audio_backend("sox_io")

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--split_file", required=True)
parser.add_argument("--output_list", required=True)
parser.add_argument("--pair_num", type=int, default=3000)
parser.add_argument("--min_seconds", type=float, default=0)
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

pairs = list(product(uttr_spkrs, uttr_spkrs))
same_spkr_pairs = [pair for pair in pairs if pair[0][0] != pair[1][0] and pair[0][1] == pair[1][1]]
diff_spkr_pairs = [pair for pair in pairs if pair[0][0] != pair[1][0] and pair[0][1] != pair[1][1]]
sampled_same_spkr_pairs = random.sample(same_spkr_pairs, k=args.pair_num // 2)
sampled_diff_spkr_pairs = random.sample(diff_spkr_pairs, k=args.pair_num // 2)

with Path(args.output_list).open("w") as output:
    for (uttr1, spkr1), (uttr2, spkr2) in sampled_same_spkr_pairs:
        assert spkr1 == spkr2
        print(1, uttr1, uttr2, file=output)

    for (uttr1, spkr1), (uttr2, spkr2) in sampled_diff_spkr_pairs:
        assert spkr1 != spkr2
        print(0, uttr1, uttr2, file=output)