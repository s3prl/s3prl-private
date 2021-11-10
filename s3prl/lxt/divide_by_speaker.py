import argparse
from pathlib import Path
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("--utt2spk", required=True)
parser.add_argument("--lxt", required=True)
parser.add_argument("--tgt", required=True)
args = parser.parse_args()
tgt = Path(args.tgt)
tgt.mkdir(exist_ok=True)

with open(args.utt2spk) as file:
    for line in file.readlines():
        utt, spk = line.split(maxsplit=1)
        utt = utt.strip()
        spk = spk.strip()
        subdir = tgt / spk
        subdir.mkdir(exist_ok=True)
        utt_path = Path(args.lxt) / f"{utt}.wav"
        copyfile(utt_path, subdir / f"{utt}.wav")