import os
import argparse
import pandas as pd
from pathlib import Path
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--lxt", required=True)
parser.add_argument("--tgt", required=True)
args = parser.parse_args()
tgt = Path(args.tgt)
tgt.mkdir(exist_ok=True)

csv = pd.read_csv(args.csv)
for row_id, row in csv.iterrows():
    spk = row["speaker (gender/id)"].replace(" ", "_")
    subdir = tgt / spk
    subdir.mkdir(exist_ok=True)
    uid = row["utterance_id"]
    utt_path = Path(args.lxt) / f"{uid}.wav"
    copyfile(utt_path, subdir / f"{uid}.wav")