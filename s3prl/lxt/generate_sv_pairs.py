import random
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from librosa.util import find_files

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--csv", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--compare_same_num", default=5, type=int)
parser.add_argument("--compare_diff_num", default=5, type=int)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()
random.seed(args.seed)

csv = pd.read_csv(args.csv)
lxt_root = Path(args.lxt)
wav_paths = [Path(path) for path in find_files(lxt_root)]

rows = []
for path in wav_paths:
    df = csv[csv["utterance_id"] == path.stem]
    assert len(df) == 1
    rows.append(df)
csv = pd.concat(rows)

spkr_column = "speaker (gender/id)"
with open(args.output, "w") as file:
    for path in tqdm(wav_paths):
        utterance_id = path.stem
        spkr = csv[csv["utterance_id"] == utterance_id][spkr_column].tolist()[0]

        same_spkr_uids = csv[csv[spkr_column] == spkr]["utterance_id"].tolist()
        sampled_same_uids = random.sample(same_spkr_uids, k=args.compare_same_num)
        for uid in sampled_same_uids:
            print("1", f"{utterance_id}.wav", f"{uid}.wav", file=file)

        diff_spkr_uids = csv[csv[spkr_column] != spkr]["utterance_id"].tolist()
        sampled_diff_uids = random.sample(diff_spkr_uids, k=args.compare_diff_num)
        for uid in sampled_diff_uids:
            print("0", f"{utterance_id}.wav", f"{uid}.wav", file=file)
