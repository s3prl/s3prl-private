import argparse
from numpy.lib.function_base import select
import pandas as pd
from collections import defaultdict
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math
import random

torchaudio.set_audio_backend("sox_io")


parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--audio", required=True)
parser.add_argument("--trials", type=int, default=10000)
parser.add_argument("--output", required=True)

args = parser.parse_args()

csv = pd.read_csv(args.csv)
secs = {uid: torchaudio.info(Path(args.audio) / f"{uid}.wav").num_frames / 16000 for uid in tqdm(csv.utterance_id.values, total=len(csv))}

original_spkrs_secs = defaultdict(lambda: 0)
spkr_column = "speaker (gender/id)"
content = defaultdict(list)
for rowid, row in csv.iterrows():
    content[row.utterance_text].append([row[spkr_column], row.utterance_id])
    original_spkrs_secs[row[spkr_column]] += secs[row.utterance_id]

def spkr_secs(uids):
    sec = 0
    for uid in uids:
        sec += secs[uid]
    return sec

start = 66666666666666666
end = 888888888888888888888888
assert start < end
selections = []
for seed in tqdm(range(start, end, round((end - start) / args.trials))):
    random.seed(seed)
    keys = list(content.keys())
    random.shuffle(keys)

    selection = defaultdict(list)
    for text in keys:
        spkrs_uids = content[text]
        spkrs, uids = zip(*spkrs_uids)
        spkrs_secs = [spkr_secs(selection[spkr]) for spkr in spkrs]
        argmin = spkrs_secs.index(min(spkrs_secs))
        selection[spkrs[argmin]].append(uids[argmin])

    trimmeds = []
    for spkr, uids in selection.items():
        trimmed = spkr_secs(uids)
        untrimmed = original_spkrs_secs[spkr]
        assert trimmed < untrimmed + 1, f"{trimmed} > {untrimmed}"
        # print(spkr, trimmed, untrimmed)
        trimmeds.append(trimmed)
    
    selections.append((seed, np.std(trimmeds), selection))

selections.sort(key=lambda x: x[1])
all_stds = [selection[1] for selection in selections]
print("stds_mean", np.mean(all_stds))
print("stds_std", np.std(all_stds))
final_selection = selections[0][-1]

for spkr, uids in final_selection.items():
    trimmed = spkr_secs(uids)
    untrimmed = original_spkrs_secs[spkr]
    assert trimmed < untrimmed + 1, f"{trimmed} > {untrimmed}"
    print(spkr, trimmed, untrimmed)

with open(args.output, "w") as file:
    print(str(dict(final_selection)), file=file)
