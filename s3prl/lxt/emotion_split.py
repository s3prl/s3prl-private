import random
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--csv", required=True)
parser.add_argument("--train_ratio", type=float, default=0.1)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
random.seed(0)

table = pd.read_csv(args.csv)
emo2utt = defaultdict(list)
for _, row in table.iterrows():
    emotion = row["emotion"]
    uid = row["utterance_id"]

    audio_path = Path(args.lxt) / (uid + ".wav")
    if not audio_path.is_file():
        content = row["utterance_text"]
        print(f"[lxt_emotion] - {audio_path} not exists, content: {content}")
        continue

    emo2utt[emotion].append(uid)

final_train = []
final_dev = []
final_test = []

def random_split(items, ratios):
    assert sum(ratios) == 1
    items = random.sample(items, k=len(items))
    start = 0
    for ratio in ratios:
        length = round(len(items) * ratio)
        yield items[start : start + length]
        start = start + length    

for emotion, utts in emo2utt.items():
    eval_ratio = (1 - args.train_ratio) / 2
    ratios = [args.train_ratio, eval_ratio, eval_ratio]
    train_utts, dev_utts, test_utts = random_split(utts, ratios)
    final_train += [(utt, emotion) for utt in train_utts]
    final_dev += [(utt, emotion) for utt in dev_utts]
    final_test += [(utt, emotion) for utt in test_utts]

Path(args.output_dir).mkdir(exist_ok=True)

with (Path(args.output_dir) / "train.txt").open("w") as file:
    for utt, emotion in final_train:
        print(utt, emotion, file=file)

with (Path(args.output_dir) / "dev.txt").open("w") as file:
    for utt, emotion in final_dev:
        print(utt, emotion, file=file)

with (Path(args.output_dir) / "test.txt").open("w") as file:
    for utt, emotion in final_test:
        print(utt, emotion, file=file)
