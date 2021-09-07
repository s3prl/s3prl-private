import argparse
from pathlib import Path
from pandas import read_csv
from s3prl.lxt.text_normalize import qbe_norm

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

table = read_csv(args.csv)
uids = table["utterance_id"].tolist()
texts = table["utterance_text"].tolist()

with open(args.output, "w") as output:
    for uid, text in zip(uids, texts):
        print(uid, qbe_norm(text), file=output)
