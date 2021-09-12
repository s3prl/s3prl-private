import re
import pandas
import argparse
from tqdm import tqdm
from pathlib import Path
from fairseq.data import Dictionary
from s3prl.lxt.text_normalize import asr_norm

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
# parser.add_argument("--dict", required=True)
parser.add_argument("--target_txt", required=True)
args = parser.parse_args()

csv = pandas.read_csv(args.csv)
# dictionary = Dictionary.load(args.dict)
target_dir = Path(args.target_txt).parent
target_dir.mkdir(exist_ok=True)

with open(args.target_txt, "w") as file:
    for row_id, row in tqdm(csv.iterrows(), total=len(csv)):
        utterance_id = row["utterance_id"]
        utterance_text = asr_norm(row["utterance_text"])
        assert len(utterance_text) > 0
        print(utterance_id, utterance_text, file=file)
