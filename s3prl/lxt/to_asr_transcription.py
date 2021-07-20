import re
import pandas
import argparse
from tqdm import tqdm
from fairseq.data import Dictionary

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--dict", required=True)
parser.add_argument("--target_txt", required=True)
args = parser.parse_args()

csv = pandas.read_csv(args.csv)
dictionary = Dictionary.load(args.dict)

with open(args.target_txt, "w") as file:
    for row_id, row in tqdm(csv.iterrows(), total=len(csv)):
        utterance_id = row["utterance_id"]
        utterance_text = row["utterance_text"].upper()
        utterance_text = re.sub(r"\.", " ", utterance_text)
        utterance_text = re.sub(r",", " ", utterance_text)
        utterance_text = re.sub(r" +", " ", utterance_text)
        utterance_text = re.sub(r" '", "'", utterance_text)
        if len(utterance_text) == 0:
            from ipdb import set_trace
            set_trace()

        truth = dictionary.string(
            [
                idx
                for idx in dictionary.encode_line(
                    " ".join(list(utterance_text.upper().replace(" ", "|"))) + " |",
                    line_tokenizer=lambda x: x.split(),
                    add_if_not_exist=False,
                )
                if idx != dictionary.unk_index
            ]
        ).replace(" ", "").replace("|", " ").strip()
        file.write(f"{utterance_id},{truth}\n")
