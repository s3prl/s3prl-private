import argparse
from tqdm import tqdm
from s3prl.downstream.ctc.metric import parse, wer, cer

parser = argparse.ArgumentParser()
parser.add_argument("--truth", required=True)
parser.add_argument("--inference", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

def read_file(filepath):
    record = {}
    with open(filepath, "r") as file:
        for line in file.readlines():
            line = line.strip()

            items = line.split(" ", maxsplit=1)
            if len(items) == 1:
                key, value = items[0], ""
                print(f"{filepath}: {key} has no transcription")
            else:
                key, value = items

            record[key] = value
    return record

truth = read_file(args.truth)
inference = read_file(args.inference)
for key in list(inference.keys()):
    if " (1)" in key:
        utterance_id = key.split()[0]
        inference[utterance_id] = inference[f"{utterance_id} (1)"] + " " + inference[f"{utterance_id} (2)"]
        del inference[f"{utterance_id} (1)"]
        del inference[f"{utterance_id} (2)"]

truths = []
infers = []
statistics = {}
for key in inference:
    truth_text = truth[key]
    infer_text = inference.get(key)

    statistics[key] = {
        "WER": wer([infer_text], [truth_text]),
        "CER": cer([infer_text], [truth_text]),
        "infer": infer_text,
        "truth": truth_text
    }

    truths.append(truth_text)
    infers.append(infer_text)

with open(args.output, "w") as file:
    print("TOTAL", file=file)
    print(f"WER: {wer(infers, truths)}", file=file)
    print(f"CER: {cer(infers, truths)}", file=file)
    print(file=file)

    sorted_keys = sorted(list(statistics.keys()), key=lambda x: statistics[x]["CER"], reverse=True)
    for key in tqdm(sorted_keys):
        stat = statistics[key]
        print(f"FILE: {key}", file=file)
        print(f"CER: {stat['CER']}", file=file)
        print(f"WER: {stat['WER']}", file=file)
        print(f"INFER: {stat['infer']}", file=file)
        print(f"TRUTH: {stat['truth']}", file=file)
        print(file=file)
