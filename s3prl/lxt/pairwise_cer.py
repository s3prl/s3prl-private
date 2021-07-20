import pickle
import argparse
from tqdm import tqdm
from Levenshtein import editops
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--truth", required=True)
parser.add_argument("--inference", required=True)
parser.add_argument("--matching", required=True)
parser.add_argument("--njobs", type=int, default=12)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

def read_file(filepath):
    record = {}
    with open(filepath, "r") as file:
        for line in file.readlines():
            line = line.strip()
            key, value = line.split(",", maxsplit=1)
            if len(value) == 0:
                print(f"{key} has no transcription")
                continue
            if key in ["quac-C_8f1052a909924795a456beaa053ffe22_0_q#6-q-0"]:
                continue
            record[key] = value
    return record

truth = read_file(args.truth)
inference = read_file(args.inference)

def find_closest_lst(name):
    truth_text = truth[name]
    if len(truth_text) == 0:
        print(name)

    records = []
    for lxt_name in inference:
        hypo_text = inference[lxt_name]
        ops = editops(hypo_text, truth_text)        
        cer = len(ops) / len(truth_text)
        
        start_insert = []
        for op in ops:
            if op[0] == "insert" and op[1] == 0:
                start_insert.append(op)
            else:
                break

        end_insert = []
        for op in reversed(ops):
            if op[0] == "insert" and op[1] == len(hypo_text):
                end_insert.append(op)
            else:
                break

        start = start_insert[-1][2] if len(start_insert) > 1 else 0
        end = end_insert[-1][2] if len(end_insert) > 1 else len(truth_text)
        truth_subtext = truth_text[start : end]

        sub_ops = editops(hypo_text, truth_subtext)
        sub_cer = len(sub_ops) / max(len(truth_subtext), 1.0e-8)

        record_variables = ["lxt_name", "hypo_text", "name", "truth_text", "truth_subtext", "cer", "sub_cer"]

        variables = locals()
        records.append({v: variables[v] for v in record_variables})

    return records

if args.debug:
    truth = {key: truth[key] for idx, key in enumerate(truth) if idx < 10}

best_lxt_matching = Parallel(n_jobs=args.njobs)(delayed(find_closest_lst)(name) for name in tqdm(truth, total=len(truth)))

if args.debug:
    from ipdb import set_trace
    set_trace()

with open(args.matching, "wb") as matching:
    pickle.dump(best_lxt_matching, matching)
