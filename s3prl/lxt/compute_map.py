import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import average_precision_score, roc_curve

def EER(labels, scores):
    """
    labels: (N,1) value: 0,1

    scores: (N,1) value: -1 ~ 1

    """

    fpr, tpr, thresholds = roc_curve(labels, scores)
    s = interp1d(fpr, tpr)
    a = lambda x : 1. - x - interp1d(fpr, tpr)(x)
    eer = brentq(a, 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer, thresh

parser = argparse.ArgumentParser()
parser.add_argument("--scores", required=True)
parser.add_argument("--labels", required=True)
parser.add_argument("--output_dir",required=True)
parser.add_argument("--query")
args = parser.parse_args()
Path(args.output_dir).mkdir(exist_ok=True)

def read_file(path):
    records = defaultdict(lambda: defaultdict(lambda: None))
    with open(path) as file:
        lines = [line.strip().split() for line in file.readlines()]
        for query_id, doc_id, score in lines:
            records[query_id][doc_id] = float(score)
    return records

scores = read_file(args.scores)
labels = read_file(args.labels)

metrics = []
for query_id in tqdm(list(scores.keys())):
    doc_scores = [scores[query_id][doc_id] for doc_id in sorted(scores[query_id].keys())]
    doc_labels = [labels[query_id][doc_id] for doc_id in sorted(scores[query_id].keys())]
    eer, threshold = EER(doc_labels.copy(), doc_scores.copy())
    ap = average_precision_score(doc_labels.copy(), doc_scores.copy())
    metrics.append((query_id, ap, eer))

    with (Path(args.output_dir) / f"{query_id}.result").open("w") as result:
        id_with_score = list(zip(sorted(scores[query_id].keys()), doc_scores))
        id_with_score.sort(key=lambda x: x[1], reverse=True)
        for qid, score in id_with_score:
            print(qid, score, file=result)

queryid2text = {}
if args.query is not None:
    with open(args.query) as file:
        for line in file.readlines():
            query_id, text = line.strip().split(maxsplit=1)
            queryid2text[query_id.strip()] = text.strip()

with (Path(args.output_dir) / "result").open("w") as result:
    metrics.sort(key=lambda x: x[1])
    for query_id, ap, eer in metrics:
        print(query_id, queryid2text.get(query_id, "None"), ap, eer, sep=" ", file=result)

with (Path(args.output_dir) / "metrics").open("w") as output:
    print("MAP:", torch.Tensor([item[1] for item in metrics]).mean().item())
    print("EER:", torch.Tensor([item[2] for item in metrics]).mean().item())
    print("MAP:", torch.Tensor([item[1] for item in metrics]).mean().item(), file=output)
    print("EER:", torch.Tensor([item[2] for item in metrics]).mean().item(), file=output)
