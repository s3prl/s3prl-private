import torch
import argparse
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
args = parser.parse_args()

def read_file(path):
    records = defaultdict(lambda: defaultdict(lambda: -10))
    with open(path) as file:
        lines = [line.strip().split() for line in file.readlines()]
        for query_id, doc_id, score in lines:
            records[query_id][doc_id] = float(score)
    return records

scores = read_file(args.scores)
labels = read_file(args.labels)

aps = []
eers = []
for query_id in scores.keys():
    doc_scores = [scores[query_id][doc_id] for doc_id in scores[query_id].keys()]
    max_score = max([abs(score) for score in doc_scores])
    doc_scores = [score / max_score for score in doc_scores]
    doc_labels = [labels[query_id][doc_id] for doc_id in scores[query_id].keys()]
    eer, threshold = EER(doc_labels, doc_scores)
    eers.append(eer)
    ap = average_precision_score(doc_labels, doc_scores)
    aps.append(ap)

print("MAP:", torch.Tensor(aps).mean().item())
print("EER:", torch.Tensor(eers).mean().item())