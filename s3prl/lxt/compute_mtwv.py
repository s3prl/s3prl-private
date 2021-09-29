import argparse
from os import P_ALL
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from traitlets.traitlets import default

parser = argparse.ArgumentParser()
parser.add_argument("--scores", required=True)
parser.add_argument("--labels", required=True)
parser.add_argument("--C_miss", type=float, default=100)
parser.add_argument("--C_fa", type=float, default=1)
parser.add_argument("--P_target", type=float, default=0.0008)
args =parser.parse_args()

def read_file(path):
    records = defaultdict(lambda: defaultdict(lambda: None))
    with open(path) as file:
        lines = [line.strip().split() for line in file.readlines()]
        for query_id, doc_id, score in lines:
            records[query_id][doc_id] = float(score)
    return records

scores = read_file(args.scores)
labels = read_file(args.labels)

all_scores = []
all_labels = []
for query_id in tqdm(list(scores.keys())):
    doc_scores = [scores[query_id][doc_id] for doc_id in sorted(scores[query_id].keys())]
    doc_labels = [labels[query_id][doc_id] for doc_id in sorted(scores[query_id].keys())]
    all_scores += doc_scores
    all_labels += doc_labels


def EER(labels, scores):
    """
    labels: (N,1) value: 0,1

    scores: (N,1) value: -1 ~ 1

    """

    fpr, tpr, thresholds = roc_curve(labels, scores)
    a = lambda x : 1. - x - interp1d(fpr, tpr)(x)
    eer = brentq(a, 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer, thresh

def MTWV(labels, scores, beta):
    """
    labels: (N,1) value: 0,1

    scores: (N,1) value: -1 ~ 1

    """

    fpr, tpr, thresholds = roc_curve(labels, scores)

    def TWV(P_miss, P_fa):
        return 1 - (P_miss + beta * P_fa)

    inter = interp1d(fpr, tpr)
    all_twvs = lambda x : TWV(1 - inter(x), x)
    mtwv = brentq(all_twvs, -beta, 1.)
    thresh = interp1d(fpr, thresholds)(mtwv)
    return mtwv, thresh

beta = (args.C_fa * (1 - args.P_target)) / (args.C_miss * args.P_target)
mtwv = MTWV(all_labels, all_scores, beta)
print(mtwv)
