import argparse

parser = argparse.ArgumentParser()
parser.add_argument("predict")
parser.add_argument("truth")
parser.add_argument("threshold")
args = parser.parse_args()

predict = [line.strip() for line in open(args.predict).readlines()]
truth = [line.strip() for line in open(args.truth).readlines()]
threshold = float(open(args.threshold).readline().strip())
for p, t in zip(predict, truth):
    score, tag1, tag2 = p.split()
    label, t1, t2 = t.split()
    assert tag1 == t1
    assert tag2 == t2
    score = float(score)
    label = int(label)
    hard_p = int(score > threshold)
    if hard_p != label:
        print(round(abs(score - threshold), 3), label, round(score, 3), tag1, tag2)
