import argparse
import numpy as np
from pathlib import Path
from shutil import rmtree
from s3prl.downstream.sv_voxceleb1.utils import EER

parser = argparse.ArgumentParser()
parser.add_argument("--truth", required=True)
parser.add_argument("--predict", required=True)
parser.add_argument("--lxt_dir", required=True)
parser.add_argument("--test_dir", required=True)
parser.add_argument("--listen_num", default=50, type=int)
args = parser.parse_args()

lxt_dir = Path(args.lxt_dir)
test_dir = Path(args.test_dir)

with open(args.truth, "r") as file:
    truth = [line.strip().split(maxsplit=1) for line in file.readlines()]

with open(args.predict, "r") as file:
    predict = [line.strip().split(maxsplit=1) for line in file.readlines()]

truth = sorted(truth, key=lambda x: x[1])
predict = sorted(predict, key=lambda x: x[1])

labels, filenames1 = zip(*truth)
scores, filenames2 = zip(*predict)
assert filenames1 == filenames2
labels = [float(item) for item in labels]
scores = [float(item) for item in scores]

eer, threshold = EER(np.array(labels), np.array(scores))
predictions = [s > threshold for s in scores]

wrong = []
for score, label, filename in zip(scores, labels, filenames1):
    prediction = int(score > threshold)
    if prediction != label:
        diff = abs(score - threshold)
        wrong.append((diff, label, filename))

wrong.sort(key=lambda x: x[0], reverse=True)
with (test_dir / "note.txt").open("w") as file:
    for idx, (diff, label, filename) in enumerate(wrong):
        print(idx, diff, label, filename, file=file)

wrong = wrong[:args.listen_num]
for idx, (diff, label, filename) in enumerate(wrong):
    first_filename, second_filename = filename.split()
    first_filename = first_filename.strip()
    second_filename = second_filename.strip()
    
    sample_dir = test_dir / str(idx)
    if sample_dir.is_dir():
        rmtree(sample_dir)
    sample_dir.mkdir()

    (sample_dir / first_filename).symlink_to((lxt_dir / first_filename).resolve())
    (sample_dir / second_filename).symlink_to((lxt_dir / second_filename).resolve())
