import pandas
import argparse
from pathlib import Path
from librosa.util import find_files

parser = argparse.ArgumentParser()
parser.add_argument("--audio_dir", required=True)
parser.add_argument("--script", required=True)
args = parser.parse_args()

def read_file(filepath):
    record = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            line = line.strip()
            key, value = line.split(",", maxsplit=1)
            record.append(key)
    return record

def read_csv(filepath):
    table = pandas.read_csv(filepath)
    record = table["utterance_id"].tolist()
    return record

audio_stems = [Path(path).stem for path in find_files(args.audio_dir)]
if Path(args.script).suffix == ".wav":
    truths = read_file(args.script)
elif Path(args.script).suffix == ".csv":
    truths = read_csv(args.script)
else:
    raise ValueError

for stem in audio_stems:
    assert stem in truths, stem

for utterance_id in truths:
    if utterance_id not in audio_stems:
        print(utterance_id)