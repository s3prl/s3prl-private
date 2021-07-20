import pandas
import argparse
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree, copy
from librosa.util import find_files

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--csv", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

filepaths = find_files(args.lxt)
csv = pandas.read_csv(args.csv)

output = Path(args.output)
if output.exists():
    rmtree(str(output))
output.mkdir()

for path in tqdm(filepaths):
    path = Path(path)
    emotion = csv[csv["utterance_id"] == path.stem]["emotion"].tolist()[0]
    new_name = path.stem + f"_{emotion}" + path.suffix
    new_path = output / new_name
    copy(path, new_path)
