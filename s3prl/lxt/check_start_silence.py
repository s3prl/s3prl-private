import argparse
import torchaudio
from tqdm import tqdm
from librosa.util import find_files
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--njobs", default=12)
args = parser.parse_args()
files = find_files(args.lxt)

def get_start_ratio(file):
    wav, sr = torchaudio.load(file)
    wav = wav.view(-1)
    start_ratio = (wav[:16000].abs() / max(wav.abs())).mean().item()
    return file, start_ratio

start_ratios = Parallel(n_jobs=args.njobs)(delayed(get_start_ratio)(file) for file in tqdm(files))
start_ratios.sort(key=lambda x: x[1])
for file, ratio in start_ratios:
    print(ratio, file)