import torch
import argparse
import torchaudio
from tqdm import tqdm
from librosa.util import find_files
torchaudio.set_audio_backend("sox_io")

parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
args = parser.parse_args()

files = find_files(args.audio)
secs = []
for file in tqdm(files):
    info = torchaudio.info(file)
    from ipdb import set_trace
    sec = info.num_frames / info.sample_rate
    secs.append(sec)

secs = torch.Tensor(secs)
print("Mean", secs.mean().item())
print("Std", secs.std().item())
print("Max", secs.max().item())
print("Min", secs.min().item())