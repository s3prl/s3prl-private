import os
import shutil
import argparse
import torchaudio
from tqdm import tqdm
from pathlib import Path
from librosa.util import find_files
torchaudio.set_audio_backend("sox_io")

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", required=True)
parser.add_argument("--tgt_dir", required=True)
parser.add_argument("--decibel", type=float, default=-25)
args = parser.parse_args()

src_dir = Path(args.src_dir)
tgt_dir = Path(args.tgt_dir)
if tgt_dir.is_dir():
    shutil.rmtree(tgt_dir)
tgt_dir.mkdir()

def normalize_wav_decibel(wav):
    '''Normalize the signal to the target level'''
    rms = wav.pow(2).mean().pow(0.5)
    scalar = (10 ** (args.decibel / 20)) / (rms + 1e-10)
    wav = wav * scalar
    return wav

files = find_files(args.src_dir)
for file in tqdm(files):
    file = Path(file)
    if file.is_symlink():
        file = Path(os.readlink(file))

    wav, sr = torchaudio.load(str(file))
    wav = wav.view(-1)
    wav = normalize_wav_decibel(wav)

    src_path = Path(file)
    tgt_path = tgt_dir.joinpath(src_path.resolve().relative_to(src_dir.resolve())).resolve()
    torchaudio.save(str(tgt_path), wav.view(1, -1), sr)
