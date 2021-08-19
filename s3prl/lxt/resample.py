import os
import shutil
import argparse
import torchaudio
from tqdm import tqdm
from pathlib import Path
from librosa.util import find_files
torchaudio.set_audio_backend("sox_io")

LXT_SAMPLE_RATE = 44100
SAMPLE_RATE = 16000

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", required=True)
parser.add_argument("--tgt_dir", required=True)
args = parser.parse_args()

src_dir = Path(args.src_dir)
tgt_dir = Path(args.tgt_dir)
if tgt_dir.is_dir():
    shutil.rmtree(tgt_dir)
tgt_dir.mkdir()

files = find_files(args.src_dir)
for file in tqdm(files):
    file = Path(file)
    if file.is_symlink():
        file = Path(os.readlink(file))

    wav, sr = torchaudio.load(str(file))
    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    if sr != LXT_SAMPLE_RATE:
        print(f"{file}: {sr}")
    wav = resampler(wav)
    wav = wav[0:1]

    src_path = Path(file)
    tgt_path = tgt_dir.joinpath(src_path.resolve().relative_to(src_dir.resolve())).resolve()
    torchaudio.save(str(tgt_path), wav, SAMPLE_RATE)
