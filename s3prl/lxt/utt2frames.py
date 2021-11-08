import argparse
import torchaudio
from librosa.util import find_files
from pathlib import Path
torchaudio.set_audio_backend("sox_io")

parser = argparse.ArgumentParser()
parser.add_argument("--lxt")
args = parser.parse_args()
files = find_files(args.lxt)

for file in files:
    frames = torchaudio.info(file).num_frames
    print(Path(file).stem, frames)