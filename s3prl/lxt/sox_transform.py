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

    effects = [
        ["channels", "1"],
        ["gain", "-n", "-3.0"],
        ["silence", "-l", "1", "0.2", "0.5%", "-1", "0.5", "0.5%"],
        ["gain", "-n", "-3.0"],
    ]

    wav, sr = torchaudio.sox_effects.apply_effects_file(str(file), effects)
    if wav.size(-1) < 16000 * 0.5:
        print(f"{file} is pass, length {wav.size(-1)}")
        continue
    src_path = Path(file)
    tgt_path = tgt_dir.joinpath(src_path.resolve().relative_to(src_dir.resolve())).resolve()
    torchaudio.save(str(tgt_path), wav, sr)
