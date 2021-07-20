import os
import torch
import shutil
import argparse
import torchaudio
from tqdm import tqdm
from shutil import copy
from pathlib import Path
from collections import defaultdict

LXT_SAMPLE_RATE = 44100

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--copy", action="store_true")
args = parser.parse_args()

delete = [
    "quac-C_1873c4f81ac24c6980a3235172ccebab_1_q#0-c-9.992(1).wav",  # duplicated
    "quac-C_1873c4f81ac24c6980a3235172ccebab_1_q#1-c-9.392(1).wav",  # duplicated
    "quac-C_1951db25f6204288ab8deda833d0c6de_0_q#5-c-7.355(1).wav",
    "quac-C_207b6af621e445289ae79262dfc3f3cb_0_q#0-c-0.208(1).wav",
    "quac-C_207b6af621e445289ae79262dfc3f3cb_0_q#0-c-11.458(1).wav",
    "quac-C_207b6af621e445289ae79262dfc3f3cb_0_q#10-c-11.320(1).wav",
]
rename = []
concat = []

src_dir = Path(args.lxt)
tgt_dir = Path(args.output)
assert src_dir.is_dir()
if tgt_dir.is_dir():
    shutil.rmtree(tgt_dir)
tgt_dir.mkdir()

src_preocessed = defaultdict(lambda: False)
for filename in delete:
    src_preocessed[filename] = True

for old_filename, new_filename in rename:
    src_preocessed[old_filename] = True
    (tgt_dir / new_filename).symlink_to((src_dir / old_filename).resolve())

for filename1, filename2, new_filename in concat:
    src_preocessed[filename1] = True
    src_preocessed[filename2] = True

    wav1, sr1 = torchaudio.load(src_dir / filename1)
    wav2, sr2 = torchaudio.load(src_dir / filename2)
    assert sr1 == sr2 == LXT_SAMPLE_RATE
    wav = torch.cat([wav1, wav2], dim=-1)
    torchaudio.save(str((tgt_dir / new_filename).resolve()), wav, LXT_SAMPLE_RATE)

for filename in src_dir.iterdir():
    if not src_preocessed[filename.name]:
        (tgt_dir / filename.name).symlink_to((src_dir / filename.name).resolve())

if args.copy:
    paths = list(tgt_dir.iterdir())
    for path in tqdm(paths, total=len(paths), desc="Copying file"):
        if path.is_symlink():
            src_path = os.readlink(str(path))
            tgt_path = str(path)
            path.unlink()
            copy(src_path, tgt_path)
