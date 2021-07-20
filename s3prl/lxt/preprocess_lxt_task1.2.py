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
    "emotionlines-train-9-22.wav",  # no transcription
    "emotionlines-train-307-21.wav",  # no transcription
    "emotionlines-train-308-19.wav",  # no transcription
    "emotionlines-train-182-15.wav",  # no transcription
    "emotionlines-train-418-20.wav",  # no transcription
    "emotionlines-train-374-9.wav",  # no transcription
    "emotionlines-train-1-15.wav",  # no recording
    "emotionlines-train-25-2.wav",  # no recording
    "emotionlines-train-325-6.wav",  # no recording
    "emotionlines-train-346-4.wav",  # no recording
    "emotionlines-train-308-1.wav",  # no recording
    "emotionlines-train-418-14.wav",  # wrong recording
    "emotionlines-train-308-1.wav",  # wrong recording
]
rename = [
    ("emotionlines-train-154-10.wav", "emotionlines-train-547-18.wav"),
    ("emotionlines-train-154-10(1).wav", "emotionlines-train-154-10.wav"),
    ("emotionlines-train-182-13(1).wav", "emotionlines-train-182-12.wav"),
    ("emotionlines-train-182-4(1).wav", "emotionlines-train-182-5.wav"),
    ("emotionlines-train-246-6.wav", "emotionlines-train-694-6.wav"),
    ("emotionlines-train-246-6(1).wav", "emotionlines-train-246-6.wav"),
    ("emotionlines-train-266-15(1).wav", "emotionlines-train-266-15.wav"),
    ("emotionlines-train-266-15.wav", "emotionlines-train-266-17.wav"),
    ("emotionlines-train-307-5(1).wav", "emotionlines-train-307-7.wav"),
    ("emotionlines-train-308-1(1).wav", "emotionlines-train-308-1.wav"),
    ("emotionlines-train-38-2.wav", "emotionlines-train-680-4.wav"),
    ("emotionlines-train-38-2(1).wav", "emotionlines-train-38-2.wav"),
    ("emotionlines-train-423-15.wav", "emotionlines-train-423-16.wav"),
    ("emotionlines-train-423-15(1).wav", "emotionlines-train-423-15.wav"),
    ("emotionlines-train-469-13.wav", "emotionlines-train-655-10.wav"),
    ("emotionlines-train-469-13(1).wav", "emotionlines-train-469-13.wav"),
    ("emotionlines-train-519-14.wav", "emotionlines-train-519-12.wav"),
    ("emotionlines-train-519-14(1).wav", "emotionlines-train-519-14.wav"),
    ("emotionlines-train-519-21_.wav", "emotionlines-train-519-21.wav"),
    ("emotionlines-train-246-10_.wav", "emotionlines-train-246-10.wav"),
    ("emotionlines-train-246-4_.wav", "emotionlines-train-246-4.wav"),
    ("emotionlines-train-145-16(1).wav", "emotionlines-train-322-16.wav")
]
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
