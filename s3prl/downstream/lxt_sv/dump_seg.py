import os
import argparse
import torchaudio
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("seg_list", help="label utt1 utt2")
parser.add_argument("lxt")
parser.add_argument("outdir")
args = parser.parse_args()

outdir = Path(args.outdir)
os.makedirs(outdir, exist_ok=True)

lxt = Path(args.lxt)
for line in open(args.seg_list).readlines():
    line = line.strip()
    label, seg1, seg2 = line.split()
    
    def get_timestamps(seg):
        start, end = seg.split("_")[-2:]
        utt = "_".join(seg.split("_")[:-2])
        return utt, int(start), int(end)
    
    def get_segment(seg):
        utt, start, end = get_timestamps(seg)
        wav, sr = torchaudio.load(str(lxt / f"{utt}.wav"))
        return wav[:, start:end]
    
    torchaudio.save(str(outdir / f"{seg1}.wav"), get_segment(seg1), 16000)
    torchaudio.save(str(outdir / f"{seg2}.wav"), get_segment(seg2), 16000)
