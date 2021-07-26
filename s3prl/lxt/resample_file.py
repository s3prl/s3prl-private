import argparse

import torchaudio
torchaudio.set_audio_backend("sox_io")

LXT_SAMPLE_RATE = 44100
SAMPLE_RATE = 16000

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--tgt", required=True)
args = parser.parse_args()

resampler = torchaudio.transforms.Resample(LXT_SAMPLE_RATE, SAMPLE_RATE)
wav, sr = torchaudio.load(str(args.src))
assert sr == LXT_SAMPLE_RATE
wav = resampler(wav)
wav = wav.mean(dim=0, keepdim=True)
torchaudio.save(str(args.tgt), wav, SAMPLE_RATE)
