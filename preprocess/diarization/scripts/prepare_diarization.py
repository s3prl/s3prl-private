import collections
import contextlib
import sys
import wave

import os
import re
import argparse


import webrtcvad


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, vad, frames):

    frame_vad = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        frame_vad.append((1 if is_speech else 0))
    return frame_vad


def get_start_end(wavpath):
    audio, sample_rate = read_wave(wavpath)
    vad = webrtcvad.Vad(1)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, vad, frames)
    assert 1 in segments
    start = segments.index(1) * 30 / 1000
    segments.reverse()
    end = (len(segments) - segments.index(1) - 1) * 30 / 1000
    return start, end


def float2str(number, size=6):
    number = str(int(number * 1000))
    return (size - len(number)) * "0" + number


def process_metadata(metadata, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    wavscp = open(os.path.join(target_dir, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(target_dir, "utt2spk"), "w", encoding="utf-8")
    spk2utt = open(os.path.join(target_dir, "spk2utt"), "w", encoding="utf-8")
    segments = open(os.path.join(target_dir, "segments"), "w", encoding="utf-8")
    rttm = open(os.path.join(target_dir, "rttm"), "w", encoding="utf-8")
    reco2dur = open(os.path.join(target_dir, "reco2dur"), "w", encoding="utf-8")

    spk2utt_cache = {}

    with open(metadata, "r", encoding="utf-8") as f:
        header = f.readline().split(",")
        assert len(header) == 6
        for linenum, line in enumerate(f, 1):
            mix_id, mix_path, source1_wav, source2_wav, _, length = line.strip().split(",")

            split_id = mix_id.find(".wav_") + 4
            source1_id, source2_id = mix_id[:split_id], mix_id[split_id + 1:]
            spk1, spk2 = source1_id.split("-")[1], source2_id.split("-")[1]
            reco1, reco2 = source1_id, source2_id
            wavscp.write("{} {}\n".format(mix_id, mix_path))
            spk1_segs, spk2_segs = get_start_end(source1_wav), get_start_end(source2_wav)

            # spk1_segs
            start, end = spk1_segs
            seg_id = "{}_{}_{}".format(mix_id, float2str(start), float2str(end))
            segments.write("{} {} {} {}\n".format(seg_id, mix_id, start, end))
            utt2spk.write("{} {}\n".format(seg_id, spk1))
            rttm.write("SPEAKER\t{}\t1\t{}\t{}\t<NA>\t<NA>\t{}\t<NA>\n".format(mix_id, start, end-start, spk1))
            spk2utt_cache[spk1] = spk2utt_cache.get(spk1, []) + [mix_id]

            # spk2_segs
            start, end = spk2_segs
            seg_id = "{}_{}_{}".format(mix_id, float2str(start), float2str(end))
            segments.write("{} {} {} {}\n".format(seg_id, mix_id, start, end))
            utt2spk.write("{} {}\n".format(seg_id, spk2))
            rttm.write("SPEAKER\t{}\t1\t{}\t{}\t<NA>\t<NA>\t{}\t<NA>\n".format(mix_id, start, end-start, spk2))
            spk2utt_cache[spk2] = spk2utt_cache.get(spk2, []) + [mix_id]

            reco2dur.write("{} {}\n".format(mix_id, float(length) / 16000))

    for spk_id in spk2utt_cache.keys():
        spk2utt.write("{} {}\n".format(spk_id, " ".join(spk2utt_cache[spk_id])))

    wavscp.close()
    utt2spk.close()
    spk2utt.close()
    segments.close()
    rttm.close()
    reco2dur.close()

    

parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, required=True, help='Path to generate kaldi_style result')
parser.add_argument('--source_dir', type=str, default="Libri2Mix/wav16k/max/metadata")

args = parser.parse_args()

process_metadata(os.path.join(args.source_dir, "mixture_tr-split_train_1hr_mix_both.csv"), os.path.join(args.target_dir, "train"))
process_metadata(os.path.join(args.source_dir, "mixture_cv-split_train_1hr_mix_both.csv"), os.path.join(args.target_dir, "dev"))
process_metadata(os.path.join(args.source_dir, "mixture_tt-split_train_1hr_mix_both.csv"), os.path.join(args.target_dir, "test"))

print("Successfully finish Kaldi-style preparation")
