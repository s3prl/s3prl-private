import os
import argparse
import soundfile as sf
import pandas as pd
import glob
from tqdm import tqdm

RATE = 16000

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('lxt_dir', type=str,
                    help='Lxt data directory')
parser.add_argument('metadata_dir', type=str,
                    help='Metadata directory')
parser.add_argument('--num_hours', default=1, type=int,
                    help='Number of hours')

def get_metadata(fname):
    utt2spk, utt2text = {}, {}
    with open(fname, 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split(',', 2)
        utt, spk, text = line_split[0], line_split[1], line_split[2]
        spk = "".join(spk.split()) 
        utt2spk[utt] = spk
        utt2text[utt] = text
    return utt2spk, utt2text 

def main(args):
    utt2spk, utt2text = get_metadata("{}/script.csv".format(args.lxt_dir))
    meta_dir = "{}/LXT{}hr".format(args.metadata_dir, args.num_hours)
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    for dset in ["train", "dev", "test"]:
        with open("{}/train_{}hr/{}.txt".format(args.lxt_dir, args.num_hours, dset), 'r') as fh:
            content = fh.readlines()
        uttlist = []
        for line in content:
            line = line.strip('\n')
            uttlist.append(line)
        with open("{}/{}.csv".format(meta_dir, dset), 'w') as fh:
            fh.write("speaker_ID,sex,subset,length,origin_path\n")
            record_list = []
            for utt in uttlist:
                spkid = utt2spk[utt]
                if spkid.startswith("male"):
                    sex = 'M'
                elif spkid.startswith("female"):
                    sex = 'F'
                else:
                    raise ValueError("Invalid condition.")
                subset = dset
                audio_path = "{}/audio/{}.wav".format(args.lxt_dir, utt)
                data, sr = sf.read(audio_path)
                assert sr == RATE
                length = len(data)
                origin_path = "audio/{}.wav".format(utt)
                record_list.append([spkid, sex, subset, length, origin_path])
            record_list = sorted(record_list, key=lambda x: x[3])
            for record in record_list:
                fh.write("{},{},{},{},{}\n".format(record[0], record[1], record[2], record[3], record[4]))
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
