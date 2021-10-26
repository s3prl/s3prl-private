import os
import argparse
import csv
import soundfile as sf
import pandas as pd
import glob
from tqdm import tqdm

# Global parameter
# We will filter out files shorter than that
NUMBER_OF_SECONDS = 1
# In hidden set all the sources are at 16K Hz
RATE = 16000

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_set_dir', type=str, required=True,
                    help='Path to hidden_set root directory')


def main(args):
    hidden_set_dir = args.hidden_set_dir
    # Create hidden_set metadata directory
    hidden_set_md_dir = os.path.join(hidden_set_dir, 'metadata')
    os.makedirs(hidden_set_md_dir, exist_ok=True)
    create_hidden_set_metadata(hidden_set_dir, hidden_set_md_dir)


def create_hidden_set_metadata(hidden_set_dir, md_dir):
    """ Generate metadata corresponding to downloaded data in hidden_set """
    # Get speakers metadata
    speakers_metadata = create_speakers_dataframe(hidden_set_dir)

    # Go through each directory and create associated metadata
    for ldir in ["split_train_1hr", "split_train_5hr"]:
        # Generate the dataframe relative to the directory
        dir_md_tr, dir_md_cv, dir_md_tt = create_hidden_set_dataframe(hidden_set_dir, ldir,
                                                    speakers_metadata)
        for dir_metadata, label in zip([dir_md_tr, dir_md_cv, dir_md_tt], ["tr", "cv", "tt"]):
            # Filter out files that are shorter than 3s
            num_samples = NUMBER_OF_SECONDS * RATE
            dir_metadata = dir_metadata[
                dir_metadata['length'] >= num_samples]
            # Sort the dataframe according to ascending Length
            dir_metadata = dir_metadata.sort_values('length')
            # Write the dataframe in a .csv in the metadata directory
            save_path = os.path.join(md_dir, "{}-".format(label) + ldir + '.csv')
            dir_metadata.to_csv(save_path, index=False)

def create_speakers_dataframe(hidden_set_dir):
    """ Read metadata from the hidden_set dataset and collect infos
    about the speakers """
    print("Reading speakers metadata")
    # Read script.csv and create a dataframe
    speakers_metadata_path = os.path.join(hidden_set_dir, 'script.csv')
    speakers_metadata = {}
    with open(speakers_metadata_path, "r", encoding="utf-8") as md:
        reader = csv.reader(md)
        for row in reader:
            if row[0] == "utterance_id":
                continue # skip the first line
            utterance_id, speaker_info, text = row
            spk_id = utterance_id.split("-")[1]
            if speaker_info[0] == "f":
                speakers_metadata[spk_id] = 0
            else:
                speakers_metadata[spk_id] = 1

    return speakers_metadata


def get_split_metadata(subdir):
    train_file = open(os.path.join(subdir, "train.txt"), "r", encoding="utf-8")
    dev_file = open(os.path.join(subdir, "dev.txt"), "r", encoding="utf-8")
    test_file = open(os.path.join(subdir, "test.txt"), "r", encoding="utf-8")
    split_meta = {}

    for tr_utt in train_file.read().split("\n"):
        split_meta[tr_utt] = "tr"
    for dev_utt in dev_file.read().split("\n"):
        split_meta[dev_utt] = "cv"
    for test_utt in test_file.read().split("\n"):
        split_meta[test_utt] = "tt"

    train_file.close(), dev_file.close(), test_file.close()
    return split_meta



def create_hidden_set_dataframe(hidden_set_dir, subdir, speakers_md):
    """ Generate a dataframe that gather infos about the sound files in a
    hidden_set subdirectory """

    print(f"Creating {subdir} metadata file in "
          f"{os.path.join(hidden_set_dir, 'metadata')}")
    # Get the current directory path
    dir_path = os.path.join(hidden_set_dir, subdir)
    # Look for wav file in "audio"
    sound_paths = os.listdir(os.path.join(hidden_set_dir, "audio"))

    split_md = get_split_metadata(dir_path)

    # Create the dataframe corresponding to this directory
    dir_md_tr = pd.DataFrame(columns=['speaker_ID', 'sex', 'subset',
                                   'length', 'origin_path'])
    dir_md_cv = pd.DataFrame(columns=['speaker_ID', 'sex', 'subset',
                                   'length', 'origin_path'])
    dir_md_tt = pd.DataFrame(columns=['speaker_ID', 'sex', 'subset',
                                   'length', 'origin_path'])

    # Go through the sound file list
    for sound_path in tqdm(sound_paths, total=len(sound_paths)):
        abs_path = os.path.join(hidden_set_dir, "audio", sound_path)
        utt_id = sound_path[:-4] # remove .wav extension
        # Get the ID of the speaker
        spk_id = sound_path.split("-")[1]
        # Find Sex according to speaker ID in hidden_set metadata
        sex = speakers_md[spk_id]
        # Find subset according to speaker ID in hidden_set metadata
        subset = split_md[utt_id]
        # Get its length
        length = len(sf.SoundFile(abs_path))
        # Get the sound file relative path
        rel_path = os.path.relpath(abs_path, hidden_set_dir)
        # Add information to the dataframe
        if subset == "tr":
            dir_md_tr.loc[len(dir_md_tr)] = [spk_id, sex, subset, length, rel_path]
        elif subset == "cv":
            dir_md_cv.loc[len(dir_md_cv)] = [spk_id, sex, subset, length, rel_path]
        else:
            dir_md_tt.loc[len(dir_md_tt)] = [spk_id, sex, subset, length, rel_path]
    return dir_md_tr, dir_md_cv, dir_md_tt


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
