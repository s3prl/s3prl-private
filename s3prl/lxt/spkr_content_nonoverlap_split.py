import argparse
import torchaudio
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
torchaudio.set_audio_backend("sox_io")

parser = argparse.ArgumentParser()
parser.add_argument("--spkr_uids", required=True)
parser.add_argument("--audio", required=True)
parser.add_argument("--dev", type=int, default=11)
parser.add_argument("--test", type=int, default=11)
parser.add_argument("--train", type=int, default=4)
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

spkr_uids = eval(open(args.spkr_uids).read())

def get_secs(uid):
    path = Path(args.audio) / f"{uid}.wav"
    assert path.exists(),  path
    return torchaudio.info(path).num_frames / 16000

def spkr_secs(spkr):
    sec = 0
    for uid in spkr_uids[spkr]:
        sec += get_secs(uid)
    return sec

start = 66666666666666666
end = 888888888888888888888888
assert start < end
selections = []

for seed in tqdm(range(start, end, round((end - start) / args.trials))):
    random.seed(seed)
    male_spkrs = [f"male {i}" for i in range(1, 31)]
    female_spkrs = [f"female {i}" for i in range(1, 31)]
    random.shuffle(male_spkrs)
    random.shuffle(female_spkrs)

    dev_end = args.dev
    test_end = args.dev + args.test
    train_end = args.dev + args.test + args.train
    dev_spkrs = male_spkrs[ : dev_end] + female_spkrs[ : dev_end]
    test_spkrs = male_spkrs[dev_end : test_end] + female_spkrs[dev_end : test_end]
    train_spkrs = male_spkrs[test_end : train_end] + female_spkrs[test_end : train_end]

    dev_secs = sum([spkr_secs(spkr) for spkr in dev_spkrs])
    test_secs = sum([spkr_secs(spkr) for spkr in test_spkrs])
    train_secs = sum([spkr_secs(spkr) for spkr in train_spkrs])

    std = np.std([dev_secs / args.dev, test_secs / args.test, train_secs / args.train])
    selections.append([std, [dev_spkrs, test_spkrs, train_spkrs]])

selections.sort(key=lambda x: x[0])
dev_spkrs, test_spkrs, train_spkrs = selections[0][1]

dev_secs = sum([spkr_secs(spkr) for spkr in dev_spkrs])
test_secs = sum([spkr_secs(spkr) for spkr in test_spkrs])
train_secs = sum([spkr_secs(spkr) for spkr in train_spkrs])

dev_nutt = [len(spkr_uids[spkr]) for spkr in dev_spkrs]
test_nutt = [len(spkr_uids[spkr]) for spkr in test_spkrs]
train_nutt= [len(spkr_uids[spkr]) for spkr in train_spkrs]

print("dev", dev_secs / args.dev, dev_secs/60/60, len(dev_spkrs), np.mean(dev_nutt), np.std(dev_nutt), sep="\t")
print("test", test_secs / args.test, test_secs/60/60, len(test_spkrs), np.mean(test_nutt), np.std(test_nutt), sep="\t")
print("train", train_secs / args.train, train_secs/60/60, len(train_spkrs), np.mean(train_nutt), np.std(train_nutt), sep="\t")

Path(args.output_dir).mkdir(exist_ok=True)
def output_split(split):
    split_dir = Path(args.output_dir) / split
    split_dir.mkdir(exist_ok=True)
    spk2utt = (split_dir / "spk2utt").open("w")
    utt2spk = (split_dir / "utt2spk").open("w")
    utt = (split_dir / "utt").open("w")

    spkrs = eval(f"{split}_spkrs")
    uids = []
    for spkr in spkrs:
        single_spkr_uids = spkr_uids[spkr]
        print(spkr, " ".join(single_spkr_uids), file=spk2utt)
        for uid in single_spkr_uids:
            print(uid, spkr, file=utt2spk)
            print(uid, file=utt)

output_split("dev")
output_split("test")
output_split("train")