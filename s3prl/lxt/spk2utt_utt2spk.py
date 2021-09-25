import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--spkr_uids", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

Path(args.output_dir).mkdir(exist_ok=True)
spkr_uids = eval(open(args.spkr_uids).read())

spk2utt = (Path(args.output_dir) / "spk2utt").open("w")
utt2spk = (Path(args.output_dir) / "utt2spk").open("w") 
for spkr, uids in spkr_uids.items():
    print(spkr, " ".join(uids), file=spk2utt)
    for uid in uids:
        print(uid, spkr, file=utt2spk)
