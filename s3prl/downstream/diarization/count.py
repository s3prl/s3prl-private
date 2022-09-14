data_path = "./data/lxt/"

#spk2utt, utt2spk
with open("../../data/superb_all/1.1_distributed/spk2utt", "r") as f:
    spk2utt = [x.split(" ") for x in f.read().split("\n")[:-1]]
    spk2utt = {x[0]+x[1]: set(x[2:]) for x in spk2utt}
    utt2spk = {}
    for k in spk2utt:
        for v in spk2utt[k]:
            utt2spk[v] = k
            
# duration
with open("../../data/superb_all/1.1_distributed/audio_duration.txt", 'r') as f:
    durations = [x.split(" ") for x in f.read().split("\n") if x]
    durations = {f.replace(".wav", ""): d for f, d in durations}
    for k in durations:
        durations[k] = float(durations[k])

splits = [
    "train",
    "dev",
    "test"
]

t = []
for split in splits:
    print(split)
    with open(data_path + split + "/wav.scp", 'r') as f:
        # file 1, file 2
        data = [x.split(maxsplit=1)[1].split("/")[-1].replace(".wav.wav", "").split(".wav_") for x in f.read().split("\n") if x]
        f1 = set(f for f, _ in data)
        f2 = set(f for _, f in data)
        utts = f1.union(f2)
        print("samples:   ", len(data))
        print("uniq utts: ", len(utts))
        print("spks:      ", sum(f in utt2spk for f in utts))
        print("")
        