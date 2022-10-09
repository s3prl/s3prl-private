data_path = "../../data/superb_all/1.1_distributed/"

# load durations
with open(data_path + "audio_duration.txt", 'r') as f:
    durations = [x.split(" ") for x in f.read().split("\n") if x]
    durations = {f.replace(".wav", ""): d for f, d in durations}
    for k in durations:
        durations[k] = float(durations[k])
        
# load spk2utt
with open(data_path + "spk2utt", "r") as f:
    spk2utt = [x.split(" ") for x in f.read().split("\n")[:-1]]
    spk2utt = {x[0]+x[1]: set(x[2:]) for x in spk2utt}
    utt2spk = {}
    for k in spk2utt:
        for v in spk2utt[k]:
            utt2spk[v] = k
            
# load transcription
# load spk2utt
with open(data_path + "normalized_transcription.txt", "r") as f:
    trans = [x.split(" ", maxsplit=1) for x in f.read().split("\n") if x]
    trans = {f: t for f, t in trans}

# alerm: the path is wrong for all, I think, previous experiments
splits = [
    "train_2hr/train/utt2spk",
    "train_2hr/dev/utt2spk",
    "train_2hr/test/utt2spk",
    "train_1hr_origin/train/utt2spk",
    "train_1hr_origin/dev/utt2spk",
    "train_1hr_origin/test/utt2spk"
]

t = []
tt = []
for split in splits:
    print("\nsplit: ", split)
    with open(data_path + split, 'r') as f:
        data = [x.split(" ") for x in f.read().split("\n") if len(x)]
        utts = len(data)
        durs = sum(durations[f] for f, s, n in data)
        spks = len(set(utt2spk[f] for f, s, n in data))
        spks2 = len(set(s+n for f, s, n in data))
        t.append(set((f,s,n) for f, s, n in data))
        tt.append(set(trans[f] for f, s, n in data))
        for f, s, n in data:
            if trans[f] == 'KEY SELLING POINTS WERE THE IMPROVEMENT IN PICTURE AND SOUND QUALITY INCREASED NUMBER OF CHANNELS AND AN INTERACTIVE SERVICE BRANDED OPEN':
                print(f)
        assert spks == spks2
        print("utts:  ", utts)
        print("durs:  ", durs)
        print("spks:  ", spks)

print(len(t[1].intersection(t[0])))
print(len(t[2].intersection(t[0])))
print(len(tt[1].intersection(tt[0])))
print(len(tt[2].intersection(tt[0])))
print(tt[2].intersection(tt[0]))
# with open(data_path + "train_1hr/dev/utt2spk", 'w') as f:
#     lines = [f"{x[0]} {x[1]} {x[2]}" for x in t[4].difference(t[0])]
#     f.write("\n".join(lines))
# with open(data_path + "train_1hr/test/utt2spk", 'w') as f:
#     lines = [f"{x[0]} {x[1]} {x[2]}" for x in t[5].difference(t[0])]
#     f.write("\n".join(lines))