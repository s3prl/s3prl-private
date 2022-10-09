data_path = "/mnt/diskc/superb-hidden/data/superb_all/"

# load durations
with open(data_path + "pr_duration.txt", 'r') as f:
    durations = [x.split(" ") for x in f.read().split("\n") if x]
    durations = {f.replace(".wav", ""): d for f, d in durations}
    for k in durations:
        durations[k] = float(durations[k])
        
# # load spk2utt
# with open(data_path + "spk2utt", "r") as f:
#     spk2utt = [x.split(" ") for x in f.read().split("\n")[:-1]]
#     spk2utt = {x[0]+x[1]: set(x[2:]) for x in spk2utt}
#     utt2spk = {}
#     for k in spk2utt:
#         for v in spk2utt[k]:
#             utt2spk[v] = k

splits = ["train", "dev", "test"]

for split in splits:
    print("split: ", split)
    with open(data_path + "er_train_0.2/" + split + ".txt", 'r') as f:
        data = [x.split(" ") for x in f.read().split("\n") if len(x)]
        utts = len(data)
        durs = sum(durations[f] for f, _ in data)
        emos = set(e for _, e in data)
        print("utts:  ", utts)
        print("durs:  ", durs)
        print("emos:", emos)
        