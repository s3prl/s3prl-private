data_path = "/mnt/diskc/superb-hidden/data/superb_all/1.1_distributed/"

# load durations
with open(data_path + "audio_duration.txt", 'r') as f:
    durations = [x.split(" ") for x in f.read().split("\n")[:-1]]
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

with open(data_path + "utt_with_valid_spk", 'r') as f:
    data = [x.split(" ") for x in f.read().split("\n") if len(x)]
    data = [(x[0], x[1]+x[2]) for x in data]
    print("utts:  ", len(data))
    print("spks:  ", len(set(x[1] for x in data)))