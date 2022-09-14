data_path = "../../data/superb_all/1.1_distributed/train_5hr"

splits = [
    "/dev/sv_seg_5000.lst",
    "/test/sv_seg_5000.lst",
    "/train/utt2spk_pruned"
]

t = []
for split in splits[:2]:
    print(split)
    with open(data_path + split, 'r') as f:
        # bool value, audio1, audio2
        data = [x.split() for x in f.read().split("\n") if x]
        f1 = set(f for _, f, _ in data)
        f2 = set(f for _, _, f in data)
        t.append(f1.union(f2))
        print("sample:  ", len(data))
        print("file1:   ", len(f1))
        print("file2:   ", len(f1))
        print("all file:", len(t[-1]))
        print("")
        
split = splits[2]
print(split)
with open(data_path + split, 'r') as f:
    # bool value, audio1, audio2
    data = [x.split(maxsplit=1) for x in f.read().split("\n") if x]
    utts = set(f for f, s in data)
    spks = set(s for f, s in data)
    t.append(utts)
    print("utts:  ", len(utts))
    print("spks:  ", len(spks))
    print("")

print(t[0].intersection(t[1]))
print(t[0].intersection(t[2]))
print(t[1].intersection(t[2]))