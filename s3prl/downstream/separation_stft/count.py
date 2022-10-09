import os
data_path = "datasets/LXT2Mix2hr5repeat/wav16k/min/"

splits = ["train", "dev", "test"]

for split in splits:
    print(split)
    with open(data_path + split + "/mix_clean/wav.scp", 'r') as f:
        print(len([x for x in f.read().split("\n") if x]))
        print()