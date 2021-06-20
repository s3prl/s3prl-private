import random
import argparse
from sys import flags
import pandas as pd
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_pair", default=1000)
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()
random.seed(args.seed)

table = pd.read_csv(args.csv)
speaker_column_name = "speaker (gender/id)"
utterance_column_name = "utterance_id"
speakers = table[speaker_column_name].tolist()
utterances = table[utterance_column_name].tolist()
speaker_ids = sorted(list(set(speakers)))
selected_utterance_pairs = set()
speaker_pair_count = {(speaker1, speaker2): 0 for speaker1, speaker2 in product(speaker_ids, speaker_ids)}

def pick_utterance_pair(same_speaker: bool):
    if same_speaker:
        speaker1 = random.choice(speaker_ids)
        speaker2 = speaker1
    else:
        speaker1, speaker2 = random.sample(speaker_ids, k=2)
    speaker_pair_count[(speaker1, speaker2)] += 1

    utterance1 = table[table[speaker_column_name] == speaker1][utterance_column_name].tolist()
    utterance2 = table[table[speaker_column_name] == speaker2][utterance_column_name].tolist()
    pairs = product(utterance1, utterance2)
    pairs = [p for p in pairs if p[0] != p[1]]
    pairs = [p for p in pairs if p not in selected_utterance_pairs]
    selected_pair = random.choice(pairs)
    selected_utterance_pairs.add(selected_pair)
    return selected_pair

outputs = []
for i in range(args.num_pair):
    same_speaker = i % 2 == 0
    pair = pick_utterance_pair(same_speaker=same_speaker)    
    outputs.append((same_speaker, pair[0], pair[1]))

print("Counting statistics.")
same_speaker_count = {k: v for k, v in speaker_pair_count.items() if k[0] == k[1] and v > 0}
diff_speaker_count = {k: v for k, v in speaker_pair_count.items() if k[0] != k[1] and v > 0}
diff_speaker_same_gender_count = {k: v for k, v in diff_speaker_count.items() if k[0].split()[0] == k[1].split()[0]}
diff_speaker_diff_gender_count = {k: v for k, v in diff_speaker_count.items() if k[0].split()[0] != k[1].split()[0]}
for statistic_name in ["same_speaker_count", "diff_speaker_count", "diff_speaker_same_gender_count", "diff_speaker_diff_gender_count"]:
    statistic = eval(statistic_name)
    print(sum(statistic.values()))
    print(statistic_name)
    print(list(statistic.values()))

print("Saving output.")
output_lines = [f"{int(answer)} {id1}.wav {id2}.wav\n" for answer, id1, id2 in outputs]
with open(args.output, "w") as file:
    file.writelines(output_lines)
