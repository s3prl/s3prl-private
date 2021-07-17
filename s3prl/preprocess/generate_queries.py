import os
import random
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_sample", type=int, default=1000)
parser.add_argument("--max_gram", type=int, default=5)
parser.add_argument("--min_chars", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)

male_spkr = ["male a", "male b", "male c", "male d", "male e"]
female_spkr = ["female a", "female b", "female c", "female d", "female e"]
gender_pairs = ["male-female", "female-male", "male-male", "female-female"]

table = pd.read_csv(args.csv)
uid_clm = "utterance_id"
spkr_clm = "speaker (gender/id)"
text_clm = "utterance_text"

final_pairs = []
for query_id in tqdm(range(args.num_sample)):
    gender1, gender2 = random.choice(gender_pairs).split("-")
    subtable = table[table[spkr_clm].apply(lambda spkr: spkr.split()[0]) == gender1]
    rowid1, rowid2 = random.sample(range(len(subtable)), k=2)
    assert rowid1 != rowid2

    def get_uid_text(rowid):
        row = subtable.iloc[rowid]
        uid = row[uid_clm]
        text = row[text_clm]
        text = " ".join(text.split())
        return uid, text

    uid1, text1 = get_uid_text(rowid1)
    uid2, text2 = get_uid_text(rowid2)
    assert uid1 != uid2
    assert text1 != text2

    def sample_query(context):
        words = context.split()
        gram = random.randint(1, min(args.max_gram, len(words)))
        start_id = random.randint(0, len(words) - gram)
        query_words = words[start_id : start_id + gram]
        query_text = " ".join(query_words)
        return query_text
    
    def sample_query_long_enough(context, min_chars=args.min_chars):
        if len(context) < 10:
            print(context)
            from ipdb import set_trace
            set_trace()
        query = sample_query(context)
        while len(query) < min_chars:
            query = sample_query(context)
        return query

    def sample_pair(match: bool):
        if match:
            return uid1, text1, f"{uid1}.{query_id}", sample_query_long_enough(text1)
        else:
            return uid1, text1, f"{uid2}.{query_id}", sample_query_long_enough(text2)

    match = query_id % 2 == 0
    while True:
        context_uid, context, query_uid, query = sample_pair(match)
        if match and (query in context):
            break
        if not match and (query not in context):
            break

    context_spkr = subtable.iloc[rowid1][spkr_clm]
    query_spkr = random.choice(eval(f"{gender2}_spkr"))
    final_pairs.append((match, context_uid, context_spkr, context, query_uid, query_spkr, query))

matches, context_uids, context_spkrs, contexts, query_ids, query_spkrs, queries = list(zip(*final_pairs))

os.makedirs(args.output_dir, exist_ok=True)

df_internal = pd.DataFrame(data={
    "match": matches,
    "context_uids": context_uids,
    "context_spkrs": context_spkrs,
    "contexts": contexts,
    "query_ids": query_ids,
    "query_spkrs": query_spkrs,
    "queries": queries,
})
df_internal.to_csv(Path(args.output_dir) / "task2_internal.csv", index=False)

df_for_lxt = pd.DataFrame(data={
    uid_clm: query_ids,
    spkr_clm: query_spkrs,
    text_clm: queries,
})
df_for_lxt.to_csv(Path(args.output_dir) / "task2_lxt.csv", index=False)
