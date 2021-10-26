import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--query", default="data/superb_all/queries.txt")
parser.add_argument("--doc", default="data/superb_all/doc.txt")
parser.add_argument("--doc_uids", default="data/superb_all/1.1_distributed/utt2spk")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--trim_doc", action="store_true")
parser.add_argument("--trim_query", action="store_true")
parser.add_argument("--doc_num", type=int, default=100)
parser.add_argument("--query_min_doc", type=int, default=1)
args = parser.parse_args()

def read_file(filepath, whitelist=None):
    mapping = {}
    with open(filepath) as file:
        lines = [line.strip() for line in file.readlines()]
        for line in lines:
            uid, text = line.split(" ", maxsplit=1)
            uid = uid.strip()
            text = text.strip()
            if whitelist is not None and not uid in whitelist:
                continue
            mapping[uid] = text
    return mapping

queries = read_file(args.query)
query_text2uid = {}
for uid, text in queries.items():
    text = tuple(text.split())
    query_text2uid[text] = uid
queries = {uid: " ".join(text) for text, uid in query_text2uid.items()}

doc_uids = None
if args.doc_uids is not None:
    doc_uids = [line.strip().split()[0] for line in open(args.doc_uids).readlines()]
docs = read_file(args.doc, doc_uids)

def is_match(query, doc):
    query = query.split()
    doc = doc.split()
    for i in range(len(doc) - len(query) + 1):
        if doc[i : i + len(query)] == query:
            return True
    return False

if args.trim_doc:
    docs_used_by = {doc_id: [] for doc_id in docs.keys()}
    for query_id, query_text in queries.items():
        for doc_id, doc_text in docs.items():
            match = is_match(query_text, doc_text)
            if match:
                docs_used_by[doc_id].append(query_id)

    docs_id_sorted = sorted(list(docs.keys()), key=lambda k: len(docs_used_by[k]), reverse=True)
    sub_docs = {doc_id: docs[doc_id] for doc_id in docs_id_sorted[:args.doc_num]}
    docs = sub_docs

queries_used_by = {query_id: [] for query_id in queries.keys()}
for query_id, query_text in queries.items():
    for doc_id, doc_text in docs.items():
        match = is_match(query_text, doc_text)
        if match:
            queries_used_by[query_id].append(doc_id)

for query_id in list(queries.keys()):
    if len(queries_used_by[query_id]) < args.query_min_doc:
        queries.pop(query_id)
        queries_used_by.pop(query_id)

if args.trim_query:
    queries_used_by_num = [len(v) for v in queries_used_by.values()]
    q1, q3 = np.percentile(queries_used_by_num, (25, 75))
    iqr = q3 - q1
    minimum = q1 - 1.5 * iqr
    maximum = q3 + 1.5 * iqr
    sub_queries = {query_id: queries[query_id] for query_id in queries.keys() if (len(queries_used_by[query_id]) <= maximum and len(queries_used_by[query_id]) >= minimum)}
    queries = sub_queries

docs_used_by = {doc_id: [] for doc_id in docs.keys()}
queries_used_by = {query_id: [] for query_id in queries.keys()}
for query_id, query_text in queries.items():
    for doc_id, doc_text in docs.items():
        match = is_match(query_text, doc_text)
        if match:
            docs_used_by[doc_id].append(query_id)
            queries_used_by[query_id].append(doc_id)

queries_id_sorted = sorted(list(queries.keys()), key=lambda k: len(queries_used_by[k]), reverse=True)
docs_id_sorted = sorted(list(docs.keys()), key=lambda k: len(docs_used_by[k]), reverse=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

with (output_dir / "detail.txt").open("w") as output:
    for idx, query_id in enumerate(queries_id_sorted):
        print(f"QUERY {idx}", file=output)
        print(query_id, queries[query_id], file=output)
        related_docs = queries_used_by[query_id]
        print(f"DOCS: {len(related_docs)} found.", file=output)
        for doc_id in related_docs:
            print(doc_id, docs[doc_id], file=output)
        print(file=output)

with (output_dir / "answer.txt").open("w") as output:
    for query_id in queries_id_sorted:
        query_text = queries[query_id]
        for doc_id in docs_id_sorted:
            doc_text = docs[doc_id]
            match = is_match(query_text, doc_text)
            print(query_id, doc_id, int(match), file=output)

with (output_dir / "queries.txt").open("w") as file:
    for query in queries_id_sorted:
        print(query, file=file)

with (output_dir / "docs.txt").open("w") as file:
    for doc in docs_id_sorted:
        print(doc, file=file)