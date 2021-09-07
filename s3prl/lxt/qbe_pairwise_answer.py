import torch
import argparse
from collections import defaultdict
from pathlib import Path

from traitlets.traitlets import default

parser = argparse.ArgumentParser()
parser.add_argument("--query", required=True)
parser.add_argument("--doc", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

def read_file(filepath):
    mapping = {}
    with open(filepath) as file:
        lines = [line.strip() for line in file.readlines()]
        for line in lines:
            uid, text = line.split(" ", maxsplit=1)
            uid = uid.strip()
            text = text.strip()
            mapping[uid] = text
    return mapping

queries = read_file(args.query)
docs = read_file(args.doc)

def is_match(query, doc):
    query = query.split()
    doc = doc.split()
    for i in range(len(doc) - len(query) + 1):
        if doc[i : i + len(query)] == query:
            return True
    return False

docs_used_by = {doc_id: [] for doc_id in docs.keys()}
queries_used_by = {query_id: [] for query_id in queries.keys()}
match_pairs = torch.BoolTensor([len(queries), len(docs)])
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