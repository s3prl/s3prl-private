import argparse
from pathlib import Path
from lxml import etree

parser = argparse.ArgumentParser()
parser.add_argument("--xml", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
Path(args.output).parent.mkdir(exist_ok=True)

tree = etree.parse(args.xml)
with open(args.output, "w") as output:
    for detected_termlist in tree.getroot():
        termid, term_search_time, oov_term_count = detected_termlist.values()
        query_id = termid
        for term in detected_termlist:
            file, channel, tbeg, dur, score, decision = term.values()
            doc_id = file
            print(query_id, doc_id, score, file=output)
