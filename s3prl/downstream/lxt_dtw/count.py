
data_path = "../../data/superb_all/qbe_nonoverlap_above0/"

# load durations
with open(data_path + "queries.txt", 'r') as f:
    queries = len([x for x in f.read().split() if x])
    
with open(data_path + "docs.txt", 'r') as f:
    docs = len([x for x in f.read().split() if x])

print(queries)
print(docs)