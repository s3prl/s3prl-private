import torch
import argparse
import s3prl.hub as hub

parser = argparse.ArgumentParser()
parser.add_argument("--upstream", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--key", default="qbe")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

upstream = getattr(hub, args.upstream)().to(args.device)
upstream.eval()
with torch.no_grad():
    result = upstream([torch.randn(16000).to(args.device)])
    key = args.key
    if key not in result:
        key = "hidden_states"

    qbe_representation = result[key]
    num_layer = len(qbe_representation)

with open(args.output, "w") as file:
    print(num_layer, file=file)
