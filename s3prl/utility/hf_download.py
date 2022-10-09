import argparse
from huggingface_hub import snapshot_download

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--org', type=str, help='organization name')
    parser.add_argument('-r', '--repo', type=str, help='repo name')
    parser.add_argument('-c', '--commit', type=str, help="commit revision")
    return parser.parse_args()

args = getargs()
filepath = snapshot_download(
    repo_id = f"{args.org}/{args.repo}",
    revision = args.commit,
    use_auth_token = True
)