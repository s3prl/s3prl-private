import torch
import fairseq
import logging
import argparse
import s3prl.hub as hub

from prepare_model import prepare_model
from summarize import summarize



logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def get_profiling_args():
    parser = argparse.ArgumentParser()
    upstreams = [attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream',  help=""
        'Upstreams with \"_local\" or \"_url\" postfix need local ckpt (-k) or config file (-g). '
        'Other upstreams download two files on-the-fly and cache them, so just -u is enough and -k/-g are not needed. '
        'Please check upstream/README.md for details. '
        f"Available options in S3PRL: {upstreams}. "
    )
    return parser.parse_args()

def profiling(model):
    model = model.cuda().eval()
    model = prepare_model(model = model)
    
    wavs = [torch.randn(160000, dtype=torch.float).to("cuda") for _ in range(1)]
    model(wavs)
    summary = summarize(model = model)
    logger.info(
        summary.to_markdown()
    )

if __name__ == "__main__":
    args = get_profiling_args()
    model = getattr(hub, args.upstream)()
    profiling(model)
    
    # FIXME: not implement: ReplicationPad1d, SamePad, LayerNorm, Embedding, MelScale, Spectrogram, TransformerLayerNorm