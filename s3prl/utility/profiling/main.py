import torch
import fairseq
import logging
import argparse
import s3prl.hub as hub

from prepare_model import prepare_model
from summarize import summarize
from deepspeed.profiling.flops_profiler import get_model_profile



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

def profiling(model_func, args):
    model = model_func().cuda().eval()
    model = prepare_model(model = model)
    
    wavs = [torch.randn(160000, dtype=torch.float).to("cuda") for _ in range(1)]
    model(wavs)
    summary = summarize(model = model)
    logger.info(
        summary.to_markdown()
    )
    # modified_cpc: ChannelNorm, LSTM
    # apc: CMVN, GRU
    # pase_plus:
    # wav2vec: ReplicationPad1d
    # vq_wav2vec: GumbelVectorQuantizer, ReplicationPad1d
    # mockingjay: MelScale, Spectrogram, TransformerLayerNorm
    # cpc: not in hub
    # vq_apc: CMVN, GRU
    # audio_albert: MelScale, Spectrogram, TransformerLayerNorm
    # wav2vec2: LayerNorm
    # tera: MelScale, Spectrogram, TransformerLayerNorm
    # npc: BatchNorm1d, CMVN
    # hubert: LayerNorm
    # decoar: CMVN, LSTM
    # decoar2: MHA, LayerNorm
    # distilhubert: LayerNorm, SplitLinear
    # wavlm: LayerNorm, SamPad, Embedding
    
    # ChannelNorm, TransformerLayerNorm, LayerNorm
    # LSTM, GRU, MHA, SplitLinear, Embedding
    # FeatureExtractor, GumbelVectorQuantizer, MelScale, Spectrogram, CMVN
    # ReplicationPad1d, SamPad
    
    
    
def myprofiling(model_func, args):
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=10,
            active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet50'),
        with_flops=True,
        record_shapes=False,
        with_stack=False,
        profile_memory=True,
    ) as p:
        model = model_func().cuda().eval()
        for _ in range(16):
            # wavs = [torch.randn(160000, dtype=torch.float, device="cuda") for _ in range(2)]
            wavs = torch.randn((64,3,224,224), dtype=torch.float, device="cuda")
            model(wavs)
            p.step()
    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    # p.export_chrome_trace("./hubert.json")

def deepspeedProfiling(model_func, args, batch_size=1):
    def s3prl_input_constructor(batch_size=1, seq_len=160000):
        return [torch.randn(seq_len, dtype=torch.float) for _ in range(batch_size)]
    
        
    with torch.cuda.device(0):
        model = model_func().eval()
        flops, macs, params = get_model_profile(model=model, # model
                                    # input_shape=(1, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=[s3prl_input_constructor(batch_size)], # list of positional arguments to the model.
                                    # kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=3, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file="./{}_deepspeed.txt".format(args.upstream), # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling

if __name__ == "__main__":
    args = get_profiling_args()
    model_func = getattr(hub, args.upstream)
    # from torchvision.models import resnet50
    # model_func = resnet50
    profiling(model_func, args)
    # myprofiling(model_func)
    
    # FIXME: not implement: ReplicationPad1d, SamePad, LayerNorm, Embedding, MelScale, Spectrogram, TransformerLayerNorm