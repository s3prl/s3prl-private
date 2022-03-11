import torch
import fairseq
import logging
import s3prl


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def compute_Conv1d_madd(module, output, *args, **kwargs):
    assert isinstance(module, torch.nn.Conv1d)
    if isinstance(args, tuple):
        input = args[0]
    else:
        input = args
    assert isinstance(input, torch.Tensor)
    assert isinstance(output, torch.Tensor)

    N, C_in, L_in = input.shape
    _, C_out, L_out = output.shape
    kernel_size = module.kernel_size
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 1
        kernel_size = kernel_size[0]
    stride = module.stride
    padding = module.padding
    if isinstance(padding, tuple):
        assert len(padding) == 1
        padding = padding[0]
    dilation = module.dilation
    if isinstance(dilation, tuple):
        assert len(dilation) == 1
        dilation = dilation[0]

    mul = kernel_size * C_in * C_out * L_out * N
    add = (kernel_size - 1) * (C_in - 1) * C_out * L_out * N 
    if module.bias is not None:
        add = add + C_out * L_out * N

    return mul + add


def compute_Dropout_madd(module, output, *args, **kwargs):
    assert isinstance(module, torch.nn.Dropout)
    if isinstance(args, tuple):
        input = args[0]
    else:
        input = args
    assert isinstance(input, torch.Tensor)

    if module.training:
        return input.numel()
    else:
        return 0


def compute_Fp32GroupNorm_madd(module, output, *args, **kwargs):
    assert isinstance(module, (fairseq.modules.Fp32GroupNorm, s3prl.upstream.wavlm.modules.Fp32GroupNorm))
    if isinstance(args, tuple):
        input = args[0]
    else:
        input = args
    assert isinstance(input, torch.Tensor)

    num_channels = module.num_channels
    num_groups = module.num_groups
    batch_size = input.size()[0] * num_channels / num_groups
    norm_size = input.numel() / batch_size

    add = norm_size - 1  # computing E[x].
    add = add + norm_size - 1 # computing Var[x].
    mul = norm_size  # computing Var[x].
    add = add + batch_size  # + epsilon.
    add = add + batch_size  # + beta.
    add = add + batch_size  # x - E[x].
    mul = mul + batch_size  # * gamma.

    return mul + add


def compute_FusedLayerNorm_madd(module, output, *args, **kwargs):
    assert isinstance(module, fairseq.modules.layer_norm.FusedLayerNorm)
    if isinstance(args, tuple):
        input = args[0]
    else:
        input = args
    assert isinstance(input, torch.Tensor)

    norm_size = module.normalized_shape.numel()
    add = norm_size - 1  # computing E[x].
    add = add + norm_size - 1 # computing Var[x].
    mul = norm_size  # computing Var[x].
    batch_size = input.numel() / norm_size
    add = add + batch_size  # + epsilon.
    add = add + batch_size  # + beta.
    add = add + batch_size  # x - E[x].
    mul = mul + batch_size  # * gamma.

    return mul + add


def compute_MultiheadAttention_madd(module, output, *args, **kwargs):
    assert isinstance(module, fairseq.modules.MultiheadAttention)
    if isinstance(output, tuple):
        output = output[0]
    assert isinstance(output, torch.Tensor)

    def _Linear_madd(_module, _input):
        batch_size = _input.numel() / _input.size()[-1]
        input_dim = _module.in_features
        output_dim = _module.out_features
        mul = batch_size * input_dim * output_dim
        add = batch_size * (input_dim - 1) * output_dim
        if _module.bias is not None:
            add = add + batch_size * output_dim
        return mul, add

    # TODO(cfyeh): double check the computation here.
    query = kwargs["query"]
    key = kwargs["key"]
    value = kwargs["value"]

    q_mul, q_add = _Linear_madd(_module=module.q_proj, _input=query)
    k_mul, k_add = _Linear_madd(_module=module.k_proj, _input=key)
    v_mul, v_add = _Linear_madd(_module=module.v_proj, _input=value)

    mul = q_mul + k_mul + v_mul
    add = q_add + k_add + v_add

    mul = mul + (query.numel() / module.q_proj.in_features * module.q_proj.out_features)  # q *= self.scaling.

    mul = mul + (query.size()[0] * query.size()[1] * module.q_proj.out_features * module.k_proj.out_features)  # torch.bmm(q, k^T).

    # TODO(cfyeh): finish the rest.
    return mul + add


def compute_GELU_madd(module, output, *args, **kwargs):
    assert isinstance(module, torch.nn.GELU)
    if isinstance(args, tuple):
        input = args[0]
    else:
        input = args
    assert isinstance(input, torch.Tensor)

    numel = input.numel()
    mul = numel * 3  # 0.044715 * (x ** 3).
    add = numel  # x + 0.044715 * (x ** 3).
    mul = mul + numel  # sqrt(2/pi) *.
    add = add + numel  # 1 +.
    mul = mul + (numel * 2)  # 0.5 * x *.

    return mul + add

def compute_ReLU_madd(module, output, *args, **kwargs):
    """
    cite from: https://github.com/Swall0w/torchstat/blob/b52a3b06c2c54c2d09ade1a18cf6c4ca5dc27510/torchstat/compute_madd.py
    MIT License
    """
    assert isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6))
    if isinstance(args, tuple):
        input = args[0]
    else:
        input = args
    assert isinstance(input, torch.Tensor)
    
    return input.numel()


def compute_SamePad_madd(module, output, *args, **kwargs):
    return 0


def compute_Linear_madd(module, output, *args, **kwargs):
    assert isinstance(module, torch.nn.Linear)
    if isinstance(args, tuple):
        input = args[0]
    else:
        input = args
    assert isinstance(input, torch.Tensor)
    assert isinstance(output, torch.Tensor)

    input_numel = input.numel()
    output_numel = output.numel()
    input_dim = input.size()[-1]
    output_dim = output.size()[-1]

    num_rows = input_numel / input_dim
    mul = num_rows * input_dim * output_dim
    add = (num_rows - 1) * input_dim * output_dim
    if module.bias is not None:
        add = add + output_numel
    return mul + add


def compute_madd(module, output, *args, **kwargs):
    
    if isinstance(module, torch.nn.Conv1d):
        return compute_Conv1d_madd(module, output, *args, **kwargs)
    if isinstance(module, torch.nn.Dropout):
        return compute_Dropout_madd(module, output, *args, **kwargs)
    if isinstance(module, torch.nn.GELU):
        return compute_GELU_madd(module, output, *args, **kwargs)
    if isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6)):
        return compute_ReLU_madd(module, output, *args, **kwargs)
    if isinstance(module, torch.nn.Linear):
        return compute_Linear_madd(module, output, *args, **kwargs)
    if isinstance(module, fairseq.modules.MultiheadAttention):
        return compute_MultiheadAttention_madd(module, output, *args, **kwargs)
    if isinstance(module, (fairseq.modules.Fp32GroupNorm, s3prl.upstream.wavlm.modules.Fp32GroupNorm)):
        return compute_Fp32GroupNorm_madd(module, output, *args, **kwargs)
    if isinstance(module, fairseq.modules.SamePad):
        return compute_SamePad_madd(module, output, *args, **kwargs)
    
    # FIXME: FusedLayerNorm has not yet been implemented in the specific version assigned by s3prl
    # if isinstance(module, fairseq.modules.layer_norm.FusedLayerNorm):
    #     return compute_FusedLayerNorm_madd(module, output, *args, **kwargs)

    logger.warning(
        "MAdd computation for module \"{}\" is not supported yet.".format(
            module.__class__.__name__
        )
    )
    return 0