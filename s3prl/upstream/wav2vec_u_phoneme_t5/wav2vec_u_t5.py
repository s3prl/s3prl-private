from enum import Enum, auto
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace
from fairseq.modules import SamePad, TransposeLast

from typing import Any, Optional, Dict, List, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Segmenter class
class Segmenter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = Namespace(**cfg)
        self.subsample_rate = self.cfg.subsample_rate

    def pre_segment(self, dense_x, dense_padding_mask):
        return dense_x, dense_padding_mask

    def logit_segment(self, logits, padding_mask):
        return logits, padding_mask


class RandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        target_num = math.ceil(dense_x.size(1) * self.subsample_rate)
        ones = torch.ones(dense_x.shape[:-1], device=dense_x.device)
        indices, _ = ones.multinomial(target_num).sort(dim=-1)
        indices_ld = indices.unsqueeze(-1).expand(-1, -1, dense_x.size(-1))
        dense_x = dense_x.gather(1, indices_ld)
        dense_padding_mask = dense_padding_mask.gather(1, index=indices)
        return dense_x, dense_padding_mask


class UniformRandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        bsz, tsz, fsz = dense_x.shape

        target_num = math.ceil(tsz * self.subsample_rate)

        rem = tsz % target_num

        if rem > 0:
            dense_x = F.pad(dense_x, [0, 0, 0, target_num - rem])
            dense_padding_mask = F.pad(
                dense_padding_mask, [0, target_num - rem], value=True
            )

        dense_x = dense_x.view(bsz, target_num, -1, fsz)
        dense_padding_mask = dense_padding_mask.view(bsz, target_num, -1)

        if self.cfg.mean_pool:
            dense_x = dense_x.mean(dim=-2)
            dense_padding_mask = dense_padding_mask.all(dim=-1)
        else:
            ones = torch.ones((bsz, dense_x.size(2)), device=dense_x.device)
            indices = ones.multinomial(1)
            indices = indices.unsqueeze(-1).expand(-1, target_num, -1)
            indices_ld = indices.unsqueeze(-1).expand(-1, -1, -1, fsz)
            dense_x = dense_x.gather(2, indices_ld).reshape(bsz, -1, fsz)
            dense_padding_mask = dense_padding_mask.gather(2, index=indices).reshape(
                bsz, -1
            )
        return dense_x, dense_padding_mask


class JoinSegmenter(Segmenter):
    def logit_segment(self, logits, padding_mask):
        preds = logits.argmax(dim=-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join:
                u[0] = 0
                u[1:] = c.cumsum(0)[:-1]
                m = c > 1
                r = torch.rand(m.sum())
                o = (c[m] * r).long()
                u[m] += o
                new_logits[b, : u.numel()] = logits[b, u]
            else:
                new_logits[b].index_add_(
                    dim=0, index=idx.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


class UniformRandomJoinSegmenter(UniformRandomSegmenter, JoinSegmenter):
    pass


class SegmentationType(Enum):
    NONE = auto()
    RANDOM = auto()
    UNIFORM_RANDOM = auto()
    UNIFORM_RANDOM_JOIN = auto()
    JOIN = auto()

SEGMENT_FACTORY = {
    SegmentationType.NONE: Segmenter,
    SegmentationType.RANDOM: RandomSegmenter,
    SegmentationType.UNIFORM_RANDOM: UniformRandomSegmenter,
    SegmentationType.UNIFORM_RANDOM_JOIN: UniformRandomJoinSegmenter,
    SegmentationType.JOIN: JoinSegmenter,
}



class Discriminator(torch.nn.Module):
    def __init__(self,
            dim, # input_dim
            discriminator_kernel: int = 3,
            discriminator_dilation: int = 1,
            discriminator_dim: int = 256,
            discriminator_causal: bool = True,
            discriminator_linear_emb: bool = False,
            discriminator_depth: int = 1,
            discriminator_max_pool: bool = False,
            discriminator_act_after_linear: bool = False,
            discriminator_dropout: float = 0.0,
            discriminator_spectral_norm: bool = False,
            discriminator_weight_norm: bool = False,
        ):
        super().__init__()
        self.discriminator_kernel = discriminator_kernel
        self.discriminator_dilation = discriminator_dilation
        self.discriminator_dim = discriminator_dim
        self.discriminator_causal = discriminator_causal
        self.discriminator_linear_emb = discriminator_linear_emb
        self.discriminator_depth = discriminator_depth
        self.discriminator_max_pool = discriminator_max_pool
        self.discriminator_act_after_linear = discriminator_act_after_linear
        self.discriminator_dropout = discriminator_dropout
        self.discriminator_spectral_norm = discriminator_spectral_norm
        self.discriminator_weight_norm = discriminator_weight_norm

        if discriminator_causal:
            padding = discriminator_kernel - 1
        else:
            padding = discriminator_kernel // 2
    
        def make_conv(in_d, out_d, k, p=0, has_dilation=True):
            conv = nn.Conv1d(
                in_d,
                out_d,
                kernel_size=k,
                padding=p,
                dilation=discriminator_dilation if has_dilation else 1,
            )
            if discriminator_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            elif discriminator_weight_norm:
                conv = nn.utils.weight_norm(conv)
            return conv
        
        inner_net = [
            nn.Sequential(
                make_conv(discriminator_dim, discriminator_dim, discriminator_kernel, padding),
                SamePad(kernel_size=discriminator_kernel, causal=discriminator_causal),
                nn.Dropout(discriminator_dropout),
                nn.GELU(),
            )
            for _ in range(discriminator_depth - 1)
        ] + [
            make_conv(discriminator_dim, 1, discriminator_kernel, padding, has_dilation=False),
            SamePad(kernel_size=discriminator_kernel, causal=discriminator_causal),
        ]

        if discriminator_linear_emb:
            emb_net = [make_conv(dim, discriminator_dim, 1)]
        else:
            emb_net = [
                make_conv(dim, discriminator_dim, discriminator_kernel, padding),
                SamePad(kernel_size=discriminator_kernel, causal=discriminator_causal),
            ]

        if discriminator_act_after_linear:
            emb_net.append(nn.GELU())

        self.net = nn.Sequential(
            *emb_net,
            nn.Dropout(discriminator_dropout),
            *inner_net,
        )
    
    def forward(self, x, padding_mask):
        x = x.transpose(1, 2)  # BTC -> BCT
        x = self.net(x)
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1)
        x = x.squeeze(-1)
        if self.max_pool:
            x, _ = x.max(dim=-1)
        else:
            x = x.sum(dim=-1)
            x = x / x_sz
        return x


class Generator(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        generator_kernel: int = 4,
        generator_dilation: int = 1,
        generator_stride: int = 1,
        generator_bias: bool = False,
        generator_dropout: float = 0.0,
        generator_batch_norm: int = 0,
        generator_residual: bool = False,
        generator_pad: int = -1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.stride = generator_stride
        self.dropout = nn.Dropout(generator_dropout)
        self.batch_norm = generator_batch_norm
        self.residual = generator_residual

        padding = generator_kernel // 2 if generator_pad < 0 else generator_pad

        self.proj = nn.Sequential(
            TransposeLast(),
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=generator_kernel,
                stride=generator_stride,
                dilation=generator_dilation,
                padding=padding,
                bias=generator_bias,
            ),
            TransposeLast(),
        )
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(input_dim)
            self.bn.weight.data.fill_(generator_batch_norm)
        if self.residual:
            self.in_proj = nn.Linear(input_dim,input_dim)

    def bn_padded_data(self, feature, padding_mask):
        normed_feature = feature.clone()
        # print(feature.size())
        # print(padding_mask.size())
        # print(feature[~padding_mask].unsqueeze(-1).size())
        # print("---" * 100)
        normed_feature[~padding_mask] = self.bn(feature[~padding_mask].unsqueeze(-1)).squeeze(-1)
        return normed_feature
    
    def forward(self, dense_x, tokens, dense_padding_mask):
        result = {}
        if self.batch_norm:
            dense_x = self.bn_padded_data(dense_x, dense_padding_mask)
        if self.residual:
            inter_x = self.in_proj(self.dropout(dense_x))
            dense_x = dense_x + inter_x
            result['inter_x'] = inter_x

        dense_x = self.dropout(dense_x)

        dense_x = self.proj(dense_x)
        if self.stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride]

        if dense_padding_mask.size(1) != dense_x.size(1):
            new_padding = dense_padding_mask.new_zeros(dense_x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)

            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding


        token_x = None
        if tokens is not None:
            token_x = dense_x.new_zeros(tokens.numel(), self.output_dim)
            token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
            token_x = token_x.view(tokens.shape + (self.output_dim,))

        result["dense_x"] = dense_x
        result["token_x"] = token_x
        result["dense_padding_mask"] = dense_padding_mask

        return result

class Upsampler(nn.Module):
    def __init__(self, downsample_rate):
        super().__init__()
        self.downsample_rate = downsample_rate
    
    def forward(self, x):
        return x.repeat_interleave(self.downsample_rate, dim=1)
    


class Wav2VecUInfer(nn.Module):
    def __init__(self, cfg, target_dict_info, downsample_rate):
        super().__init__()
        self.cfg = Namespace(**cfg)
        self.target_dict_info = Namespace(**target_dict_info)

        # dict information
        print(self.cfg)
        self.zero_index = self.target_dict_info.symbols.index("<SIL>") if "<SIL>" in self.target_dict_info.symbols else 0
        output_size = len(self.target_dict_info.symbols)
        self.pad = self.target_dict_info.pad
        self.eos = self.target_dict_info.eos
        self.no_softmax = self.cfg.no_softmax
        self.gumbel = self.cfg.gumbel
        self.hard_gumbel = self.cfg.hard_gumbel

        self.mmi_weight = self.cfg.mmi_weight
        self.blank_weight = self.cfg.blank_weight
        self.blank_mode = self.cfg.blank_mode
        self.blank_index = zero_index if self.cfg.blank_is_sil else 0
        
        self.discriminator = Discriminator(
            output_size,
            discriminator_kernel = self.cfg.discriminator_kernel,
            discriminator_dilation = self.cfg.discriminator_dilation,
            discriminator_dim = self.cfg.discriminator_dim,
            discriminator_causal = self.cfg.discriminator_causal,
            discriminator_linear_emb = self.cfg.discriminator_linear_emb,
            discriminator_depth = self.cfg.discriminator_depth,
            discriminator_max_pool = self.cfg.discriminator_max_pool,
            discriminator_act_after_linear = self.cfg.discriminator_act_after_linear,
            discriminator_dropout = self.cfg.discriminator_dropout,
            discriminator_spectral_norm = self.cfg.discriminator_spectral_norm,
            discriminator_weight_norm = self.cfg.discriminator_weight_norm,
        )

        self.segmenter = SEGMENT_FACTORY[
            SegmentationType(5) if self.cfg.segmentation["type"] == "JOIN" else 0 
        ](self.cfg.segmentation)

        self.generator = Generator(
            self.cfg.input_dim,
            output_size,
            generator_kernel = self.cfg.generator_kernel,
            generator_dilation = self.cfg.generator_dilation,
            generator_stride = self.cfg.generator_stride,
            generator_bias = self.cfg.generator_bias,
            generator_dropout = self.cfg.generator_dropout,
            generator_batch_norm = self.cfg.generator_batch_norm,
            generator_residual = self.cfg.generator_residual,
            generator_pad = self.cfg.generator_pad,
        )

        if self.mmi_weight > 0:
            self.target_downsample_rate = self.cfg.target_downsample_rate
            self.decoder = nn.Linear(self.cfg.input_dim, self.cfg.target_dim)
    
        self.downsample_rate = downsample_rate
        self.upsampler = Upsampler(downsample_rate)
    
    def get_logits(
        self,
        net_output: Optional[Dict[str, List[Optional[torch.Tensor]]]],
        normalize: bool = False,
    ):
        logits = net_output["logitis"]

        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., self.blank_index] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., self.blank_index] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        padding = net_output["padding_mask"]
        if padding.any():
            logits[padding] = float("-inf")
            logits[padding][..., self.blank_index] = float("inf")

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits.transpose(0, 1)


    def get_normalized_probs(
        self,
        net_output: Tuple[
            torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]
        ],
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ):
        logits = self.get_logits(net_output)

        probs = super().get_normalized_probs(logits, log_probs, sample)
        # BTC -> TBC for ctc
        probs = probs.transpose(0, 1)
        return probs

    def normalize(self, dense_x):

        bsz, tsz, csz = dense_x.shape

        if dense_x.numel() == 0:
            raise Exception(dense_x.shape)
        _, k = dense_x.max(-1)
        hard_x = (
            dense_x.new_zeros(bsz * tsz, csz)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(-1, csz)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)

        if not self.no_softmax:
            dense_x = dense_x.softmax(-1)

        return dense_x

    def forward(
        self,
        features,
        padding_mask,
        random_label=None,
        dense_x_only=False,
        segment=True,
    ):
        if segment:
            features, padding_mask = self.segmenter.pre_segment(features, padding_mask)

        orig_size = features.size(0) * features.size(1) - padding_mask.sum()

        gen_result = self.generator(features, random_label, padding_mask)

        orig_dense_x, token_x = gen_result["dense_x"], gen_result["token_x"]
        orig_dense_padding_mask = gen_result["dense_padding_mask"]

        if segment:
            dense_x, dense_padding_mask = self.segmenter.logit_segment(
                orig_dense_x, orig_dense_padding_mask
            )
        else:
            dense_x = orig_dense_x
            dense_padding_mask = orig_dense_padding_mask

        dense_logits = dense_x

        if not (self.no_softmax and dense_x_only):
            dense_x = self.normalize(dense_logits)

        if dense_x_only or self.discriminator is None:
            return {
                "logits": dense_x,
                "padding_mask": dense_padding_mask,
            }

        token_padding_mask = random_label == self.pad

        if self.mmi_weight > 0:
            inter_x = self.decoder(gen_result['inter_x'])
        
        gen_result["upsampled_x"] = self.upsampler(gen_result["dense_x"])


        return gen_result


### A simpler dictionary processor
class SimpleDict:
    def __init__(
        self, 
        dict_file,
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
    ):
        """
        Expected format:
        
            Format1:
                A\\n
                B\\n
                C\\n
            
            Format2:
                A 100\\n
                B 102\\n
                C 884\\n
        """
        self.symbols = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)

        # reading files
        dict_file = open(dict_file, "r", encoding="utf-8")
        lines = dict_file.readlines()
        for line in lines:
            line = line.split(" ")
            if len(line) == 1:
                symbol = line
            elif len(line) == 2:
                symbol = line[0]
            else:
                raise RuntimeError("Too many fields, check if using the correct file")
            
            if symbol in self.symbols:
                raise RuntimeError("Dumplicate symbols found in dictionary file: {} (the symbol is {})".format(dict_file, symbol))
            self.add_symbol(symbol, overwrite=False)
            
    
    def add_symbol(self, symbol, overwrite=False):
        if symbol in self.indices and not overwrite:
            logging.info("word exist, no update")
            idx = self.indices[word]
        else:
            idx = len(self.symbols)
            self.indices[symbol] = idx
            self.symbols.append(symbol)
        return idx
    
    def pack_to_dict_info(self):
        return {
            "symbols": self.symbols,
            "pad": self.pad_index,
            "eos": self.eos_index,
        }


class PhonemeT5(torch.nn.Module):
    def __init__(self, downsample_rate):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("voidful/phoneme_byt5_v2")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("voidful/phoneme_byt5_v2")
        self.dict = {'ð': 'T',
                    'ə': '@',
                    'ʌ': 'u',
                    'v': 'v',
                    'æ': 'a',
                    'n': 'n',
                    'd': 'd',
                    'ɪ': 'i',
                    't': 't',
                    'uː': 'U',
                    'eɪ': 'A',
                    'w': 'w',
                    'z': 'z',
                    'ɔ': 'o',
                    'f': 'f',
                    'ɔːɹ': 'Wr',
                    'ɹ': 'r',
                    'ɔː': 'W',
                    'b': 'b',
                    'aɪ': 'I',
                    'h': 'h',
                    'iː': 'E',
                    'i': 'E',
                    'm': 'm',
                    'ɜː': '3',
                    'tʃ': 'tS',
                    'ʃ': 'S',
                    's': 's',
                    'ɑːɹ': '~r',
                    'ɑː': '~',
                    'ɛɹ': 'er',
                    'ɛ': 'e',
                    'ɚ': 'er',
                    'l': 'l',
                    'oʊ': 'O',
                    'ʊ': 'U',
                    'ʊɹ': 'Ur',
                    'ŋ': 'N',
                    'oːɹ': 'Or',
                    'oː': 'O',
                    'ɡ': 'g',
                    'ɾ': 'd',
                    'p': 'p',
                    'θ': 'D',
                    'ɐ': '@',
                    'aʊ': '^U',
                    'ᵻ': 'i',
                    'j': 'j',
                    'ɪɹ': 'ir',
                    'k': 'k',
                    'əl': '@l',
                    'iə': 'E@',
                    'dʒ': 'dZ',
                    'ʒ': 'Z',
                    'ɔɪ': 'oi',
                    'ʔ': '|',
                    'n̩': 'n',
                    'aɪɚ': 'Ier',
                    'aɪə': 'I@',
                    'iːː': 'E',
                    'x': 'H',
                    'r': 'r',
                    'ɑ̃': 'A',
                    'ɡʲ': 'g',
                    }
        
        self.dict_values = list(self.dict.values())
        self.downsample_rate = downsample_rate
        self.upsampler = Upsampler(downsample_rate)
    
    def forward(self, prediction):
        tokens = prediction["dense_x"].argmax(-1)
        # t5_input = []
        # for token in tokens:
        #     phones = [self.dict_values[int(i)] for i in token]
        #     phones = "".join(phones)
        #     ids = self.tokenizer(phones, return_tensors="pt").input_ids
        #     t5_input.append(ids)
       
        # import logging
        # logging.info("t5_input: {}".format([t5.shape for t5 in t5_input]))
        # t5_input = torch.cat(t5_input, dim=0).to(tokens.device)
        result = self.model.encoder(tokens)
        return self.upsampler(result["last_hidden_state"])
        
        
