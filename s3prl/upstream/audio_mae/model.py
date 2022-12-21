# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

from math import ceil
import torch
import torch.nn as nn
import torchaudio
import timm.models.vision_transformer

from .module import PatchEmbed_new

# TODO (Tzu-hsun Feng): I took the mean and std of audioset to use, since the model is pretrained on audioset and in the scenario of SUPERB, we won't finetune the upstream model. Is this setting correct? 
TARGET_LENGTH = 1024
FBANK_MEAN = -9.087865 # audioset: -4.2677393
FBANK_2STD = 4.703878 * 2 # audioset: 4.5689974 * 2
NUM_MEL_BINS = 128

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        use_custom_patch = kwargs.pop("use_custom_patch")
        super(VisionTransformer, self).__init__(**kwargs)
        
        img_size=(TARGET_LENGTH, NUM_MEL_BINS) # 1024, 128
        emb_dim = kwargs.get("emb_dim", 768)
        if use_custom_patch:
            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=16, in_chans=1, embed_dim=emb_dim, stride=10)
            self.pos_embed = nn.Parameter(torch.zeros(1, 1212 + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
        else:
            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
            num_patches = self.patch_embed.num_patches
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
    
    @staticmethod    
    def _wav2fbank(waveform):
        # unlike the original one, here only convert waveform to normalized fbank
        waveform -= waveform.mean()
        # 498 128, 998, 128
        fbank = torchaudio.compliance.kaldi.fbank(waveform.unsqueeze(0), htk_compat=True, sample_frequency=16000, use_energy=False, window_type='hanning', num_mel_bins=NUM_MEL_BINS, dither=0.0, frame_shift=10)
        # use the mean and std * 2 of audioset
        fbank = (fbank - FBANK_MEAN) / FBANK_2STD
        
        return fbank
    
    @staticmethod
    def _split_fbank(fbank):
        # if an fbank is too long, split it into multiple pieces, and padding those is shorter than target_length
        # TODO (Tzu-hsun Feng): Current method doesn't consider the relation between each piece, is there any better way for splitting those fbanks longer than target_length? One thought may be splitting it with some overlap, but I am not sure how to combime the outputs properly.
        # cut and pad
        while True:
            p = TARGET_LENGTH - fbank.shape[0] # n_frames
            if p > 0:
                m = torch.nn.ConstantPad2d((0, 0, 0, p), -FBANK_MEAN / FBANK_2STD)
                yield m(fbank)
                break
            elif p < 0:
                yield fbank[:TARGET_LENGTH]
                fbank = fbank[TARGET_LENGTH:]
            else:
                yield fbank
                break
            
    def _trim_or_cat(self, fbank_lengths, resolution, old_out):
        # concat each pieces which is from the same audio, trim the padding, and reorder feature to t0_highest_freq, ..., t0_lowest_freq, t1_highest_freq, ..., t1_lowest_freq, ...
        cls_out = old_out[:, :1].mean(dim=2)
        patch_size = self.patch_embed.patch_size
        seq_len = TARGET_LENGTH // patch_size[0]
        mel_len = NUM_MEL_BINS // patch_size[1]
        assert patch_size[0] == patch_size[1], "Double check here if the patch is not a square"
        old_out = old_out[:, 1:].view(old_out.shape[0], seq_len, mel_len, resolution, old_out.shape[-1]).transpose_(2,3).flatten(3, 4)
        new_out = []
        cls_idx = [0]
        for l in fbank_lengths:
            num_blocks = ceil(l / TARGET_LENGTH)
            cls_idx.append(num_blocks + cls_idx[-1])
            # get all outputs from the same audio
            tmp = old_out[:num_blocks]
            # cat the sequence
            tmp = tmp.flatten(0, 1)
            # trim the padding
            tmp = tmp[:ceil(l / patch_size[0])]
            # make the order to t0_highest_freq, ..., t0_lowest_freq, t1_highest_freq, ..., t1_lowest_freq, ...
            # tmp = tmp.transpose_(0,1).reshape(-1, old_out.shape[-1])
            new_out.append(tmp.flatten(0, 1))
            old_out = old_out[num_blocks:]
            
        return nn.utils.rnn.pad_sequence(new_out, batch_first=True), cls_out[cls_idx[:-1]]

    # overwrite original timm
    def forward(self, wavs, resolution=2):
        origin_fbanks = list(map(self._wav2fbank, wavs))
        patch_size = self.patch_embed.patch_size
        assert patch_size[0] == patch_size[1], "Double check here if the patch is not a square"
        len_diff = patch_size[0] - patch_size[0] // (1<<(resolution-1))
        fbank_lengths = list(map(lambda f: f.shape[0] - len_diff, origin_fbanks))
        outss = []
        for r in range(resolution):
            l_offset = r * (16>>(resolution-1))
            fbanks = []
            for fbank in origin_fbanks:
                r_offset = -(len_diff - l_offset) if len_diff - l_offset else len(fbank)
                
                fbanks.extend(self._split_fbank(fbank[l_offset: r_offset]))
            
            x = torch.stack(fbanks).unsqueeze(1)

            x = self.patch_embed(x) + self.pos_embed[:, 1:]
            cls_token = self.cls_token + self.pos_embed[:, :1]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

            outs = [x]
            for blk in self.blocks:
                outs.append(blk(outs[-1]))
            
            outss.append(outs)
            
        outss = [torch.stack(layer_out, dim=2) for layer_out in zip(*outss)]
        seq_outs, scene_outs = zip(*map(partial(self._trim_or_cat, fbank_lengths, resolution), outss))
            
            
        # TODO (Tzu-hsun Feng): use the first cls token for SID, ER, and ASV, does it more make sense?
        return {
            "hidden_states": seq_outs,
            "SID": scene_outs[-1],
            "ASV": scene_outs,
            "ER": scene_outs,
            "feature_lengths": [ceil(l / patch_size[0]) * (NUM_MEL_BINS // patch_size[1]) for l in fbank_lengths]
        }