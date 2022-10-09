"""
    Distiller Model
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import torch
from torch import nn
from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel
from fairseq.modules import GradMultiply
from .module import TransformerEncoder, SplitLinear
import torchaudio

class DistillerConfig:
    """
    Configuration class
    """

    def __init__(self, config: dict):
        # Feature extractor
        self.extractor_mode = str(config.get("extractor_mode", "default"))
        self.extractor_conv_feature_layers = str(
            config.get(
                "extractor_conv_feature_layers",
                "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
            )
        )
        if config.get("wav_rct", 1.0) > 0.0:
            from .module import ReconstructionModel
            self.reconstuctor_convtrans_feature_layers = str(
                config.get(
                    "reconstuctor_convtrans_feature_layers",
                    "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
                )
            )
        self.extractor_dropout = float(config.get("extractor_dropout", 0.0))
        self.feature_grad_mult = float(config.get("feature_grad_mult", 1.0))

        # Convolutional relative positional encoding
        self.conv_pos = int(config.get("conv_pos", 128))
        self.conv_pos_groups = int(config.get("conv_pos_groups", 16))

        # Transformer encoder
        self.encoder_layers = int(config.get("encoder_layers", 1))
        self.encoder_embed_dim = int(config.get("encoder_embed_dim", 768))
        self.encoder_ffn_embed_dim = int(config.get("encoder_ffn_embed_dim", 3072))
        self.encoder_attention_heads = int(config.get("encoder_attention_heads", 12))
        self.activation_fn = str(config.get("activation_fn", "gelu"))
        self.layer_norm_first = bool(config.get("layer_norm_first", False))
        self.attention_type = str(config.get("attention_type", "original"))

        # Dropout
        self.dropout = float(config.get("dropout", 0.1))
        self.attention_dropout = float(config.get("attention_dropout", 0.1))
        self.activation_dropout = float(config.get("activation_dropout", 0.1))
        self.encoder_layerdrop = float(config.get("encoder_layerdrop", 0.0))

        # Output
        self.final_dim = int(config.get("final_dim", 768))
        self.out_layer_type = str(config.get("out_layer_type", "expand-last"))
        self.out_layer_inter_dim = int(config.get("out_layer_inter_dim", -1))

        # Task & loss
        self.n_tasks = int(config.get("n_tasks", 12))
        self.task_emb_type = str(config.get("task_emb_type", "expand-last"))
        self.task_emb_size = int(config.get("task_emb_size", 0))
        self.layer_emb_size = int(config.get("layer_emb_size", 0))
        self.loss_type = str(config.get("loss_type", "l1"))
        self.feat_pen_loss = float(config.get("feat_pen_loss", 0.0))
        self.cosine_loss = float(config.get("cosine_loss", 0.0))
        self.wav_rct = float(config.get("wav_rct", 0.0))

        # When task_emb_type == 'expand-last' only
        self.pred_layer_id = list(
            config.get("pred_layer_id", range(1, self.n_tasks + 1))
        )

        self.multi_pred_head = int(config.get("multi_pred_head", 1))

        # Initialization
        self.init_teacher_conv_layers = bool(
            config.get("init_teacher_conv_layers", False)
        )
        self.init_teacher_encoder_layers = bool(
            config.get("init_teacher_encoder_layers", False)
        )

        # Pre-trained Model Init.
        self.init_melhubert_encoder_layers = bool(
            config.get("init_melhubert_encoder_layers", False)
        )

        self.melhubert_ckpt = str(
            config.get("melhubert_ckpt","")
        )

class DistillerModel(nn.Module):
    """
    Distiller Model
    """

    def __init__(self, config: DistillerConfig):
        super().__init__()

        self.config = config

        # [Project Chimera] Feature Extractor
        # win_length default is n_fft
        # hop_length default is win_length // 2
        # if hop_length = 320, downsample rate = 160 (HuBERT downsample rate)
        # if hop_length = 160, downsample rate = 160
        # n_mels = 80 // 2, we follow Mel-HuBERT, concating afterwards
        if config.extractor_mode == 'mel':
            feat_emb_dim = 80
            self.feature_extractor = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels= feat_emb_dim // 2, n_fft=400, hop_length=160, window_fn=torch.hamming_window)

        else:
            self.conv_layers = eval(config.extractor_conv_feature_layers)
            feat_emb_dim = self.conv_layers[-1][0]
            self.feature_extractor = ConvFeatureExtractionModel(
                self.conv_layers,
                dropout=config.extractor_dropout,
                mode=config.extractor_mode,
                conv_bias=False,
            )
        self.feature_grad_mult = config.feature_grad_mult

        self.n_tasks = config.n_tasks
        self.task_emb_type = config.task_emb_type

        final_emb_size = config.encoder_embed_dim
        if self.task_emb_type == "add":
            self.task_embedding = nn.Embedding(config.n_tasks, config.encoder_embed_dim)
            nn.init.normal_(self.task_embedding.weight, 0.0, 0.1)
        elif self.task_emb_type == "concat":
            assert config.task_emb_size > 0
            feat_emb_dim += config.task_emb_size
            self.task_embedding = nn.Embedding(config.n_tasks, config.task_emb_size)
        elif self.task_emb_type == "concat-last":
            assert config.task_emb_size > 0
            self.task_embedding = nn.Embedding(config.n_tasks, config.task_emb_size)
            final_emb_size += config.task_emb_size
        elif self.task_emb_type == "expand-last":
            self.pred_layer_id = config.pred_layer_id
            assert self.n_tasks == len(self.pred_layer_id)
            print(
                f"[DistillerModel] - Expands the output dimension by {self.n_tasks} times"
            )
            print(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "self-hidden":
            self.pred_layer_id = config.pred_layer_id
            assert self.n_tasks == len(self.pred_layer_id)
            assert self.n_tasks == config.encoder_layers + 1
            print("[DistillerModel] - Predicting with self-hidden layers")
            print(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "none":
            print(
                f"[DistillerModel] - Disabled task embedding (predicts only layer {self.n_tasks})"
            )
        else:
            raise NotImplementedError(f"Unknown task emb type {self.task_emb_type}")

        self.post_extract_proj = (
            nn.Linear(feat_emb_dim, config.encoder_embed_dim)
            if feat_emb_dim != config.encoder_embed_dim
            else None
        )

        # Encoder
        if config.encoder_layers > 0:
            self.encoder = TransformerEncoder(config)
        else:
            self.encoder = nn.GELU()

        final_dim = config.final_dim * (
            1 if self.task_emb_type != "expand-last" else self.n_tasks
        )

        inter_dim = config.out_layer_inter_dim
        inter_dim = inter_dim if inter_dim > 0 else final_emb_size

        print(f"[DistillerModel] - Out layer type: {config.out_layer_type}")
        if config.out_layer_type == "expand-last":
            assert self.task_emb_type == "expand-last"
            print(f"[DistillerModel] - Inter dim = {inter_dim}")
            if hasattr(config, 'multi_pred_head') and config.multi_pred_head > 1:
                # different pred heads for each teacher model
                # out: B x T x N_task x final_dim
                for idx in range(config.multi_pred_head):
                    output_layer = nn.Sequential(
                        nn.Linear(final_emb_size, inter_dim * self.n_tasks),
                        nn.GELU(),
                        SplitLinear(inter_dim, self.n_tasks, config.final_dim),
                    )
                    setattr(self, 'output_layer{}'.format(idx), output_layer)
            else:
                # out: B x T x N_task x final_dim
                self.output_layer = nn.Sequential(
                    nn.Linear(final_emb_size, inter_dim * self.n_tasks),
                    nn.GELU(),
                    SplitLinear(inter_dim, self.n_tasks, config.final_dim),
                )
        elif config.out_layer_type in {"none", "self-hidden"}:
            self.output_layer = None
        else:
            raise NotImplementedError(f"Unknown out layer type {config.out_layer_type}")

        if config.wav_rct:
            self.pre_rct_proj = nn.Linear(config.encoder_embed_dim, feat_emb_dim)
            self.reconstuctor_convtrans_feature_layers = eval(config.reconstuctor_convtrans_feature_layers)
            self.reconstruct_module = ReconstructionModel(self.reconstuctor_convtrans_feature_layers)


    def forward_feature(self, wave, pad_mask):
        """Forward feature extractor"""

        if self.feature_grad_mult > 0:
            feat = self.feature_extractor(wave)
            if self.feature_grad_mult != 1.0:
                feat = GradMultiply.apply(feat, self.feature_grad_mult)
        else:
            with torch.no_grad():
                feat = self.feature_extractor(wave)

        feat = feat.transpose(1, 2)  # B x T x D
        # [Project Chimera] downsample 160 -> downsample 320
        # ref: Mel-HuBERT
        if self.config.extractor_mode == 'mel':
            odd_feat = feat[:,::2,:]
            even_feat = feat[:,1::2,:]
            if odd_feat.shape[1] != even_feat.shape[1]:
                even_feat = torch.cat((even_feat, torch.zeros(even_feat.shape[0], 1, even_feat.shape[2]).to(feat.device)), dim=1)
            feat = torch.cat((odd_feat, even_feat), dim=2)

        pad_mask = self.cal_pad_mask(pad_mask, feat.shape[1])

        return feat, pad_mask

    def forward_rct(self, hidden):
        hidden = self.pre_rct_proj(hidden)
        rct_wavs = self.reconstruct_module(hidden.transpose(1, 2))
        return rct_wavs

    def forward(self, wave, pad_mask, task_id=None, get_hidden=False, no_pred=False):
        """
        Forward function
        Input:
            wave (FloatTensor): B x T_wave
            pad_mask (BoolTensor): B x T_wave
            task_id (LongTensor): N >= 1
        """

        feat, pad_mask = self.forward_feature(wave, pad_mask)

        if self.task_emb_type not in ["none", "expand-last", "self-hidden"]:
            if task_id is None:
                task_id = self.generate_task_id(feat.device)
            elif isinstance(task_id, list):
                task_id = torch.LongTensor(task_id).to(feat.device)
            task_embs = self.task_embedding(task_id)
            # N x D
            n_sz = len(task_id)
        else:
            n_sz = 1
        b_sz, t_sz, _ = feat.shape

        if self.task_emb_type == "add":
            # Add embs to feature
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1) + task_embs.unsqueeze(0).unsqueeze(2)
        elif self.task_emb_type == "concat":
            # Concatenates embs to feature
            feat_final = torch.cat(
                [
                    feat.unsqueeze(1).expand(-1, n_sz, -1, -1),
                    task_embs.unsqueeze(0).unsqueeze(2).expand(b_sz, -1, t_sz, -1),
                ],
                dim=-1,
            )
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat_final)
        else:
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1)
        # feat_final: B x N x T x D or B x 1 x T x D

        pad_mask = pad_mask.unsqueeze(1).expand(-1, n_sz, -1).reshape(b_sz * n_sz, t_sz)
        # BN x T
        feat_final = feat_final.reshape(b_sz * n_sz, t_sz, -1)
        # BN x T x D

        layer_hiddens = []
        if self.config.encoder_layers > 0:
            get_hidden_tmp = (
                True if (self.task_emb_type == "self-hidden") else get_hidden
            )
            hidden, layer_hiddens = self.encoder(
                feat_final, ~pad_mask.bool(), get_hidden=get_hidden_tmp
            )
        else:
            hidden = self.encoder(feat_final)

        if not no_pred:
            if self.task_emb_type == "self-hidden":
                pred = torch.stack([feat_final] + layer_hiddens, dim=1)
            elif self.config.multi_pred_head > 1:
                pred_list = []
                for idx in range(self.config.multi_pred_head):
                    pred_list.append(
                        getattr(self, 'output_layer{}'.format(idx))(hidden).reshape(b_sz, n_sz, t_sz, -1)
                    )
            else:
                pred = self.output_layer(hidden).reshape(b_sz, n_sz, t_sz, -1)
            # B x N x T x D
        else:
            pred = None

        if (not no_pred) and self.task_emb_type == "expand-last":
            assert n_sz == 1, n_sz
            if self.config.multi_pred_head > 1:
                pred_list = [pred.squeeze(1).reshape(b_sz, t_sz, self.n_tasks, -1).permute(0, 2, 1, 3) for pred in pred_list]
                pred = pred_list
            else:
                pred = (
                    pred.squeeze(1)
                    .reshape(b_sz, t_sz, self.n_tasks, -1)
                    .permute(0, 2, 1, 3)
                )
                # B x N x T x D
        ret = [feat, feat_final, pred, pad_mask]
        if get_hidden:
            ret.append(layer_hiddens)
        if self.config.wav_rct:
            rct_wavs = self.forward_rct(hidden)
            ret.append(rct_wavs)
        return ret

    def cal_pad_mask(self, pad_mask, max_len):
        """Calculates pad mask after conv."""
        pad_len = (pad_mask > 0).sum(1).long()

        # [Project Chimera]
        # mel pad mask
        if self.config.extractor_mode == 'mel':
            pad_len = pad_len // self.feature_extractor.hop_length

        # conv. pad mask
        else:
            for _, k_size, s_size in self.conv_layers:
                pad_len = (pad_len - k_size) // s_size + 1

        new_pad_mask = torch.ones(
            (pad_mask.shape[0], max_len), dtype=pad_mask.dtype, device=pad_mask.device
        )

        for idx in range(pad_len.shape[0]):
            new_pad_mask[idx, pad_len[idx] :] = 0

        return new_pad_mask

    def generate_task_id(self, device):
        return torch.arange(self.n_tasks, device=device, dtype=torch.long)
