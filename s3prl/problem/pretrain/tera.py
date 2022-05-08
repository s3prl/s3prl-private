from s3prl.corpus.librispeech_for_pretrain import LibriSpeechForPretrain
from s3prl.dataset.pretrain_tera_pipe import (
    PretrainTaskPipe,
)
from s3prl.sampler import (
    MaxTimestampBatchSampler,
    FixedBatchSizeBatchSampler,
)
from s3prl.task.feat_reconstruction_task import FeatReconstructionTask
from s3prl.nn.tera_transformer import (
    TransformerConfig,
    TransformerModel,
    TransformerSpecPredictionHead
)
from torch.nn import L1Loss
from s3prl import Container


class Tera:
    Corpus = LibriSpeechForPretrain
    TrainData = PretrainTaskPipe
    TrainSampler = MaxTimestampBatchSampler
    ValidData = PretrainTaskPipe
    ValidSampler = FixedBatchSizeBatchSampler
    TestData = PretrainTaskPipe
    TestSampler = FixedBatchSizeBatchSampler
    ModelConfig = TransformerConfig
    Body = TransformerModel
    Head = TransformerSpecPredictionHead
    Task = FeatReconstructionTask
    Loss = L1Loss

    default_config = Container(
        Corpus=dict(
            train_split=["train-clean-100", "train-clean-360", "train-other-500"]
        ),
        TrainData=dict(
            mask_args = dict(
                position_encoding_size=768,    # int, this should be identical to `hidden_size`
                mask_proportion=0.15,          # float, mask this percentage of all spectrogram frames in each sequence at random during MAM training
                mask_consecutive_min=7,        # int, mask this amount of consecutive frames
                mask_consecutive_max=7,        # int, mask this amount of consecutive frames
                mask_allow_overlap=True,       # bool, allow overlap masking
                mask_bucket_ratio=1.5,         # float, only used when overlap is not allowed. sample a mask from each bucket in size of [sampled mask_consecutive * mask_bucket_ratio]
                mask_frequency=0.2,            # int, mask maximum this percentage of frequency bands, set to 0 for no frequency mask  
            ),
            noise_args = dict(
                noise_proportion=0.0,          # float, for this percentage of the time, Gaussian noise will be applied on all frames during MAM training, set to 0 for no noise
            ),
            audio_config = dict(
                win_ms=25,
                hop_ms=10,
                n_freq=201,
                n_mels=80,
                n_mfcc=13,
                input = {'channel': 0,
                        'cmvn': True,
                        'delta': 0,
                        'feat_type': 'mel',
                        'log': True},
                target = {'channel': 1,
                        'cmvn': True,
                        'delta': 0,
                        'feat_type': 'mel',
                        'log': True},
            ),
            target_level=-25,
        ),
        TrainSampler=dict(
            max_timestamp=16000 * 20,
            shuffle=True,
        ),
        ValidData=dict(),
        ValidSampler=dict(
            batch_size=2,
        ),
        TestData=dict(),
        TestSampler=dict(
            batch_size=2,
        ),
        ModelConfig=dict(
            hidden_size=768,                        # Size of the encoder layers and the pooler layer.
            num_hidden_layers=3,                    # Number of hidden layers in the Transformer encoder.
            num_attention_heads=12,                 # Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size=3072,                 # The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act="gelu",                      # The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob=0.1,                # The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob=0.1,       # The dropout ratio for the attention probabilities.
            initializer_range=0.02,                 # The sttdev of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps=1.e-12,                  # The epsilon used by LayerNorm.
            share_layer=False,                      # Share layer weights
            pre_layer_norm=False,                   # To apply the pre layer normalization technique introduced in: https://arxiv.org/abs/2002.04745
        ),
        Body=dict(
            input_dim=80,
            output_attentions=False, 
            keep_multihead_output=False, 
            with_input_module=True
        ),
        Head=dict(
            output_dim=80,
            input_dim=None, # automatically use hidden_state dim
        ),
        Task=dict(),
        Loss=dict(),
        Optimizer=dict(
            cls="torch.optim.AdamW",
            lr=2.0e-4,
        ),
        Trainer=dict(
            total_steps=1000000,
            log_step=50000,
            valid_step=50000,
            save_step=50000,
            gradient_clipping=5.0,
            gradient_accumulate_steps=4,
            use_valid=True,
            valid_metric="loss",
            valid_higher_better=False,
        ),
    )