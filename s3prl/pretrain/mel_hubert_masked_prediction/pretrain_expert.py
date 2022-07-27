"""
MelHuBERT pretrain expert.
Author: Tzu-Quan Lin (https://github.com/nervjack2)
Reference: (https://github.com/s3prl/s3prl/blob/master/s3prl/pretrain/apc/pretrain_expert.py)
Reference author:  Andy T. Liu (https://github.com/andi611)
"""

import re
import yaml
import copy
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#-------------#
from pretrain.mel_hubert_masked_prediction.dataset import MelFeatDataset
from utility.audio import plot_spectrogram_to_numpy


class UpstreamPretrainExpert(nn.Module):
    """
    The MelHuBERT pretrain expert
    """

    def __init__(self, datarc, upstream_config, initial_weight=None, device='cuda', multi_gpu=False, **kwargs):
        super(UpstreamPretrainExpert, self).__init__()

        self.datarc = datarc
        self.initial_weight = initial_weight
        self.device = device
        self.multi_gpu = multi_gpu
        
        # prune
        self.prune_regax = r".*encoder\.layers\.[0-9]+\.((self_attn\.([qkv]|out)_proj)|fc[12])\.weight"

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(open(upstream_config, 'r'), Loader=yaml.FullLoader)
            print('[UpstreamPretrainExpert] - Using upstream config from:', upstream_config)
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print('[UpstreamPretrainExpert] - Using upstream config from the previous experiment.')
        else:
            raise ValueError
        
        preprocessor = self._init_model()
        self._get_train_dataloader(preprocessor)

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[UpstreamPretrainExpert] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[UpstreamPretrainExpert] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _init_model(self):
        from upstream.mel_hubert_masked_prediction.audio import create_transform
        from upstream.mel_hubert_masked_prediction.model import MelHuBERTModel, MelHuBERTConfig

        try:
            print('[UpstreamPretrainExpert] - Using the preprocessor, on-the-fly feature preprocessing')
            preprocessor, feat_dim = create_transform(copy.deepcopy(self.upstream_config['data']['audio']))
        except Exception as e:
            raise NotImplementedError('Our upstream wrapper currently does not support other feature extracters\n' + str(e))
        
        print('[UpstreamPretrainExpert] - Initializing model...')
        self.config = MelHuBERTConfig(self.upstream_config['hubert'])
        self.model = MelHuBERTModel(self.config)
   
        if self.initial_weight:
            all_states = torch.load(self.initial_weight, map_location="cpu")
            try:             
                self.model.load_state_dict(all_states["model"])
                print(f'[UpstreamPretrainExpert] Load initilization model weight from {self.initial_weight}')
            except:
                raise NotImplementedError('Could not load the initilization weight')
    
        if 'adjust_init_weight' in self.upstream_config.keys():
            self.model.adjust_arch(self.upstream_config['adjust_init_weight'])
            self.upstream_config['hubert']['num_cluster'] = self.upstream_config['adjust_init_weight']['num_cluster']
            print(f'[UpstreamPretrainExpert] Adjust model architecture according to the config.')
 
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100, size_average=True)
        return preprocessor

    def _get_train_dataloader(self, preprocessor):
        dataset = MelFeatDataset(
            preprocessor,
            self.upstream_config['task'],
            self.datarc['train_batch_size'],
            **self.datarc
        )
        self.dataloader = DataLoader(
            dataset, 
            batch_size=1, # for bucketing
            shuffle=True, 
            num_workers=self.datarc['num_workers'],
            drop_last=False, 
            pin_memory=True, 
            collate_fn=dataset.collate_fn
        )

    # Interface
    def load_model(self, init_ckpt, strict=True):
        assert 'model' in init_ckpt
        if self.multi_gpu:
            self.model.module.load_state_dict(init_ckpt['model'], strict=strict)
        else:
            self.model.load_state_dict(init_ckpt['model'], strict=strict)

    # Interface
    def loss_to_device(self):
        self.loss.to(self.device)

    # Interface
    def add_state_to_save(self, all_states):
        all_states['model'] = self.model.state_dict() if not self.multi_gpu else \
                                 self.model.module.state_dict()
        all_states['Upstream_Config'] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [audio feature, cluster id, padding mask, audio length]
            
            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step
        Return:
            loss        
        """
        audio_feat, label, pad_mask, audio_len = data[0], data[1], data[2], data[3]
        audio_feat = audio_feat.to(self.device)
        label = label.to(self.device)
        pad_mask = pad_mask.to(self.device)
  
        _, logit_m, logit_u, label_m, label_u, _, _ = self.model(audio_feat, pad_mask, label, mask=True)

        loss = 0.0 
        if logit_m != None and label_m != None and self.config.pred_masked_weight > 0: 
            loss += self.config.pred_masked_weight * self.loss(logit_m, label_m)
        if logit_u != None and label_u != None and self.config.pred_nomask_weight > 0: 
            loss += self.config.pred_nomask_weight * self.loss(logit_u, label_u)
        
        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass
    
    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended
            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents
            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'
            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                logger.add_image(f"{prefix}{key}", values, global_step=global_step)
            elif isinstance(values, float):
                logger.add_scalar(f"{prefix}{key}", values, global_step=global_step)
               
    # interface 
    def get_params_to_prune(self):
        params_to_prune = tuple()

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        for layer in model.encoder.layers:
            params_to_prune = (
                *params_to_prune,
                # self attention layer
                (layer.self_attn.q_proj, "weight"),
                (layer.self_attn.k_proj, "weight"),
                (layer.self_attn.v_proj, "weight"),
                (layer.self_attn.out_proj, "weight"),
                # fc layer
                (layer.fc1, "weight"),
                (layer.fc2, "weight"),
            )
            
        def name_filter(name):
            return re.fullmatch(self.prune_regax, name)
            
        return params_to_prune, name_filter