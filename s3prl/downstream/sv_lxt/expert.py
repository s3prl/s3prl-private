# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
from sys import excepthook
import torch
import random
import pathlib
from pathlib import Path
from argparse import Namespace
#-------------#
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized, get_rank, get_world_size
#-------------#
from utility.helper import is_leader_process
from .model import Model, AMSoftmaxLoss, SoftmaxLoss, UtteranceExtractor
from .dataset import LxtSvTrain, LxtSvEval
from .utils import EER


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super().__init__()
        # config
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        self.train_dataset = LxtSvTrain(**self.datarc)

        # module
        self.connector = nn.Linear(self.upstream_dim, self.modelrc['input_dim'])

        # downstream model
        agg_dim = self.modelrc["module_config"][self.modelrc['module']].get(
            "agg_dim",
            self.modelrc['input_dim']
        )
        
        ModelConfig = {
            "input_dim": self.modelrc['input_dim'],
            "agg_dim": agg_dim,
            "agg_module_name": self.modelrc['agg_module'],
            "module_name": self.modelrc['module'], 
            "hparams": self.modelrc["module_config"][self.modelrc['module']],
            "utterance_module_name": self.modelrc["utter_module"]
        }
        # downstream model extractor include aggregation module
        self.model = Model(**ModelConfig)


        # SoftmaxLoss or AMSoftmaxLoss
        objective_config = {
            "speaker_num": self.train_dataset.speaker_num, 
            "hidden_dim": self.modelrc['input_dim'], 
            **self.modelrc['LossConfig'][self.modelrc['ObjectiveLoss']]
        }

        self.objective = eval(self.modelrc['ObjectiveLoss'])(**objective_config)
        # utils
        self.score_fn  = nn.CosineSimilarity(dim=-1)
        self.eval_metric = EER

        self.save_best_on = self.datarc.get("save_best_on", "lxt_dev")
        self.register_buffer('best_score', torch.ones(1) * 100)

    def get_dataloader(self, split, epoch=0):
        if "train" in split:
            return self._get_train_dataloader(self.train_dataset, epoch)
        else:
            dataset_name = f"{split}_dataset"
            if not hasattr(self, dataset_name):
                dataset = LxtSvEval(split, **self.datarc)
                setattr(self, dataset_name, dataset)
            dataset = getattr(self, dataset_name)
            return self._get_eval_dataloader(dataset)

    def _get_train_dataloader(self, dataset, epoch):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        if sampler is not None:
            sampler.set_epoch(epoch)

        return DataLoader(
            dataset,
            batch_size=self.datarc['train_batch_size'], 
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def forward(self, split, features, utter_idx, labels, records, **kwargs):
        features_pad = pad_sequence(features, batch_first=True)
        
        if self.modelrc['module'] == "XVector":
            # TDNN layers in XVector will decrease the total sequence length by fixed 14
            attention_mask = [torch.ones((feature.shape[0] - 14)) for feature in features]
        else:
            attention_mask = [torch.ones((feature.shape[0])) for feature in features]

        attention_mask_pad = pad_sequence(attention_mask,batch_first=True)
        attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        features_pad = self.connector(features_pad)

        if "train" in split:
            agg_vec = self.model(features_pad, attention_mask_pad.cuda())
            labels = torch.LongTensor(labels).to(features_pad.device)
            loss, logits = self.objective(agg_vec, labels)
            records['loss'].append(loss.item())
            records["acc"] += (logits.argmax(dim=-1) == labels).cpu().long().tolist()
            return loss
        
        else:
            agg_vec = self.model.inference(features_pad, attention_mask_pad.cuda())
            agg_vec = agg_vec / (torch.norm(agg_vec, dim=-1).unsqueeze(-1))

            # separate batched data to pair data.
            vec1, vec2 = self.separate_data(agg_vec)
            names1, names2 = self.separate_data(utter_idx)

            scores = self.score_fn(vec1, vec2).cpu().detach().tolist()
            records['scores'].extend(scores)
            records['labels'].extend(labels)
            records['pair_names'].extend(list(zip(names1, names2)))

            return torch.tensor(0)

    def log_records(self, split, records, logger, global_step, **kwargs):
        save_names = []

        if "train" in split:
            for key in ["loss", "acc"]:
                avg = torch.FloatTensor(records[key]).mean().item()
                logger.add_scalar(f"sv_lxt/{split}-{key}", avg, global_step=global_step)
                print(f"sv_lxt/{split}-{key}: {avg}")

        else:
            err, *others = self.eval_metric(np.array(records['labels']), np.array(records['scores']))
            logger.add_scalar(f'sv_lxt/{split}-EER', err, global_step=global_step)
            print(f'sv_lxt/{split}-ERR: {err}')

            if err < self.best_score and split == self.save_best_on:
                self.best_score = torch.ones(1) * err
                save_names.append(f'{split}-best.ckpt')

            with open(Path(self.expdir) / f"{split}_predict.txt", "w") as file:
                for (name1, name2), score in zip(records["pair_names"], records["scores"]):
                    print(score, name1, name2, file=file)

            with open(Path(self.expdir) / f"{split}_truth.txt", "w") as file:
                for (name1, name2), score in zip(records["pair_names"], records["labels"]):
                    print(score, name1, name2, file=file)

        return save_names

    def separate_data(self, agg_vec):
        assert len(agg_vec) % 2 == 0
        total_num = len(agg_vec) // 2
        feature1 = agg_vec[:total_num]
        feature2 = agg_vec[total_num:]
        return feature1, feature2
