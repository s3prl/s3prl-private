# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DistributedSampler

from ..model import *
from .dataset import LxtSid

LXT_SPEAKER_NUM = 60

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})

        projector_dim = self.modelrc['projector_dim']
        if projector_dim <= 0:
            self.projector = lambda x: x
            latest_dim = upstream_dim
        else:
            self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
            latest_dim = projector_dim

        self.model = model_cls(
            input_dim = latest_dim,
            output_dim = LXT_SPEAKER_NUM,
            **model_conf,
        )
        self.objective = nn.CrossEntropyLoss()

        self.expdir = Path(expdir)
        self.save_best_on = downstream_expert.get("save_best_on", "dev")
        self.save_best_metric = downstream_expert.get("save_best_metric", "acc")
        self.register_buffer('best_score', torch.ones(1) * -1<<31)

    def _get_train_dataloader(self, dataset, epoch):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn,
        )

    def get_dataloader(self, split, epoch=0):
        key = f"{split}_dataset"
        if not hasattr(self, key):
            dataset = LxtSid(split, **self.datarc)
            setattr(self, key, dataset)
        dataset = getattr(self, key)

        if "train" in split:
            return self._get_train_dataloader(dataset, epoch)
        else:
            return self._get_eval_dataloader(dataset)

    def forward(self, split, features, labels, uids, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)
        labels = torch.LongTensor(labels).to(features.device)
        
        frame_level = False
        if predicted.dim() == 3:
            frame_level = True
            frames, frame_labels, frame_uids = [], [], []
            for p, l, label, uid in zip(predicted, features_len, labels, uids):
                frames.append(p[:l])
                frame_labels.append(label.expand(l))
                frame_uids.extend([f"{uid}:{t}" for t in range(l)])
            predicted = torch.cat(frames, dim=0)
            labels = torch.cat(frame_labels, dim=0)
            uids = frame_uids

        loss = self.objective(predicted, labels)
        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())
        if frame_level:
            utt_predicteds = predicted.split(features_len.cpu().tolist())
            utt_labels = labels.split(features_len.cpu().tolist())
            def split_uids(uids, features_len):
                start = 0
                for length in features_len:
                    yield uids[start : start + length]
                    start = start + length
            utt_uids = list(split_uids(uids, features_len))
            for utt_predicted, utt_label, utt_uid in zip(utt_predicteds, utt_labels, utt_uids):
                classids = (utt_predicted.max(dim=-1).indices.tolist())
                classid = sorted([(c, classids.count(c)) for c in list(set(classids))], key=lambda x: x[1], reverse=True)[0][0]
                records["utt-acc"].append(float(classid == utt_label[0]))
                records["utt-predict_speaker"].append(classid)
                records["utt-truth_speaker"].append(utt_label[0])
                records["utt-uid"].append(utt_uid[0].split(":")[0])

        records["filename"] += uids
        records["predict_speaker"] += list(predicted_classid)
        records["truth_speaker"] += list(labels)

        return loss

    def log_records(self, split, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss", "utt-acc"]:
            if key not in records:
                continue
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'sid_lxt/{split}-{key}',
                average,
                global_step=global_step
            )
            print(f"{split} {key}: {average}")
            with open(self.expdir / "log.log", 'a') as f:
                if key == self.save_best_metric:
                    f.write(f'{split} at step {global_step}: {average}\n')
                    if split == self.save_best_on and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {split} at step {global_step}: {average}\n')
                        save_names.append(f'{split}-best.ckpt')

        if split in ["dev", "test"]:
            with open(Path(self.expdir) / f"{split}_predict.txt", "w") as file:
                lines = [f"{f} {getattr(self, f'{split}_dataset').spkrs[p]}\n" for f, p in zip(records["filename"], records["predict_speaker"])]
                file.writelines(lines)

            with open(Path(self.expdir) / f"{split}_truth.txt", "w") as file:
                lines = [f"{f} {getattr(self, f'{split}_dataset').spkrs[l]}\n" for f, l in zip(records["filename"], records["truth_speaker"])]
                file.writelines(lines)
            
            if "utt-acc" in records:
                with open(Path(self.expdir) / f"{split}_utt_predict.txt", "w") as file:
                    lines = [f"{f} {getattr(self, f'{split}_dataset').spkrs[p]}\n" for f, p in zip(records["utt-uid"], records["utt-predict_speaker"])]
                    file.writelines(lines)

                with open(Path(self.expdir) / f"{split}_utt_truth.txt", "w") as file:
                    lines = [f"{f} {getattr(self, f'{split}_dataset').spkrs[l]}\n" for f, l in zip(records["utt-uid"], records["utt-truth_speaker"])]
                    file.writelines(lines)

        return save_names
