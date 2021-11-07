#!/bin/bash

set -x
set -e

if [ $# -lt "6" ]; then
    echo $0 [upstream] [n_train] [seed] [total steps] [expdir_root]
    exit 1
fi

upstream=$1
n_train=$2
seed=$3
total_steps=$4
expdir_root=$5
shift 5

min=1
max=2
granularity=UtteranceLevel
projector_dim=0

if [ -z "$*" ]; then
    lrs=("1.0e-1" "1.0e-2" "1.0e-3" "1")
else
    lrs=($*)
fi

for lr in "${lrs[@]}";
do
    expdir=$expdir_root/${min}_${max}secs/projector${projector_dim}/${granularity}/train_seg${n_train}/${upstream}/lr${lr}/${seed}seed/
    python3 run_downstream.py -a -m train -u $upstream -s SID -d lxt_sid2 -o config.optimizer.lr=$lr,,config.downstream_expert.modelrc.projector_dim=$projector_dim,,config.downstream_expert.modelrc.select=$granularity,,config.runner.total_steps=$total_steps,,config.downstream_expert.datarc.min_secs=$min,,config.downstream_expert.datarc.max_secs=$max,,config.downstream_expert.datarc.seed=$seed,,config.downstream_expert.datarc.n_train=$n_train \
        -p $expdir

    last_ckpt=$(ls -t $expdir | grep -P ".*states.*\.ckpt" | head -n 1) # take the last checkpoint for train acc
    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    python3 run_downstream.py -m evaluate -e $expdir/$last_ckpt -t train > $expdir/train.result
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t dev > $expdir/dev.result
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t test > $expdir/test.result
done
