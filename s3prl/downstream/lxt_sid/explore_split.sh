#!/bin/bash

set -x
set -e

if [ $# -lt "2" ]; then
    echo $0 [upstream] [frame/utterance] [split_dir] [total steps] [expdir_root]
    exit 1
fi

upstream=$1
granularity=$2
split_dir=$3
total_steps=$4
expdir_root=$5
shift 5

min=1
max=2
projector_dim=0

if [ -z "$*" ]; then
    lrs=("1" "1.0e-1" "1.0e-2" "1.0e-3")
else
    lrs=($*)
fi

for lr in "${lrs[@]}";
do
    expdir=$expdir_root/$(basename $split_dir)/${min}_${max}secs/projector${projector_dim}/${granularity}/${upstream}/lr${lr}
    python3 run_downstream.py -a -m train -u $upstream -s SID -d lxt_sid -o config.optimizer.lr=$lr,,config.downstream_expert.modelrc.projector_dim=$projector_dim,,config.downstream_expert.modelrc.select=$granularity,,config.runner.total_steps=$total_steps,,config.downstream_expert.datarc.min_secs=$min,,config.downstream_expert.datarc.max_secs=$max,,config.downstream_expert.datarc.split_dir=$split_dir \
        -p $expdir

    last_ckpt=$(ls -t $expdir | grep -P ".*states.*\.ckpt" | head -n 1) # take the last checkpoint for train acc
    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    python3 run_downstream.py -m evaluate -e $expdir/$last_ckpt -t train > $expdir/train.result
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t dev > $expdir/dev.result
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t test > $expdir/test.result
done
