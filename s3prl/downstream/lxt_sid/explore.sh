#!/bin/bash

set -x
set -e

if [ $# -lt "2" ]; then
    echo $0 [upstream] [min] [max] [projector_dim] [frame/utterance] [total steps] [expdir_root]
    exit 1
fi

upstream=$1
min=$2
max=$3
projector_dim=$4
granularity=$5
total_steps=$6
expdir_root=$7
shift 7

if [ -z "$*" ]; then
    lrs=("1" "1.0e-1" "1.0e-2" "1.0e-3")
else
    lrs=($*)
fi

for lr in "${lrs[@]}";
do
    expdir=$expdir_root/${min}_${max}secs/projector${projector_dim}/${granularity}/${upstream}/lr${lr}
    python3 run_downstream.py -a -m train -u $upstream -s SID -d lxt_sid -o config.optimizer.lr=$lr,,config.downstream_expert.modelrc.projector_dim=$projector_dim,,config.downstream_expert.modelrc.select=$granularity,,config.runner.total_steps=$total_steps,,config.downstream_expert.datarc.min_secs=$min,,config.downstream_expert.datarc.max_secs=$max \
        -p $expdir

    last_ckpt=$(ls -t $expdir | grep -P ".*states.*\.ckpt" | head -n 1) # take the last checkpoint for train acc
    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    python3 run_downstream.py -m evaluate -e $expdir/$last_ckpt -t train > $expdir/train.result
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t dev > $expdir/dev.result
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t test > $expdir/test.result
done
