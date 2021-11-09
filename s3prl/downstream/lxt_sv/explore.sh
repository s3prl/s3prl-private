#!/bin/bash

set -x
set -e

if [ $# -lt "5" ]; then
    echo $0 [upstream] [config] [total steps] [seed] [expdir_root]
    exit 1
fi

upstream=$1
config=$2
total_steps=$3
seed=$4
expdir_root=$5
shift 5

if [ -z "$*" ]; then
    lrs=("1.0e-1" "1.0e-2" "1.0e-3" "1")
else
    lrs=($*)
fi

for lr in "${lrs[@]}";
do
    expdir=$expdir_root/${upstream}/lr${lr}/seed${seed}/
    python3 run_downstream.py -a -m train -u $upstream -s ASV -d lxt_sv -o config.optimizer.lr=$lr,,config.downstream_expert.datarc.seed=$seed,,config.runner.total_steps=$total_steps \
        -p $expdir --config $config

    last_ckpt=$(ls -t $expdir | grep -P ".*states.*\.ckpt" | head -n 1) # take the last checkpoint for train acc
    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    python3 run_downstream.py -m evaluate -e $expdir/$last_ckpt -t train > $expdir/train.result
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t dev > $expdir/dev.result
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t test > $expdir/test.result
done
