#!/bin/bash

set -x
set -e

if [ $# -lt "4" ]; then
    echo $0 [upstream] [train_utt] [seed] [expdir_root]
    exit 1
fi

upstream=$1
train_utt=$2
seed=$3
expdir_root=$4
shift 4

if [ -z "$*" ]; then
    lrs=("1.0e-3" "1.0e-4")
else
    lrs=($*)
fi

for lr in "${lrs[@]}";
do
    expdir=$expdir_root/train_utt$train_utt/$upstream/lr$lr/seed$seed
    python3 run_downstream.py -a -m train -u $upstream -s SID -d voxceleb1 -o config.optimizer.lr=$lr,,config.downstream_expert.datarc.train_utt=$train_utt,,config.downstream_expert.datarc.seed=$seed \
        -p $expdir

    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t dev > $expdir/dev.result    
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t test > $expdir/test.result
done
