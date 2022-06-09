#!/bin/bash

set -x
set -e

if [ $# -lt "2" ]; then
    echo $0 [upstream] [expdir_root]
    exit 1
fi

upstream=$1
shift
expdir_root=$1
shift

if [ -z "$*" ]; then
    lrs=("1.0e-3" "1.0e-4" "1.0e-5")
else
    lrs=($*)
fi

for lr in "${lrs[@]}";
do
    expdir=$expdir_root/$upstream/lr$lr
    python3 run_downstream.py --upstream_feature_normalize -a -m train -u $upstream -s ASR -d lxt_asr -o config.optimizer.lr=$lr,,config.runner.total_steps=10000 \
        -p $expdir

    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    python3 run_downstream.py --upstream_feature_normalize -m evaluate -e $expdir/$dev_ckpt -t lxt_dev > $expdir/dev.result    
    python3 run_downstream.py --upstream_feature_normalize -m evaluate -e $expdir/$dev_ckpt -t lxt_test > $expdir/test.result
done

