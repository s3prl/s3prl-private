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
    lrs=("1.0e-4" "1.0e-3")
else
    lrs=($*)
fi

for lr in "${lrs[@]}";
do
    expdir=$expdir_root/$upstream/lr$lr
    python3 run_downstream.py -a -m train -u $upstream -s SE -d enhancement_stft -o config.optimizer.lr=$lr \
        -p $expdir -c downstream/enhancement_stft/configs/cfg_LXT1mix.yaml

    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev

    if [ ! -f "$expdir/dev.result" ] || [ "$(cat $expdir/dev.result | grep "pesq" | wc -l)" -lt 1 ]; then
        python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t dev > $expdir/dev.result
    fi
    if [ ! -f "$expdir/test.result" ] || [ "$(cat $expdir/test.result | grep "pesq" | wc -l)" -lt 1 ]; then
        python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t test > $expdir/test.result
    fi
done
