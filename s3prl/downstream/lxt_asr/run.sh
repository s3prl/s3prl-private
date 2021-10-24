#!/bin/bash

set -x
set -e

if [ $# != "2" ]; then
    echo $0 [upstream] [expdir_root]
    exit 1
fi

upstream=$1
expdir_root=$2

for lr in "1.0e-3" "1.0e-4" "1.0e-5";
do
    expdir=$expdir_root/$upstream/lr$lr
    python3 run_downstream.py -a -m train -u $upstream -d lxt_asr -o config.optimizer.lr=$lr \
        -p $expdir

    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t lxt_dev > $expdir/dev.result    
    python3 run_downstream.py -m evaluate -e $expdir/$dev_ckpt -t lxt_test > $expdir/test.result
done
