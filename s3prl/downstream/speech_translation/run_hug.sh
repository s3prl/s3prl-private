#!/bin/bash

set -x
set -e

if [ $# -lt "4" ]; then
    echo $0 [org] [repo] [revision] [expdir_root]
    exit 1
fi

org=$1
repo=$2
revision=$3
expdir_root=$4
shift 4

if [ -z "$*" ]; then
    lrs=("1.0e-3")
else
    lrs=($*)
fi

for lr in "${lrs[@]}";
do
    expdir=$expdir_root/${org}__${repo}__${revision}/lr$lr
    python3 run_downstream.py --upstream_feature_normalize -a -m train -u $org/$repo --upstream_revision $revision --hub huggingface -s ST -d speech_translation -o config.optimizer.lr=$lr \
        -p $expdir

    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    python3 run_downstream.py --upstream_feature_normalize -m evaluate -e $expdir/$dev_ckpt -t dev > $expdir/dev.result
    python3 run_downstream.py --upstream_feature_normalize -m evaluate -e $expdir/$dev_ckpt -t test > $expdir/test.result
done
