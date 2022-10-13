#!/bin/bash

set -x
set -e

if [ $# -ge "4" ]; then
    org=$1
    repo=$2
    revision=$3
    expdir_root=$4
    args="-u ${org}/${repo} --hub huggingface --upstream_revision ${revision}"
    upstream=${org}__${repo}__${revision}
    shift 4
elif [ $# -ge "2" ]; then
    upstream=$1
    expdir_root=$2
    args="-u ${upstream}"
    shift 2
else
    echo $0 \([org] [repo] [revision] [expdir_root]\) \| \([upstream] [expdir_root]\)
    exit 1
fi

if [ -z "$*" ]; then
    lrs=("1" "1.0e-1" "1.0e-2" "1.0e-3")
else
    lrs=($*)
fi

for lr in ${lrs[@]};
do
    expdir=$expdir_root/$upstream/lr$lr
    python3 run_downstream.py \
    --upstream_feature_normalize \
    -a -m train \
    $args \
    -s SID -d lxt_sid4 \
    -o config.optimizer.lr=$lr \
    -p $expdir

    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    python3 run_downstream.py \
    -m evaluate -e $expdir/$dev_ckpt \
    -t dev > $expdir/dev.result
    python3 run_downstream.py \
    -m evaluate -e $expdir/$dev_ckpt \
    -t test > $expdir/test.result
done