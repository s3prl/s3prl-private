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
    lrs=("1.0e-2")
else
    lrs=($*)
fi

# extract
if [ ! -d "$expdir_root/$upstream/extracted_feats/dev/all_data.ckpt" ]; then
    python3 run_downstream.py \
    --use_extracted_feature --extract_to_single_file --extract_scene_feature \
    -m extract -s SID -d voxceleb1 $args \
    -p $expdir_root/$upstream
fi
for lr in ${lrs[@]};
do
    expdir=$expdir_root/$upstream/lr$lr
    # train
    python3 run_downstream.py \
    --upstream_feature_normalize \
    --use_extracted_feature --extract_to_single_file --extract_scene_feature \
    --extracted_path $expdir_root/$upstream \
    -a -m train -s SID -d voxceleb1 $args \
    -o config.optimizer.lr=$lr \
    -p $expdir

    dev_ckpt=$(ls -t $expdir | grep -P ".*dev.*\.ckpt" | head -n 1)  # take the best checkpoint on dev
    # dev
    if [ ! -e "$expdir/dev.result" ]; then
        python3 run_downstream.py \
        -m evaluate -e $expdir/$dev_ckpt \
        -t dev > $expdir/dev.result
    fi
    # test
    if [ ! -e "$expdir/test.result" ]; then
        python3 run_downstream.py \
        -m evaluate -e $expdir/$dev_ckpt \
        -t test > $expdir/test.result
    fi
    # if [ -e "$expdir/dev.result" && -e "$expdir/test.result" ]; then
    #     rm -r $expdir_root/$upstream/extracted_feats
    # fi
done
