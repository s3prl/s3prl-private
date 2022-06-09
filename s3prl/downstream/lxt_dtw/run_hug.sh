#!/bin/bash

set -e
set -x

if [ $# -lt "4" ]; then
    echo $0 [org] [repo] [revision] [expdir_root] [layer1] [layer2] ...
    exit 1
fi

org=$1
repo=$2
revision=$3
expdir_root=$4
shift 4
upstream=${org}__${repo}__${revision}

if [ -z "$*" ]; then
    tmpdir=$(mktemp -d)

    python3 run_downstream.py --upstream_feature_normalize -a -m train -u $org/$repo --upstream_revision $revision -s QbE -d example \
        -o config.runner.total_steps=1 \
        -p $tmpdir --hub huggingface &> $tmpdir/log
    layer_num=$(cat $tmpdir/log | grep "Take a list of" | cut -d " " -f 7)
    echo [LAYER INFO] $upstream has $layer_num layers.

    rm -rf $tmpdir

    layers=$(seq 0 $(($layer_num-1)))
else
    layers=("$*")
fi

for layer in ${layers[@]};
do
    expdir=$expdir_root/$upstream/layer$layer
    python3 run_downstream.py --upstream_feature_normalize -m evaluate -u $org/$repo --upstream_revision $revision -s QbE -l $layer -d lxt_dtw \
        -p $expdir --hub huggingface

    score_dir=$expdir/scoring
    mkdir -p $score_dir
    python3 lxt/compute_map.py --scores $expdir/scores.txt --output_dir $score_dir
done
