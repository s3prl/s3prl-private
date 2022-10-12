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
    tmpdir=$(mktemp -d)

    python3 run_downstream.py \
    -m train \
    $args \
    -s QbE -d example \
    -o config.runner.total_steps=1 \
    -p $tmpdir &> $tmpdir/log
    layer_num=$(cat $tmpdir/log | grep "Take a list of" | cut -d " " -f 7)
    echo [LAYER INFO] $upstream has $layer_num layers.

    rm $layer_info

    layers=$(seq 0 $(($layer_num-1)))
else
    layers=("$*")
fi

for layer in ${layers[@]};
do
    expdir=$expdir_root/$upstream/layer$layer
    if [ ! -f "$expdir/scoring/test.result" ] || [ "$(cat $expdir/scoring/test.result | grep "EER" | wc -l)" -lt 1 ]; then
        python3 run_downstream.py \
        --upstream_feature_normalize \
        -m evaluate \
        $args \
        -s QbE -l $layer -d lxt_dtw \
        -p $expdir

        score_dir=$expdir/scoring
        mkdir -p $score_dir
        python3 lxt/compute_map.py --scores $expdir/scores.txt --output_dir $score_dir
    fi
done
