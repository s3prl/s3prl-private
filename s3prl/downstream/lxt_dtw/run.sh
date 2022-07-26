#!/bin/bash

set -e
set -x

if [ $# -lt "2" ]; then
    echo $0 [upstream] [expdir_root] [layer1] [layer2] ...
    exit 1
fi

upstream=$1
expdir_root=$2
shift 2

if [ -z "$*" ]; then
    layer_info=$(mktemp)

    python3 downstream/lxt_dtw/get_layer_num.py --upstream $upstream --key QbE --output $layer_info
    layer_num=$(cat $layer_info)
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
        python3 run_downstream.py --upstream_feature_normalize -m evaluate -u $upstream -s QbE -l $layer -d lxt_dtw -p $expdir

        score_dir=$expdir/scoring
        mkdir -p $score_dir
        python3 lxt/compute_map.py --scores $expdir/scores.txt --output_dir $score_dir
    fi
done
