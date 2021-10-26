#!/bin/bash

set -e
set -x

if [ $# != "2" ]; then
    echo $0 [upstream] [expdir_root]
    exit 1
fi

upstream=$1
expdir_root=$2

layer_info=$(mktemp)
python3 downstream/lxt_dtw/get_layer_num.py --upstream $upstream --key qbe --output $layer_info
layer_num=$(cat $layer_info)
rm $layer_info

echo [LAYER INFO] $upstream has $layer_num layers.
for layer in $(seq 0 $(($layer_num-1)));
do
    expdir=$expdir_root/$upstream/layer$layer
    python3 run_downstream.py -m evaluate -u $upstream -s qbe -l $layer -d lxt_dtw -p $expdir

    score_dir=$expdir/scoring
    mkdir -p $score_dir
    python3 lxt/compute_map.py --scores $expdir/scores.txt --output_dir $score_dir
done
