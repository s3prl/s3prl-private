#!/bin/bash

set -x
set -e

if [ $# != "6" ]; then
    echo $0 upstream split_dir projector_dim model_select min_secs max_secs.
    exit 1
fi

upstream=$1
split_dir=$2
projector_dim=$3
model_select=$4
min_secs=$5
max_secs=$6

for lr in 1.0e-3 1.0e-2 1.0e-1 1;
do
    python3 run_downstream.py -a -m train -u $upstream -d sid_lxt -o "\
        config.optimizer.lr=$lr,,\
        config.downstream_expert.datarc.split_dir=$split_dir,,\
        config.downstream_expert.datarc.min_secs=$min_secs,,\
        config.downstream_expert.datarc.max_secs=$max_secs,,\
        config.downstream_expert.modelrc.projector_dim=$projector_dim,,\
        config.downstream_expert.modelrc.select=$model_select" \
        -p result/downstream/lxt/sid_all/$(basename $split_dir)/${min_secs}_${max_secs}secs/project$projector_dim/$model_select/${upstream}_${lr}
done