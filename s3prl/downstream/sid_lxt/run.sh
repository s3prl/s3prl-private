#!/bin/bash

set -x
set -e

if [ $# != "5" ]; then
    echo $0 split_dir projector_dim model_select min_secs max_secs.
    exit 1
fi

split_dir=$1
projector_dim=$2
model_select=$3
min_secs=$4
max_secs=$5

for upstream in hubert_large_ll60k modified_cpc;
do
    for lr in "1.0e-3" "1.0e-2" "1.0e-1" "1";
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
done
