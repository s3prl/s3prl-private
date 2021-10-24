#!/bin/bash

set -x
set -e

if [ $# != "2" ]; then
    echo $0 [upstream] [expdir_root]
    exit 1
fi

upstream=$1
expdir_root=$2

for lr in "1" "1.0e-1" "1.0e-2" "1.0e-3" "1.0e-4";
do
    for seed in 1 2 3;
    do
        python3 run_downstream.py -m train -a -u $upstream -d sv_lxt -p $expdir_root/${upstream}/lr${lr}_seed${seed} \
            -o config.optimizer.lr=$lr \
            --seed $seed
    done
done
