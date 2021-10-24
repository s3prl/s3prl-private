#!/bin/bash

set -x
set -e

if [ $# != "2" ]; then
    echo $0 [upstream] [expdir_root]
    exit 1
fi

upstream=$1
expdir_root=$2

for lr in "1" "1.0e-1" "1.0e-2" "1.0e-3";
do
    python3 run_downstream.py -a -m train -u $upstream -d sid_lxt -o config.optimizer.lr=$lr -p $expdir_root/${upstream}/${lr}
done
