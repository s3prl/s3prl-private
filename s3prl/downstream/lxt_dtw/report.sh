#!/bin/bash

if [ $# != "1" ]; then
    echo $0 [upstream_dir]
    exit 1
fi

upstream_dir=$1
if [ ! -d $upstream_dir ]; then
    echo $upstream_dir is not a directory
    exit 1
fi

# MAP
best_dev_line=$(grep "MAP" $upstream_dir/*/scoring/dev.result /dev/null | sort -nk 3 | tail -n 1)
best_dev_layer=$([[ $best_dev_line =~ layer([0-9]+) ]] && echo ${BASH_REMATCH[1]})
best_dev_score=$([[ $best_dev_line =~ ([0-9]+\.[0-9]+) ]] && echo ${BASH_REMATCH[1]})
test_line=$(grep "MAP" $upstream_dir/layer$best_dev_layer/scoring/test.result)
test_score=$([[ $test_line =~ ([0-9]+\.[0-9]+) ]] && echo ${BASH_REMATCH[1]})
echo MAP: $(basename $upstream_dir) layer $best_dev_layer: dev $best_dev_score, test $test_score

# EER
best_dev_line=$(grep "EER" $upstream_dir/*/scoring/dev.result /dev/null | sort -nrk 3 | tail -n 1)
best_dev_layer=$([[ $best_dev_line =~ layer([0-9]+) ]] && echo ${BASH_REMATCH[1]})
best_dev_score=$([[ $best_dev_line =~ ([0-9]+\.[0-9]+) ]] && echo ${BASH_REMATCH[1]})
test_line=$(grep "EER" $upstream_dir/layer$best_dev_layer/scoring/test.result)
test_score=$([[ $test_line =~ ([0-9]+\.[0-9]+) ]] && echo ${BASH_REMATCH[1]})
echo EER: $(basename $upstream_dir) layer $best_dev_layer: dev $best_dev_score, test $test_score
