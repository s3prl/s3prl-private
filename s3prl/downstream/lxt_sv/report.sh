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

for lr_dir in $(ls -d $upstream_dir/*);
do
    dev_score=$(cat $lr_dir/*/dev.result | grep EER | cut -d " " -f 2 | num mean)

    test_score=$(cat $lr_dir/*/test.result | grep EER | cut -d " " -f 2 | num mean)

    echo $(basename $lr_dir): dev $dev_score, test $test_score
done

