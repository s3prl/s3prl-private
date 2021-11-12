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

lr_eer=$(mktemp)
for lr_dir in $(ls -d $upstream_dir/*);
do
    dev_score=$(cat $lr_dir/*/dev.result | grep EER | cut -d " " -f 2 | num avg)
    test_score=$(cat $lr_dir/*/test.result | grep EER | cut -d " " -f 2 | num avg)

    echo $(basename $lr_dir): dev $dev_score test $test_score >> $lr_eer
done

cat $lr_eer | sort -grk 3
rm $lr_eer
