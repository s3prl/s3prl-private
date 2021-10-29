#!/bin/bash

# lxt_dir: the directory to LXT data
lxt_dir=/export/c12/hzili1/dataset/lxt_v1/1.1_distributed
metadata_dir=`pwd`/LibriMix/metadata
# wham_dir: the directory to wham noise data
wham_dir=/export/c12/hzili1/dataset/wham_noise
# number of speakers (1 for enhancement, 2 for separation)
n_srcs=2
# the data split we use for experiment (1 for train_1hr, 2 for train_2hr, 5 for train_5hr)
n_hours=2
# number of times to repeat each utterance
n_repeat=5

python LibriMix/scripts/create_lxt_metadata.py \
	--num_hours $n_hours $lxt_dir $metadata_dir

python LibriMix/scripts/create_lxtmix_metadata.py \
	--lxt_dir $lxt_dir \
	--lxt_md_dir $metadata_dir/LXT${n_hours}hr \
	--wham_dir $wham_dir \
	--wham_md_dir $metadata_dir/Wham_noise \
	--metadata_outdir $metadata_dir/LXT${n_srcs}Mix${n_hours}hr${n_repeat}repeat \
	--n_src $n_srcs \
	--n_repeat $n_repeat
