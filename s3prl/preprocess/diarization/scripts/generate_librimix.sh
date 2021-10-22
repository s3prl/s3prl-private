#!/bin/bash
set -eu  # Exit on error

storage_dir=$1
hidden_set_dir=$storage_dir/1.1_distributed
wham_dir=$storage_dir/wham_noise
librimix_outdir=$storage_dir/

function Data() {
    if ! test -e $hidden_set_dir; then
        echo "Do not find hidden set at ${hidden_set_dir}"
        exit
    fi
}

function wham() {
    if ! test -e $wham_dir; then
        echo "Download wham_noise into $storage_dir"
        # If downloading stalls for more than 20s, relaunch from previous state.
        wget -c --tries=0 --read-timeout=20 https://storage.googleapis.com/whisper-public/wham_noise.zip -P $storage_dir
        unzip -qn $storage_dir/wham_noise.zip -d $storage_dir
        rm -rf $storage_dir/wham_noise.zip
    fi
}


Data &
wham & 

wait

# Path to python
python_path=python

# If you wish to rerun this script in the future please comment this line out.
# $python_path augment_train_noise.py --wham_dir $wham_dir

for n_src in 2; do
  metadata_dir=metadata
  $python_path create_mixture_from_metadata.py --librispeech_dir $hidden_set_dir \
    --wham_dir $wham_dir \
    --metadata_dir $metadata_dir \
    --librimix_outdir $librimix_outdir \
    --n_src $n_src \
    --freqs 16k \
    --modes max \
    --types mix_clean mix_both
done
