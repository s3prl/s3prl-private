#!/bin/bash

data_root=/path/to/hidden/set/train_1hr
kaldi_root=/path/to/kaldi
s3prl_root=/path/to/s3prl

cd ${data_root}

# Get normalized transcriptions
awk '(ARGIND==1){a[$1]=1;} (ARGIND==2) {if ($1 in a) {print($0)}}' train.txt ../normalized_transcription.txt > train_transcriptions.txt
awk '(ARGIND==1){a[$1]=1;} (ARGIND==2) {if ($1 in a) {print($0)}}' dev.txt ../normalized_transcription.txt > dev_transcriptions.txt
awk '(ARGIND==1){a[$1]=1;} (ARGIND==2) {if ($1 in a) {print($0)}}' test.txt ../normalized_transcription.txt > test_transcriptions.txt

cat *_transcriptions.txt | cut -d" " -f 2- | tr " " "\n" | sort -u > train_1hr_word_units.txt

# get G2P model
mkdir g2p
wget https://www.openslr.org/resources/11/g2p-model-5

${kaldi_root}/kaldi/egs/librispeech/s5/local/g2p.sh train_1hr_word_units.txt g2p-model-5 train_1hr_lexicon_g2p.txt

mkdir ${s3prl_root}/s3prl/downstream/ctc_lxt/lexicon
cp train_1hr_lexicon_g2p.txt ${s3prl_root}/s3prl/downstream/ctc_lxt/lexicon/lxt_lexicon_g2p.txt

mkdir ${s3prl_root}/s3prl/downstream/ctc_lxt/vocab
cat ${s3prl_root}/s3prl/downstream/ctc_lxt/lexicon/lxt_lexicon_g2p.txt | sed '1d' | awk -d" " '{$1=""; print $0;}' | tr -s " " | tr " " "\n" | sort -u > ${s3prl_root}/s3prl/downstream/ctc_lxt/vocab/vocab_phone.txt
