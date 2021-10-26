# Instructions for Diarization Preparation

The hidden-set preparation for diarization is almost the same as LibriMix (therefore, align with the training in our published S3PRL version). The processes are as follows:

### Step1: Generate Mixture from MetaData

```
cd ${Your_s3prl-private_repo}/preprocess/diarization/scripts

./generate_librimix.sh ${Your_unzipped_hiddenset}
```

### Step2: Prepare Kaldi-like Directory for dataset

```
python prepare_diarization.py \
    --target_dir ${Your_s3prl-private_repo}/s3prl/downstream/diarization/data \
    --source_dir "Libri2Mix/wav16k/max/metadata"
```

After you see the log "Successfully finish Kaldi-style preparation", the data is ready to use.
