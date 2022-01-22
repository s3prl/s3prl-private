## SS: Speech Separation

#### Prepare data

```bash
cd scripts/LXT

# change the settings in prepare_LXT_sep.sh
./generate_LXT_sep.sh

cd s3prl/s3prl
python3 downstream/separation_stft/scripts/LXT/data_prepare_LXT2Mix.py data/superb_all/LXT2Mix2hr5repeat downstream/separation_stft/datasets/LXT2Mix2hr5repeat --part train
python3 downstream/separation_stft/scripts/LXT/data_prepare_LXT2Mix.py data/superb_all/LXT2Mix2hr5repeat downstream/separation_stft/datasets/LXT2Mix2hr5repeat --part dev
python3 downstream/separation_stft/scripts/LXT/data_prepare_LXT2Mix.py data/superb_all/LXT2Mix2hr5repeat downstream/separation_stft/datasets/LXT2Mix2hr5repeat --part test
```

#### Training
```bash
python3 run_downstream.py --mode train \
        -d separation_stft \
        -c downstream/separation_stft/configs/cfg_LXT1mix.yaml \
        -u wav2vec2 \
        -n ExpName \
```
