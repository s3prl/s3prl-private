## SS: Speech Separation

#### Prepare data

```bash
cd scripts/LXT

# change the settings in prepare_LXT_sep.sh
./scripts/LXT/prepare_LXT_sep.sh

python data_prepare_LXT2Mix.py output_dir/LXT2Mix2hr5repeat ../../datasets/LXT2Mix2hr5repeat --part train
python data_prepare_LXT2Mix.py output_dir/LXT2Mix2hr5repeat ../../datasets/LXT2Mix2hr5repeat --part dev
python data_prepare_LXT2Mix.py output_dir/LXT2Mix2hr5repeat ../../datasets/LXT2Mix2hr5repeat --part test
```

#### Training
```bash
python3 run_downstream.py --mode train \
        -d separation_stft \
        -c downstream/separation_stft/configs/cfg_LXT1mix.yaml \
        -u wav2vec2 \
        -n ExpName \
```
