## SE: Speech Enhancement

#### Prepare data

```bash
cd scripts/LXT

# change the settings in prepare_LXT_enh.sh
./generate_LXT_enh.sh

python data_prepare_LXT1Mix.py output_dir/LXT1Mix5hr2repeat ../../datasets/LXT1Mix5hr2repeat --part train
python data_prepare_LXT1Mix.py output_dir/LXT1Mix5hr2repeat ../../datasets/LXT1Mix5hr2repeat --part dev
python data_prepare_LXT1Mix.py output_dir/LXT1Mix5hr2repeat ../../datasets/LXT1Mix5hr2repeat --part test
```

#### Training
```bash
python3 run_downstream.py --mode train \
        -d enhancement_stft \
        -c downstream/enhancement_stft/configs/cfg_LXT1mix.yaml \
        -u wav2vec2 \
        -n ExpName \
```
