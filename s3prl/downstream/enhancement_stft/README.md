## SE: Speech Enhancement

#### Prepare data

```bash
cd scripts/LXT

# change the settings in prepare_LXT_enh.sh
./generate_LXT_enh.sh

cd s3prl/s3prl
python3 downstream/enhancement_stft/scripts/LXT/data_prepare_LXT1Mix.py data/superb_all/LXT1Mix5hr2repeat downstream/enhancement_stft/datasets/LXT1Mix5hr2repeat --part train
python3 downstream/enhancement_stft/scripts/LXT/data_prepare_LXT1Mix.py data/superb_all/LXT1Mix5hr2repeat downstream/enhancement_stft/datasets/LXT1Mix5hr2repeat --part dev
python3 downstream/enhancement_stft/scripts/LXT/data_prepare_LXT1Mix.py data/superb_all/LXT1Mix5hr2repeat downstream/enhancement_stft/datasets/LXT1Mix5hr2repeat --part test
```

#### Training
```bash
python3 run_downstream.py --mode train \
        -d enhancement_stft \
        -c downstream/enhancement_stft/configs/cfg_LXT1mix.yaml \
        -u wav2vec2 \
        -n ExpName \
```
