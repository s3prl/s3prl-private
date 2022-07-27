#  Mel-HuBERT pre-training on Original HuBERT objective but with hard label instead 
## Basic information
This is the implementation of HuBERT which takes fbank feature as input. And its pre-training objective is to predict the hard clustering label of fbank feature on masked time steps.

## Data Preparing
### Libri-360
1. Download fbank features and clustering labels of libri-360 by the following command:
    ```
    wget http://140.112.30.56:9999/dataset/libri-360.tar.gz
    ```

2. Run the following script to prepare numpy data array for pre-training:
    ```
    python3 tools/tidy_libri360_kaldi_data.py [KALDI_DATA_DIR] [OUT_NUMPY_DIR] [DATA_CSV_FILE]
    ```
    Where KALDI_DATA_DIR is the directory you will get after decompressing in step 1. OUT_NUMPY_DIR is the output directory of numpy data array. DATA_CSV_FILE is the file recording data path and its corresponding label path which will be used in pre-trianing phase. The mean and standard variance of libri-360 will be saved at OUT_NUMPY_DIR/mean-std.npy

    This script will do normalization automatically for you.
    
    Notes: Please use absolute path when running the command.

3. Adjust your pre-training config file:
    - config_model.yaml: Adjust **data.audio.mean_std_npy_path** to OUT_NUMPY_DIR/mean-std.npy
    - config_runner.yaml: Adjust **pretrain_expert.datarc.sets**. Assume that your DATA_CSV_FILE is /path/to/data/csv/file/libri-360.csv, then **pretrain_expert.datarc.sets** should be ['/path/to/data/csv/file/libri-360.csv']    

### Libri-960
#### Stage 1 Pretraining
1. Download fbank features and clustering labels of libri-960 by the following command:
    ```
    wget http://140.112.30.56:9999/dataset/libri-960.tar.gz
    ```
   After decompressing, the folder structure should look like this:
   ```
   - libri-960
      - fbank
      - kmeans
      - stage2-cluster
   ```
   
2. Run the following script to prepare numpy data array for pre-training:
    ```
    python3 tools/tidy_libri960_kaldi_data.py [KALDI_FBANK_DIR] [KALDI_KMEANS_DIR] [OUT_NUMPY_DIR] [DATA_CSV_FILE]
    ```
    Where KALDI_FBANK_DIR is **libri-960/fbank**, KALDI_KMEANS_DIR is **libri-960/kmeans**. OUT_NUMPY_DIR is the output directory of numpy data array. DATA_CSV_FILE is the file recording data path and its corresponding label path which will be used in stage 1 pre-trianing phase. The mean and standard variance of libri-960 will be saved at OUT_NUMPY_DIR/mean-std.npy

    This script will do normalization automatically for you.
    
    Notes: Please use absolute path when running the command.

3. Adjust your pre-training config file:
    - config_model.yaml: Adjust **data.audio.mean_std_npy_path** to OUT_NUMPY_DIR/mean-std.npy
    - config_runner.yaml: Adjust **pretrain_expert.datarc.sets**. Assume that your DATA_CSV_FILE is /path/to/data/csv/file/libri-960-stage1.csv, then **pretrain_expert.datarc.sets** should be ['/path/to/data/csv/file/libri-960-stage1.csv']  

#### Stage 2 Pretraining
1. Please finish the preprocessing steps of stage 1 pretraining first (except step 3) to extract fbank feature and do normalize. You will only need OUT_NUMPY_DIR/feature/ for stage 2 pretraining, because OUT_NUMPY_DIR/cluster/ is the cluster assignments for stage 1 pretraining.

2. Run the following script to prepare the cluster assignments for stage 2 pretraining. (clustering result of the 8th layer representation of the first stage model)
    ```
    python3 tools/extract_label.py [KALDI_STAGE2_KMEANS_DIR] [NUMPY_STAGE2_KMEANS_DIR]
    ```
    Where KALDI_STAGE2_KMEANS_DIR is **libri-960/stage2-cluster**, NUMPY_STAGE2_KMEANS_DIR is the output numpy directory of the stage 2 cluster assignments.
3. Run the following script to create a .csv file for stage 2 pretraining
    ```
    python3 tools/create_csv.py [NUMPY_STAGE2_KMEANS_DIR] [NUMPY_FBANK_DIR] [DATA_STAGE2_CSV_FILE]
    ```
    Where NUMPY_STAGE2_KMEANS_DIR is the diretory you created in step 2, NUMPY_FBANK_DIR is OUT_NUMPY_DIR/feature/, DATA_STAGE2_CSV_FILE is the file recording data path and its corresponding label path which will be used in stage 2 pre-trianing phase.
4. Adjust your pre-training config file:
    - config_model.yaml: Adjust **data.audio.mean_std_npy_path** to OUT_NUMPY_DIR/mean-std.npy
    - config_runner.yaml: Adjust **pretrain_expert.datarc.sets**. Assume that your DATA_STAGE2_CSV_FILE is /path/to/data/csv/file/libri-960-stage2.csv, then **pretrain_expert.datarc.sets** should be ['/path/to/data/csv/file/libri-960-stage2.csv']  

5. You can see [here](#pre-training-with-weight-initialized) to know how to train the stage 2 model with the stage 1 model's weight initialized.

## Pre-training Command 
### Pre-training from scratch
Execute the following command to pretrain from scratch with default configuration
```
python3 run_pretrain.py -u mel_hubert_masked_prediction -g pretrain/mel_hubert_masked_prediction/config_model.yaml -c pretrain/mel_hubert_masked_prediction/config_runner.yaml -n EXP_NAME
```
-u: upstream name, corresponding to the directory name in pretrain/ \
-g: model config \
-c: runner config \
-n: experiment name, the default experiment result will be saved at result/pretrain/<exp_name> 
### Pre-training with weight initialized
Execute the following command to pretrain with weight initialized
```
python3 run_pretrain.py -u mel_hubert_masked_prediction -i Path/to/CkptFile -g pretrain/mel_hubert_masked_prediction/config_model.yaml -c pretrain/mel_hubert_masked_prediction/config_runner.yaml -n EXP_NAME
```
-u: upstream name, corresponding to the directory name in pretrain/ \
-i: initial weight will be loaded from this .ckpt file \
-g: model config \
-c: runner config \
-n: experiment name, the default experiment result will be saved at result/pretrain/<exp_name> 

Add --init_optimizer_from_initial_weight if you also want to initialize the optimizer from -i .ckpt file

### Resume training 
Execute the following command to resume the pre-training process which had been interrupted
```
python3 run_pretrain.py -e EXP_NAME
```

This function can only be used to resume the interrupted precess which was running on the same machine.

## Pretrained Models 
Pretrained models are saved at [here](http://140.112.30.56:9999/pretrained_model/) \
You can use wget to download the models \
Their pretraining config are saved at pretraining-config/
### Load models
```
import torch
from s3prl.upstream.mel_hubert_masked_prediction.model import MelHuBERTConfig, MelHuBERTModel
    
all_states = torch.load(model_ckpt, map_location="cpu")
upstream_config = all_states["Upstream_Config"]["hubert"]  
upstream_config = MelHuBERTConfig(upstream_config)
upstream_model = MelHuBERTModel(upstream_config).to(device)
state_dict = all_states["model"]
upstream_model.load_state_dict(state_dict)
upstream_model.eval() # If you are only used to extract representation
last_layer_feat, _, _, _, _, hidden_states, _ = upstream_model(mel_input, input_pad_mask, get_hidden=True, no_pred=True)
```
### Pretrained models performance
|                  Model name                  | Params | Phone Classification(PER%) | Phone Recognition(PER%) | Speaker Identificaiton(ACC%) |
|:--------------------------------------------:|:------:|:--------------------------:|:-----------------------:|:----------------------------:|
| causal-melhubert-APC-objective-30epochs.ckpt |  ~45M  |            23.1            |          35.85          |             57.8             |
|    melhubert-stage1-libri360-50epochs.ckpt   |  ~90M  |            16.5            |          23.04          |             63.59            |
|   melhubert-stage1-libri360-150epochs.ckpt   |  ~90M  |            14.12           |          16.08          |             65.81            |
| melhubert-stage1-libri360-200epochs.ckpt     | ~90M   |            13.61           |          15.10          |             64.75            |
|melhubert-stage2-libri360-100epochs-oldmask.ckpt|~90M| x |12.35|62.50|
|melhubert-stage1-libri960-100epochs.ckpt|~90M|x|13.68|72.48|
|melhubert-stage1-libri960-200epochs.ckpt|~90M|x|12.11|70.81|
### Notes 
- melhubert-stage2-libri360-100epochs-oldmask.ckpt is trained with an old masking strategy that we are not plannig to use anymore. The old masking strategy is dependent with the training batch size, which means that it will produce different amount of mask when using different batch size. This will make the training loss incomparable when using different batch size. **Please set [this line](https://github.com/JSALT-2022-SSL/s3compression/blob/clean_masking/s3prl/s3prl/upstream/mel_hubert_masked_prediction/model.py#L116) to True in order to switch to the old masking strategies if you want to utilize this checkpoint**. Also, please refer to specific config file located in pretraining-config/ to know the batch size we used to train this model. **You have to use exatly same batch size in order to make the training loss comparable with ours**.
