#!/bin/bash
#SBATCH --job-name=Hello_TWCC    ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --mem=65536
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        ## 該 task 索取 4 CPUs
#SBATCH --gres=gpu:1             ## 每個節點索取 1 GPUs
#SBATCH --time=96:00:00          ## 最長跑 10 分鐘 (測試完這邊記得改掉，或是直接刪除該行)
#SBATCH --account=ENT211069      ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gp2d         ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、gp4d(最長跑4天)

echo "0. setup environment"
module purge
module load miniconda3
conda activate superb

echo "1. excute"
declare -a path=(lxt_pr lxt_sid4 lxt_emotion lxt_asr lxt_dtw lxt_sv diarization enhancement_stft separation_stft speech_translation)
declare -a name=(PR SID ER ASR QBE ASV SD SE SS ST)

hug="huggingface"
org=""
repo=""
revision=""
model_name=""
task=$1
lr=$2

for i in {0..9}
do
    if [ $task == "all" ] || [ $task == ${name[i]} ]
    then
        echo start task: ${name[i]}
        start=$(date +%s)
        if [$hug == "huggingface"]
        then
            bash downstream/${path[i]}/run_hug.sh $org $repo $revision /work/twsacsq997/joseph1227/$model_name/${name[i]} $lr
        else
            bash downstream/${path[i]}/run.sh $model_name /work/twsacsq997/joseph1227/$model_name/${name[i]} $lr
        fi

        end=$(date +%s)
        echo $model_name,${name[i]},$((end - start)),sec >> /work/twsacsq997/joseph1227/runtime.log
        echo $model_name,${name[i]},$((end - start)),sec >> /work/twsacsq997/joseph1227/$model_name/runtime.log
    fi
done