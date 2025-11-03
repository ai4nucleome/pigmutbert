#!/bin/bash
set -euo pipefail

name=$1
cuda=$2
use_lora=$3
lr=3e-5

echo "The provided name is $name"
echo "The provided cuda is $cuda"
echo "The provided use_lora is $use_lora"
echo "The provided lr is $lr"

for seed in 42
do
    for task in ATAC CTCF
    do
        for data in $(ls data/${task}); do 
            CUDA_VISIBLE_DEVICES=${cuda} python train_pig.py \
                --model_name_or_path models/${name} \
                --data_path  data/${task}/${data} \
                --kmer -1 \
                --run_name ${task}/${data} \
                --model_max_length 1024 \
                --factor 1.0 \
                --use_lora ${use_lora} \
                --lora_r 8 \
                --lora_dropout 0.05 \
                --lora_alpha 16 \
                --lora_target_modules 'query,value,key,dense' \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --gradient_accumulation_steps 1 \
                --learning_rate ${lr} \
                --num_train_epochs 3 \
                --fp16 \
                --save_steps 200 \
                --output_dir output/${name} \
                --eval_strategy steps \
                --eval_steps 200 \
                --warmup_steps 50 \
                --logging_steps 100000 \
                --tb_name pig_${task}_${data} \
                --overwrite_output_dir True \
                --log_level info \
                --seed ${seed} \
                --find_unused_parameters False
        done
    done

    for task in enhancer promoter
    do
        for data in $(ls data/${task}); do 
            CUDA_VISIBLE_DEVICES=${cuda} python train_pig.py \
                --model_name_or_path models/${name} \
                --data_path  data/${task}/${data} \
                --kmer -1 \
                --run_name ${task}/${data} \
                --model_max_length 512 \
                --factor 1.0 \
                --use_lora ${use_lora} \
                --lora_r 8 \
                --lora_dropout 0.05 \
                --lora_alpha 16 \
                --lora_target_modules 'query,value,key,dense' \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --gradient_accumulation_steps 1 \
                --learning_rate ${lr} \
                --num_train_epochs 3 \
                --fp16 \
                --save_steps 200 \
                --output_dir output/${name} \
                --eval_strategy steps \
                --eval_steps 200 \
                --warmup_steps 50 \
                --logging_steps 100000 \
                --tb_name pig_${task}_${data} \
                --overwrite_output_dir True \
                --log_level info \
                --seed ${seed} \
                --find_unused_parameters False
        done
    done

    for task in H3K27ac H3K27me1 H3K27me3
    do
        for data in $(ls data/${task}); do 
            CUDA_VISIBLE_DEVICES=${cuda} python train_pig.py \
                --model_name_or_path models/${name} \
                --data_path  data/${task}/${data} \
                --kmer -1 \
                --run_name ${task}/${data} \
                --model_max_length 1024 \
                --factor 2.0 \
                --use_lora ${use_lora} \
                --lora_r 8 \
                --lora_dropout 0.05 \
                --lora_alpha 16 \
                --lora_target_modules 'query,value,key,dense' \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --gradient_accumulation_steps 1 \
                --learning_rate ${lr} \
                --num_train_epochs 3 \
                --fp16 \
                --save_steps 200 \
                --output_dir output/${name} \
                --eval_strategy steps \
                --eval_steps 200 \
                --warmup_steps 50 \
                --logging_steps 100000 \
                --tb_name pig_${task}_${data} \
                --overwrite_output_dir True \
                --log_level info \
                --seed ${seed} \
                --find_unused_parameters False
        done
    done
done
