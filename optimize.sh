#!/bin/bash

model_name_or_path="klue/roberta-small"
study_name="study_${model_name_or_path//\//_}"

python src/optimize.py \
    --model_name_or_path ${model_name_or_path} \
    --train_file data/train_data.csv \
    --valid_file data/valid_data.csv \
    --label2id_path tools/dict_label_to_num.pkl \
    --id2label_path tools/dict_num_to_label.pkl \
    --model_dir models \
    --study_name ${study_name} \
    --n_trials 10