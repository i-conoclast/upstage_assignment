python src/train.py \
    --model_name_or_path klue/roberta-large \
    --train_file data/train_data.csv \
    --valid_file data/valid_data.csv \
    --label2id_path tools/dict_label_to_num.pkl \
    --id2label_path tools/dict_num_to_label.pkl \
    --model_dir models \
    --num_epochs 10 \
    --save_model \
    --focal_loss \
    --label_smoothing 0.09 \
    --alpha 1.0 \
    --gamma 2.0 \
    --scheduler cosine \
    --use_entity_markers \
    --use_entity_types \
    --use_span_pooling 
