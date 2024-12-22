python src/ensemble.py \
 --model_dir models \
 --model_files best_model_20241221_1,best_model_20241221_2 \
 --ensemble_mode logit \
 --output_dir outputs \
 --test_file data/test_data.csv \
 --label2id_path tools/dict_label_to_num.pkl \
 --id2label_path tools/dict_num_to_label.pkl 