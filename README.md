
## 1. Summary


## 2. Experimental Results


## 3. Instructions

**Environment Settings**

**Usage Instructions**

1) Data Preparation
    - data/ 디렉토리에 train_data.csv, valid_data.csv, test_data.csv 파일 준비
        - 기존 train.csv를 사용하여 sklearn의 train_test_split 함수를 통해 train_data.csv와 valid_data.csv 생성
        ```
        train, valid = train_test_split(train_data, test_size=0.2, 
        stratify=train_data["label"])
        
        train.reset_index(drop=True, inplace=True)
        valid.reset_index(drop=True, inplace=True)
        
        train.to_csv("data/train_data.csv", index=False)
        valid.to_csv("data/valid_data.csv", index=False)
        ```
2) Environment Setup
    ```
    poetry install
    poetry shell
    ````

3) Model Training
    ```
    bash train.sh
    ```

    OR

    ```
    python src/train.py \
    --train_file data/train_data.csv \
    --valid_file data/valid_data.csv \
    --model_name_or_path klue/roberta-large \
    --max_length 128 \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 3e-5 \
    --use_cuda True \
    --use_entity_markers True \
    --use_entity_types True \
    --use_span_pooling True \
    --dropout 0.1 \
    --label2id_path utils/dict_label_to_num.pkl \
    --id2label_path utils/dict_num_to_label.pkl \
    --model_dir models \
    --output_dir outputs \
    --focal_loss \
    --label_smoothing 0.09 \
    --alpha 1.0 \
    --gamma 2.0 \
    ```

    - 세부값은 조정 가능
    - 학습 결과는 models/ 디렉토리에 저장

4) Model Inference
    ```
    bash inference.sh
    ```

    OR

    ```
    python src/inference.py \
    --model_file models/{모델명}.pth \
    --output_dir outputs \
    --test_file data/test_data.csv \
    ```

    - 추론 결과는 csv 파일로 outputs/ 디렉토리에 저장
        - id, pred_label, probs 컬럼 포함
        - id는 원본 데이터의 id와 동일
        - pred_label은 예측 레이블
        - probs는 예측 확률

## 4. Approach

**Exploratory Data Analysis(EDA)**

**Model & Architecture**

**Training/Evaluation Scheme**

**Literature & Relevant Works**

**Future Work**