
## 1. Summary


## 2. Experimental Results


## 3. Instructions

**Environment Settings**

- 파이썬 버전 관리 도구로 `pyenv` 사용 (.python-version 파일 참고)
- 의존성 관리 도구로 `Poetry` 사용
- 상세 내용은 Usage Instructions - 2) Environment Setup 참고

**Code Structure**

```
📦 upstage_assignment
├─ data
│  ├─ train.csv (원본)
│  ├─ train_data.csv
│  ├─ valid_data.csv
│  └─ test_data.csv
├─ models
├─ notebooks
│  └─ plots
├─ outputs
├─ src
│  ├─ tools
│  │  ├─ __init__.py
│  │  ├─ dict_label_to_num.pkl
│  │  ├─ dict_num_to_label.pkl
│  │  └─ utils.py
│  ├─ config.py
│  ├─ dataset.py
│  ├─ inference.py
│  ├─ loss.py
│  ├─ metrics.py
│  ├─ model.py
│  └─ train.py
├─ train.sh
├─ inference.sh
├─ README.md
├─ .python_version
├─ poetry.lock
└─ pyproject.toml
```

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
    - pyenv 설치 후 진행
        ```
        pyenv install 3.10.13
        pyenv local 3.10.13
        ```

    - poetry 설치 후 진행

        ```
        poetry install # 의존성 설치
        poetry shell # 가상환경 활성화
        ```

    - `pyproject.toml` 파일 내용

        ```
        [tool.poetry]
        name = "upstage-assignment"
        version = "0.1.0"
        description = ""
        authors = ["Your Name <you@example.com>"]
        readme = "README.md"

        [tool.poetry.dependencies]
        python = "^3.10"
        requests = "^2.32.3"
        pandas = "^2.2.3"
        tqdm = "^4.67.1"
        torch = "^2.5.1"
        transformers = "^4.47.1"
        scikit-learn = "^1.6.0"
        numpy = "^2.2.0"


        [build-system]
        requires = ["poetry-core"]
        build-backend = "poetry.core.masonry.api"
        ```
    
    - (선택) pyenv의 파이썬 버전을 poetry가 사용하도록 설정
        ```
        poetry env use 3.10.13
        ```


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