
* 자세한 내용은 pdf 파일 참고([문장 내 개체간 관계 추출 분석 보고서.pdf](./문장%20내%20개체간%20관계%20추출%20분석%20보고서.pdf))

## 1. Summary

본 레포지토리에서는 한국어 문장에서 두 엔티티 간 관계를 추출하는 태스크를 다루고 있습니다.  
데이터 탐색(EDA) 결과, 특히 `no_relation` 클래스로 인해 심각한 불균형 문제가 발생한다는 점을 확인했습니다.   
또한, 특정 엔티티 타입 조합(예: ORG-PER)에서 특정 관계가 집중적으로 발생한다는 패턴이 발견되었고,   
일부 존재하는 긴 문장에 대한 정보 손실을 줄이지 않도록 max_length를 적절한 수준(128~160)으로 설정했습니다.   
이러한 분석에 따라 엔티티 마커(타입 정보 포함 가능), Focal Loss, 그리고 linear/cosine/polynomial LR 스케줄러 등을 적용해 모델 성능을 개선했습니다.   
평가에서는 과제의 평가지표인 no_relation을 제외한 Micro-F1, AUPRC를 지표로 사용했습니다.   
추가적으로, 엔티티 스팬 풀링과 Attention 기반 풀링 등을 추가 적용해 엔티티 내부 토큰에의 집중도를 높여 성능을 높이고자 했습니다.

## 2. Experimental Results

* 검증 데이터셋 기준
- base model : klue/roberta-base

| Model Setting                                           | Micro-F1 (no_relation excl.) | AUPRC  |
|---------------------------------------------------------|------------------------------|--------|
| (A) Model Only                                          | 0.5942                       | 0.4824 |
| (B) Entity Marker                                       | 0.8288                       | 0.7757 |
| (C) Entity Marker + Focal Loss                          | 0.8327                       | 0.7890 |
| (D) Entity Marker + Focal Loss + Span Pooling(mean)     | 0.8505                       | 0.7972 |
| (E) Entity Marker + Focal Loss + Span Pooling(attention)| 0.8513                       | 0.8050 |

* 검증 데이터셋 기준
- 앙상블 대상 모델 (단일 모델)

| Model                                | Optimized Parameters                                                                                                                                                                                                                                                                                                                                   | Micro-F1 (no_relation excl.) | AUPRC  |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|--------|
| (A) deberta-korean-base (optimized)  | batch_size : 8 <br> num_epochs : 8 <br> learning_rate : 2.9361705292170236e-05 <br> max_length : 128 <br> dropout : 0.2 <br> scheduler : linear <br> focal_loss : true <br> label_smoothing : 0.05 <br> alpha : 0.25 <br> gamma : 2.0 <br> span_pooling : false <br> entity_markers : true <br> entity_types : false                             | 0.8191                         | 0.7431 |
| (B) koelectra-v3-discriminator (optimized) | batch_size : 16 <br> num_epochs : 6 <br> learning_rate : 2.8262768586139738e-05 <br> max_length : 128 <br> dropout : 0.0 <br> scheduler : linear <br> focal_loss : false <br> label_smoothing : 0.2 <br> span_pooling : false <br> entity_markers : true <br> entity_types : true                                                              | 0.8389                         | 0.7648 |
| (C) roberta-large (optimized)        | batch_size : 32 <br> num_epochs : 3 <br> learning_rate : 2.80026895611933e-05 <br> max_length : 200 <br> dropout : 0.0 <br> scheduler : linear <br> focal_loss : false <br> label_smoothing : 0.15000000000000002 <br> span_pooling : true <br> attention_pooling : false <br> entity_markers : true <br> entity_types : false | 0.8593                         | 0.8235 |

- 앙상블 모델

| Model                                                                              | Micro-F1 (no_relation excl.) | AUPRC   |
|------------------------------------------------------------------------------------|------------------------------|---------|
| (D) (A) + (B) + (C) soft voting                                                    | 69.5315                      | 73.0182 |
| (E) (A) + (B) + (C) logit averaging                                                | 69.4768                      | 73.4883 |
| (F) (A) + (B) + (C) weighted logit averaging ( (A) : 0.1, (B) : 0.2, (C): 0.7 )     | 71.7391                      | 75.7794 |


* 테스트 데이터 셋 평가 결과

| Model Variant            | Loss Function            | Label Smoothing | LR Scheduler | Entity Marker (Type) | Span Pooling | Attention Pooling | Micro-F1 (no_relation excl.) | AUPRC   |
|--------------------------|--------------------------|-----------------|-------------|----------------------|--------------|-------------------|------------------------------|---------|
| RoBERTa base            | Focal Loss               | 0.1             | Linear      | Yes (With Type)      | Yes          | No                | 65.4346                       | 59.9928 |
| RoBERTa large           | Focal Loss               | 0.1             | Linear      | Yes (With Type)      | Yes          | No                | 67.3219                       | 66.0349 |
| RoBERTa large           | Focal Loss (Optimized)   | 0.09            | Linear      | Yes (With Type)      | Yes          | No                | 68.7285                       | 68.9031 |
| RoBERTa large           | Focal Loss (Optimized)   | 0.09            | Cosine      | Yes (With Type)      | Yes          | No                | 71.1004                       | 72.6258 |

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
│  ├─ 01.EDA.ipynb
│  ├─ 02.Result Analysis.ipynb
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
│  ├─ loss.py
│  ├─ metrics.py
│  ├─ model.py
│  ├─ train.py
│  ├─ inference.py
│  ├─ optimize.py
│  └─ ensemble.py
├─ train.sh
├─ inference.sh
├─ optimize.sh
├─ ensemble.sh
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
        authors = ["sora kim <icon_o_clast@naver.com>"]
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
        matplotlib = "^3.10.0"
        seaborn = "^0.13.2"
        optuna = "^4.1.0"
        optuna-dashboard = "^0.17.0"
        accelerate = "^1.2.1"
        safetensors = "^0.4.5"


        [tool.poetry.group.dev.dependencies]
        ipykernel = "^6.29.5"

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
    sh train.sh
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
    --use_cuda \
    --use_entity_markers \
    --use_entity_types \
    --use_span_pooling \
    --use_attention_pooling \
    --dropout 0.1 \
    --label2id_path tools/dict_label_to_num.pkl \
    --id2label_path tools/dict_num_to_label.pkl \
    --model_dir models \
    --focal_loss \
    --label_smoothing 0.09 \
    --alpha 1.0 \
    --gamma 2.0 \
    ```

    - 세부값은 조정 가능
    - 학습 결과는 models/ 디렉토리에 저장

4) Model Inference
    ```
    sh inference.sh
    ```

    OR

    ```
    python src/inference.py \
    --model_file {모델명} \
    --model_dir models \
    --output_dir outputs \
    --test_file data/test_data.csv \
    --use_cuda 
    ```

    - 추론 결과는 csv 파일로 outputs/ 디렉토리에 저장
        - id, pred_label, probs 컬럼 포함
        - id는 원본 데이터의 id와 동일
        - pred_label은 예측 레이블
        - probs는 예측 확률

5) Model Optimization
    ```
    bash optimize.sh
    ```

    OR

    ```
    python src/optimize.py \
    --train_file data/train_data.csv \
    --valid_file data/valid_data.csv \
    --model_name_or_path klue/roberta-large \
    --model_dir models \
    --study_name {study_name} \
    --n_trials 10 \
    --save_model \
    --use_cuda 
    ```

6) Model Ensemble
    ```
    sh ensemble.sh
    ```

    OR

    ```
    python src/ensemble.py \
    --model_dir models \
    --model_files best_model_20241224_100000,best_model_20241224_100001 \
    --ensemble_mode logit \
    --output_dir outputs \
    --test_file data/test_data.csv \
    --use_cuda 
    ```

## 4. Approach

**Exploratory Data Analysis(EDA)**
- 라벨 분포: `no_relation` 클래스가 우세하고, 다수의 희소 클래스가 존재함을 확인했습니다. 모델이 쉽게 `no_relation`으로만 예측하는 편향을 막기 위해 불균형을 해소하는 전략이 필요한 것을 확인했습니다.

- 문장 길이 & 엔티티 타입: 평균 100자내외, 최장 400자 이상인 경우도 있어 max_length 128-160 정도가 적절하다고 판단했습니다. ORG-PER 등 특정 타입 조합에서 특정 관계(org:top_members/employees)가 집중되는 패턴이 드러나, 타입 정보를 활용하면 성능 향상이 있을 것이라 예상했습니다. 이를 통해 엔티티 타입 정보를 직접적으로 모델에 알려주면 (E1-ORG, E2-PER 등) 관계 추출 성능이 향상될 것이라는 가정을 세웠습니다.

- 엔티티 내부 토큰 분석: 특정 관계 라벨에서 엔티티 내부 토큰이 핵심 정보를 포함하는 것을 확인했습니다. 이를 통해 엔티티 스팬 풀링 방식을 적용하면 성능 향상이 있을 것이라 예상했습니다.

**Model & Architecture**
- Backbone Model
    - KoELECTRA, KLUE-RoBERTa, KoBERT 등 한국어 특화 사전학습모델(PLM) 중 KLUE-RoBERTa를 주로 사용했습니다.
    - 문장 입력 후 마지막 hidden state를 활용하되, 단순 [CLS] 임베딩 대신 엔티티 스팬 임베딩이나 Attention Pooling을 사용하도록 개선하였습니다.

- Entity Marker
    - EDA 결과, 엔티티 타입이 특정 관계와 밀접하므로 마커에 타입까지 명시하였습니다.([E1-ORG], [E2-PER] 등)
    - 모델 인코딩 시, 이 마커 정보를 별도의 스페셜 토큰으로 처리해 엔티티 주변 토큰에 Attention이 집중되도록 유도했습니다.

- Span Pooling / Attention Pooling
    - Span Pooling: 엔티티 구간(예: [E1] ~ [/E1]) 임베딩을 평균 pooling으로 모으고, 이를 최종 관계 분류에 활용했습니다.
    - Attention Pooling: 엔티티 스팬 내 토큰마다 learnable weight(Attention)를 적용해 중요한 토큰에 더 큰 가중치를 부여, 엔티티 정보를 더욱 정확히 추출했습니다.


**Training/Evaluation Scheme**

- Loss Function & 불균형 대응
    - Focal Loss 사용: $\gamma$와 $\alpha$를 탐색하며 희소 클래스에 대한 학습을 강화했습니다.
    - 기존 Cross Entropy 대비 희소 클래스 식별력이 향상됨을 확인했습니다.
    
- Learning Rate Scheduler
    - Linear, Cosine, Polynomial 스케줄러 모두 실험하였습니다.
    
- 하이퍼파라미터 튜닝
    - max_length: [128, 160, 200] 값을 실험하였습니다.

- 평가 지표
    - no_relation 제외 Micro-F1: 희소 클래스 인식력을 중시하는 지표입니다.    
    - AUPRC: 모든 클래스(포함 no_relation)에 대한 정밀도-재현율 곡선을 종합적으로 평가합니다.

- 전체 학습 프로세스
    - Baseline(RoBERTa+CE) → Focal Loss 추가 → 엔티티 마커/타입 표기 → 스팬 풀링/Attention 기반 pooling → 스케줄러 변경(Linear→Cosine) 등 단계적으로 개선하는 과정을 거쳤습니다.
    - validation set에서 Micro-F1, AUPRC 측정 후 best 모델을 선정했습니다.
    - 이후 `Optuna`라이브러리를 이용하여 각 모델의 하이퍼파라미터를 튜닝 후, 최적 모델을 각각 선정 후 앙상블 진행

**Literature & Relevant Works**

1.	KLUE Benchmark 및 RE 태스크([KLUE-RE](https://github.com/KLUE-benchmark/KLUE/tree/main))
    - 한국어 전용 벤치마크로, KLUE-RE에서 다양한 baseline 및 SOTA 모델이 제시되었습니다.

2.	Focal Loss([Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002))
    - Tsung-Yi Lin (2017)에서 제안된 기법으로, $\gamma$, $\alpha$ 조정 시 불균형 데이터에서의 희소 클래스 식별력이 개선된 것을 보여주고 있습니다.

3.	Entity Marker 연구([An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/abs/2102.01373))
    - 기존 NER, RE 연구에서 [E1], [E2] 마커를 삽입 시 모델이 엔티티 위치와 타입을 더욱 명확히 인지해 성능 향상을 보인다고 보고되었습니다.

**Future Work**

1.	데이터 증강
    - EDA, AEDA, Honorific Transformation, Masking & Infilling 등 다양한 데이터 증강 기법을 결합해 데이터 다변화를 도모할 수 있습니다.
    - 희소 클래스 샘플을 중심으로 증강하면 불균형 완화 효과를 기대할 수 있습니다.

2.	Span-based 모델 한국어 파인튜닝
    - EDA 결과, 스팬 내부 단어가 중요한 것으로 확인되었습니다.
    - 사전 학습 자체나 모델 구조 자체가 span-aware일 경우 Relation Extraction 성능이 향상될 것이라 예상합니다.

