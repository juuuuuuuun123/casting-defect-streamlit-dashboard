# Casting Defect Image Classification Project

주조품 이미지가 정상(`ok_front`)인지 불량(`def_front`)인지 분류하는 제조 품질검사 프로젝트입니다. EDA 기반 전처리, 5-fold stratified CV, CNN/transfer learning/임베딩 기반 ML 비교, 불균형 처리 실험, 통계검증, Streamlit 대시보드를 포함합니다.

## 1. 환경 준비

Streamlit Cloud 배포와 대시보드 실행에는 최소 런타임 의존성만 설치합니다.

```powershell
pip install -r requirements.txt
```

학습, Kaggle 다운로드, 실험 재현까지 모두 실행하려면 분석용 의존성을 설치합니다.

```powershell
pip install -r requirements-train.txt
```

CUDA가 잡히는지 확인합니다.

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## 2. 데이터 준비

`.env`에는 다음 값이 필요합니다.

```text
KAGGLE_USERNAME=...
KAGGLE_API_TOKEN=...
```

Kaggle 데이터셋을 다시 내려받아야 하면 다음 명령을 사용합니다.

```powershell
.\download_dataset.ps1
```

메타데이터와 fold를 생성합니다.

```powershell
python -m src.prepare_data
```

생성 파일:

- `data/processed/metadata.csv`
- `data/processed/data_summary.json`
- `data/processed/label_mapping.json`
- `data/processed/data_quality_issues.csv` (문제가 있을 때만)

## 3. 5-Fold CV 실험

CPU에서도 바로 실행 가능한 빠른 비교실험입니다. 32x32 grayscale 이미지 피처와 밝기/대비 피처를 사용해 Logistic Regression, Random Forest, LightGBM, XGBoost를 비교하고, 통계검증 및 holdout 평가 파일까지 생성합니다.

```powershell
python -m src.run_feature_experiments
```

CNN/transfer learning 본 실험은 시간이 더 오래 걸립니다.

빠른 smoke test:

```powershell
python -m src.train_cv --epochs 1 --batch-size 16
```

본 실험:

```powershell
python -m src.train_cv --epochs 15 --batch-size 32 --image-size 224
```

비교 모델:

- `B1_small_cnn_basic`: 기본 CNN baseline
- `B2_small_cnn_weighted`: class weighted CNN
- `T1_resnet18_weighted`: ResNet18 transfer learning
- `T2_mobilenet_v3_weighted`: MobileNetV3 transfer learning
- `T3_efficientnet_focal`: EfficientNet-B0 + focal loss
- `A1_resnet18_sampler`: weighted sampler ablation
- `embed_logreg`, `embed_random_forest`, `embed_lightgbm`, `embed_xgboost`: CNN 임베딩 기반 ML 비교군

## 4. Holdout 평가

CV에서 가장 좋은 실험을 fold ensemble로 holdout test에 평가합니다.

```powershell
python -m src.evaluate_holdout
```

생성 파일:

- `reports/experiments.csv`
- `reports/oof_predictions.csv`
- `reports/statistical_tests.csv`
- `reports/holdout_predictions.csv`
- `reports/holdout_summary.json`
- `reports/mcnemar_best_vs_baseline.json`

## 5. Streamlit 대시보드

```powershell
streamlit run streamlit_app.py
```

대시보드에서는 데이터 EDA, fold별 성능 분포, baseline 대비 통계검증, ROC/PR curve, holdout confusion matrix, 오분류 이미지를 확인할 수 있습니다.

### Streamlit Cloud 배포 설정

GitHub 저장소를 Streamlit Cloud에 연결할 때 아래 값으로 설정합니다.

```text
Main file path: streamlit_app.py
Python dependencies: requirements.txt
```

현재 대시보드는 저장소에 포함된 `data/processed`와 `reports` 산출물을 읽어 실행됩니다. 원본 이미지(`data/raw`)와 `.env`는 저장소에 올리지 않습니다.

## 해석 포인트

불량품 탐지에서는 accuracy보다 false negative rate와 recall을 우선 확인합니다. 즉, 실제 불량품을 정상으로 놓치는 케이스를 얼마나 줄였는지가 현장 적용 관점에서 더 중요합니다.

## CUDA 메모

현재 확인된 NVIDIA 드라이버는 CUDA 11.7이고, 설치된 Python 3.12용 PyTorch wheel은 CPU 전용입니다. GPU 학습을 하려면 드라이버를 최신화한 뒤 PyTorch 공식 안내에 맞는 CUDA wheel을 설치하거나, Python 3.10 환경에서 CUDA 11.7 호환 PyTorch를 별도로 구성하는 것을 권장합니다.
