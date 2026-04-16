# 멀티모달 퓨전 기반 피부 병변 AI 진단 보조 시스템

피부경(dermoscopy) 이미지와 환자 임상 정보(나이, 성별, 병변 위치)를 결합한 **멀티모달 딥러닝 모델**로 7가지 피부 병변을 분류하는 의료 AI 프로젝트입니다.

## 프로젝트 개요

### 해결하고자 하는 문제
- 피부암은 조기 발견 시 생존율이 크게 높아지지만, 육안으로는 양성/악성 구분이 어려움
- 기존 CNN 분류는 이미지만 사용하여 환자의 임상 정보를 무시
- 실제 피부과 진단에서는 의사가 이미지 + 환자 정보를 종합적으로 판단
- **목표**: 이미지와 임상 데이터를 결합한 멀티모달 모델로 단일 모달리티 대비 성능 향상 검증

### 데이터셋
**HAM10000** (Human Against Machine with 10000 training images)
- 10,015장의 피부경 이미지 + 환자 메타데이터
- 7가지 진단 클래스: 멜라노마, 멜라닌세포 모반, 기저세포암, 광선각화증, 양성 각화증, 피부섬유종, 혈관 병변

## 모델 아키텍처

```
       이미지 입력 [B,3,224,224]          테이블 입력 [B,18]
              |                                |
    EfficientNet-B0 (pretrained)       Linear(18->64) + BN + ReLU
      부분 동결 (70%)                   Linear(64->64) + BN + ReLU
              |                                |
       특징 [B, 1280]                     특징 [B, 64]
              |                                |
    Linear(1280->256) + BN + ReLU              |
              |                                |
         [B, 256] ------- concatenate ------- [B, 64]
                           |
                      [B, 320]
                           |
               Linear(320->128) + BN + ReLU + Dropout
                           |
                    Linear(128->7)
```

## 평가 결과

| 지표 | 멀티모달 | Image-Only | 개선 |
|------|---------|-----------|------|
| Balanced Accuracy | **0.6789** | 0.4457 | +23.3%p |
| Macro F1-Score | **0.6150** | 0.3437 | +27.1%p |
| Weighted F1-Score | **0.7582** | 0.5982 | +16.0%p |
| ROC-AUC (macro) | **0.9442** | 0.8270 | +11.7%p |
| Melanoma Recall | **0.6687** | 0.6319 | +3.7%p |
| Cohen's Kappa | **0.5637** | 0.3415 | +22.2%p |

멀티모달 퓨전 모델이 모든 지표에서 Image-Only 모델을 크게 상회합니다.

## 실행 방법

### 환경 설정
```bash
uv sync
```

### 데이터 준비
1. [Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) 다운로드
2. `data/raw/`에 배치:
   - `HAM10000_images_part_1/`
   - `HAM10000_images_part_2/`
   - `HAM10000_metadata.csv`

### 전처리 및 학습
```bash
uv run python -m src.preprocess
uv run python -m src.train --model multimodal
uv run python -m src.evaluate --model multimodal
```

### Streamlit 앱 실행
```bash
uv run streamlit run app.py
```

## 기술 스택
- **모델**: PyTorch + timm (EfficientNet-B0)
- **학습**: Mixed Precision (AMP), AdamW, 차등 학습률
- **시각화**: GradCAM, Matplotlib, Seaborn
- **앱**: Streamlit
- **환경**: uv, Python 3.12
