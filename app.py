import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image

from src.config import (
    CLASS_NAMES, CLASS_LABELS, CLASS_MALIGNANCY,
    LOCALIZATION_CATEGORIES, SEX_CATEGORIES, CHECKPOINT_DIR,
    PROJECT_ROOT,
)
from src.predict import load_model, predict_single

# === Page Config ===
st.set_page_config(
    page_title="피부 병변 AI 진단 보조 시스템",
    page_icon="🔬",
    layout="wide",
)

# === Korean Labels ===
SEX_LABELS = {"female": "여성", "male": "남성", "unknown": "알 수 없음"}
SEX_KEYS = list(SEX_LABELS.keys())
SEX_DISPLAY = list(SEX_LABELS.values())

LOCALIZATION_LABELS = {
    "abdomen": "복부", "back": "등", "chest": "가슴", "ear": "귀",
    "face": "얼굴", "foot": "발", "genital": "생식기", "hand": "손",
    "lower extremity": "하지", "neck": "목", "scalp": "두피",
    "trunk": "몸통", "upper extremity": "상지", "unknown": "알 수 없음",
}
LOC_DISPLAY = [LOCALIZATION_LABELS[k] for k in LOCALIZATION_CATEGORIES]

# === Sample Data ===
SAMPLE_CASES = [
    {
        "name": "Sample 1 - 멜라노마",
        "image": "assets/samples/ISIC_0033819.jpg",
        "age": 60, "sex": "male", "localization": "back",
        "description": "60세 남성, 등 부위",
    },
    {
        "name": "Sample 2 - 모반",
        "image": "assets/samples/ISIC_0030934.jpg",
        "age": 40, "sex": "female", "localization": "trunk",
        "description": "40세 여성, 몸통 부위",
    },
    {
        "name": "Sample 3 - 기저세포암",
        "image": "assets/samples/ISIC_0034066.jpg",
        "age": 70, "sex": "male", "localization": "chest",
        "description": "70세 남성, 가슴 부위",
    },
    {
        "name": "Sample 4 - 광선각화증",
        "image": "assets/samples/ISIC_0031198.jpg",
        "age": 55, "sex": "female", "localization": "neck",
        "description": "55세 여성, 목 부위",
    },
    {
        "name": "Sample 5 - 혈관 병변",
        "image": "assets/samples/ISIC_0031201.jpg",
        "age": 45, "sex": "male", "localization": "trunk",
        "description": "45세 남성, 몸통 부위",
    },
]


@st.cache_resource
def get_model():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = load_model(device)
    return model, device


# === Header ===
st.title("🔬 멀티모달 피부 병변 AI 진단 보조 시스템")
st.markdown(
    "피부경(dermoscopy) 이미지와 환자 임상 정보를 결합한 "
    "**멀티모달 딥러닝 모델**을 활용하여 피부 병변을 분류합니다."
)

# Check if model exists
checkpoint_path = CHECKPOINT_DIR / "best_model_multimodal.pth"
if not checkpoint_path.exists():
    st.error(
        "학습된 모델이 없습니다. 먼저 학습을 실행해 주세요:\n\n"
        "```\nuv run python -m src.train\n```"
    )
    st.stop()

# Load model
model, device = get_model()

# Determine current values from sample selection or defaults
selected = st.session_state.get("selected_sample", None)
if selected is not None:
    sample = SAMPLE_CASES[selected]
    default_age = sample["age"]
    default_sex_idx = SEX_KEYS.index(sample["sex"])
    default_loc_idx = LOCALIZATION_CATEGORIES.index(sample["localization"])
    sample_image_path = PROJECT_ROOT / sample["image"]
else:
    default_age = 50
    default_sex_idx = 0
    default_loc_idx = 0
    sample_image_path = None

# === Input Section ===
st.markdown("---")
col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("📋 환자 정보 입력")

    uploaded_file = st.file_uploader(
        "피부 병변 이미지 업로드",
        type=["jpg", "jpeg", "png"],
        help="피부경(dermoscopy) 이미지를 업로드하세요. 또는 아래 샘플 케이스를 선택하세요.",
    )

    # Determine which image to use
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="업로드된 이미지", use_container_width=True)
        image_ready = True
    elif sample_image_path is not None and sample_image_path.exists():
        image = Image.open(sample_image_path).convert("RGB")
        st.image(image, caption=f"샘플 이미지: {SAMPLE_CASES[selected]['name']}", use_container_width=True)
        image_ready = True
    else:
        image = None
        image_ready = False

    st.markdown("#### 임상 정보")
    age = st.slider("나이", min_value=0, max_value=100, value=default_age, step=1)

    sex_selected = st.radio("성별", SEX_DISPLAY, index=default_sex_idx, horizontal=True)
    sex = SEX_KEYS[SEX_DISPLAY.index(sex_selected)]

    loc_selected = st.selectbox("병변 위치", LOC_DISPLAY, index=default_loc_idx)
    localization = LOCALIZATION_CATEGORIES[LOC_DISPLAY.index(loc_selected)]

    # === Sample Cases Section (below input) ===
    st.markdown("#### 📂 샘플 케이스")
    st.caption("클릭하면 환자 정보가 자동 입력됩니다.")

    sample_cols = st.columns(5)
    for i, sample_case in enumerate(SAMPLE_CASES):
        with sample_cols[i]:
            img_path = PROJECT_ROOT / sample_case["image"]
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            if st.button(sample_case["name"], key=f"sample_{i}", use_container_width=True):
                st.session_state["selected_sample"] = i
                st.rerun()
            st.caption(sample_case["description"])

    predict_btn = st.button("🔍 분석 시작", type="primary", use_container_width=True)

# === Result Section ===
with col_result:
    st.subheader("📊 분석 결과")

    if predict_btn and image_ready:
        with st.spinner("모델 추론 중..."):
            result = predict_single(
                model, device, image, age, sex, localization, with_gradcam=True,
            )

        pred_class = result["predicted_class"]
        pred_label = result["predicted_label"]
        confidence = result["confidence"]
        is_malignant = result["is_malignant"]

        # Prediction result
        if is_malignant:
            st.error(f"⚠️ **{pred_label}** (악성 가능성)")
        else:
            st.success(f"✅ **{pred_label}** (양성)")

        st.metric("예측 확신도", f"{confidence * 100:.1f}%")

        # Probability bar chart
        st.markdown("#### 클래스별 예측 확률")
        probs = result["probabilities"]
        chart_data = {
            CLASS_LABELS[cls]: prob * 100
            for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)
        }
        st.bar_chart(chart_data, horizontal=True)

        # GradCAM
        if result.get("gradcam_overlay") is not None:
            st.markdown("#### GradCAM 시각화")
            st.caption("모델이 주목한 영역을 히트맵으로 표시합니다")
            st.image(result["gradcam_overlay"], caption="GradCAM 히트맵", use_container_width=True)

        # Medical disclaimer
        st.markdown("---")
        st.warning(
            "⚠️ **의료 면책 조항**: 이 시스템은 연구 및 교육 목적으로 개발되었으며, "
            "실제 의료 진단을 대체할 수 없습니다. 정확한 진단을 위해 반드시 "
            "전문 의료진과 상담하세요."
        )

    elif predict_btn and not image_ready:
        st.warning("이미지를 먼저 업로드하거나 샘플 케이스를 선택해 주세요.")
    else:
        st.info("이미지를 업로드하거나 샘플 케이스를 선택한 후 '분석 시작' 버튼을 눌러주세요.")

# === Sidebar Info ===
with st.sidebar:
    st.markdown("### 모델 정보")
    st.markdown(
        "- **아키텍처**: 멀티모달 Late Fusion\n"
        "- **이미지 백본**: EfficientNet-B0\n"
        "- **입력**: 피부경 이미지 + 환자 메타데이터\n"
        "- **분류**: 7가지 피부 병변\n"
    )

    st.markdown("### 분류 가능한 병변")
    for cls in CLASS_NAMES:
        malignancy = "🔴 악성" if CLASS_MALIGNANCY[cls] else "🟢 양성"
        st.markdown(f"- {CLASS_LABELS[cls]} ({malignancy})")

    st.markdown("---")
    st.markdown("### 개발자 정보")
    st.markdown(
        "**이태희**\n\n"
        "연세대학교 의과대학 융합의학과\n\n"
        "박사과정 | 2025323230"
    )
    st.markdown("---")
    st.caption("HAM10000 데이터셋으로 학습된 모델입니다.")
