"""Streamlit demo app for MobileViT flower classification."""

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import MobileViTForImageClassification, MobileViTImageProcessor

from mia_vpc_iii.config import DATA_DIR, MODELS_DIR

MODEL_PATH = MODELS_DIR / "mobilevit-small-tf-flowers-final-colab"
DEMO_DIR = DATA_DIR / "demo"


@st.cache_resource
def load_model():
    processor = MobileViTImageProcessor.from_pretrained(MODEL_PATH)
    model = MobileViTForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return processor, model


def run_classify(image: Image.Image, processor, model) -> tuple[str, float, dict[str, float]]:
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    pred_id = int(logits.argmax(-1))
    id2label = model.config.id2label
    label = id2label[pred_id]
    confidence = float(probs[pred_id])
    all_probs = {
        id2label[i]: float(probs[i])
        for i in range(len(probs))
    }
    return label, confidence, all_probs


def get_demo_images() -> list[Path]:
    if DEMO_DIR.is_dir():
        return sorted(p for p in DEMO_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return []


def _on_upload_change():
    st.session_state["source"] = "upload"
    st.session_state["results"] = None


def _on_demo_change():
    st.session_state["source"] = "demo"
    st.session_state["results"] = None


def main():
    st.set_page_config(
        page_title="Flower Classifier — MobileViT",
        page_icon="🌸",
        layout="centered",
    )

    for key in ("results_upload", "results_demo"):
        if key not in st.session_state:
            st.session_state[key] = None

    st.title("Flower Classifier")
    st.markdown(
        "Classify flower images using a **MobileViT-Small** model fine-tuned on the "
        "[tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) dataset. "
        "The model recognises **5 classes**: dandelion, daisy, tulips, sunflowers, and roses."
    )

    processor, model = load_model()

    tab_upload, tab_demo = st.tabs(["Upload an image", "Use a demo image"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Choose a flower image",
            type=["jpg", "jpeg", "png"],
            on_change=lambda: st.session_state.update(results_upload=None),
        )
        if uploaded is not None:
            image = Image.open(uploaded)
            _show_image_and_results(image, processor, model, "upload")

    with tab_demo:
        demo_images = get_demo_images()
        if demo_images:
            selected = st.selectbox(
                "Pick a demo image",
                options=demo_images,
                format_func=lambda p: p.name,
                on_change=lambda: st.session_state.update(results_demo=None),
                key="demo_select",
            )
            image = Image.open(selected)
            _show_image_and_results(image, processor, model, "demo")
        else:
            st.info(
                "No demo images found. Run `python -m mia_vpc_iii.save_demo_images` "
                "to download sample images into `data/demo/`."
            )


def _show_image_and_results(image: Image.Image, processor, model, key_suffix: str):
    results_key = f"results_{key_suffix}"

    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Input image", use_container_width=True)

    with col_results:
        if st.button("Classify", type="primary", key=f"classify_{key_suffix}"):
            with st.spinner("Classifying..."):
                label, confidence, all_probs = run_classify(image, processor, model)
            st.session_state[results_key] = (label, confidence, all_probs)

        results = st.session_state[results_key]
        if results is not None:
            label, confidence, all_probs = results
            st.metric("Predicted class", label.capitalize(), f"{confidence:.1%}")
            st.markdown("**Class probabilities**")
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1])
            df = pd.DataFrame(
                {"Probability": [v for _, v in sorted_probs]},
                index=[k.capitalize() for k, _ in sorted_probs],
            )
            st.bar_chart(df, horizontal=True)


if __name__ == "__main__":
    main()
