"""BreastVisionAI Streamlit entry point.

This is a lightweight web UI around the existing PSO-selected CNN ensemble.
It is intentionally independent of Django so it can run on Streamlit
Community Cloud with: streamlit run streamlit_app.py
"""

import base64
import io

import streamlit as st
from PIL import Image

from prediction.services.model_registry import PSOModelRegistry
from prediction.services.prediction import predict_single


st.set_page_config(
    page_title="BreastVisionAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_ensemble():
    """Load the Keras ensemble once per Streamlit process."""
    return PSOModelRegistry.initialize()


def render_heatmap(data_uri):
    if not data_uri:
        return None
    _, encoded = data_uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def main():
    st.title("BreastVisionAI")
    st.caption("Research-support tool for benign/malignant breast-image classification")

    with st.sidebar:
        st.header("Analysis settings")
        image_type = st.radio(
            "Input type",
            options=["raw", "processed"],
            format_func=lambda value: value.title(),
            help="Use Raw for ordinary uploaded images. Choose Processed only when the image has already been normalized.",
        )
        explain = st.checkbox(
            "Generate Grad-CAM explanation",
            value=True,
            help="Shows the image regions that influenced the primary model. It adds CPU processing time.",
        )
        st.info("This tool is for research support and is not a medical diagnosis.")

    uploaded = st.file_uploader(
        "Upload a breast image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="The image is resized to 224×224 for the ensemble models.",
    )

    if uploaded is None:
        st.markdown(
            "Upload an image to run the PSO-selected ensemble using EfficientNet, "
            "ResNet, VGG16, and the GradientBoosting meta-learner."
        )
        return

    image = Image.open(uploaded)
    preview_col, details_col = st.columns([1, 1])
    with preview_col:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Loading models and analysing the image…"):
        try:
            load_ensemble()
            uploaded.seek(0)
            result = predict_single(
                uploaded,
                image_type=image_type,
                generate_heatmap=explain,
            )
        except Exception as error:
            st.error("The image could not be analysed.")
            st.exception(error)
            return

    with details_col:
        prediction = result["prediction"].title()
        confidence = result["confidence"]
        if prediction == "Malignant":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")
        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Malignant probability", f"{result['prediction_probability']:.1%}")
        st.caption("Confidence is the model's confidence in the displayed class.")

    st.subheader("Model breakdown")
    cols = st.columns(len(result["model_breakdown"]))
    for col, (name, values) in zip(cols, result["model_breakdown"].items()):
        with col:
            st.metric(name, f"{values['probability']:.1%}")
            st.caption(f"Fusion weight: {values['weight']:.1%}")

    heatmap = render_heatmap(result.get("heatmap_base64"))
    if heatmap is not None:
        st.subheader("Grad-CAM explanation")
        st.image(
            heatmap,
            caption="Highlighted areas show regions used by the primary ensemble model.",
            use_container_width=True,
        )
    elif explain:
        st.warning("A Grad-CAM explanation could not be generated for this image.")

    with st.expander("Technical details"):
        st.write({
            "selected_models": result["selected_models"],
            "ensemble_method": result["ensemble_method"],
            "meta_learner": result["meta_learner"],
            "input_type": result["image_type"],
        })


if __name__ == "__main__":
    main()
