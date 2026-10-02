"""Streamlit demo for post-harvest fruit defect segmentation."""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import (  # noqa: E402
    CLASS_LABELS,
    FRUITS,
    color_for,
    load_image,
    model_path,
    overlay,
    predict,
    summarize,
)

SAMPLES_DIR = ROOT / "app" / "samples"
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]

st.set_page_config(page_title="Fruit Defect Segmentation", layout="wide")


@st.cache_resource(show_spinner="Loading model")
def load_model(path: str):
    from ultralytics import YOLO

    return YOLO(path)


def sample_images(fruit: str):
    images = []
    for name in FRUITS[fruit]["samples"]:
        folder = SAMPLES_DIR / name
        if folder.is_dir():
            images += sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return images


def legend(names):
    chips = []
    for index, name in enumerate(names):
        r, g, b = color_for(name, index)
        label = CLASS_LABELS.get(name, name)
        chips.append(
            f"<span style='display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});"
            f"margin:0 6px 0 14px;border-radius:2px'></span>{label}"
        )
    st.markdown(" ".join(chips), unsafe_allow_html=True)


st.title("Post-Harvest Fruit Defect Segmentation")
st.caption(
    "Two-stage YOLO26 instance segmentation by CIPHER (AI2 AM3). "
    "Masks mark visible surface defects only. They are not a disease diagnosis."
)

with st.sidebar:
    st.header("Settings")
    fruit = st.radio("Fruit", list(FRUITS), horizontal=True)
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Masks below this confidence are hidden")
    alpha = st.slider("Mask opacity", 0.1, 0.9, 0.45, 0.05)
    use_fruit_mask = st.checkbox("Measure coverage with the Stage 1 fruit mask", value=True)
    show_fruit = st.checkbox("Outline the fruit", value=True, disabled=not use_fruit_mask)

stage2_path = model_path(fruit, "stage2")
stage1_path = model_path(fruit, "stage1")
if not stage2_path.is_file():
    st.warning(f"The {fruit.lower()} defect model is not available yet. Place it at `models/{stage2_path.name}`.")
    st.stop()

defect_model = load_model(str(stage2_path))
fruit_model = load_model(str(stage1_path)) if use_fruit_mask and stage1_path.is_file() else None
if use_fruit_mask and fruit_model is None:
    st.sidebar.info(f"`models/{stage1_path.name}` not found so fruit coverage is skipped")

with st.sidebar:
    st.subheader("Show classes")
    visible = {name: st.checkbox(CLASS_LABELS.get(name, name), value=True, key=f"show_{name}")
               for name in defect_model.names.values()}

upload_tab, sample_tab = st.tabs(["Upload an image", "Use a sample"])
with upload_tab:
    upload = st.file_uploader("JPG PNG WEBP or BMP", type=UPLOAD_TYPES)
with sample_tab:
    samples = sample_images(fruit)
    by_name = {f"{p.parent.name}/{p.name}": p for p in samples}
    options = ["None"] + list(by_name)
    requested = st.query_params.get("sample")
    # Accept either "apple/rot_a.jpg" or a bare file name
    matches = [o for o in options if requested and (o == requested or o.endswith("/" + requested))]
    start = options.index(matches[0]) if matches else 0
    picked = st.selectbox("Sample image", options, index=start) if samples else None
    if not samples:
        st.write("No sample images for this fruit.")

image_source, image_name = None, None
if upload is not None:
    image_source, image_name = upload, upload.name
elif picked and picked != "None":
    image_source, image_name = by_name[picked], picked

if image_source is None:
    st.info("Upload a photo or pick a sample to run the model.")
    st.stop()

try:
    image = load_image(image_source)
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image. Try another JPG or PNG.")
    st.stop()

if min(image.shape[:2]) < 32:
    st.error("The image is too small to segment. Use a photo at least 32 pixels on each side.")
    st.stop()

with st.spinner("Segmenting"):
    prediction = predict(defect_model, image, conf=conf, fruit_model=fruit_model)

rendered = overlay(prediction, visible, alpha=alpha, show_fruit=show_fruit)
left, right = st.columns(2)
left.image(image, caption=f"Input: {image_name}", width="stretch")
right.image(rendered, caption="Predicted defect masks", width="stretch")
legend(list(defect_model.names.values()))

shown = [d for d in prediction.detections if visible.get(d.class_name, True)]
if fruit_model is None:
    fruit_status = "Not checked"
else:
    fruit_status = prediction.fruit_name.capitalize() if prediction.fruit_mask is not None else "No"
metric_cols = st.columns(3)
metric_cols[0].metric("Defect regions", len(shown))
metric_cols[1].metric("Inference time", f"{prediction.seconds * 1000:.0f} ms")
metric_cols[2].metric("Fruit found", fruit_status)

if not prediction.detections:
    st.success("No defect above the confidence threshold. Lower the threshold to see weaker predictions.")
else:
    st.subheader("Summary")
    st.dataframe(pd.DataFrame(summarize(prediction)), hide_index=True, width="stretch")

buffer = io.BytesIO()
Image.fromarray(rendered).save(buffer, format="PNG")
st.download_button("Download result PNG", buffer.getvalue(),
                   file_name=f"{Path(image_name).stem}_defects.png", mime="image/png")

with st.expander("How to read this"):
    st.markdown(
        "- Each coloured region is one predicted defect instance\n"
        "- Confidence is the model score for that mask. It is not accuracy\n"
        "- Coverage uses the Stage 1 fruit mask when it is enabled\n"
        "- Validation and test scores are in the notebook and paper"
    )
