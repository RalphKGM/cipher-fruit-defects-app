"""Streamlit demo for post-harvest fruit defect segmentation."""
import io
import sys
import zipfile
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
MAX_BATCH = 30

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

def damaged_share(prediction, shown):
    """Percent of the Stage 1 fruit covered by the shown defect masks."""
    if prediction.fruit_mask is None or not prediction.fruit_mask.any():
        return None
    if not shown:
        return 0.0
    union = shown[0].mask.copy()
    for det in shown[1:]:
        union |= det.mask
    return round(100 * (union & prediction.fruit_mask).sum() / prediction.fruit_mask.sum(), 2)


def fruit_status(prediction):
    if fruit_model is None:
        return "Not checked"
    return prediction.fruit_name.capitalize() if prediction.fruit_mask is not None else "No"


def show_single(source, name):
    try:
        image = load_image(source)
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Try another JPG or PNG.")
        return
    if min(image.shape[:2]) < 32:
        st.error("The image is too small to segment. Use a photo at least 32 pixels on each side.")
        return

    with st.spinner("Segmenting"):
        prediction = predict(defect_model, image, conf=conf, fruit_model=fruit_model)

    rendered = overlay(prediction, visible, alpha=alpha, show_fruit=show_fruit)
    left, right = st.columns(2)
    left.image(image, caption=f"Input: {name}", width="stretch")
    right.image(rendered, caption="Predicted defect masks", width="stretch")
    legend(list(defect_model.names.values()))

    shown = [d for d in prediction.detections if visible.get(d.class_name, True)]
    metric_cols = st.columns(3)
    metric_cols[0].metric("Defect regions", len(shown))
    metric_cols[1].metric("Inference time", f"{prediction.seconds * 1000:.0f} ms")
    metric_cols[2].metric("Fruit found", fruit_status(prediction))

    if not prediction.detections:
        st.success("No defect above the confidence threshold. Lower the threshold to see weaker predictions.")
    else:
        st.subheader("Summary")
        st.dataframe(pd.DataFrame(summarize(prediction)), hide_index=True, width="stretch")

    buffer = io.BytesIO()
    Image.fromarray(rendered).save(buffer, format="PNG")
    st.download_button("Download result PNG", buffer.getvalue(),
                       file_name=f"{Path(name).stem}_defects.png", mime="image/png", key=f"dl_{name}")


def show_batch():
    files = st.file_uploader(f"Select up to {MAX_BATCH} images", type=UPLOAD_TYPES, accept_multiple_files=True,
                             key="batch_files")
    if not files:
        st.info("Select several photos at once to segment them in one run.")
        return
    if len(files) > MAX_BATCH:
        st.warning(f"Only the first {MAX_BATCH} images are processed.")
        files = files[:MAX_BATCH]

    run_key = (fruit, conf, fruit_model is not None, tuple(getattr(f, "file_id", f.name) for f in files))
    if st.session_state.get("batch_key") != run_key:
        if not st.button(f"Run on {len(files)} images", type="primary"):
            return
        results = []
        bar = st.progress(0.0)
        for i, f in enumerate(files):
            bar.progress(i / len(files), text=f"Segmenting {f.name} ({i + 1} of {len(files)})")
            try:
                image = load_image(f)
            except (UnidentifiedImageError, OSError):
                results.append((f.name, None, "Could not be read"))
                continue
            if min(image.shape[:2]) < 32:
                results.append((f.name, None, "Too small"))
                continue
            results.append((f.name, predict(defect_model, image, conf=conf, fruit_model=fruit_model), ""))
        bar.empty()
        st.session_state.batch_key = run_key
        st.session_state.batch_results = results
    results = st.session_state.batch_results

    rows, gallery = [], []
    for name, prediction, note in results:
        if prediction is None:
            rows.append({"File": name, "Fruit found": "-", "Defect regions": 0, "Defects": "-",
                         "% of fruit damaged": None, "Time (ms)": None, "Note": note})
            continue
        shown = [d for d in prediction.detections if visible.get(d.class_name, True)]
        labels = sorted({CLASS_LABELS.get(d.class_name, d.class_name) for d in shown})
        rows.append({"File": name, "Fruit found": fruit_status(prediction), "Defect regions": len(shown),
                     "Defects": ", ".join(labels) or "None", "% of fruit damaged": damaged_share(prediction, shown),
                     "Time (ms)": round(prediction.seconds * 1000), "Note": note})
        gallery.append((name, overlay(prediction, visible, alpha=alpha, show_fruit=show_fruit)))

    table = pd.DataFrame(rows)
    shares = table["% of fruit damaged"].dropna()
    metric_cols = st.columns(3)
    metric_cols[0].metric("Images", len(rows))
    metric_cols[1].metric("With defects", int((table["Defect regions"] > 0).sum()))
    metric_cols[2].metric("Mean % of fruit damaged", f"{shares.mean():.1f}%" if len(shares) else "-")
    st.dataframe(table, hide_index=True, width="stretch")

    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, rendered in gallery:
            png = io.BytesIO()
            Image.fromarray(rendered).save(png, format="PNG")
            archive.writestr(f"{Path(name).stem}_defects.png", png.getvalue())
    dl_cols = st.columns(2)
    dl_cols[0].download_button("Download table CSV", table.to_csv(index=False).encode(), file_name="batch_results.csv",
                               mime="text/csv")
    dl_cols[1].download_button("Download masked images ZIP", zipped.getvalue(), file_name="batch_results.zip",
                               mime="application/zip")

    legend(list(defect_model.names.values()))
    for start in range(0, len(gallery), 4):
        cols = st.columns(4)
        for col, (name, rendered) in zip(cols, gallery[start:start + 4]):
            col.image(rendered, caption=name, width="stretch")


upload_tab, sample_tab, batch_tab = st.tabs(["Upload an image", "Use a sample", "Batch upload"])
with upload_tab:
    upload = st.file_uploader("JPG PNG WEBP or BMP", type=UPLOAD_TYPES)
    if upload is None:
        st.info("Upload a photo to run the model.")
    else:
        show_single(upload, upload.name)
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
    elif picked and picked != "None":
        show_single(by_name[picked], picked)
    else:
        st.info("Pick a sample to run the model.")
with batch_tab:
    show_batch()

with st.expander("How to read this"):
    st.markdown(
        "- Each coloured region is one predicted defect instance\n"
        "- Confidence is the model score for that mask. It is not accuracy\n"
        "- Coverage uses the Stage 1 fruit mask when it is enabled\n"
        "- Validation and test scores are in the notebook and paper"
    )
