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

from evaluation import LABEL_FORMATS, label_prediction, read_labels, score  # noqa: E402
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
TEST_SET_DIR = ROOT / "app" / "test_set"
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


def test_set_items(fruit_choice: str):
    """Built-in test images with their team label files."""
    sets = ["apple", "tomato"] if fruit_choice.startswith("Both") else [fruit_choice.lower()]
    per_set = []
    for name in sets:
        folder = TEST_SET_DIR / name
        found = []
        for img in sorted((folder / "images").glob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                label = folder / "labels" / (img.stem + ".txt")
                found.append((img.name, img, label.read_text() if label.is_file() else None))
        per_set.append(found)
    # Alternate fruits so a short run of the combined model shows both
    items = []
    for i in range(max(map(len, per_set), default=0)):
        items += [found[i] for found in per_set if i < len(found)]
    return items


def show_batch():
    source = st.radio("Images", ["Built-in test set (with team labels)", "Upload my own"], horizontal=True,
                      key="batch_source")
    label_format = list(LABEL_FORMATS)[0]
    if source.startswith("Built-in"):
        pool = test_set_items(fruit)
        if not pool:
            st.warning("The built-in test set is not available.")
            return
        count = st.slider("Number of test images", 1, len(pool), min(20, len(pool)),
                          help="About 1 to 2 seconds per image on CPU")
        items = pool[:count]
        st.caption(f"{len(pool)} held-out test images for {fruit.lower()}. The model never saw them in training "
                   "or model selection.")
    else:
        files = st.file_uploader(f"Select up to {MAX_BATCH} images. Add their YOLO label .txt files to score "
                                 "each image", type=UPLOAD_TYPES + ["txt"], accept_multiple_files=True,
                                 key="batch_files")
        if not files:
            st.info("Select several photos at once. Without label files the app shows predictions and confidence "
                    "but cannot compute mAP50 or IoU.")
            return
        label_files = {Path(f.name).stem: f for f in files if f.name.lower().endswith(".txt")}
        images = [f for f in files if not f.name.lower().endswith(".txt")]
        if not images:
            st.warning("Only label files were selected. Add the images too.")
            return
        if len(images) > MAX_BATCH:
            st.warning(f"Only the first {MAX_BATCH} images are processed.")
            images = images[:MAX_BATCH]
        if label_files:
            label_format = st.selectbox("Label format", list(LABEL_FORMATS),
                                        index=1 if fruit.startswith("Both") else 0,
                                        help="Class ids differ between the single-fruit and combined datasets")
        items = []
        for f in images:
            lf = label_files.get(Path(f.name).stem)
            items.append((f.name, f, lf.getvalue().decode("utf-8", errors="ignore") if lf is not None else None))

    run_key = (fruit, conf, fruit_model is not None, label_format, source,
               tuple(getattr(src, "file_id", str(src)) for _, src, _ in items))
    if st.session_state.get("batch_key") != run_key:
        if not st.button(f"Run on {len(items)} images", type="primary"):
            return
        results = []
        bar = st.progress(0.0)
        for i, (name, src, label_text) in enumerate(items):
            bar.progress(i / len(items), text=f"Segmenting {name} ({i + 1} of {len(items)})")
            try:
                image = load_image(src)
            except (UnidentifiedImageError, OSError):
                results.append({"name": name, "prediction": None, "note": "Could not be read"})
                continue
            if min(image.shape[:2]) < 32:
                results.append({"name": name, "prediction": None, "note": "Too small"})
                continue
            prediction = predict(defect_model, image, conf=conf, fruit_model=fruit_model)
            labels = None
            if label_text is not None:
                labels = read_labels(label_text, image.shape[1], image.shape[0], LABEL_FORMATS[label_format])
            results.append({"name": name, "prediction": prediction, "labels": labels,
                            "scores": score(prediction, labels) if labels is not None else {}, "note": ""})
        bar.empty()
        st.session_state.batch_key = run_key
        st.session_state.batch_results = results
        st.session_state.batch_index = 0
    results = st.session_state.batch_results

    rows = []
    for r in results:
        prediction = r["prediction"]
        if prediction is None:
            rows.append({"File": r["name"], "Note": r["note"]})
            continue
        shown = [d for d in prediction.detections if visible.get(d.class_name, True)]
        labels = sorted({CLASS_LABELS.get(d.class_name, d.class_name) for d in shown})
        top = max((d.confidence for d in shown), default=None)
        row = {"File": r["name"], "Fruit found": fruit_status(prediction), "Defect regions": len(shown),
               "Top confidence": None if top is None else round(top, 2),
               "Defects": ", ".join(labels) or "None", "% of fruit damaged": damaged_share(prediction, shown)}
        if any(x.get("labels") is not None for x in results):
            if r["labels"] is None:
                row["Note"] = "No label file"
            elif not r["labels"]:
                row["Note"] = "No labeled defect"
            row.update({k: v for k, v in r["scores"].items()})
        row["Time (ms)"] = round(prediction.seconds * 1000)
        rows.append(row)
    table = pd.DataFrame(rows)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Images", len(rows))
    metric_cols[1].metric("With defects", int((table.get("Defect regions", pd.Series(dtype=int)) > 0).sum()))
    shares = table.get("% of fruit damaged", pd.Series(dtype=float)).dropna()
    metric_cols[2].metric("Mean % of fruit damaged", f"{shares.mean():.1f}%" if len(shares) else "-")
    maps = table.get("mAP50", pd.Series(dtype=float)).dropna()
    metric_cols[3].metric("Mean per-image mAP50", f"{maps.mean():.3f}" if len(maps) else "-",
                          help="Average of per-image scores at the current confidence threshold. "
                               "It is not the official test mAP from the paper")

    viewable = [r for r in results if r["prediction"] is not None]
    if viewable:
        index = min(st.session_state.get("batch_index", 0), len(viewable) - 1)
        nav = st.columns([1, 6, 1])
        if nav[0].button("◀ Previous", width="stretch", disabled=index == 0):
            index -= 1
        if nav[2].button("Next ▶", width="stretch", disabled=index == len(viewable) - 1):
            index += 1
        names = [r["name"] for r in viewable]
        index = nav[1].selectbox("Image", range(len(names)), index=index, format_func=lambda i: f"{i + 1} of "
                                 f"{len(names)}: {names[i]}", label_visibility="collapsed")
        st.session_state.batch_index = index
        current = viewable[index]
        prediction = current["prediction"]
        rendered = overlay(prediction, visible, alpha=alpha, show_fruit=show_fruit)
        if current.get("labels") is not None:
            cols = st.columns(3)
            cols[0].image(prediction.image, caption="Input", width="stretch")
            truth = overlay(label_prediction(prediction, current["labels"]), visible, alpha=alpha, show_fruit=False)
            cols[1].image(truth, caption="Team label", width="stretch")
            cols[2].image(rendered, caption="Model prediction", width="stretch")
        else:
            cols = st.columns(2)
            cols[0].image(prediction.image, caption="Input", width="stretch")
            cols[1].image(rendered, caption="Model prediction", width="stretch")
        legend(list(defect_model.names.values()))
        shown = [d for d in prediction.detections if visible.get(d.class_name, True)]
        share = damaged_share(prediction, shown)
        stats = st.columns(7)
        stats[0].metric("Defect regions", len(shown))
        stats[1].metric("% of fruit damaged", "-" if share is None else f"{share:.1f}%")
        top = max((d.confidence for d in shown), default=None)
        stats[2].metric("Top confidence", "-" if top is None else f"{top:.2f}")
        sc = current.get("scores") or {}
        for col, key in zip(stats[3:], ["mAP50", "Pixel IoU", "Precision", "Recall"]):
            value = sc.get(key)
            col.metric(key, "-" if value is None else f"{value:.3f}")
        if current.get("labels") is None:
            st.caption("No team label for this image, so mAP50 and IoU cannot be computed. Use the built-in test set "
                       "or upload the matching label .txt files.")
        elif current.get("labels") == []:
            st.caption("This image has no labeled defect, so mAP50 and IoU are not defined.")
        if shown:
            fruit_px = prediction.fruit_mask.sum() if prediction.fruit_mask is not None else 0
            st.dataframe(pd.DataFrame([{
                "Defect": CLASS_LABELS.get(d.class_name, d.class_name),
                "Confidence": round(d.confidence, 2),
                "% of fruit": round(100 * (d.mask & prediction.fruit_mask).sum() / fruit_px, 2) if fruit_px else None,
            } for d in sorted(shown, key=lambda d: -d.confidence)]), hide_index=True, width="stretch")

    st.subheader("All images")
    st.dataframe(table, hide_index=True, width="stretch")
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as archive:
        for r in viewable:
            png = io.BytesIO()
            Image.fromarray(overlay(r["prediction"], visible, alpha=alpha, show_fruit=show_fruit)).save(png, "PNG")
            archive.writestr(f"{Path(r['name']).stem}_defects.png", png.getvalue())
    dl_cols = st.columns(2)
    dl_cols[0].download_button("Download table CSV", table.to_csv(index=False).encode(), file_name="batch_results.csv",
                               mime="text/csv")
    dl_cols[1].download_button("Download masked images ZIP", zipped.getvalue(), file_name="batch_results.zip",
                               mime="application/zip")


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
