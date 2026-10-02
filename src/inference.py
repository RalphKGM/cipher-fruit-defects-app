"""Run the two-stage fruit defect models on one image and summarize the masks."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Okabe-Ito colours stay readable for colour-blind viewers
CLASS_COLORS = {
    "bruise_discoloration": (230, 159, 0),
    "rot_mold_decay": (204, 121, 167),
    "surface_damage": (0, 114, 178),
    "apple": (0, 158, 115),
    "tomato": (0, 158, 115),
}
FALLBACK_COLORS = [(204, 121, 167), (86, 180, 233), (240, 228, 66)]

CLASS_LABELS = {
    "bruise_discoloration": "Bruise or discoloration",
    "rot_mold_decay": "Rot mold or decay",
    "surface_damage": "Surface damage",
}

FRUITS = {
    "Apple": {"stage1": "apple_stage1.pt", "stage2": "apple_stage2.pt", "samples": ["apple"]},
    "Tomato": {"stage1": "tomato_stage1.pt", "stage2": "tomato_stage2.pt", "samples": ["tomato"]},
    "Both (one model)": {"stage1": "apple_tomato_stage1.pt", "stage2": "apple_tomato_stage2.pt",
                         "samples": ["apple", "tomato"]},
}


@dataclass
class Detection:
    class_name: str
    confidence: float
    mask: np.ndarray
    area_px: int


@dataclass
class Prediction:
    image: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    fruit_mask: Optional[np.ndarray] = None
    fruit_name: Optional[str] = None
    seconds: float = 0.0


def model_path(fruit: str, stage: str) -> Path:
    return MODELS_DIR / FRUITS[fruit][stage]


def color_for(name: str, index: int = 0) -> tuple:
    return CLASS_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def load_image(source) -> np.ndarray:
    """Open a path or file object as an upright RGB array."""
    image = Image.open(source)
    image = ImageOps.exif_transpose(image)
    return np.array(image.convert("RGB"))


def _masks_to_image_size(result, height: int, width: int) -> List[np.ndarray]:
    if result.masks is None:
        return []
    masks = []
    for polygon in result.masks.xy:
        canvas = Image.new("L", (width, height), 0)
        if len(polygon) >= 3:
            ImageDraw.Draw(canvas).polygon([tuple(p) for p in polygon], fill=1)
        masks.append(np.array(canvas, dtype=bool))
    return masks


def predict(defect_model, image: np.ndarray, conf: float, imgsz: int = 864,
            fruit_model=None) -> Prediction:
    """Segment defects and optionally the whole fruit for coverage."""
    height, width = image.shape[:2]
    start = time.perf_counter()
    # Ultralytics expects BGR arrays
    bgr = image[:, :, ::-1]
    result = defect_model.predict(bgr, conf=conf, imgsz=imgsz, retina_masks=True, verbose=False)[0]

    detections = []
    masks = _masks_to_image_size(result, height, width)
    for i, mask in enumerate(masks):
        cls_id = int(result.boxes.cls[i])
        detections.append(Detection(
            class_name=result.names[cls_id],
            confidence=float(result.boxes.conf[i]),
            mask=mask,
            area_px=int(mask.sum()),
        ))

    fruit_mask, fruit_name = None, None
    if fruit_model is not None:
        fruit_result = fruit_model.predict(bgr, conf=0.5, imgsz=imgsz, retina_masks=True, verbose=False)[0]
        fruit_masks = _masks_to_image_size(fruit_result, height, width)
        if fruit_masks:
            fruit_mask = np.logical_or.reduce(fruit_masks)
            # The combined Stage 1 model names the fruit. Take its most confident mask.
            best = int(fruit_result.boxes.conf.argmax())
            fruit_name = fruit_result.names[int(fruit_result.boxes.cls[best])]

    seconds = time.perf_counter() - start
    return Prediction(image=image, detections=detections, fruit_mask=fruit_mask, fruit_name=fruit_name,
                      seconds=seconds)


def overlay(prediction: Prediction, visible: Dict[str, bool], alpha: float = 0.45,
            show_fruit: bool = True) -> np.ndarray:
    """Blend coloured masks on the image and draw their outlines."""
    import cv2

    canvas = prediction.image.astype(np.float32).copy()
    outlines = []
    if show_fruit and prediction.fruit_mask is not None:
        outlines.append((prediction.fruit_mask, CLASS_COLORS["apple"]))
    for index, det in enumerate(prediction.detections):
        if not visible.get(det.class_name, True):
            continue
        color = np.array(color_for(det.class_name, index), dtype=np.float32)
        canvas[det.mask] = canvas[det.mask] * (1 - alpha) + color * alpha
        outlines.append((det.mask, color_for(det.class_name, index)))

    result = canvas.clip(0, 255).astype(np.uint8)
    thickness = max(2, round(min(result.shape[:2]) / 300))
    for mask, color in outlines:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, thickness)
    return result


def summarize(prediction: Prediction) -> List[dict]:
    """One row per class with mask count and coverage."""
    image_px = prediction.image.shape[0] * prediction.image.shape[1]
    fruit_px = int(prediction.fruit_mask.sum()) if prediction.fruit_mask is not None else 0
    rows = {}
    for det in prediction.detections:
        row = rows.setdefault(det.class_name, {"masks": 0, "union": np.zeros_like(det.mask), "best": 0.0})
        row["masks"] += 1
        row["union"] |= det.mask
        row["best"] = max(row["best"], det.confidence)

    summary = []
    for name, row in rows.items():
        area = int(row["union"].sum())
        on_fruit = int((row["union"] & prediction.fruit_mask).sum()) if fruit_px else 0
        summary.append({
            "Defect": CLASS_LABELS.get(name, name),
            "Regions": row["masks"],
            "Highest confidence": round(row["best"], 2),
            "% of image": round(100 * area / image_px, 2),
            "% of fruit": round(100 * on_fruit / fruit_px, 2) if fruit_px else None,
        })
    return sorted(summary, key=lambda r: r["% of image"], reverse=True)
