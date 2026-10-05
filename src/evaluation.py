"""Per-image scores against YOLO polygon labels for the batch viewer."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from inference import Detection, Prediction

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

DEFECTS = ["bruise_discoloration", "rot_mold_decay", "surface_damage"]
# Class ids in the exported datasets. Single-fruit sets start with the fruit, the combined set with both fruits.
LABEL_FORMATS = {
    "Apple or tomato dataset (0 fruit, 1 bruise, 2 rot, 3 surface)": {1: DEFECTS[0], 2: DEFECTS[1], 3: DEFECTS[2]},
    "Combined dataset (0 apple, 1 tomato, 2 bruise, 3 rot, 4 surface)": {2: DEFECTS[0], 3: DEFECTS[1],
                                                                        4: DEFECTS[2]},
}


def read_labels(text: str, width: int, height: int, id_to_name: Dict[int, str]) -> List[Tuple[str, np.ndarray]]:
    """Rasterize the defect polygons of one YOLO segmentation label file."""
    masks = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7 or int(float(parts[0])) not in id_to_name:
            continue
        xy = np.array(parts[1:], dtype=float).reshape(-1, 2) * (width, height)
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in xy], fill=1)
        masks.append((id_to_name[int(float(parts[0]))], np.array(canvas, dtype=bool)))
    return masks


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _match(preds: List[Detection], gts: List[np.ndarray]) -> List[int]:
    """Greedy match by confidence at IoU 0.5. Returns 1 for a true positive, 0 for a false positive."""
    used, hits = set(), []
    for det in sorted(preds, key=lambda d: -d.confidence):
        best, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if j not in used:
                iou = _iou(det.mask, gt)
                if iou > best:
                    best, best_j = iou, j
        if best >= 0.5:
            used.add(best_j)
            hits.append(1)
        else:
            hits.append(0)
    return hits


def _ap(hits: List[int], n_gt: int) -> float:
    """Area under the precision-recall curve with 101-point interpolation, as in Ultralytics."""
    if not hits:
        return 0.0
    tp = np.cumsum(hits)
    recall = tp / n_gt
    precision = tp / np.arange(1, len(hits) + 1)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapezoid(np.interp(x, mrec, mpre), x))


def score(prediction: Prediction, labels: List[Tuple[str, np.ndarray]]) -> Dict[str, Optional[float]]:
    """mAP50, precision, recall and class-aware pixel IoU for one image at the current confidence threshold."""
    out = {"mAP50": None, "Precision": None, "Recall": None, "Pixel IoU": None}
    gt_classes = {name for name, _ in labels}
    preds = prediction.detections
    if not gt_classes:
        return out

    aps, tp_total = [], 0
    for name in DEFECTS:
        gts = [m for n, m in labels if n == name]
        hits = _match([d for d in preds if d.class_name == name], gts)
        tp_total += sum(hits)
        if gts:
            aps.append(_ap(hits, len(gts)))
    out["mAP50"] = round(float(np.mean(aps)), 3)
    out["Precision"] = round(tp_total / len(preds), 3) if preds else 0.0
    out["Recall"] = round(tp_total / len(labels), 3)

    shape = prediction.image.shape[:2]
    ious = []
    for name in gt_classes | {d.class_name for d in preds}:
        gt_union = np.zeros(shape, bool)
        for n, m in labels:
            if n == name:
                gt_union |= m
        pr_union = np.zeros(shape, bool)
        for d in preds:
            if d.class_name == name:
                pr_union |= d.mask
        ious.append(_iou(gt_union, pr_union))
    out["Pixel IoU"] = round(float(np.mean(ious)), 3)
    return out


def label_prediction(prediction: Prediction, labels: List[Tuple[str, np.ndarray]]) -> Prediction:
    """Wrap team labels as a Prediction so the same overlay code can draw them."""
    detections = [Detection(class_name=n, confidence=1.0, mask=m, area_px=int(m.sum())) for n, m in labels]
    return Prediction(image=prediction.image, detections=detections, fruit_mask=prediction.fruit_mask,
                      fruit_name=prediction.fruit_name)
