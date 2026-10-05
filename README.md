# Post-Harvest Fruit Defect Segmentation

Streamlit app for "Computer Vision-Driven Two-Stage Instance Segmentation of Post-Harvest Surface Defects in Apple and
Tomato Using YOLO26". Group CIPHER, Artificial Intelligence 2, AM3, Mapúa University. Adviser: Dr. Lysa V. Comia.

Upload a photo of an apple or a tomato. Stage 1 finds the whole fruit and Stage 2 outlines each visible defect:
bruise or discoloration, rot mold or decay, and surface damage. The app reports each defect and its share of the fruit.

The Batch upload tab runs up to 30 photos at once. Previous and Next buttons step through each image. Add the matching YOLO label .txt files (for example from `test/labels`) to see the team label next to the prediction and per-image mAP50, pixel IoU, precision and recall. Results export as a CSV table and a ZIP of masked images.

| Model option | Checkpoints | Test mask mAP50 |
|---|---|---:|
| Apple | `models/apple_stage1.pt`, `models/apple_stage2.pt` (Runs 21, 22) | 0.613 |
| Tomato | `models/tomato_stage1.pt`, `models/tomato_stage2.pt` (Runs 23, 24) | 0.393 |
| Both (one model) | `models/apple_tomato_stage1.pt`, `models/apple_tomato_stage2.pt` (Runs 25, 26) | 0.586 apple, 0.352 tomato |

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

Main file `streamlit_app.py`, Python 3.11. `requirements.txt` installs CPU-only PyTorch on Linux and `packages.txt`
adds the system libraries OpenCV needs.
