# ShelfAugment AI — Streamlit Workspace

Retail shelf-image augmentation and YOLO dataset preparation application.

## Deploy with GitHub + Streamlit Community Cloud

1. Extract this ZIP.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `.streamlit/` folder to the root of your GitHub repository.
3. Open Streamlit Community Cloud and connect the repository.
4. Select the branch and set the main file path to `app.py`.
5. Deploy. Streamlit installs packages from `requirements.txt`.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Modules

- **Overview:** session-based activity metrics and recent actions.
- **Image Studio:** single-image Lighting, Camera angle, Perspective and Combined transformations; green YOLO overlays; side-by-side original/output comparison; updated image and label downloads.
- **Batch Processing:** ZIP-based processing, matching image/label names, row-level annotation validation, metadata CSV, validation CSV, report JSON and output ZIP.
- **Dataset QA:** accepts split YOLO datasets (`images/train`, `images/test`, `labels/train`, `labels/test`) and flat datasets (`images/`, `labels/`); reports exact invalid annotation rows.
- **TensorFlow preparation:** creates 640×640 RGB letterboxed input data, normalized pixels, YOLO box arrays, padded `tf.data` pipeline source, and a JSON report. Invalid YOLO rows are excluded from exported label files and counted in the report. Augmented images are added only to the training split when their source image is assigned to training.

## Scope and limitations

This application performs augmentation and dataset preparation. It does **not** train or evaluate an object detector or calculate detector metrics such as mAP, precision, recall, or F1. TensorFlow is not installed in the Streamlit deployment because the app exports a TensorFlow pipeline script for execution in Kaggle or Colab. The generated script requires TensorFlow, OpenCV and NumPy in that environment.

Perspective and combined augmentation strengths have been increased from the prior build. Inspect transformed images and green annotation overlays before using generated data for model training; augmentation strength is configurable and overly aggressive settings can distort products or filter boxes.

## Theme palette

- Navy: `#27374D`
- Slate blue: `#526D82`
- Steel blue: `#9DB2BF`
- Light blue-grey: `#DDE6ED`
