# ShelfAugment AI — Streamlit Workspace

A Streamlit application for retail shelf-image augmentation and YOLO dataset preparation.

## Deploy with GitHub + Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Open Streamlit Community Cloud and connect your GitHub account.
4. Select the repository and branch.
5. Set the main file path to `app.py`, then deploy.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Included modules

- Dashboard with session-based processing metrics and activity.
- Image Studio for individual shelf images.
- Lighting, camera-angle, perspective, and combined augmentation.
- Batch processing with downloadable outputs and metadata.
- YOLO annotation-aware transformations and dataset quality checks.
- Dataset export/preparation utilities, including TensorFlow pipeline scaffolding where provided in the app.

## Important scope notes

This app focuses on image augmentation and dataset preparation. It does not train a planogram-compliance detector or independently calculate detector metrics such as mAP, precision, recall, or F1. Dashboard metrics reflect actions performed during the current app session.

The TensorFlow export feature prepares a pipeline script; TensorFlow itself is not required to run this Streamlit app. Run the generated TensorFlow script in an environment where TensorFlow is installed.

## Theme palette

- Navy: `#27374D`
- Slate blue: `#526D82`
- Steel blue: `#9DB2BF`
- Light blue-grey: `#DDE6ED`
