# ShelfAugment AI — Streamlit

A Streamlit application for retail shelf-image augmentation with optional YOLO annotation handling.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and this `README.md` to the repository root.
3. In Streamlit Community Cloud, select **New app**, choose the repository and `app.py`.
4. Deploy.

## Single image

Upload JPG/PNG and optionally a matching YOLO `.txt` annotation. Select Lighting, Camera angle, Perspective, or Combined, then download the augmented image and updated labels.

## Batch ZIP structure

```text
dataset.zip
├── images/
│   ├── shelf_001.jpg
│   └── shelf_002.jpg
└── labels/
    ├── shelf_001.txt
    └── shelf_002.txt
```

The `labels/` directory is optional. Image and label filenames must match (apart from extension).

## Notebook class mapping

| ID | Class |
|---:|---|
| 0 | Coca Cola |
| 1 | Sprite |
| 2 | Pepsi |
| 3 | Mountain Dew |
| 4 | 7UP |
| 5 | Fanta |

## Scope

This is an augmentation and dataset-preparation interface. It does not train a detector or calculate planogram compliance.
