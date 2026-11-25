# ShelfAugment AI

Streamlit application for retail shelf-image augmentation, YOLO annotation validation, batch processing, and TensorFlow dataset export.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with GitHub and Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and this README to your GitHub repository.
2. Create an app in Streamlit Community Cloud from that repository.
3. Set the main file path to `app.py` and deploy.

TensorFlow is not installed in the Streamlit deployment. The app exports a TensorFlow input package and pipeline script for execution in Kaggle or Colab.
