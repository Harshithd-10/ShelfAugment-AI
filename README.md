# ShelfAugment AI

Streamlit app for retail shelf-image augmentation, YOLO annotation validation, batch processing, and TensorFlow dataset export.

Deploy through GitHub and Streamlit Community Cloud with `app.py` as the main file.

Local run:
```bash
pip install -r requirements.txt
streamlit run app.py
```


To deploy an updated ZIP, extract its contents into your existing Git repository and replace the existing files, especially `app.py` and `.streamlit/config.toml`. Commit and push those files, then redeploy or restart the Streamlit app. Uploading the ZIP itself to the repository does not replace the app source.
