# Streamlit Cloud Deployment

## Files
- app.py: Streamlit app (loads ONNX and metadata from models/ subfolder)
- models/covid_multitask.onnx: Exported ONNX model
- models/covid_multitask.onnx.meta.json: Metadata (class names, image size)
- requirements.txt: Python dependencies

## Deploy Steps
1. Push this new_streamlit/ folder to a Git repository.
2. On Streamlit Cloud, create an app:
   - Repo: your repo
   - Branch: main (or your branch)
   - File: new_streamlit/app.py
3. Deploy.

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a .png/.jpg and the app will show side-by-side input and annotated output (classification + segmentation overlay).
