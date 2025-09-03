"""
COVID-19 Multi-task Medical Image Analysis Application.

This module provides functionality for COVID-19 medical image analysis using
both classification and segmentation with an ONNX model through Streamlit.
"""

import json
import numpy as np
import cv2
import streamlit as st
import onnxruntime as ort


def load_metadata(meta_path: str) -> dict:
    """
    Load model metadata from a JSON file.

    Args:
        meta_path (str): Path to the metadata JSON file.

    Returns:
        dict: Metadata containing model parameters and class names.
    """
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_session(onnx_path: str) -> ort.InferenceSession:
    """
    Initialize an ONNX Runtime inference session.

    Args:
        onnx_path (str): Path to the ONNX model file.

    Returns:
        ort.InferenceSession: Initialized ONNX Runtime session.
    """
    sess_options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']
    return ort.InferenceSession(onnx_path, sess_options, providers=providers)


def preprocess(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """
    Preprocess an input image for model inference.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.
        size (int): Target size for resizing.

    Returns:
        np.ndarray: Preprocessed image in NCHW format.
    """
    image_resized = cv2.resize(image_bgr, (size, size))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image_norm, (2, 0, 1))
    return chw[np.newaxis, ...]


def postprocess(segmentation: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Post-process model outputs to get final predictions.

    Args:
        segmentation (np.ndarray): Raw segmentation output from model.
        logits (np.ndarray): Raw classification logits from model.

    Returns:
        tuple[np.ndarray, np.ndarray]: Processed segmentation and probabilities.
    """
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return segmentation, probs


def overlay_segmentation(image_bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """
    Create a visualization by overlaying segmentation mask on the image.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.
        seg (np.ndarray): Segmentation mask.

    Returns:
        np.ndarray: Image with segmentation overlay in red.
    """
    pred = (seg[0] > 0.5).astype(np.float32)
    overlay = image_bgr.copy().astype(np.float32) / 255.0
    overlay[..., 2] = np.maximum(overlay[..., 2], pred)  # red
    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return overlay


def annotate(overlay_bgr: np.ndarray, text: str) -> np.ndarray:
    """
    Add text annotation to an image.

    Args:
        overlay_bgr (np.ndarray): Input image in BGR format.
        text (str): Text to annotate on the image.

    Returns:
        np.ndarray: Annotated image.
    """
    out = overlay_bgr.copy()
    cv2.rectangle(out, (5, 5), (5 + 520, 50), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def side_by_side(original_bgr: np.ndarray, annotated_bgr: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side comparison image.

    Args:
        original_bgr (np.ndarray): Original image in BGR format.
        annotated_bgr (np.ndarray): Annotated image in BGR format.

    Returns:
        np.ndarray: Combined side-by-side image.
    """
    h = max(original_bgr.shape[0], annotated_bgr.shape[0])
    w = original_bgr.shape[1] + annotated_bgr.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:original_bgr.shape[0], :original_bgr.shape[1]] = original_bgr
    canvas[:annotated_bgr.shape[0],
           original_bgr.shape[1]:original_bgr.shape[1]+annotated_bgr.shape[1]] = annotated_bgr
    return canvas


# Constants
DEFAULT_ONNX = 'models/covid_multitask.onnx'
DEFAULT_META = 'models/covid_multitask.onnx.meta.json'

# Streamlit UI Setup
st.set_page_config(page_title='COVID Multi-task (ONNX)', layout='centered')
st.title('COVID Multi-task: Classification + Segmentation (ONNX)')
st.caption('Upload an image to see predicted class and segmentation overlay (red)')

# Sidebar Configuration
with st.sidebar:
    st.header('Model Files (local)')
    onnx_path = st.text_input('ONNX Model Path', DEFAULT_ONNX)
    meta_path = st.text_input('Metadata JSON Path', DEFAULT_META)

@st.cache_resource(show_spinner=False)
def _load(onnx_path: str, meta_path: str) -> tuple[ort.InferenceSession, dict]:
    """
    Load model and metadata with caching.

    Args:
        onnx_path (str): Path to ONNX model file.
        meta_path (str): Path to metadata JSON file.

    Returns:
        tuple[ort.InferenceSession, dict]: Loaded model session and metadata.
    """
    meta = load_metadata(meta_path)
    session = load_session(onnx_path)
    return session, meta

# Initialize model and metadata
session = None
meta = None
try:
    session, meta = _load(onnx_path, meta_path)
except Exception as e:
    st.error(f'Failed to load model/meta: {e}')

uploaded = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

if uploaded is not None and session is not None and meta is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error('Invalid image file')
    else:
        class_names = meta['class_names']
        img_size = int(meta.get('img_size', 256))
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        inp = preprocess(image_bgr, img_size)
        seg, logits = session.run(output_names, {input_name: inp})
        seg, probs = postprocess(seg, logits)
        pred_idx = int(np.argmax(probs[0]))
        pred_name = class_names[pred_idx]
        prob_str = ', '.join([f'{c}:{probs[0][i]:.2f}' for i, c in enumerate(class_names)])

        resized = cv2.resize(image_bgr, (img_size, img_size))
        overlay = overlay_segmentation(resized, seg[0])
        annotated = annotate(overlay, f'Pred: {pred_name} | {prob_str}')
        combined = side_by_side(resized, annotated)

        st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), caption=f'Prediction: {pred_name}', use_column_width=True)

st.markdown('---')
st.caption('Deploy on Streamlit Cloud: set the repo to this folder and entry point to app.py')


