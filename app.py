"""COVID-19 Multi-task Classification and Segmentation Streamlit App.

This app provides a user-friendly interface for COVID-19 chest X-ray analysis
using an ONNX model for both classification and segmentation tasks.
"""

import os
import json
import time
from typing import Tuple, Optional
import numpy as np
import cv2
import streamlit as st
import onnxruntime as ort


def load_metadata(meta_path: str) -> dict:
    """Load model metadata from JSON file.
    
    Args:
        meta_path: Path to metadata JSON file
        
    Returns:
        Dictionary containing model metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        json.JSONDecodeError: If metadata file is invalid JSON
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_session(onnx_path: str) -> ort.InferenceSession:
    """Load ONNX Runtime inference session.
    
    Args:
        onnx_path: Path to ONNX model file
        
    Returns:
        ONNX Runtime inference session
        
    Raises:
        FileNotFoundError: If ONNX file doesn't exist
        RuntimeError: If ONNX model loading fails
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    sess_options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']
    
    try:
        return ort.InferenceSession(onnx_path, sess_options, providers=providers)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")


def preprocess(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """Preprocess image for model input.
    
    Args:
        image_bgr: Input BGR image
        size: Target size for resizing
        
    Returns:
        Preprocessed image tensor in NCHW format
    """
    image_resized = cv2.resize(image_bgr, (size, size))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image_norm, (2, 0, 1))
    return chw[np.newaxis, ...]


def postprocess(segmentation: np.ndarray, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Postprocess model outputs.
    
    Args:
        segmentation: Raw segmentation output
        logits: Raw classification logits
        
    Returns:
        Tuple of (segmentation, probabilities)
    """
    # Apply softmax to logits
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return segmentation, probs


def overlay_segmentation(image_bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Overlay segmentation mask on image.
    
    Args:
        image_bgr: Input BGR image
        seg: Segmentation mask
        
    Returns:
        Image with segmentation overlay
    """
    pred = (seg[0] > 0.5).astype(np.float32)
    overlay = image_bgr.copy().astype(np.float32) / 255.0
    overlay[..., 2] = np.maximum(overlay[..., 2], pred)  # Red channel
    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return overlay


def annotate(overlay_bgr: np.ndarray, text: str) -> np.ndarray:
    """Add text annotation to image.
    
    Args:
        overlay_bgr: Input BGR image
        text: Text to add
        
    Returns:
        Annotated image
    """
    out = overlay_bgr.copy()
    cv2.rectangle(out, (5, 5), (5 + 520, 50), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def side_by_side(original_bgr: np.ndarray, annotated_bgr: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison image.
    
    Args:
        original_bgr: Original BGR image
        annotated_bgr: Annotated BGR image
        
    Returns:
        Combined side-by-side image
    """
    h = max(original_bgr.shape[0], annotated_bgr.shape[0])
    w = original_bgr.shape[1] + annotated_bgr.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Place original image on left
    canvas[:original_bgr.shape[0], :original_bgr.shape[1]] = original_bgr
    
    # Place annotated image on right
    start_x = original_bgr.shape[1]
    canvas[:annotated_bgr.shape[0], start_x:start_x + annotated_bgr.shape[1]] = annotated_bgr
    
    return canvas


def is_xray_like(image_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """Validate if image appears to be a chest X-ray.
    
    Args:
        image_bgr: Input BGR image or None
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image_bgr is None:
        return False, "Failed to read the image file."
    
    # Check image dimensions
    h, w = image_bgr.shape[:2]
    if min(h, w) < 128:
        return False, "Image is too small. Minimum dimension should be 128 pixels."
    
    # Check if image is grayscale-like (low colorfulness)
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    diff_bg = np.mean(np.abs(b - g))
    diff_br = np.mean(np.abs(b - r))
    diff_gr = np.mean(np.abs(g - r))
    avg_diff = (diff_bg + diff_br + diff_gr) / 3.0
    
    if avg_diff >= 10.0:
        return False, "Image appears to be colorful. X-ray images should be grayscale."
    
    # Check overall brightness and contrast
    mean_intensity = np.mean(image_bgr)
    std_intensity = np.std(image_bgr)
    
    if mean_intensity < 20 or mean_intensity > 235:
        return False, "Image is too dark or too bright for an X-ray."
    if std_intensity < 10:
        return False, "Image has very low contrast. Unlikely to be an X-ray."
    
    return True, ""


def run_inference(session: ort.InferenceSession, image_bgr: np.ndarray, 
                 img_size: int, repeats: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run model inference with timing.
    
    Args:
        session: ONNX Runtime session
        image_bgr: Input BGR image
        img_size: Target image size
        repeats: Number of inference repeats for timing
        
    Returns:
        Tuple of (segmentation, probabilities, average_inference_time_ms)
    """
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    inp = preprocess(image_bgr, img_size)
    
    # Warm-up run
    session.run(output_names, {input_name: inp})
    
    # Timed inference
    start = time.perf_counter()
    for _ in range(repeats):
        seg, logits = session.run(output_names, {input_name: inp})
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    
    seg, probs = postprocess(seg, logits)
    return seg, probs, elapsed_ms


# Streamlit app configuration
st.set_page_config(
    page_title='COVID Multi-task (ONNX)',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.title('COVID Multi-task: Classification + Segmentation (ONNX)')
st.caption('Upload a chest X-ray to see predicted class and segmentation overlay (red)')

# Constants
DEFAULT_ONNX = 'models/covid_multitask.onnx'
DEFAULT_META = 'models/covid_multitask.onnx.meta.json'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Sidebar configuration
with st.sidebar:
    st.header('Model Configuration')
    onnx_path = st.text_input('ONNX Model Path', DEFAULT_ONNX)
    meta_path = st.text_input('Metadata JSON Path', DEFAULT_META)
    repeats = st.number_input('Timing repeats', min_value=1, max_value=20, value=1, step=1)
    
    st.header('About')
    st.info("""
    This app analyzes chest X-ray images for COVID-19 detection.
    
    **Features:**
    - Multi-class classification
    - Lung segmentation
    - Performance timing
    - Input validation
    """)

# Load model
@st.cache_resource(show_spinner=False)
def _load_onnx(onnx_path: str, meta_path: str) -> Tuple[ort.InferenceSession, dict]:
    """Load ONNX model and metadata with caching."""
    meta = load_metadata(meta_path)
    session = load_session(onnx_path)
    return session, meta

# Model loading with error handling
try:
    session, meta = _load_onnx(onnx_path, meta_path)
except Exception as e:
    st.error(f'Failed to load ONNX model: {e}')
    st.stop()

# File uploader
uploaded = st.file_uploader(
    'Upload a chest X-ray image (PNG/JPG)',
    type=['png', 'jpg', 'jpeg'],
    help='Maximum file size: 50MB'
)

if uploaded is not None:
    # File size validation
    if uploaded.size > MAX_FILE_SIZE:
        st.error(f'File too large ({uploaded.size / 1024 / 1024:.1f}MB). Maximum size: 50MB')
        st.stop()
    
    # Load and validate image
    try:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = is_xray_like(image_bgr)
        if not is_valid:
            st.error(f'Invalid input: {error_msg}\n\nPlease upload a valid chest X-ray image.')
            st.stop()
        
        # Run inference
        with st.spinner('Processing image...'):
            class_names = meta['class_names']
            img_size = int(meta.get('img_size', 256))
            resized = cv2.resize(image_bgr, (img_size, img_size))
            
            seg, probs, elapsed_ms = run_inference(session, image_bgr, img_size, repeats)
            
            # Process results
            pred_idx = int(np.argmax(probs[0]))
            pred_name = class_names[pred_idx]
            confidence = probs[0][pred_idx]
            
            # Create probability string
            prob_str = ', '.join([f'{c}:{probs[0][i]:.3f}' for i, c in enumerate(class_names)])
            
            # Generate output
            overlay = overlay_segmentation(resized, seg[0])
            annotated = annotate(overlay, f'Pred: {pred_name} | {prob_str}')
            combined = side_by_side(resized, annotated)
            
            # Display results
            st.success(f'✅ **Prediction: {pred_name}** (Confidence: {confidence:.1%})')
            st.image(
                cv2.cvtColor(combined, cv2.COLOR_BGR2RGB),
                caption=f'Input vs. Output: {pred_name}',
                use_column_width=True
            )
            
            # Performance metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Inference Time", f"{elapsed_ms:.1f} ms")
            with col2:
                st.metric("Model Confidence", f"{confidence:.1%}")
            
            # Detailed results
            with st.expander("Detailed Results"):
                st.write("**Classification Probabilities:**")
                for i, (class_name, prob) in enumerate(zip(class_names, probs[0])):
                    st.write(f"- {class_name}: {prob:.3f}")
                
                st.write(f"**Segmentation:** Dice threshold applied at 0.5")
                st.write(f"**Performance:** {repeats} inference(s) averaged")
                
    except Exception as e:
        st.error(f'Error processing image: {e}')
        st.stop()

# Footer
st.markdown('---')
st.caption('Built with Streamlit • ONNX Runtime • OpenCV')


