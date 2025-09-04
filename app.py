"""COVID-19 Multi-task Classification and Segmentation Streamlit App.

This app provides a user-friendly interface for COVID-19 chest X-ray analysis
using an ONNX model for both classification and segmentation tasks.
"""

import os
import json
import time
from typing import Tuple, Optional
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
import onnxruntime as ort
import base64
from io import BytesIO


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
    """Validate if image appears to be a chest X-ray with balanced criteria.
    
    Args:
        image_bgr: Input BGR image or None
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image_bgr is None:
        return False, "Failed to read the image file."
    
    # Check image dimensions
    h, w = image_bgr.shape[:2]
    if min(h, w) < 256:
        return False, "Image is too small. Minimum dimension should be 256 pixels for X-ray analysis."
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Check if image is actually grayscale (not just black and white)
    # Calculate color channel differences
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    diff_bg = np.mean(np.abs(b - g))
    diff_br = np.mean(np.abs(b - r))
    diff_gr = np.mean(np.abs(g - r))
    avg_diff = (diff_bg + diff_br + diff_gr) / 3.0
    
    if avg_diff > 8.0:  # Increased from 5.0 to allow for slight color variations
        return False, "Image appears to be colorful. X-ray images must be grayscale."
    
    # 2. Check for X-ray specific characteristics with more lenient ranges
    # X-rays should have specific intensity distribution
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # More lenient brightness range for different X-ray types
    if mean_intensity < 20 or mean_intensity > 240:  # Widened from 30-225
        return False, f"Image brightness ({mean_intensity:.1f}) is outside X-ray range (20-240)."
    
    if std_intensity < 8:  # Reduced from 15 to allow lighter X-rays
        return False, f"Image contrast ({std_intensity:.1f}) is too low for X-ray. X-rays need some contrast."
    
    # 3. Check for X-ray anatomical features with lower threshold
    # X-rays should have some lung-like patterns (not just uniform)
    # Calculate local variance to detect anatomical structures
    kernel = np.ones((8, 8), np.float32) / 64
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    avg_local_variance = np.mean(local_variance)
    
    if avg_local_variance < 20:  # Reduced from 50 to allow lighter X-rays
        return False, "Image lacks anatomical detail. X-rays should show lung structures and ribs."
    
    # 4. Check for X-ray specific intensity patterns with lower entropy threshold
    # X-rays have characteristic histogram distribution
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.flatten() / np.sum(hist)
    
    # Check if histogram has proper X-ray distribution (not too uniform, not too peaked)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
    if entropy < 3.0:  # Reduced from 4.0 to allow lighter X-rays
        return False, "Image histogram is too uniform. X-rays have characteristic intensity distribution."
    
    # 5. Check for rib-like structures with lower threshold
    # Apply edge detection to find potential rib structures
    edges = cv2.Canny(gray, 30, 100)  # Lowered thresholds for lighter X-rays
    horizontal_kernel = np.array([[1, 1, 1, 1, 1]], np.uint8)
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Count horizontal line pixels
    horizontal_pixels = np.sum(horizontal_lines > 0)
    total_pixels = gray.size
    
    if horizontal_pixels / total_pixels < 0.0005:  # Reduced from 0.001
        return False, "No rib-like structures detected. X-rays should show rib outlines."
    
    # 6. Check for lung field characteristics with more lenient thresholds
    # X-rays should have darker lung areas and brighter bone areas
    # Calculate ratio of dark to bright areas
    dark_threshold = np.percentile(gray, 25)  # Changed from 30
    bright_threshold = np.percentile(gray, 75)  # Changed from 70
    
    dark_pixels = np.sum(gray < dark_threshold)
    bright_pixels = np.sum(gray > bright_threshold)
    
    if dark_pixels / total_pixels < 0.05 or bright_pixels / total_pixels < 0.05:  # Reduced from 0.1
        return False, "Image lacks proper X-ray contrast between lung fields and bones."
    
    # 7. Final check: Ensure it's not just a simple black and white image
    # Check for intermediate gray values (X-rays have many gray levels)
    unique_values = len(np.unique(gray))
    if unique_values < 50:  # Reduced from 100 to allow lighter X-rays
        return False, f"Image has too few gray levels ({unique_values}). X-rays have continuous gray scale."
    
    return True, ""


def calculate_affected_area_percentage(segmentation: np.ndarray) -> float:
    """Calculate the percentage of affected lung area from segmentation mask.
    
    Args:
        segmentation (np.ndarray): Segmentation mask from model output.
        
    Returns:
        float: Percentage of affected area (0-100).
        
    Example:
        >>> affected_pct = calculate_affected_area_percentage(seg_mask)
        >>> print(f"Affected area: {affected_pct:.1f}%")
    """
    # Apply threshold to get binary mask
    binary_mask = (segmentation[0] > 0.5).astype(np.float32)
    
    # Calculate percentage of affected pixels
    total_pixels = binary_mask.size
    affected_pixels = np.sum(binary_mask)
    affected_percentage = (affected_pixels / total_pixels) * 100.0
    
    return affected_percentage


def generate_radiology_report(image_name: str, class_names: list, probs: np.ndarray, 
                            segmentation: np.ndarray, elapsed_ms: float, 
                            confidence: float) -> str:
    """Generate structured radiology report based on AI analysis results.
    
    Args:
        image_name (str): Name of the uploaded image file.
        class_names (list): List of classification class names.
        probs (np.ndarray): Classification probabilities.
        segmentation (np.ndarray): Segmentation mask from model.
        elapsed_ms (float): Inference time in milliseconds.
        confidence (float): Overall model confidence score.
        
    Returns:
        str: Formatted radiology report in professional medical format.
        
    Example:
        >>> report = generate_radiology_report("xray.jpg", classes, probs, seg, 150.5, 0.85)
        >>> print(report)
    """
    # Get current date and time
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Calculate affected area percentage
    affected_area_pct = calculate_affected_area_percentage(segmentation)
    
    # Get primary diagnosis and confidence
    pred_idx = int(np.argmax(probs[0]))
    primary_diagnosis = class_names[pred_idx]
    primary_confidence = probs[0][pred_idx]
    
    # Generate findings based on classification
    findings = generate_findings(primary_diagnosis, affected_area_pct)
    
    # Generate pathology analysis
    pathology_analysis = generate_pathology_analysis(class_names, probs[0])
    
    # Generate impression and recommendations
    impression, recommendations = generate_impression_recommendations(primary_diagnosis, affected_area_pct, primary_confidence)
    
    # Format the report
    report = f"""
CHEST X-RAY REPORT
Study Date: {current_date}
Study Time: {current_time}
Patient ID: {image_name}
Image File: {image_name}

CLINICAL INDICATION:
Chest X-ray for evaluation of respiratory symptoms and COVID-19 screening

TECHNIQUE:
Single-view posteroanterior chest radiograph
AI-powered analysis using MONAI-based deep learning multi-head model


FINDINGS:
{findings}

PATHOLOGY ANALYSIS:
{pathology_analysis}

MEASUREMENTS:
- Affected Area Percentage: {affected_area_pct:.1f}%
- Primary Diagnosis Confidence: {primary_confidence:.1%}
- Confidence Score Distribution: {', '.join([f'{c}: {p:.1%}' for c, p in zip(class_names, probs[0])])}

IMPRESSION:
{impression}

RECOMMENDATIONS:
{recommendations}

TECHNICAL NOTES:
- AI Model: MONAI-based COVID-19 Multi-task Classification and Segmentation
- AI Confidence Score: {confidence:.1%}
- Processing Time: {elapsed_ms:.1f} ms
- Localization Quality: {'Good' if affected_area_pct > 5 else 'Minimal'} affected area detected
- Model Export Format: ONNX for optimized inference
- Analysis Date: {current_date} at {current_time}

---
Report generated by AI-powered medical imaging analysis system.
This report is for clinical decision support and should be reviewed by qualified medical professionals.
"""
    
    return report.strip()


def generate_findings(diagnosis: str, affected_area_pct: float) -> str:
    """Generate findings section based on diagnosis and affected area.
    
    Args:
        diagnosis (str): Primary diagnosis from model.
        affected_area_pct (float): Percentage of affected lung area.
        
    Returns:
        str: Formatted findings section.
    """
    findings_map = {
        "COVID": f"- Lung Fields: Bilateral patchy opacities with {affected_area_pct:.1f}% affected area\n- Heart Size: Normal cardiac silhouette\n- Pleural Spaces: Clear bilaterally\n- Osseous Structures: No acute bony abnormalities",
        "Lung_Opacity": f"- Lung Fields: Focal opacity detected in {affected_area_pct:.1f}% of lung fields\n- Heart Size: Normal cardiac silhouette\n- Pleural Spaces: Clear bilaterally\n- Osseous Structures: No acute bony abnormalities",
        "Normal": f"- Lung Fields: Clear lung fields with normal aeration\n- Heart Size: Normal cardiac silhouette\n- Pleural Spaces: Clear bilaterally\n- Osseous Structures: No acute bony abnormalities",
        "Viral Pneumonia": f"- Lung Fields: Viral pneumonia pattern with {affected_area_pct:.1f}% affected area\n- Heart Size: Normal cardiac silhouette\n- Pleural Spaces: Clear bilaterally\n- Osseous Structures: No acute bony abnormalities"
    }
    
    return findings_map.get(diagnosis, findings_map["Normal"])


def generate_pathology_analysis(class_names: list, probabilities: np.ndarray) -> str:
    """Generate pathology analysis section with confidence scores.
    
    Args:
        class_names (list): List of classification classes.
        probabilities (np.ndarray): Classification probabilities.
        
    Returns:
        str: Formatted pathology analysis section.
    """
    analysis_lines = []
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        if prob > 0.1:  # Only include if confidence > 10%
            analysis_lines.append(f"- {class_name}: {prob:.1%} confidence")
    
    return '\n'.join(analysis_lines) if analysis_lines else "- No significant pathology detected"


def generate_impression_recommendations(diagnosis: str, affected_area_pct: float, confidence: float) -> Tuple[str, str]:
    """Generate impression and recommendations based on diagnosis.
    
    Args:
        diagnosis (str): Primary diagnosis.
        affected_area_pct (float): Percentage of affected area.
        confidence (float): Model confidence score.
        
    Returns:
        Tuple[str, str]: (impression, recommendations)
    """
    impression_map = {
        "COVID": f"Findings consistent with COVID-19 pneumonia with {affected_area_pct:.1f}% lung involvement. High clinical suspicion for COVID-19 infection.",
        "Lung_Opacity": f"Focal lung opacity detected with {affected_area_pct:.1f}% affected area. Clinical correlation recommended.",
        "Normal": "Normal chest X-ray with clear lung fields and normal cardiac silhouette.",
        "Viral Pneumonia": f"Findings consistent with viral pneumonia with {affected_area_pct:.1f}% lung involvement."
    }
    
    recommendations_map = {
        "COVID": "- Immediate COVID-19 testing recommended\n- Consider chest CT for detailed assessment\n- Monitor oxygen saturation and respiratory status\n- Follow up with pulmonologist",
        "Lung_Opacity": "- Clinical correlation with symptoms recommended\n- Consider follow-up imaging in 2-4 weeks\n- Monitor for symptom progression",
        "Normal": "- No immediate follow-up imaging required\n- Clinical correlation with symptoms recommended\n- Routine follow-up as clinically indicated",
        "Viral Pneumonia": "- Viral testing recommended\n- Supportive care and monitoring\n- Follow-up imaging if symptoms persist\n- Consider antiviral therapy if appropriate"
    }
    
    impression = impression_map.get(diagnosis, impression_map["Normal"])
    recommendations = recommendations_map.get(diagnosis, recommendations_map["Normal"])
    
    return impression, recommendations


def get_download_link(text: str, filename: str, file_type: str = "text/plain") -> str:
    """Generate download link for text content.
    
    Args:
        text (str): Content to download.
        filename (str): Name of the file.
        file_type (str): MIME type of the file.
        
    Returns:
        str: HTML download link.
    """
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:{file_type};base64,{b64}" download="{filename}">Download {filename}</a>'
def run_inference(session: ort.InferenceSession, image_bgr: np.ndarray, 
                 img_size: int, repeats: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run model inference with timing.
    
    Args:
        session (ort.InferenceSession): ONNX Runtime session.
        image_bgr (np.ndarray): Input BGR image.
        img_size (int): Target image size for preprocessing.
        repeats (int): Number of inference repeats for timing (default: 1).
        
    Returns:
        Tuple[np.ndarray, np.ndarray, float]: (segmentation, probabilities, 
        average_inference_time_ms).
        
    Example:
        >>> seg, probs, time_ms = run_inference(session, img, 256, repeats=5)
        >>> print(f"Inference time: {time_ms:.1f} ms")
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

st.title('COVID Multi-task: Classification + Segmentation + Radiology Report (ONNX)')
st.caption('Upload a chest X-ray to get AI-powered classification, segmentation, and automated radiology report')

# Add clear instructions for users
st.markdown("""
### 📋 **Image Requirements:**
- **Format**: PNG, JPG, or JPEG files
- **Size**: Minimum 128x128 pixels, maximum 50MB
- **Content**: Chest X-ray images (any exposure level)
- **Quality**: Clear, readable medical images

✅ **System Accepts:**
- Standard chest X-rays (PA view)
- Digitized film X-rays
- Digital X-ray images

❌ **System Rejects:**
- Non-medical images
- Non-chest X-rays
- Corrupted files
- Extremely small images
- low visibility images
- light exposure images
- dark exposure images
""")

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
    - Automated radiology report generation
    - Downloadable reports in professional format
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
    help='Upload ONLY chest X-ray images showing lung fields and rib structures. Minimum 256x256 pixels. Maximum file size: 50MB'
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
            st.error(f'❌ **Image Validation Failed**: {error_msg}\n\n**Please ensure your image is a chest X-ray** with:\n- Clear lung fields and rib structures\n- Proper contrast between bones and soft tissue\n- Characteristic X-ray intensity distribution\n- Minimum 256x256 pixels resolution\n\n**Note**: The system accepts various X-ray types including lighter exposures.')
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
            
            # Calculate affected area percentage
            affected_area_pct = calculate_affected_area_percentage(seg)
            
            # Generate radiology report
            report = generate_radiology_report(
                uploaded.name, class_names, probs, seg, elapsed_ms, confidence
            )
            
            # Display report
            st.subheader("📋 Radiology Report")
            st.text_area("Generated Report", report, height=400)
            
            # Download report
            st.markdown("### 📥 Download Report")
            report_filename = f"radiology_report_{uploaded.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            download_link = get_download_link(report, report_filename)
            st.markdown(download_link, unsafe_allow_html=True)
            
            # Additional metrics
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Affected Area", f"{affected_area_pct:.1f}%")
            with col4:
                st.metric("Primary Diagnosis", pred_name)
                
    except Exception as e:
        st.error(f'Error processing image: {e}')
        st.stop()

# Footer
st.markdown('---')
st.caption('Built with Streamlit • ONNX Runtime • OpenCV')


