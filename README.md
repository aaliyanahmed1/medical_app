# COVID-19 Multi-task Medical Image Analysis Application

A Streamlit-based web application for COVID-19 medical image analysis that performs both classification and segmentation using an ONNX model.

## Features

- Image Upload: Supports PNG, JPG, and JPEG formats
- Multi-task Analysis:
  - Classification: Predicts COVID-19 related classes
  - Segmentation: Highlights regions of interest in red overlay
- Real-time Processing: Instant results upon image upload
- Side-by-side Visualization: Shows original and annotated images
- Configurable Model Paths: Allows custom model and metadata file selection

## Technical Components

### Libraries and Dependencies

1. **Core Libraries**
   - `streamlit`: Web application framework for the user interface
   - `onnxruntime`: Deep learning model inference engine
   - `opencv-python` (cv2): Image processing and visualization
   - `numpy`: Numerical computations and array operations

2. **File Requirements**
   - Main application: `app.py`
   - Model file: `models/covid_multitask.onnx`
   - Model metadata: `models/covid_multitask.onnx.meta.json`
   - Dependencies: `requirements.txt`

### Key Functions

1. **Model Management**
   - `load_metadata(meta_path)`: Loads model metadata (class names, image size)
   - `load_session(onnx_path)`: Initializes ONNX Runtime session with CPU provider

2. **Image Processing**
   - `preprocess(image_bgr, size)`: 
     - Resizes image to model input size
     - Converts BGR to RGB
     - Normalizes pixel values (0-1)
     - Transposes to channel-first format (CHW)

3. **Inference and Post-processing**
   - `postprocess(segmentation, logits)`:
     - Converts logits to probabilities using softmax
     - Returns segmentation mask and class probabilities

4. **Visualization**
   - `overlay_segmentation(image_bgr, seg)`: Overlays segmentation mask in red
   - `annotate(overlay_bgr, text)`: Adds prediction text to image
   - `side_by_side(original_bgr, annotated_bgr)`: Creates comparison view

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aaliyanahmed1/medical_app.git
cd medical_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface:
   - Open browser at http://localhost:8501
   - Upload a medical image
   - View predictions and segmentation results

### Streamlit Cloud Deployment

1. Push your repository to GitHub
2. On Streamlit Cloud:
   - Connect to your repository
   - Set branch: main
   - Set main file path: app.py
   - Deploy

## Configuration

The application provides configurable paths for model files in the sidebar:
- ONNX Model Path: Location of the ONNX model file
- Metadata JSON Path: Location of the model metadata file

## Cache Management

The application uses Streamlit's caching mechanism (`@st.cache_resource`) to optimize:
- Model loading
- Session initialization
- Metadata parsing

## Error Handling

The application includes robust error handling for:
- Invalid file uploads
- Model loading failures
- Inference errors

## Technical Requirements

- Python 3.6+
- ONNX Runtime
- OpenCV
- NumPy
- Streamlit

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

[Specify your license here]
