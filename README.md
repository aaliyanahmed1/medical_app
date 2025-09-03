# COVID-19 Multi-task Medical Image Analysis

A Streamlit web application that analyzes chest X-ray images using ONNX models for COVID-19 detection and lung segmentation.Model is primarily developed using MONAI framework and has been decoupled with for the faster inference and has been exported to ONNX format. 

## 🚀 Quick Start

```bash
git clone https://github.com/aaliyanahmed1/medical_app
cd medical_app
pip install -r requirements.txt
streamlit run app.py
```

## ✨ What It Does

**Classification**: Predicts one of four classes:
- **COVID**: COVID-19 positive cases
- **Lung_Opacity**: Lung opacity (non-COVID)
- **Normal**: Healthy lungs
- **Viral Pneumonia**: Viral pneumonia cases

**Segmentation**: Highlights lung regions with red overlay for visual analysis

**Validation**: Automatically checks if uploaded images are valid chest X-rays

## 🔧 How It Works

1. **Upload**: User uploads a chest X-ray image (PNG/JPG, max 50MB)
2. **Validate**: App checks if image is a valid X-ray (grayscale, proper contrast)
3. **Process**: Image is resized to 256x256 and normalized
4. **Predict**: ONNX model runs inference for classification and segmentation
5. **Display**: Shows prediction with confidence and lung segmentation overlay

## 📊 Usage

**For Medical Professionals**: Upload chest X-ray images to get instant COVID-19 classification and lung segmentation for diagnostic assistance.


**Repository**: [https://github.com/aaliyanahmed1/medical_app](https://github.com/aaliyanahmed1/medical_app)

---

Built with using Streamlit, ONNX Runtime, and OpenCV

