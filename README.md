# COVID-19 Multi-task Medical Image Analysis

A Streamlit web application that analyzes chest X-ray images using ONNX models for COVID-19 detection, lung segmentation, and automated radiology report generation. Model is primarily developed using MONAI framework and has been decoupled for faster inference and exported to ONNX format.

[Web Application](https://medicalapp-nvxwpnutdybrfbujnbfed4.streamlit.app/)


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

**Radiology Report**: Generates professional medical reports with:
- Structured findings and pathology analysis
- Quantitative measurements (affected area percentage)
- Clinical impressions and recommendations
- Downloadable reports in professional format

### 📋 Report Structure

The generated radiology report follows professional medical standards and includes:

**Header Information:**
- Study Date and Time
- Patient ID (image filename)
- Clinical Indication
- Technique Description

**Findings Section:**
- Lung Fields analysis with affected area percentage
- Heart Size assessment
- Pleural Spaces evaluation
- Osseous Structures review

**Pathology Analysis:**
- COVID-19 confidence score
- Lung Opacity detection
- Normal findings confirmation
- Viral Pneumonia assessment
- All confidence scores >10%

**Measurements:**
- Affected Area Percentage (calculated from segmentation)
- Primary Diagnosis Confidence
- Complete confidence score distribution

**Clinical Sections:**
- **Impression**: Summary of key findings with clinical significance
- **Recommendations**: Follow-up actions based on diagnosis
  - COVID-19: Immediate testing, CT consideration, monitoring
  - Lung Opacity: Clinical correlation, follow-up imaging
  - Normal: Routine follow-up as clinically indicated
  - Viral Pneumonia: Viral testing, supportive care

**Technical Notes:**
- AI Model details (MONAI-based)
- Processing time and confidence scores
- Localization quality assessment
- Model export format (ONNX)
- Analysis timestamp

**Downloadable Format:**
- Professional text file (.txt)
- Timestamped filename
- Ready for medical records integration

**Validation**: Automatically checks if uploaded images are valid chest X-rays

## 📋 Image Requirements

**Accepted Formats**: PNG, JPG, or JPEG files
**Size Limits**: Minimum 128x128 pixels, maximum 50MB
**Content**: Chest X-ray images (any exposure level)
**Quality**: Clear, readable medical images

✅ **System Accepts:**
- Standard chest X-rays (PA view)
- Light or dark exposures
- Various image qualities
- Digitized film X-rays
- Digital X-ray images

❌ **System Rejects:**
- Non-medical images
- Non-chest X-rays
- Corrupted files
- Extremely small images

## 🔧 How It Works

1. **Upload**: User uploads a chest X-ray image (PNG/JPG, max 50MB)
2. **Validate**: App checks if image is a valid X-ray (grayscale, proper contrast)
3. **Process**: Image is resized to 256x256 and normalized
4. **Predict**: ONNX model runs inference for classification and segmentation
5. **Report**: Generates professional radiology report with findings and recommendations
6. **Display**: Shows prediction with confidence, segmentation overlay, and downloadable report

## 📊 Usage

**For Medical Professionals**: Upload chest X-ray images to get instant COVID-19 classification, lung segmentation, and automated radiology reports for diagnostic assistance.

**For Researchers**: Use the ONNX model for efficient inference and analysis of chest X-ray datasets.

**For Developers**: Integrate the ONNX model into your medical imaging applications.

## 🤝 Contributing

**Repository**: [https://github.com/aaliyanahmed1/medical_app](https://github.com/aaliyanahmed1/medical_app)

---

Built with Streamlit, ONNX Runtime, and OpenCV

