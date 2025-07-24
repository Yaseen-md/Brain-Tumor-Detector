# 🧠 Brain Tumor Detection with MRI Scans

A deep learning-powered web app built with **PyTorch** and **Streamlit** to detect brain tumor types from MRI scans. The model predicts one of four classes and provides a visual explanation using **Grad-CAM** heatmaps.

---

## 🚀 Features

- 🔍 Detects brain tumors: **Glioma**, **Meningioma**, **Pituitary**, **No Tumor**
- 📤 Upload your own MRI or paste a direct image URL
- 🧠 Visual explanations using Grad-CAM
- 📈 Displays class probabilities and prediction confidence
- 🌐 User-friendly web interface built with Streamlit

---

## 🧪 Model Info

- Architecture: `ResNet18`
- Framework: `PyTorch`
- Dataset: [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- Trained on: 4-class classification (glioma, meningioma, pituitary, no tumor)

---

## 📦 How to Run Locally

1. **Clone this repository**
```bash
git clone https://github.com/your-username/brain-tumor-detector.git
cd brain-tumor-detector
