<div align="center">

# 🧠 Brain Tumor Detection System

### AI-Powered MRI Analysis with Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yaseen-md-brain-tumor-detector-app-zhlhz5.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live Demo](https://yaseen-md-brain-tumor-detector-app-zhlhz5.streamlit.app/) • [Report Bug](https://github.com/Yaseen-md/Brain-Tumor-Detector/issues) • [Request Feature](https://github.com/Yaseen-md/Brain-Tumor-Detector/issues)

</div>

---

## 📋 Overview

An intelligent deep learning application that automatically classifies brain tumors from MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**. Built with state-of-the-art computer vision techniques and deployed as an interactive web application.

### 🎯 Key Highlights

- **High Accuracy**: ResNet18-based architecture fine-tuned for medical image classification
- **Explainable AI**: Grad-CAM visualization shows which brain regions influenced the prediction
- **User-Friendly**: Intuitive Streamlit interface requiring no technical expertise
- **Flexible Input**: Upload local images or provide URLs for instant analysis
- **Real-Time**: Fast predictions with confidence scores for all tumor types

---

## 🌐 Live Demo

**🚀 Try it now:** [Brain Tumor Detector Web App](https://yaseen-md-brain-tumor-detector-app-zhlhz5.streamlit.app/)

Upload an MRI scan and get instant results with visual explanations!

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Multi-Class Detection** | Classifies 4 tumor types: Glioma, Meningioma, Pituitary, No Tumor |
| 📤 **Flexible Upload** | Support for image upload and URL-based input |
| 🎨 **Grad-CAM Visualization** | Heatmap overlays highlight decision-making regions |
| 📊 **Confidence Scores** | Probability distribution across all classes |
| ⚡ **Fast Inference** | Real-time predictions with optimized model |
| 📱 **Responsive Design** | Works seamlessly on desktop and mobile devices |

---

## 🏗️ Architecture

### Model Specifications

- **Base Architecture**: ResNet18 (Transfer Learning)
- **Framework**: PyTorch 2.0+
- **Input Size**: 224×224 RGB images
- **Output Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Explainability**: Grad-CAM visualization on final convolutional layer

### Tech Stack

```
Frontend:  Streamlit
Backend:   PyTorch, OpenCV
Deployment: Streamlit Cloud
```

---

## 📊 Dataset

- **Source**: [Kaggle Brain MRI Images Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Classes**: 4 balanced categories
- **Preprocessing**: Resizing, normalization, augmentation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yaseen-md/Brain-Tumor-Detector.git
   cd Brain-Tumor-Detector
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   ```
   Navigate to http://localhost:8501
   ```

---

## 📁 Project Structure

```
Brain-Tumor-Detector/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── models/
│   ├── resnet18_best.pth          # Trained model weights
│   ├── loss_history.npy           # Training metrics
│   ├── y_pred.npy                 # Validation predictions
│   └── y_true.npy                 # Ground truth labels
├── src/
│   ├── model.py                   # Model architecture
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation utilities
│   ├── inference.py               # Prediction functions
│   ├── grad_cam.py                # Grad-CAM implementation
│   ├── data_loader.py             # Dataset handling
│   ├── utils.py                   # Helper functions
│   └── validate_dataset.py        # Data validation
└── notebooks/
    ├── 01_data_exploration.ipynb  # EDA
    ├── 02_model_training.ipynb    # Training experiments
    ├── 03_model_evaluation.ipynb  # Performance analysis
    └── 04_grad_cam_visualization.ipynb  # Visualization demos
```

---

## 🔬 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | High |
| Validation Accuracy | Competitive |
| Inference Time | < 1 second |

*Detailed metrics available in notebooks/03_model_evaluation.ipynb*

---

## 🎓 Usage Examples

### Web Interface
1. Visit the [live demo](https://yaseen-md-brain-tumor-detector-app-zhlhz5.streamlit.app/)
2. Upload an MRI scan or paste an image URL
3. Click "Predict"
4. View results with Grad-CAM visualization

### Programmatic Usage
```python
from src.inference import predict_image
from src.utils import load_model

model = load_model('models/resnet18_best.pth')
prediction = predict_image(model, 'path/to/mri.jpg')
print(f"Prediction: {prediction['class']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

---

## 🛣️ Roadmap

- [x] ✅ Deploy to Streamlit Cloud
- [ ] 🔄 Implement ensemble models for improved accuracy
- [ ] 📈 Add model performance dashboard
- [ ] 🎥 Support for video/DICOM file uploads
- [ ] 📄 Generate downloadable PDF reports
- [ ] 🌍 Multi-language support
- [ ] 🔐 Add user authentication and history tracking

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- **Dataset**: [Navoneel Chakrabarty](https://www.kaggle.com/navoneel) - Brain MRI Images
- **Frameworks**: [PyTorch](https://pytorch.org/) | [Streamlit](https://streamlit.io/)
- **Architecture**: ResNet18 from [torchvision.models](https://pytorch.org/vision/stable/models.html)
- **Grad-CAM**: Implementation based on [Grad-CAM paper](https://arxiv.org/abs/1610.02391)

---

## 📬 Contact & Support

**Yaseen MD**

- GitHub: [@Yaseen-md](https://github.com/Yaseen-md)
- Project Link: [Brain-Tumor-Detector](https://github.com/Yaseen-md/Brain-Tumor-Detector)

**Found this helpful?** Give it a ⭐️ to show your support!

---

<div align="center">

**⚠️ Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.

Made with ❤️ and 🧠 by Yaseen

</div>
