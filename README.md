# 🧠 Brain Tumor Detection with MRI Scans

A deep learning-powered web app built with **PyTorch** and **Streamlit** to detect brain tumor types from MRI scans. The model predicts one of four classes and provides visual explanations using **Grad-CAM** heatmaps.

---

## 🚀 Features

- 🔍 Detects brain tumor types: **Glioma**, **Meningioma**, **Pituitary**, **No Tumor**
- 📤 Upload an MRI image or paste an image URL (supports random images from the internet)
- 🧠 Visual explanations with **Grad-CAM** overlays
- 📊 Class-wise prediction confidence and probabilities
- 🖥️ Modern, user-friendly interface with **Streamlit**

---

## 🧠 Model Details

- **Architecture**: `ResNet18`
- **Framework**: `PyTorch`
- **Visualization**: Grad-CAM
- **Dataset**: [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Classes**:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor

---

## 🛠️ Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Yaseen-md/brain-tumor-detector.git
cd brain-tumor-detector


2️⃣ (Optional) Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app.py
```

🔒 Note on Dataset
To keep the repo lightweight, the dataset/ and models/ directories are excluded via .gitignore. Please download the dataset from the Kaggle link above if needed.

✅ TODOs
 Add Streamlit Cloud or Hugging Face Spaces deployment

 Improve model accuracy with larger data

 Add webcam/image capture support

 Add downloadable prediction reports


 🙌 Acknowledgments
Navoneel Chakrabarty — Brain MRI Dataset

PyTorch and Streamlit — Powerful open-source frameworks



📬 Contact
Made with ❤️ by Yaseen
📫 Feel free to open issues or suggestions.
