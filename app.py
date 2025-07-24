import streamlit as st
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO
import os
import sys

# Custom modules
sys.path.append("src")
from model import get_resnet18
from grad_cam import GradCAM
from utils import load_model, predict_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once
@st.cache_resource
def load_trained_model():
    model = get_resnet18(num_classes=4)
    model = load_model(model, "models/resnet18_best.pth")
    model.to(device)
    model.eval()
    return model

# Preprocessing for general image input
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# App main logic
def main():
    st.set_page_config(page_title="Brain Tumor Detection", layout="wide", page_icon="🧠")
    st.title("🧠 Brain Tumor Detection Web App")
    st.markdown("Upload an MRI image or paste a URL to detect tumor type and view AI explanation.")

    with st.sidebar:
        st.markdown("### ℹ️ How to Use")
        st.markdown("- Upload a **clean MRI image** (JPG/PNG)")
        st.markdown("- Or paste a **direct image URL**")
        st.markdown("- Supports: **Glioma**, **Meningioma**, **Pituitary**, **No Tumor**")
        st.markdown("---")
        st.markdown("🔬 Powered by ResNet18 + Grad-CAM")
        st.markdown("📁 Dataset: [Kaggle MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)")

    # Upload or URL input
    uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("🌐 Or paste an image URL:")

    image = None
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except:
            st.error("⚠️ Unable to read uploaded image. Please upload a valid MRI scan.")
            st.stop()

    elif image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("⚠️ Couldn't load image from the URL. Ensure it's a direct link to a JPG/PNG file.")
            st.stop()

    if image:
        model = load_trained_model()
        classes = ['Glioma', 'Meningioma', 'Pituitary', 'Non-Tumor']
        tensor_image = preprocess_image(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="🧾 Uploaded MRI", use_container_width=True)

        with st.spinner("🔍 Analyzing MRI..."):
            probs, predicted_idx = predict_image(model, image)
            predicted_class = classes[predicted_idx]
            confidence = probs[predicted_idx]

        st.subheader("📊 Prediction Result")
        st.success(f"**Predicted Tumor Type:** `{predicted_class}`")
        st.info(f"**Confidence Level:** `{confidence:.2%}`")

        st.subheader("📈 Class Probabilities")
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
        st.bar_chart(prob_dict)

        with st.spinner("🧠 Generating Grad-CAM Heatmap..."):
            target_layer = model.layer4[-1].conv2
            grad_cam = GradCAM(model, target_layer)
            cam = grad_cam.generate_cam(tensor_image, predicted_idx)

            heatmap = cv2.resize(cam, (image.size[0], image.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

        with col2:
            st.image(superimposed_img, caption="📍 Grad-CAM Heatmap", use_container_width=True, clamp=True, channels="BGR")

if __name__ == "__main__":
    main()
