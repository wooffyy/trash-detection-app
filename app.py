import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Trash Detection App", layout='centered')
st.title("Trash Detection App with YOLOv8")

@st.cache_resource
def load_model():
    model = os.path.join("model", "best.pt")
    return YOLO(model)

model = load_model()

st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.30, 0.01)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 1.0, 0.5, 0.05)

st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing..."):
        results = model.predict(
            source=np.array(image),
            conf=conf_threshold,
            iou=iou_threshold,
            show_labels=True,
            show_conf=True,
            save=False
        )
        
        rendered_img = results[0].plot()  # NumPy array
        st.image(rendered_img, caption="Hasil Deteksi", use_column_width=True)

        # Tampilkan daftar deteksi
        names = model.names
        boxes = results[0].boxes
        if boxes:
            st.subheader("📋 Deteksi Terdeteksi:")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                st.markdown(f"- **{names[cls]}** ({conf:.2f})")
        else:
            st.info("Tidak ada objek terdeteksi.")
