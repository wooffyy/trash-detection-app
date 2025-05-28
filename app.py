import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np

model = YOLO("model/best.pt")  
names = model.model.names


st.sidebar.title("Trash Detection Model")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

st.title("♻️ Trash Detection App with YOLOv8")
st.markdown("Upload your trash image and let the model do the rest.")

image = st.file_uploader("Upload gambar", type=['jpg', 'jpeg', 'png'])

if image:
    image = Image.open(image).convert('RGB')
    st.image(image, caption='🖼️ Uploaded image', use_column_width=True)

    # Optional: resize 2x
    if st.checkbox("Zoom image 2x"):
        w, h = image.size
        image = image.resize((w*2, h*2))

    # Predict
    results = model.predict(source=np.array(image), conf=confidence_threshold, imgsz=1024)
    boxes = results[0].boxes

    # Draw & show
    if boxes:
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detect...", use_column_width=True)

        st.subheader("📋 Detection Result:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{names[cls]} ({conf:.2f})"
            if conf < 0.5:
                st.markdown(f"- ❗ *Low confidence* — **{label}**")
            else:
                st.markdown(f"- ✅ **{label}**")
    else:
        st.warning("🤖 No trash detected. Try another image or use lower threshold")