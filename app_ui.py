import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.title("Face Mask Detection System")
st.write("Choose Image Upload or Live Camera Detection")

# ------------------ LOAD MODEL SAFELY ------------------
model = None
model_path = "face_mask_detector.h5"

if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("Model Loaded Successfully")
else:
    st.error("Model file not found! Check path.")
    st.stop()

# ------------------ SIDEBAR ------------------
option = st.sidebar.selectbox(
    "Select Mode",
    ["Image Upload", "Live Camera"]
)

# ------------------ IMAGE UPLOAD ------------------
if option == "Image Upload":
    st.subheader("📂 Upload an Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = cv2.resize(image, (224, 224))
        img = img / 255.0
        img = np.reshape(img, (1, 224, 224, 3))

        prediction = model.predict(img)

        if prediction[0][0] > 0.5:
            st.error("No Mask Detected")
        else:
            st.success("Mask Detected")

# ------------------ LIVE CAMERA ------------------
elif option == "Live Camera":
    st.subheader("🎥 Live Camera Detection")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not working")
            break

        # Preprocess
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.reshape(img, (1, 224, 224, 3))

        prediction = model.predict(img)

        label = "Mask" if prediction[0][0] < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()