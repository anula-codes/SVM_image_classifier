import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load model
with open("model_svm.pkl", "rb") as f:
    model = pickle.load(f)

st.title("AI vs Real Image Classifier")

st.write("Upload an image to check if it's AI-generated or real.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((64, 64))
    image = np.array(image)

    # Flatten image → convert to 1D
    image = image.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(image)

    # Output
    if prediction[0] == 1:
        st.success("This image is AI-generated!")
    else:
        st.success("This image is Real!")
