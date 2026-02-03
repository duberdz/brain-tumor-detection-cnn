import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from gradcam_explainer import generate_gradcam

# seteamos la pagina
st.set_page_config(page_title="Detección de tumores cerebrales", layout="wide")

MODEL_PATH = "modelo_tumores_vgg16_V2.0.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
LAST_CONV_LAYER = "block5_conv3"

@st.cache_resource
def load_cnn_model():
    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# interfaz
st.title("Clasificación automática de tumores cerebrales")
st.write("Sube una imagen de resonancia magnética para obtener la predicción del modelo y el mapa Grad-CAM.")

uploaded_file = st.file_uploader("Selecciona una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Imagen cargada")
    st.image(image, use_container_width=False, width=300)

    if st.button("Analizar imagen"):
        with st.spinner("Ejecutando modelo..."):
            #pred_label, gradcam_img = generate_gradcam(image, model)
            pred_label, gradcam_img = generate_gradcam(
                image_pil=image,
                model=model,
                class_names=CLASS_NAMES,
                img_size=IMG_SIZE,
                last_conv_layer_name=LAST_CONV_LAYER
)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Imagen original**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown(f"**Mapa Grad-CAM ({pred_label})**")
            st.image(gradcam_img, use_container_width=True)

        st.success(f"Predicción del modelo: **{pred_label}**")
else:
    st.info("Esperando una imagen...")
