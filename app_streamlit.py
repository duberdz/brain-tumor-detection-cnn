import streamlit as st
import io
import numpy as np
import time
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

uploaded_file = st.file_uploader(
    "Selecciona una imagen (JPG, PNG)", 
    type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    allowed_mime = {"image/jpg", "image/jpeg", "image/png"}
    if uploaded_file.type not in allowed_mime:
        st.error("Formato no válido. Solo se permiten imágenes JPG/JPEG o PNG.")
        st.stop()

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
    except Exception:
        st.error("El archivo no es una imagen válida o está dañado.")
        st.stop()
    
    st.subheader("Imagen cargada")
    st.image(image, use_container_width=False, width=300)

    if st.button("Analizar imagen"):
        with st.spinner("Ejecutando modelo..."):
            start_time = time.perf_counter()
            #pred_label, gradcam_img = generate_gradcam(image, model)
            pred_label, confidence, gradcam_img = generate_gradcam(
                image_pil=image,
                model=model,
                class_names=CLASS_NAMES,
                img_size=IMG_SIZE,
                last_conv_layer_name=LAST_CONV_LAYER
            )
            end_time = time.perf_counter()
            inference_time = end_time - start_time

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Imagen original**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown(f"**Mapa Grad-CAM ({pred_label})**")
            st.image(gradcam_img, use_container_width=True)

        st.success(
            f"Predicción del modelo: **{pred_label}** "
            f"(confianza: **{confidence*100:.2f}%**)"
        )
        st.info(f"Tiempo de inferencia: {inference_time:.4f} segundos")

        st.subheader("Exportar resultados")
        gradcam_pil = Image.fromarray(gradcam_img)
        buf = io.BytesIO()
        gradcam_pil.save(buf, format="PNG")

        st.download_button(
            label="Descargar imagen Grad-CAM (PNG)",
            data=buf.getvalue(),
            file_name=f"gradcam_{pred_label}.png",
            mime="image/png"
        )

        report_txt = (
            "Resultado de clasificación\n"
            f"- Predicción: {pred_label}\n"
            f"- Confianza: {confidence*100:.2f}%\n"
        )

        st.download_button(
            label="Descargar informe (TXT)",
            data=report_txt.encode("utf-8"),
            file_name="resultado_clasificacion.txt",
            mime="text/plain"
        )
else:
    st.info("Esperando una imagen...")
