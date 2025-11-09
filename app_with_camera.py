
# app.py (con cámara)
# Streamlit MVP para clasificar gestos de lenguaje de señas (A-Z) con el modelo entrenado.
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import json, pathlib, io

st.set_page_config(page_title="ASL Classifier (MVP)", page_icon="🤟")

st.title("🤟 ASL Classifier — MVP con Cámara")
st.write("Sube una imagen **o** usa la **cámara** para clasificar una letra A–Z.")

artifacts = pathlib.Path("artifacts")
model_path = artifacts / "model.h5"
labels_path = artifacts / "labels.json"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    with open(labels_path) as f:
        labels = json.load(f)
    # labels: {index: class_name}  ó  {"0":"A","1":"B",...}
    # Asegurar mapeo index->label ordenado
    try:
        max_idx = max(int(k) for k in labels.keys())
        index_to_label = [labels[str(i)] for i in range(max_idx+1)]
    except Exception:
        # Si vino invertido {label: index}
        inv = {v:k for k,v in labels.items()}
        max_idx = max(int(v) for v in inv.values())
        index_to_label = [None]*(max_idx+1)
        for k,v in inv.items():
            index_to_label[int(v)] = k
    return model, index_to_label

if not model_path.exists() or not labels_path.exists():
    st.error("No se encontró el modelo. Ejecuta primero `python train_mvp.py` para generar artifacts/model.h5 y labels.json.")
    st.stop()

model, index_to_label = load_model()

def preprocess(img: Image.Image) -> np.ndarray:
    """Convierte a 64x64 escala de grises normalizado y da forma (1,64,64,1)."""
    img = img.convert("L")
    img = ImageOps.fit(img, (64,64), method=Image.Resampling.LANCZOS)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, -1))
    return x, img

def predict_image(img: Image.Image):
    x, view = preprocess(img)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds))
    label = index_to_label[idx] if idx < len(index_to_label) else str(idx)
    return label, prob, view

tab1, tab2 = st.tabs(["📤 Subir imagen", "📷 Cámara"])

with tab1:
    uploaded = st.file_uploader("Cargar imagen", type=["png","jpg","jpeg"])
    if uploaded:
        try:
            img = Image.open(uploaded)
            label, prob, view = predict_image(img)
            st.image(view, caption="Entrada (64x64, escala de grises)")
            st.subheader(f"Predicción: **{label}**")
            st.write(f"Confianza: {prob:.2%}")
            st.progress(prob)
        except Exception as e:
            st.error(f"No se pudo procesar la imagen: {e}")

with tab2:
    st.caption("Usa el botón **Take Photo** para capturar un cuadro desde tu cámara.")
    capture = st.camera_input("Tomar foto")
    if capture is not None:
        try:
            img = Image.open(capture)
            label, prob, view = predict_image(img)
            st.image(view, caption="Entrada (64x64, escala de grises)")
            st.subheader(f"Predicción: **{label}**")
            st.write(f"Confianza: {prob:.2%}")
            st.progress(prob)
        except Exception as e:
            st.error(f"No se pudo procesar la captura: {e}")
