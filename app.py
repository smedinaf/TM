import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image
from keras.models import load_model
import platform

# CONFIG
st.set_page_config(
    page_title="💖 Girlie AI Recognition",
    page_icon="🌸",
    layout="wide"
)

# 🎀 ESTILOS GIRLIE ROSADOS
st.markdown("""
<style>

/* FONDO */
.main {
    background: linear-gradient(135deg, #ffe4ec, #fff1f5);
    color: #5c3a3a;
}

/* TITULOS */
h1, h2, h3 {
    color: #d63384;
    font-weight: 700;
}

/* TEXTO */
p, span, label {
    color: #6b4c4c;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #fff0f6;
}

section[data-testid="stSidebar"] * {
    color: #d63384 !important;
}

/* BOTONES */
.stButton>button {
    background: linear-gradient(90deg, #ff8fab, #ffc2d1);
    color: white;
    border-radius: 15px;
    border: none;
    padding: 10px 20px;
    transition: 0.3s;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #ff6f91, #ffb3c6);
    transform: scale(1.05);
}

/* METRICS */
[data-testid="stMetric"] {
    background: #ffe4ec;
    padding: 15px;
    border-radius: 15px;
}

/* ALERTAS */
.stAlert {
    border-radius: 15px;
}

/* IMAGEN */
img {
    border-radius: 20px;
}

/* TARJETAS */
.card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(255, 182, 193, 0.4);
}

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("## 💖 Girlie AI Recognition 🌸")
st.caption("Reconoce objetos con vibes cute ✨🧸")

# INFO SISTEMA
st.markdown(f"🌷 **Python version:** {platform.python_version()}")

# MODELO
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# SIDEBAR
with st.sidebar:
    st.markdown("### 🎀 About this app")
    st.write("Usa tu modelo de Teachable Machine para identificar objetos 💕")
    st.write("Solo toma una foto y deja que la magia pase ✨📸")

# IMAGEN REFERENCIA
st.markdown("### 🧸 Ejemplo")
image = Image.open('gatito.jpg')
st.image(image, width=300)

# INPUT
st.markdown("### 📸 Toma una fotito")
img_file_buffer = st.camera_input("Haz click aquí 💖")

# PROCESAMIENTO
if img_file_buffer is not None:

    st.markdown("### ✨ Analizando tu imagen...")

    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))

    img_array = np.array(img)

    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    st.markdown("### 💕 Resultado")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Tu fotito 🌸", use_container_width=True)

    with col2:
        # RESULTADOS BONITOS
        if prediction[0][0] > 0.5:
            st.success(f"🧸 Mini Bambi detected 💖\n\nProbabilidad: {prediction[0][0]:.2f}")

        if prediction[0][1] > 0.5:
            st.success(f"👜 Bolso detected ✨\n\nProbabilidad: {prediction[0][1]:.2f}")

        if prediction[0][0] <= 0.5 and prediction[0][1] <= 0.5:
            st.warning("😢 No estoy segura... intenta otra foto 💕")

# FOOTER
st.markdown("---")
st.caption("🌸 Made with love + AI 💖✨")


