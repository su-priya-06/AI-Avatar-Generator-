import streamlit as st
import cv2
import numpy as np
from filters import list_styles
from PIL import Image
import io
import random

st.set_page_config(page_title="AI Avatar Generator", layout="centered")
st.title("🎨 AI Avatar Generator")
st.markdown("Upload a photo and apply fun styles!")

styles = list_styles()

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
style_choice = st.selectbox("🎭 Choose a Style", ["Random"] + list(styles.keys()))

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV

    if style_choice == "Random":
        style_name, style_func = random.choice(list(styles.items()))
    else:
        style_name = style_choice
        style_func = styles[style_choice]

    st.subheader(f"✨ Applying: {style_name}")
    output = style_func(img_np)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(output[:, :, ::-1], caption=style_name, use_column_width=True)

    img_out = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button("📥 Download Stylized Image", data=byte_im,
                       file_name=f"{style_name.lower().replace(' ', '_')}_avatar.png",
                       mime="image/png")
