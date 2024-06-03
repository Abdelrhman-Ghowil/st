import os
import streamlit as st
from PIL import Image
from transformers import pipeline

def remove_background(image, log_func=None):
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    output_img = pipe(image)
    return output_img

st.title("Background Remover App")
st.write("Upload an image to remove its background")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_img = Image.open(uploaded_file).convert("RGB")
    st.image(input_img, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Removing background...'):
        output_img = remove_background(input_img)
    
    st.image(output_img, caption='Background Removed Image', use_column_width=True)
    
    # Provide an option to download the processed image
    output_filename = os.path.splitext(uploaded_file.name)[0] + '_no_bg.png'
    output_img.save(output_filename, "PNG")
    
    with open(output_filename, "rb") as file:
        btn = st.download_button(
            label="Download image",
            data=file,
            file_name=output_filename,
            mime="image/png"
        )
