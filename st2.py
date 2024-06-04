import streamlit as st
from PIL import Image
from transformers import pipeline
import zipfile
import io
from io import BytesIO
import os


def remove_background(image):
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    output_img = pipe(image)
    return output_img

def resize_image(image, size=(1024, 1024)):
    return image.resize(size, Image.Resampling.LANCZOS)

def download_all_images_as_zip(images_info):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for file_name, image in images_info:
            with io.BytesIO() as img_buffer:
                image.save(img_buffer, "PNG")
                zf.writestr(file_name, img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

st.title("Background Remover App")
st.write("Upload images to remove their backgrounds")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} images")
    
    output_images = []
    for uploaded_file in uploaded_files:
        input_img = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(resize_image(input_img), caption='Uploaded Image', use_column_width=True)
        
        output_img = remove_background(input_img)
        output_img = resize_image(output_img)
        output_images.append((os.path.splitext(uploaded_file.name)[0] + '.png', output_img))
        
        with col2:
            st.image(output_img, caption='Background Removed Image', use_column_width=True)
            output_filename = os.path.splitext(uploaded_file.name)[0] + '.png'
            img_buffer = BytesIO()
            output_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            st.download_button(
                label=f"Download {os.path.splitext(uploaded_file.name)[0]}.png",
                data=img_buffer,
                file_name=output_filename,
                mime="image/png"
            )
    
    # Provide an option to download all processed images as a ZIP file
    zip_buffer = download_all_images_as_zip(output_images)
    
    st.download_button(
        label="Download All Images as ZIP",
        data=zip_buffer,
        file_name="processed_images.zip",
        mime="application/zip"
    )
