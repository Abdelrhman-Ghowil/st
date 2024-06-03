import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np

# Initialize the segmentation pipeline once
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

# Function to perform image segmentation
def segment_image(image):
    try:
        # Perform image segmentation
        results = pipe(image)
        # Get the segmented image and mask
        segmented_image = results[0]['segmentation']
        mask = results[0]['mask']
        # Convert results back to PIL images for display
        segmented_image_pil = Image.fromarray(segmented_image.astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        return segmented_image_pil, mask_pil
    except Exception as e:
        st.error(f"An error occurred during segmentation: {e}")
        return None, None

# Streamlit UI
st.title("Image Segmentation with Transformers")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("Segment Image"):
        segmented_image, mask = segment_image(image)
        
        if segmented_image and mask:
            st.image(segmented_image, caption="Segmented Image", use_column_width=True)
            st.image(mask, caption="Mask", use_column_width=True)
