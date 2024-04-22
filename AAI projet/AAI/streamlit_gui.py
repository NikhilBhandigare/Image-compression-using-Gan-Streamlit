import streamlit as st
from PIL import Image
import io
import cv2
import base64 
import numpy as np

# Function to compress and decompress images
def compress_decompress_image(image):
    # Implement image compression and decompression logic here
    return image

# Function to compress and decompress videos
def compress_decompress_video(video_path):
    # Implement video compression and decompression logic here
    return video_path

# Streamlit GUI
def main():
    st.title("Image and Video Compression using GAN")
    st.sidebar.title("Options")
    
    option = st.sidebar.radio("Select Option", ("Image", "Video"))

    if option == "Image":
        uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.sidebar.button("Compress"):
                compressed_image = compress_decompress_image(image)
                st.image(compressed_image, caption="Compressed Image", use_column_width=True)
                # Add download button for compressed image
                compressed_img_data = io.BytesIO()
                compressed_image.save(compressed_img_data, format='PNG')
                st.sidebar.download_button(
                    label="Download Compressed Image",
                    data=compressed_img_data.getvalue(),
                    file_name='compressed_image.png'
                )

            if st.sidebar.button("Decompress"):
                decompressed_image = compress_decompress_image(image)  # Implement decompression logic
                st.image(decompressed_image, caption="Decompressed Image", use_column_width=True)
                # Add download button for decompressed image
                decompressed_img_data = io.BytesIO()
                decompressed_image.save(decompressed_img_data, format='PNG')
                st.sidebar.download_button(
                    label="Download Decompressed Image",
                    data=decompressed_img_data.getvalue(),
                    file_name='decompressed_image.png'
                )

    elif option == "Video":
        uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4"])
        if uploaded_video is not None:
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

            if st.sidebar.button("Compress"):
                compressed_video_path = compress_decompress_video(video_path)
                st.video(compressed_video_path)
                # Add download button for compressed video
                st.sidebar.markdown(get_video_download_link(compressed_video_path), unsafe_allow_html=True)

            if st.sidebar.button("Decompress"):
                decompressed_video_path = compress_decompress_video(video_path)  # Implement decompression logic
                st.video(decompressed_video_path)
                # Add download button for decompressed video
                st.sidebar.markdown(get_video_download_link(decompressed_video_path), unsafe_allow_html=True)

# Function to generate download link for video
def get_video_download_link(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_encoded = base64.b64encode(video_data).decode()
    href = f'<a href="data:file/mp4;base64,{video_encoded}" download="output_video.mp4">Download Decompressed Video</a>'
    return href

if __name__ == "__main__":
    main()
