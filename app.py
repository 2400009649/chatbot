import os
import requests
import streamlit as st

# Set the path to the saved model directory and Google Drive link
MODEL_DIR = "blenderbot_model"
MODEL_URL = "https://drive.google.com/uc?id=1JgPI7RjiQ57zin9HxMcLOdQ-C4o1swoZ"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to download the model file
def download_model(url, dest_path):
    st.info("Downloading model... This may take a while.")
    response = requests.get(url, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    st.success("Model downloaded successfully!")

# Main logic
st.title("Model Downloader")

if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found. Downloading now...")
    try:
        download_model(MODEL_URL, MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to download model: {e}")
else:
    st.success("Model file already exists.")

# Check file size
if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH)
    st.write(f"Model file size: {file_size} bytes")
    if file_size < 1_400_000_000:  # Expected size for the model is ~1.4GB
        st.error("Model file appears to be incomplete. Please re-download.")
    else:
        st.success("Model file size is valid.")
else:
    st.error("Model file is missing after download attempt.")
