import os
import requests

MODEL_DIR = "blenderbot_model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URL = "https://drive.google.com/uc?id=1A2B3C4D5E6F7G8H"  # Thay File ID ở đây
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")

# Tải file nếu chưa có
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# Load mô hình sau khi tải
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_DIR)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_DIR)
