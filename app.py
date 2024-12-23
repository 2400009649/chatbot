import os
import gdown
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Model download and initialization
MODEL_DIR = "blenderbot_model"
MODEL_URL = "https://drive.google.com/uc?id=1JgPI7RjiQ57zin9HxMcLOdQ-C4o1swoZ"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_DIR)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_DIR, use_safetensors=True)

# FastAPI setup
app = FastAPI()

# Request body schema
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        inputs = tokenizer([request.message], return_tensors="pt", padding=True, truncation=True, max_length=512)
        reply_ids = model.generate(**inputs)
        reply_text = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return {"response": reply_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
