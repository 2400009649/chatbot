import os
import requests
import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Set the path to the saved model directory and Google Drive link
MODEL_DIR = "blenderbot_model"
MODEL_URL = "https://drive.google.com/uc?id=1JgPI7RjiQ57zin9HxMcLOdQ-C4o1swoZ"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download the model if it does not exist
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... This might take a while."):
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded successfully!")

# Load the tokenizer and model
try:
    tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_DIR)
    model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_DIR, use_safetensors=True)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    st.error(f"Failed to load model: {e}")
    model = None

# Function to generate a response
def get_response(input_text):
    # Encode the input text
    inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Generate a response
    reply_ids = model.generate(**inputs)
    reply_text = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    return reply_text

# Initialize session state for chat history if not exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Local paths to avatars
user_avatar = "img/person.png"  # Local path to the user avatar
bot_avatar = "img/bot.png"  # Local path to the bot avatar

# Streamlit UI setup
st.title("Chatbot with BlenderBot")

# Using a form to handle inputs and button
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", value="", key="input")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    # Add user input with avatar to chat history
    st.session_state.chat_history.append(("user", user_input))
    
    # Display loading message while processing
    with st.spinner("Bot is thinking..."):
        response = get_response(user_input)
    
    # Add the bot response with avatar to chat history
    st.session_state.chat_history.append(("bot", response))

# Display the chat history with avatars
for sender, message in st.session_state.chat_history:
    col1, col2 = st.columns([1, 9])
    if sender == "user":
        col1.image(user_avatar, width=30)  # Adjust size as needed
    else:
        col1.image(bot_avatar, width=30)  # Adjust size as needed
    col2.markdown(f"**{message}**")  # Use markdown for better text formatting
