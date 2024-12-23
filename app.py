import os
import gdown
import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Set the paths
MODEL_DIR = "blenderbot_model"
MODEL_URL = "https://drive.google.com/uc?id=1JgPI7RjiQ57zin9HxMcLOdQ-C4o1swoZ"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
IMG_DIR = "img"
USER_AVATAR = os.path.join(IMG_DIR, "person.png")
BOT_AVATAR = os.path.join(IMG_DIR, "bot.png")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Function to download the model
def download_model_with_gdown(url, dest_path):
    st.info("Downloading model... This may take a while.")
    gdown.download(url, dest_path, quiet=False)
    st.success("Model downloaded successfully!")

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found. Downloading now...")
    try:
        download_model_with_gdown(MODEL_URL, MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        raise
else:
    file_size = os.path.getsize(MODEL_PATH)
    if file_size < 1_400_000_000:  # Expected size ~1.4GB
        st.error("Model file appears to be incomplete. Please re-download.")
        os.remove(MODEL_PATH)
        download_model_with_gdown(MODEL_URL, MODEL_PATH)
    else:
        if "validation_message_displayed" not in st.session_state:
            st.success("Model file size is valid.")
            st.session_state.validation_message_displayed = True

# Check if avatar images exist
if not os.path.exists(USER_AVATAR):
    st.error(f"User avatar image not found at {USER_AVATAR}. Please upload it.")
if not os.path.exists(BOT_AVATAR):
    st.error(f"Bot avatar image not found at {BOT_AVATAR}. Please upload it.")

# Load the tokenizer and model
try:
    tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_DIR)
    model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_DIR, use_safetensors=True)
    st.success("Model and tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model or tokenizer: {e}")
    model = None

# Function to generate a response
def get_response(input_text):
    if model is None:
        return "Sorry, the chatbot is not available right now."
    try:
        inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        reply_ids = model.generate(**inputs)
        reply_text = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return reply_text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "An error occurred while generating a response."

# Initialize session state for chat history if not exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        if os.path.exists(USER_AVATAR):
            col1.image(USER_AVATAR, width=30)  # Adjust size as needed
        else:
            col1.write("👤")  # Fallback if image is missing
    else:
        if os.path.exists(BOT_AVATAR):
            col1.image(BOT_AVATAR, width=30)  # Adjust size as needed
        else:
            col1.write("🤖")  # Fallback if image is missing
    col2.markdown(f"**{message}**")  # Use markdown for better text formatting
