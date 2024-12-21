import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Kiểm tra khả năng khởi chạy ứng dụng
st.title('Chatbot with BlenderBot')

# Load mô hình và tokenizer
try:
    MODEL_DIR = "blenderbot_model"
    tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_DIR)
    model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_DIR)
    st.success("Model and tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model or tokenizer: {e}")

# Kiểm tra giao diện cơ bản
user_input = st.text_input("Enter your message:")
if st.button("Send"):
    if user_input:
        try:
            # Tạo câu trả lời từ bot
            inputs = tokenizer([user_input], return_tensors='pt', padding=True, truncation=True, max_length=512)
            reply_ids = model.generate(**inputs)
            response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            st.write(f"BlenderBot: {response}")
        except Exception as e:
            st.error(f"Failed to generate response: {e}")
