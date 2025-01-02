import streamlit as st
import speech_recognition as sr
import paho.mqtt.client as mqtt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Tạo bộ dữ liệu cho mô hình phân loại
data = [
    ("Bật đèn", "Bật", "Đèn"),
    ("Tắt đèn", "Tắt", "Đèn"),
    ("Bật tivi", "Bật", "Tivi"),
    ("Tắt tivi", "Tắt", "Tivi"),
    ("Bật quạt", "Bật", "Quạt"),
    ("Tắt quạt", "Tắt", "Quạt"),
    ("Bật tủ lạnh", "Bật", "Tủ lạnh"),
    ("Tắt tủ lạnh", "Tắt", "Tủ lạnh"),
]

# Tách dữ liệu thành câu lệnh, hành động và thiết bị
X = [item[0] for item in data]  # Các câu lệnh
actions = [item[1] for item in data]  # Các hành động: Bật, Tắt
devices = [item[2] for item in data]  # Các thiết bị: Đèn, Tivi, Quạt, ...

# Tạo mô hình học máy cho hành động và thiết bị
model_action = make_pipeline(TfidfVectorizer(), MultinomialNB())  # Mô hình cho hành động
model_device = make_pipeline(TfidfVectorizer(), MultinomialNB())  # Mô hình cho thiết bị

# Huấn luyện mô hình
model_action.fit(X, actions)
model_device.fit(X, devices)

# MQTT client configuration
broker = "broker.mqtt-dashboard.com"  # Broker MQTT công cộng
port = 1883
topic = "home/lighting"

client = mqtt.Client()

# Kết nối tới MQTT broker
client.connect(broker, port, 60)

# Hàm nhận diện giọng nói từ microphone
def recognize_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Đang lắng nghe, xin hãy nói lệnh...")
        recognizer.adjust_for_ambient_noise(source)  # Điều chỉnh độ ồn môi trường
        audio = recognizer.listen(source)

        try:
            # Nhận diện giọng nói và chuyển thành văn bản
            command = recognizer.recognize_google(audio, language='vi-VN')
            return command
        except sr.UnknownValueError:
            return "Xin lỗi, tôi không thể hiểu lệnh."
        except sr.RequestError:
            return "Xin lỗi, tôi không thể kết nối với dịch vụ nhận diện giọng nói."

# Giao diện Streamlit
st.title("Ứng dụng Nhận diện Lệnh Giọng nói")

if st.button("Nhận diện lệnh"):
    command = recognize_voice()
    st.write(f"Lệnh nhận diện: {command}")
    
    if command:
        # Dự đoán hành động và thiết bị từ câu lệnh
        predicted_action = model_action.predict([command])[0]
        predicted_device = model_device.predict([command])[0]
        
        st.write(f"Hành động: {predicted_action}")
        st.write(f"Thiết bị: {predicted_device}")
        
        # Gửi thông điệp qua MQTT
        message = f"{predicted_action} {predicted_device}"
        client.publish(topic, message)
        st.write(f"Lệnh đã gửi qua MQTT: {message}")
