# Sử dụng Python tối giản
FROM python:3.10-slim

# Tạo thư mục làm việc trong container
WORKDIR /app

# Copy và cài đặt thư viện
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . /app

# Mở cổng mặc định
EXPOSE 8080

# Chạy ứng dụng FastAPI bằng Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
