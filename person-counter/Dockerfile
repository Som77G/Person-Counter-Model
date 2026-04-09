FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (important for OpenCV + RTSP)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

CMD ["python", "main.py"]