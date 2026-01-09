FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt ni copy qilamiz
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Butun projectni copy qilamiz
COPY . .

# FastAPI port
EXPOSE 8000

# main.py app ichida bo‘lgani uchun:
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
