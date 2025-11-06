# CPU build (simple). For CUDA, see comments below.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose API
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ===== CUDA (optional) =====
# Use an NVIDIA CUDA base image and install torch with matching cu* version, e.g.:
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
# RUN pip install --no-cache-dir torch==2.3.* --index-url https://download.pytorch.org/whl/cu121
# (Then install the rest of requirements)