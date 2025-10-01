# syntax=docker/dockerfile:1

# Use official Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# System deps for OpenCV, Pillow, and Torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libjpeg62-turbo \
        libpng16-16 \
        libopencv-core-dev \
        libopencv-imgproc-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (CPU-only torch)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy app code
COPY . /app

# Expose port
EXPOSE 8000

# Default environment
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    DEBUG=False

# Healthcheck for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; import os; import sys;\
import time;\
import json;\
import urllib.request as r;\
u=f'http://127.0.0.1:{os.environ.get("API_PORT","8000")}/health';\
req=r.Request(u);\
req.add_header('User-Agent','healthcheck');\
resp=r.urlopen(req, timeout=5);\
import sys; sys.exit(0 if resp.getcode()==200 else 1)" || exit 1

# Run the app with uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


