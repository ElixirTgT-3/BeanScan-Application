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

# Expose default port (Render sets PORT dynamically)
EXPOSE 8000

# Default environment
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    DEBUG=False

# Healthcheck uses PORT if provided (fallback 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python - <<'PY' || exit 1
import os, urllib.request, sys
port = os.environ.get('PORT') or os.environ.get('API_PORT', '8000')
url = f'http://127.0.0.1:{port}/health'
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        sys.exit(0 if r.getcode() == 200 else 1)
except Exception:
    sys.exit(1)
PY

# Run the app with uvicorn, binding to Render PORT when present
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]


