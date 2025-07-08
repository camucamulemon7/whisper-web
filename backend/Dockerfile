# Stage 1: Base image with system dependencies (rarely changes)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Install system dependencies (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Stage 2: Python dependencies (changes when requirements change)
FROM base AS python-deps

WORKDIR /tmp

# Copy only requirements file first (better cache utilization)
COPY requirements.txt .

# Upgrade pip once
RUN python3 -m pip install --upgrade pip wheel setuptools

# Install PyTorch separately (large, rarely changes)
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Optional: Install additional backends (uncomment if needed)
# For OpenAI Whisper:
# RUN python3 -m pip install --no-cache-dir openai-whisper==20250625
#
# For WhisperX:
# RUN python3 -m pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git@v3.1.1
RUN python3 -m pip install --no-cache-dir ctranslate2==4.4.0

# Stage 3: Final application image
FROM base AS final

# Copy Python packages from deps stage
COPY --from=python-deps /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Create models directory
RUN mkdir -p /app/models

# Copy application code (changes frequently)
COPY . .

# Pre-download model (uncomment for production, comment for development)
# RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16', download_root='/app/models')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
