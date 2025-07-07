FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install -r requirements.txt

# Install WhisperX (with dependencies)
RUN python3 -m pip install git+https://github.com/m-bain/whisperx.git@v3.1.1
RUN python3 -m pip install ctranslate2==4.4.0
# Pre-download model (comment out for faster builds during development)
# RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16', download_root='/app/models')"

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
