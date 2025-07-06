#!/bin/bash

# CUDAとcuDNNの診断スクリプト

echo "=== CUDA and cuDNN Diagnostic ==="
echo

# Check NVIDIA driver
echo "1. NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "NVIDIA driver not found"
echo

# Check CUDA version
echo "2. CUDA Version:"
nvcc --version 2>/dev/null || echo "CUDA toolkit not found"
echo

# Check PyTorch CUDA
echo "3. PyTorch CUDA Status:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
" 2>/dev/null || echo "PyTorch not installed or error"
echo

# Check libraries
echo "4. CUDA Libraries:"
ldconfig -p | grep cuda || echo "No CUDA libraries found in ldconfig"
echo

echo "5. cuDNN Libraries:"
ldconfig -p | grep cudnn || echo "No cuDNN libraries found in ldconfig"
echo

# Check environment variables
echo "6. Environment Variables:"
echo "CUDA_HOME=$CUDA_HOME"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
