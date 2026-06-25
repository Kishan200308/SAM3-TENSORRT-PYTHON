# Base Image: CUDA 12.6 Runtime on Ubuntu 24.04
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# Environment Variables
ENV GRADIO_PORT=7860 \
    RESOLUTION=1008 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive

# Install System Dependencies & TensorRT
RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
    git wget sudo libgl1 libglib2.0-0 \
    python3 python3-pip \
    libnvinfer10=10.14.1.48-1+cuda12.9 \
    libnvinfer-bin=10.14.1.48-1+cuda12.9 \
    libnvinfer-dispatch10=10.14.1.48-1+cuda12.9 \
    libnvinfer-lean10=10.14.1.48-1+cuda12.9 \
    libnvinfer-plugin10=10.14.1.48-1+cuda12.9 \
    libnvinfer-vc-plugin10=10.14.1.48-1+cuda12.9 \
    libnvonnxparsers10=10.14.1.48-1+cuda12.9 \
    && rm -rf /var/lib/apt/lists/*

# Add TensorRT binary utilities to PATH and libraries to LD_LIBRARY_PATH
ENV PATH="/usr/src/tensorrt/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/tensorrt:$LD_LIBRARY_PATH"

# Upgrade pip, Pre-install Python Requirements, Download Patch, Apply Patch, and Cleanup
RUN wget https://raw.githubusercontent.com/Kishan200308/SAM3-TENSORRT-PYTHON/main/requirements.txt \
    && pip3 install --no-cache-dir -r requirements.txt \
    && wget https://raw.githubusercontent.com/Kishan200308/SAM3-TENSORRT-PYTHON/refs/heads/main/Patch_Sam3_Interp_Rope.py \
    && python3 Patch_Sam3_Interp_Rope.py \
    && rm Patch_Sam3_Interp_Rope.py requirements.txt

# Setup Working Directory
WORKDIR /home/ubuntu

# Copy startup script
COPY startup.sh /home/ubuntu/startup.sh
RUN chmod +x /home/ubuntu/startup.sh

# Expose Default Gradio Port
EXPOSE 7860

# Execute Startup Script
ENTRYPOINT ["/bin/bash", "/home/ubuntu/startup.sh"]