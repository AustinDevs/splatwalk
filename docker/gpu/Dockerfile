# Multi-stage build for SplatWalk GPU processing
# Stage 1: Builder - Use PyTorch base image (includes CUDA + PyTorch pre-installed)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /opt

# Clone repositories (shallow clone to save space)
RUN git clone --depth 1 https://github.com/Drexubery/ViewCrafter.git && \
    git clone --depth 1 https://github.com/NVlabs/InstantSplat.git && \
    git clone --depth 1 https://github.com/naver/dust3r.git && \
    git clone --depth 1 https://github.com/naver/mast3r.git

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy scipy pillow tqdm einops timm transformers \
    boto3 awscli plyfile

# Install project dependencies (ignore errors for optional deps)
WORKDIR /opt/ViewCrafter
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

WORKDIR /opt/dust3r
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

WORKDIR /opt/mast3r
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

WORKDIR /opt/InstantSplat
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# Install 3DGS converter
RUN pip install --no-cache-dir git+https://github.com/francescofugazzi/3dgsconverter.git 2>/dev/null || \
    pip install --no-cache-dir 3dgsconverter 2>/dev/null || true

# Copy pipeline scripts
COPY run_pipeline.sh /opt/run_pipeline.sh
COPY convert_to_ksplat.py /opt/convert_to_ksplat.py
RUN chmod +x /opt/run_pipeline.sh

WORKDIR /data

ENTRYPOINT ["/opt/run_pipeline.sh"]
