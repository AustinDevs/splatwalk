# Multi-stage build for SplatWalk GPU processing
# Stage 1: Builder - Install dependencies and download models
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone repositories
RUN git clone https://github.com/Drexubery/ViewCrafter.git && \
    git clone https://github.com/NVlabs/InstantSplat.git && \
    git clone https://github.com/naver/dust3r.git && \
    git clone https://github.com/naver/mast3r.git

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install base dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install numpy scipy pillow tqdm einops timm transformers && \
    pip install boto3 awscli

# Install ViewCrafter dependencies
WORKDIR /opt/ViewCrafter
RUN pip install -r requirements.txt || true

# Install DUSt3R/MASt3R dependencies
WORKDIR /opt/dust3r
RUN pip install -r requirements.txt || true

WORKDIR /opt/mast3r
RUN pip install -r requirements.txt || true

# Install InstantSplat dependencies
WORKDIR /opt/InstantSplat
RUN pip install -r requirements.txt || true

# Install 3DGS converter
RUN pip install plyfile && \
    pip install git+https://github.com/francescofugazzi/3dgsconverter.git || \
    pip install 3dgsconverter || true

# Download model weights
RUN mkdir -p /opt/weights && \
    echo "Model weights will be downloaded at runtime if needed"

# Stage 2: Runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y \
    python3.10 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and code from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/ViewCrafter /opt/ViewCrafter
COPY --from=builder /opt/InstantSplat /opt/InstantSplat
COPY --from=builder /opt/dust3r /opt/dust3r
COPY --from=builder /opt/mast3r /opt/mast3r
COPY --from=builder /opt/weights /opt/weights

# Copy pipeline scripts
COPY run_pipeline.sh /opt/run_pipeline.sh
COPY convert_to_ksplat.py /opt/convert_to_ksplat.py
RUN chmod +x /opt/run_pipeline.sh

WORKDIR /data

ENTRYPOINT ["/opt/run_pipeline.sh"]
