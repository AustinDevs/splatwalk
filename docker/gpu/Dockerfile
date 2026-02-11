# SplatWalk GPU processing with InstantSplat
# Use devel image for CUDA compilation of gaussian splatting extensions
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Pin numpy<2 to avoid compatibility issues with CUDA extensions
RUN pip install --no-cache-dir "numpy<2" scipy pillow tqdm einops timm boto3 awscli plyfile

WORKDIR /opt

# Clone InstantSplat and its gaussian splatting dependency
RUN git clone --depth 1 --recursive https://github.com/NVlabs/InstantSplat.git

# Build diff-gaussian-rasterization CUDA extension
WORKDIR /opt/InstantSplat/submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir .

# Build simple-knn CUDA extension
WORKDIR /opt/InstantSplat/submodules/simple-knn
RUN pip install --no-cache-dir .

# Install InstantSplat requirements
WORKDIR /opt/InstantSplat
RUN pip install --no-cache-dir -r requirements.txt || true

# Install additional dependencies for dust3r/mast3r (used by InstantSplat)
RUN pip install --no-cache-dir roma opencv-python transformers

# Copy pipeline scripts
COPY run_pipeline.sh /opt/run_pipeline.sh
COPY convert_to_ksplat.py /opt/convert_to_ksplat.py
RUN chmod +x /opt/run_pipeline.sh

WORKDIR /data

ENTRYPOINT ["/opt/run_pipeline.sh"]
