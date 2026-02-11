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

# Clone InstantSplat and its submodules (includes DUSt3R)
RUN git clone --recursive https://github.com/NVlabs/InstantSplat.git

# Build diff-gaussian-rasterization CUDA extension
WORKDIR /opt/InstantSplat/submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir .

# Build simple-knn CUDA extension
WORKDIR /opt/InstantSplat/submodules/simple-knn
RUN pip install --no-cache-dir .

# Build fused-ssim if available
WORKDIR /opt/InstantSplat/submodules
RUN if [ -d "fused-ssim" ]; then cd fused-ssim && pip install --no-cache-dir .; fi

# Build DUSt3R's RoPE CUDA kernels for faster inference
WORKDIR /opt/InstantSplat/submodules/dust3r/croco/models/curope
RUN python setup.py build_ext --inplace || true

# Install InstantSplat and DUSt3R requirements
WORKDIR /opt/InstantSplat
RUN pip install --no-cache-dir -r requirements.txt || true
WORKDIR /opt/InstantSplat/submodules/dust3r
RUN pip install --no-cache-dir -r requirements.txt || true

# Download DUSt3R model weights
RUN mkdir -p /opt/InstantSplat/submodules/dust3r/checkpoints && \
    wget -q -O /opt/InstantSplat/submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# Download MASt3R model weights (required for init_test_pose.py)
# Create both possible checkpoint locations for compatibility
RUN mkdir -p /opt/InstantSplat/submodules/mast3r/checkpoints && \
    mkdir -p /opt/InstantSplat/mast3r/checkpoints && \
    wget -q -O /opt/InstantSplat/submodules/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" && \
    ln -sf /opt/InstantSplat/submodules/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    /opt/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# Install additional dependencies
RUN pip install --no-cache-dir roma opencv-python transformers

# Install instantsplat PyPI package as alternative
RUN pip install --no-cache-dir instantsplat || true

# ViewCrafter for Stage 4 diffusion enhancement
RUN git clone https://github.com/Drexubery/ViewCrafter.git /opt/ViewCrafter
WORKDIR /opt/ViewCrafter
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir pytorch3d || \
    pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git" || true
RUN mkdir -p checkpoints && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Drexubery/ViewCrafter_25_512', local_dir='checkpoints/ViewCrafter_25_512')" || true

# Copy conversion script (rarely changes, fine to bake in)
COPY convert_to_ksplat.py /opt/convert_to_ksplat.py

# Pipeline Python scripts for walkable pipeline (Stages 3-5)
COPY render_descent.py /opt/render_descent.py
COPY enhance_with_viewcrafter.py /opt/enhance_with_viewcrafter.py
COPY quality_gate.py /opt/quality_gate.py

# Pipeline script is fetched at runtime from GitHub so we don't
# have to rebuild this heavy image for script-only changes.
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/opt/entrypoint.sh"]
