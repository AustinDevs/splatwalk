# SplatWalk GPU processing with InstantSplat + ViewCrafter
# Use devel image for CUDA compilation of gaussian splatting extensions
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg unzip pkg-config \
    libgl1-mesa-glx libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python deps (one layer)
RUN pip install --no-cache-dir \
    "numpy<2" scipy pillow tqdm einops timm boto3 awscli plyfile \
    roma opencv-python transformers huggingface_hub icecream \
    open3d trimesh "pyglet<2" evo matplotlib tensorboard imageio gdown

WORKDIR /opt

# InstantSplat: clone, build CUDA extensions, install deps, strip .git (~500MB saved)
RUN git clone --recursive https://github.com/NVlabs/InstantSplat.git && \
    cd /opt/InstantSplat/submodules/diff-gaussian-rasterization && pip install --no-cache-dir . && \
    cd /opt/InstantSplat/submodules/simple-knn && pip install --no-cache-dir . && \
    cd /opt/InstantSplat/submodules && \
    (if [ -d "fused-ssim" ]; then cd fused-ssim && pip install --no-cache-dir .; fi) && \
    cd /opt/InstantSplat/submodules/dust3r/croco/models/curope && \
    (python setup.py build_ext --inplace || true) && \
    cd /opt/InstantSplat && (pip install --no-cache-dir -r requirements.txt || true) && \
    cd /opt/InstantSplat/submodules/dust3r && (pip install --no-cache-dir -r requirements.txt || true) && \
    find /opt/InstantSplat -name ".git" -type d -exec rm -rf {} + 2>/dev/null; true

# Model weights: DUSt3R + MASt3R (one layer, ~2.5GB)
RUN mkdir -p /opt/InstantSplat/submodules/dust3r/checkpoints \
             /opt/InstantSplat/submodules/mast3r/checkpoints \
             /opt/InstantSplat/mast3r/checkpoints && \
    wget -q -O /opt/InstantSplat/submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" && \
    wget -q -O /opt/InstantSplat/submodules/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" && \
    ln -sf /opt/InstantSplat/submodules/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    /opt/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# ViewCrafter: clone, install deps, strip .git (one layer)
RUN git clone https://github.com/Drexubery/ViewCrafter.git /opt/ViewCrafter && \
    cd /opt/ViewCrafter && \
    (pip install --no-cache-dir -r requirements.txt || true) && \
    (pip install --no-cache-dir pytorch3d || \
     pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git" || true) && \
    rm -rf /opt/ViewCrafter/.git

# ViewCrafter checkpoint (~14GB) — separate layer so it caches independently
RUN pip install --no-cache-dir huggingface_hub && \
    mkdir -p /opt/ViewCrafter/checkpoints && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Drexubery/ViewCrafter_25_512', local_dir='/opt/ViewCrafter/checkpoints/ViewCrafter_25_512')" && \
    rm -rf /root/.cache/huggingface

# Pipeline scripts (tiny layer, changes often)
COPY convert_to_ksplat.py /opt/convert_to_ksplat.py
COPY render_descent.py /opt/render_descent.py
COPY enhance_with_viewcrafter.py /opt/enhance_with_viewcrafter.py
COPY quality_gate.py /opt/quality_gate.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/opt/entrypoint.sh"]
