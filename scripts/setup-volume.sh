#!/bin/bash
set -euo pipefail
#
# Setup script for SplatWalk runtime Volume — REPLACES snapshot-based approach.
#
# Installs the full SplatWalk runtime onto a DO Volume at /mnt/splatwalk/:
#   Miniconda, PyTorch 2.4+CUDA 12.4, InstantSplat, ViewCrafter, PyTorch3D,
#   model weights, pipeline scripts.
#
# INCREMENTAL: Safe to re-run on an existing Volume. Each step checks
# whether its work is already present and skips if so.
#
# Usage:
#   1. Create a 100GB Volume in tor1 via DO console or API:
#        curl -s -X POST \
#          -H "Authorization: Bearer $DO_API_TOKEN" \
#          -H "Content-Type: application/json" \
#          -d '{"size_gigabytes":100,"name":"splatwalk-runtime","region":"tor1",
#               "filesystem_type":"ext4"}' \
#          "https://api.digitalocean.com/v2/volumes"
#
#   2. Create a GPU droplet in tor1 and attach the Volume:
#        # Create droplet, attach volume, SSH in...
#        mount -o discard,defaults,noatime /dev/disk/by-id/scsi-0DO_Volume_splatwalk-* /mnt/splatwalk
#
#   3. Run this script:
#        bash /root/setup-volume.sh
#
#   4. When done, unmount + detach Volume, destroy the droplet.
#      The Volume persists and is reattached by pipeline droplets at runtime.
#

VOLUME_ROOT="/mnt/splatwalk"

if [ ! -d "$VOLUME_ROOT" ]; then
    echo "ERROR: $VOLUME_ROOT does not exist. Mount the Volume first."
    echo "  mkdir -p /mnt/splatwalk"
    echo "  mount -o discard,defaults,noatime /dev/disk/by-id/scsi-0DO_Volume_splatwalk-* /mnt/splatwalk"
    exit 1
fi

echo "============================================"
echo "SplatWalk Volume Setup"
echo "Target: $VOLUME_ROOT"
echo "============================================"

# --- Step 1: Verify GPU ---
echo ""
echo "=== Step 1/8: Verify GPU ==="
if ! nvidia-smi; then
    echo "ERROR: No NVIDIA GPU detected. This script must run on a GPU droplet."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"

# --- Step 2: System packages ---
echo ""
echo "=== Step 2/8: System packages ==="
if command -v ffmpeg &>/dev/null && command -v git &>/dev/null && dpkg -s build-essential &>/dev/null 2>&1; then
    echo "System packages already installed — skipping"
else
    echo "Installing system packages..."
    apt-get update -qq
    apt-get install -y -qq \
        git wget curl ffmpeg unzip pkg-config \
        libgl1-mesa-glx libglib2.0-0 build-essential
    apt-get clean
    rm -rf /var/lib/apt/lists/*
fi

# --- Step 3: Miniconda + Python 3.11 ---
echo ""
echo "=== Step 3/8: Miniconda + Python 3.11 ==="
if [ -x "$VOLUME_ROOT/conda/bin/python" ]; then
    echo "Miniconda already installed on Volume"
else
    echo "Installing Miniconda to $VOLUME_ROOT/conda..."
    wget -q -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$VOLUME_ROOT/conda"
    rm /tmp/miniconda.sh
fi
export PATH="$VOLUME_ROOT/conda/bin:$PATH"
# Pin Python 3.11 to match PyTorch 2.4 compatibility (3.13 is too new)
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PYVER" != "3.11" ]; then
    echo "Python $PYVER detected, downgrading to 3.11..."
    conda install -y python=3.11
fi
echo "Python: $(python --version 2>&1)"

# --- Step 4: PyTorch + CUDA + all Python deps ---
echo ""
echo "=== Step 4/8: PyTorch + CUDA + Python packages ==="
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

if python -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__} CUDA OK')" 2>/dev/null; then
    echo "PyTorch with CUDA already installed — skipping"
else
    echo "Installing PyTorch 2.4 with CUDA 12.4..."
    pip install --no-cache-dir \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124
fi

# Ensure build tools are available for CUDA extension compilation
pip install --no-cache-dir setuptools wheel 2>&1 | tail -1

# Runtime Python packages
echo "Installing runtime Python packages..."
pip install --no-cache-dir \
    icecream open3d trimesh "pyglet<2" evo matplotlib tensorboard imageio gdown \
    roma opencv-python transformers huggingface_hub \
    omegaconf pytorch-lightning open-clip-torch kornia decord \
    imageio-ffmpeg scikit-image moviepy "av<14" 2>&1 | tail -5

# Base Python packages
pip install --no-cache-dir "numpy<2" scipy pillow tqdm einops timm boto3 awscli plyfile 2>&1 | tail -3

# Diffusers + FLUX deps
if python -c "import diffusers" 2>/dev/null; then
    echo "diffusers already installed — skipping"
else
    echo "Installing diffusers + accelerate..."
    pip install --no-cache-dir diffusers accelerate safetensors sentencepiece protobuf 2>&1 | tail -3
fi

# Gemini API SDK (for Gemini-based captioning — replaces BLIP)
echo "Installing google-genai SDK..."
pip install --no-cache-dir google-genai 2>&1 | tail -1

# --- Step 5: InstantSplat + CUDA extensions ---
echo ""
echo "=== Step 5/8: InstantSplat + CUDA extensions ==="

if [ -d "$VOLUME_ROOT/InstantSplat" ]; then
    echo "InstantSplat repo already cloned — skipping clone"
else
    echo "Cloning InstantSplat..."
    git clone --recursive https://github.com/NVlabs/InstantSplat.git "$VOLUME_ROOT/InstantSplat"
fi

# Build CUDA extensions
if python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "diff-gaussian-rasterization already built — skipping"
else
    echo "Building diff-gaussian-rasterization..."
    cd "$VOLUME_ROOT/InstantSplat/submodules/diff-gaussian-rasterization"
    pip install --no-cache-dir --no-build-isolation .
fi

if python -c "import simple_knn" 2>/dev/null; then
    echo "simple-knn already built — skipping"
else
    echo "Building simple-knn..."
    cd "$VOLUME_ROOT/InstantSplat/submodules/simple-knn"
    pip install --no-cache-dir --no-build-isolation .
fi

if [ -d "$VOLUME_ROOT/InstantSplat/submodules/fused-ssim" ]; then
    if python -c "import fused_ssim" 2>/dev/null; then
        echo "fused-ssim already built — skipping"
    else
        echo "Building fused-ssim..."
        cd "$VOLUME_ROOT/InstantSplat/submodules/fused-ssim"
        pip install --no-cache-dir --no-build-isolation .
    fi
fi

# Build curope extension
CUROPE_DIR="$VOLUME_ROOT/InstantSplat/croco/models/curope"
if [ -d "$CUROPE_DIR" ]; then
    if ls "$CUROPE_DIR"/*.so &>/dev/null; then
        echo "curope extension already built — skipping"
    else
        echo "Building curope extension..."
        cd "$CUROPE_DIR"
        python setup.py build_ext --inplace || true
    fi
fi

# Install InstantSplat requirements
echo "Installing InstantSplat requirements..."
cd "$VOLUME_ROOT/InstantSplat"
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -3 || true

# Strip .git dirs to save space
find "$VOLUME_ROOT/InstantSplat" -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

# ViewCrafter (kept for future re-enablement)
if [ -d "$VOLUME_ROOT/ViewCrafter" ]; then
    echo "ViewCrafter repo already cloned — skipping clone"
else
    echo "Cloning ViewCrafter..."
    git clone https://github.com/Drexubery/ViewCrafter.git "$VOLUME_ROOT/ViewCrafter"
    rm -rf "$VOLUME_ROOT/ViewCrafter/.git"
fi

echo "Installing ViewCrafter requirements..."
cd "$VOLUME_ROOT/ViewCrafter"
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -3 || true

# PyTorch3D with CUDA kernels
echo ""
echo "Building PyTorch3D from source with CUDA support..."
echo "FORCE_CUDA=1  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

EXISTING_SO=$(find "$VOLUME_ROOT/conda/lib/python"*/site-packages/pytorch3d -name "_C*.so" 2>/dev/null | head -1)
NEED_REBUILD=1
if [ -n "$EXISTING_SO" ]; then
    SO_SIZE=$(stat -c%s "$EXISTING_SO" 2>/dev/null || stat -f%z "$EXISTING_SO" 2>/dev/null || echo 0)
    SO_SIZE_MB=$((SO_SIZE / 1024 / 1024))
    echo "Existing _C.so: ${SO_SIZE_MB}MB ($EXISTING_SO)"
    if [ "$SO_SIZE_MB" -ge 15 ]; then
        echo "PyTorch3D already has CUDA kernels (${SO_SIZE_MB}MB) — skipping rebuild"
        NEED_REBUILD=0
    else
        echo "Existing _C.so is only ${SO_SIZE_MB}MB — needs rebuild"
    fi
fi

if [ "$NEED_REBUILD" -eq 1 ]; then
    pip uninstall -y pytorch3d 2>/dev/null || true
    rm -rf "$VOLUME_ROOT/conda/lib/python"*/site-packages/pytorch3d* 2>/dev/null || true

    echo "Compiling PyTorch3D from source (takes ~15-20 min on H100)..."
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
        pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git" \
        > /tmp/pytorch3d_build.log 2>&1 || {
            echo "PyTorch3D build failed! Last 50 lines:"
            tail -50 /tmp/pytorch3d_build.log
            exit 1
        }
    echo "Build finished. Checking _C.so..."

    PYTORCH3D_SO=$(find "$VOLUME_ROOT/conda/lib/python"*/site-packages/pytorch3d -name "_C*.so" 2>/dev/null | head -1)
    if [ -n "$PYTORCH3D_SO" ]; then
        SO_SIZE=$(stat -c%s "$PYTORCH3D_SO" 2>/dev/null || stat -f%z "$PYTORCH3D_SO" 2>/dev/null)
        SO_SIZE_MB=$((SO_SIZE / 1024 / 1024))
        echo "_C.so size: ${SO_SIZE_MB}MB"
        if [ "$SO_SIZE_MB" -lt 15 ]; then
            echo "ERROR: _C.so is only ${SO_SIZE_MB}MB — CUDA kernels were NOT compiled!"
            tail -50 /tmp/pytorch3d_build.log
            exit 1
        fi
        echo "Size looks good (>15MB = CUDA kernels present)"
    else
        echo "ERROR: _C.so not found!"
        exit 1
    fi
fi

# Patch ANTIALIAS -> LANCZOS for Pillow 10+
echo ""
echo "Patching ViewCrafter for Pillow 10+ (ANTIALIAS -> LANCZOS)..."
grep -rl "Image.ANTIALIAS" "$VOLUME_ROOT/ViewCrafter/" 2>/dev/null | while read f; do
    sed -i "s/Image.ANTIALIAS/Image.LANCZOS/g" "$f"
    echo "  Patched $f"
done || echo "  No files needed patching"

# --- Step 6: Model weights ---
echo ""
echo "=== Step 6/8: Model weights ==="

mkdir -p "$VOLUME_ROOT/models/dust3r" "$VOLUME_ROOT/models/mast3r" "$VOLUME_ROOT/models/flux"

DUST3R_CKPT="$VOLUME_ROOT/models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
MAST3R_CKPT="$VOLUME_ROOT/models/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

if [ -f "$DUST3R_CKPT" ]; then
    echo "DUSt3R checkpoint already present — skipping"
else
    echo "Downloading DUSt3R checkpoint (~2.5GB)..."
    wget -q -O "$DUST3R_CKPT" \
        "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
fi

if [ -f "$MAST3R_CKPT" ]; then
    echo "MASt3R checkpoint already present — skipping"
else
    echo "Downloading MASt3R checkpoint (~2.5GB)..."
    wget -q -O "$MAST3R_CKPT" \
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
fi

# Symlink checkpoints to expected locations
mkdir -p "$VOLUME_ROOT/InstantSplat/dust3r/checkpoints" "$VOLUME_ROOT/InstantSplat/mast3r/checkpoints"
ln -sf "$DUST3R_CKPT" "$VOLUME_ROOT/InstantSplat/dust3r/checkpoints/" 2>/dev/null || true
ln -sf "$MAST3R_CKPT" "$VOLUME_ROOT/InstantSplat/mast3r/checkpoints/" 2>/dev/null || true

# Alternate lookup paths
MAST3R_ALT="$VOLUME_ROOT/InstantSplat/submodules/mast3r/checkpoints"
mkdir -p "$MAST3R_ALT"
ln -sf "$MAST3R_CKPT" "$MAST3R_ALT/" 2>/dev/null || true

# ViewCrafter checkpoint links
mkdir -p "$VOLUME_ROOT/ViewCrafter/checkpoints" 2>/dev/null || true
ln -sf "$DUST3R_CKPT" "$VOLUME_ROOT/ViewCrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" 2>/dev/null || true

# ViewCrafter sparse checkpoint
if [ -f "$VOLUME_ROOT/ViewCrafter/checkpoints/model_sparse.ckpt" ]; then
    echo "ViewCrafter sparse checkpoint already present — skipping"
else
    echo "Downloading ViewCrafter sparse checkpoint..."
    python -c "
from huggingface_hub import hf_hub_download
import os
ckpt_dir = '$VOLUME_ROOT/ViewCrafter/checkpoints'
os.makedirs(ckpt_dir, exist_ok=True)
try:
    path = hf_hub_download(repo_id='Drexubery/ViewCrafter_25', filename='model.ckpt',
                           local_dir=ckpt_dir, local_dir_use_symlinks=False)
    sparse_path = os.path.join(ckpt_dir, 'model_sparse.ckpt')
    if not os.path.exists(sparse_path):
        os.rename(path, sparse_path)
    print(f'Sparse checkpoint: {sparse_path} ({os.path.getsize(sparse_path) / 1e9:.1f}GB)')
except Exception as e:
    print(f'WARNING: Could not download sparse checkpoint: {e}')
" || true
fi

# Pre-download FLUX models
echo "Pre-downloading FLUX models (this takes a while)..."
export HF_HOME="$VOLUME_ROOT/models/flux"
python -c "
from diffusers import FluxControlNetPipeline, FluxControlNetModel
import torch

print('Downloading FLUX ControlNet-depth...')
cn = FluxControlNetModel.from_pretrained(
    'XLabs-AI/flux-controlnet-depth-diffusers', torch_dtype=torch.bfloat16)

print('Downloading FLUX.1-dev + pipeline...')
pipe = FluxControlNetPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev', controlnet=cn,
    torch_dtype=torch.bfloat16)

print('All FLUX models cached.')
" || echo "WARNING: FLUX model pre-download failed (models will download at runtime)"

# Pre-download Depth-Anything-V2 (replaces MiDaS for depth estimation)
echo "Pre-downloading Depth-Anything-V2..."
python -c "
from transformers import pipeline as tf_pipeline
print('Downloading Depth-Anything-V2-Large...')
p = tf_pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf')
del p
print('Depth-Anything-V2 cached.')
" || echo "WARNING: Depth-Anything-V2 pre-download failed (will download at runtime)"

# Pre-download XLabs IP-Adapter v2 + CLIP encoder for FLUX (aerial texture matching)
echo "Pre-downloading XLabs FLUX IP-Adapter v2 + CLIP encoder..."
python -c "
from huggingface_hub import hf_hub_download
print('Downloading XLabs IP-Adapter v2...')
hf_hub_download('XLabs-AI/flux-ip-adapter-v2', 'ip_adapter.safetensors')
print('Downloading CLIP ViT-L/14 image encoder...')
from transformers import CLIPVisionModelWithProjection
CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
print('IP-Adapter + CLIP cached.')
" || echo "WARNING: IP-Adapter pre-download failed (will download at runtime)"

# --- Step 7: Pipeline scripts ---
echo ""
echo "=== Step 7/8: Pipeline scripts ==="

mkdir -p "$VOLUME_ROOT/scripts"

# Try to find scripts from the repo (if run from repo root or SCP'd alongside)
SCRIPT_DIR=""
if [ -d "$(dirname "$0")/../docker/gpu" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")/../docker/gpu" && pwd)"
elif [ -d "/root/pipeline-scripts" ]; then
    SCRIPT_DIR="/root/pipeline-scripts"
fi

if [ -n "$SCRIPT_DIR" ]; then
    echo "Copying pipeline scripts from $SCRIPT_DIR to $VOLUME_ROOT/scripts/..."
    for script in entrypoint.sh run_pipeline.sh render_descent.py generate_ground_views.py enhance_with_viewcrafter.py quality_gate.py convert_to_ksplat.py compress_splat.py score_renders.py; do
        if [ -f "$SCRIPT_DIR/$script" ]; then
            cp "$SCRIPT_DIR/$script" "$VOLUME_ROOT/scripts/$script"
            echo "  Copied $script"
        else
            echo "  WARNING: $script not found in $SCRIPT_DIR"
        fi
    done
    chmod +x "$VOLUME_ROOT/scripts/entrypoint.sh" "$VOLUME_ROOT/scripts/run_pipeline.sh" 2>/dev/null || true
else
    echo "WARNING: Pipeline scripts directory not found."
    echo "  Expected either repo at $(dirname "$0")/../docker/gpu"
    echo "  or SCP'd scripts at /root/pipeline-scripts/"
    echo "  Scripts will be fetched from GitHub at runtime instead."
fi

# --- Step 8: Verification ---
echo ""
echo "=== Step 8/8: Verification ==="

echo "Checking key imports..."
python -c "
import torch; print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import open3d; print(f'Open3D {open3d.__version__}')
import trimesh; print(f'Trimesh {trimesh.__version__}')
import kornia; print(f'Kornia {kornia.__version__}')
import transformers; print(f'Transformers {transformers.__version__}')
import diffusers; print(f'Diffusers {diffusers.__version__}')
print('All imports OK')
"

echo ""
echo "Testing PyTorch3D GPU rasterization..."
python -c "
import torch
import pytorch3d
print(f'PyTorch3D {pytorch3d.__version__}')
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PointsRasterizer, PointsRasterizationSettings, PerspectiveCameras
pts = torch.randn(1, 100, 3).cuda()
features = torch.randn(1, 100, 3).cuda()
pc = Pointclouds(points=pts, features=features)
cameras = PerspectiveCameras(device='cuda')
settings = PointsRasterizationSettings(image_size=64, radius=0.01, points_per_pixel=1)
rasterizer = PointsRasterizer(cameras=cameras, raster_settings=settings)
fragments = rasterizer(pc)
print(f'Rasterized: idx shape={fragments.idx.shape}')
print('PyTorch3D GPU rasterization: PASSED')
" || {
    echo "ERROR: PyTorch3D GPU rasterization FAILED"
    exit 1
}

# Cleanup
echo ""
echo "Cleaning up caches..."
pip cache purge 2>/dev/null || true
rm -rf /root/.cache/pip /tmp/*

# Report disk usage
echo ""
echo "Volume disk usage:"
du -sh "$VOLUME_ROOT"/* 2>/dev/null | sort -rh
echo ""
du -sh "$VOLUME_ROOT"
echo ""

echo "============================================"
echo "Volume setup complete!"
echo "============================================"
echo ""
echo "Volume contents at $VOLUME_ROOT:"
ls -la "$VOLUME_ROOT/"
echo ""
echo "To use this Volume with pipeline droplets:"
echo "  1. Set DO_VOLUME_ID in .env to this Volume's ID"
echo "  2. Set GPU_DROPLET_IMAGE to any stock DO GPU base image"
echo "  3. Ensure GPU_DROPLET_REGION=tor1 (same region as Volume)"
echo ""
echo "To update dependencies later:"
echo "  1. Launch any GPU droplet in tor1, attach + mount this Volume"
echo "  2. export PATH=$VOLUME_ROOT/conda/bin:\$PATH"
echo "  3. pip install <package> or conda install <package>"
echo "  4. Unmount + detach Volume, destroy droplet"
echo ""
