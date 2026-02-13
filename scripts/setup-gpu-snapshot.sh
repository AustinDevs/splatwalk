#!/bin/bash
set -euo pipefail
#
# Setup script for SplatWalk GPU snapshot — NATIVE installation (no Docker).
#
# Installs everything directly on the droplet filesystem:
#   Miniconda, PyTorch 2.4+CUDA 12.4, InstantSplat, ViewCrafter, PyTorch3D, etc.
#
# INCREMENTAL: Safe to re-run on an existing snapshot. Each step checks
# whether its work is already present and skips if so.
#
# Usage (from your local machine):
#   1. Create a GPU droplet (from base image or existing snapshot):
#        export $(grep -v '^#' .env | xargs)
#        DROPLET_ID=$(curl -s -X POST \
#          -H "Authorization: Bearer $DO_API_TOKEN" \
#          -H "Content-Type: application/json" \
#          -d '{"name":"splatwalk-snapshot-builder","region":"nyc2",
#               "size":"gpu-h100x1-80gb","image":"'"$GPU_DROPLET_IMAGE"'",
#               "ssh_keys":["'"$DO_SSH_KEY_ID"'"]}' \
#          "https://api.digitalocean.com/v2/droplets" | python3 -c "import sys,json; print(json.load(sys.stdin)['droplet']['id'])")
#        echo "Droplet: $DROPLET_ID"
#
#   2. Wait for it to boot, get IP, upload repo scripts + run:
#        scp -i ~/.ssh/splatwalk_gpu scripts/setup-gpu-snapshot.sh root@<IP>:/root/
#        scp -i ~/.ssh/splatwalk_gpu docker/gpu/*.sh docker/gpu/*.py root@<IP>:/root/pipeline-scripts/
#        ssh -i ~/.ssh/splatwalk_gpu root@<IP> "bash /root/setup-gpu-snapshot.sh"
#
#   3. After completion, power off + snapshot + update .env (script prints commands)
#

echo "============================================"
echo "SplatWalk GPU Snapshot Setup (native, no Docker)"
echo "============================================"

# --- Step 1: Verify GPU ---
echo ""
echo "=== Step 1/12: Verify GPU ==="
if ! nvidia-smi; then
    echo "ERROR: No NVIDIA GPU detected. This script must run on a GPU droplet."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"

# --- Step 2: System packages ---
echo ""
echo "=== Step 2/12: System packages ==="
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
echo "=== Step 3/12: Miniconda + Python 3.11 ==="
if [ -x /opt/conda/bin/python ]; then
    echo "Miniconda already installed"
else
    echo "Installing Miniconda to /opt/conda..."
    wget -q -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p /opt/conda
    rm /tmp/miniconda.sh
fi
export PATH="/opt/conda/bin:$PATH"
# Pin Python 3.11 to match PyTorch 2.4 compatibility (3.13 is too new)
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PYVER" != "3.11" ]; then
    echo "Python $PYVER detected, downgrading to 3.11..."
    conda install -y python=3.11
fi
echo "Python: $(python --version 2>&1)"

# --- Step 4: PyTorch 2.4 + CUDA 12.4 ---
echo ""
echo "=== Step 4/12: PyTorch + CUDA ==="
if python -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__} CUDA OK')" 2>/dev/null; then
    echo "PyTorch with CUDA already installed — skipping"
else
    echo "Installing PyTorch 2.4 with CUDA 12.4..."
    pip install --no-cache-dir \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124
fi

# --- Step 5: InstantSplat + CUDA extensions ---
echo ""
echo "=== Step 5/12: InstantSplat + CUDA extensions ==="
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Ensure build tools are available for CUDA extension compilation
pip install --no-cache-dir setuptools wheel 2>&1 | tail -1

if [ -d /opt/InstantSplat ]; then
    echo "InstantSplat repo already cloned — skipping clone"
else
    echo "Cloning InstantSplat..."
    git clone --recursive https://github.com/NVlabs/InstantSplat.git /opt/InstantSplat
fi

# Build CUDA extensions (check if already compiled)
if python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "diff-gaussian-rasterization already built — skipping"
else
    echo "Building diff-gaussian-rasterization..."
    cd /opt/InstantSplat/submodules/diff-gaussian-rasterization
    pip install --no-cache-dir --no-build-isolation .
fi

if python -c "import simple_knn" 2>/dev/null; then
    echo "simple-knn already built — skipping"
else
    echo "Building simple-knn..."
    cd /opt/InstantSplat/submodules/simple-knn
    pip install --no-cache-dir --no-build-isolation .
fi

if [ -d /opt/InstantSplat/submodules/fused-ssim ]; then
    if python -c "import fused_ssim" 2>/dev/null; then
        echo "fused-ssim already built — skipping"
    else
        echo "Building fused-ssim..."
        cd /opt/InstantSplat/submodules/fused-ssim
        pip install --no-cache-dir --no-build-isolation .
    fi
fi

# Build curope extension
CUROPE_DIR="/opt/InstantSplat/croco/models/curope"
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
cd /opt/InstantSplat
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -3 || true

# Strip .git dirs to save space
find /opt/InstantSplat -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

# --- Step 6: Model checkpoints (DUSt3R + MASt3R) ---
echo ""
echo "=== Step 6/12: Model checkpoints ==="
DUST3R_CKPT="/opt/InstantSplat/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
MAST3R_CKPT="/opt/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

mkdir -p "$(dirname "$DUST3R_CKPT")" "$(dirname "$MAST3R_CKPT")"

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

# Symlinks for alternate lookup paths (old Dockerfile layout + ViewCrafter)
MAST3R_ALT="/opt/InstantSplat/submodules/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
mkdir -p "$(dirname "$MAST3R_ALT")"
ln -sf "$MAST3R_CKPT" "$MAST3R_ALT" 2>/dev/null || true

# ViewCrafter's DUSt3R code looks for the checkpoint under its own checkpoints dir
mkdir -p /opt/ViewCrafter/checkpoints 2>/dev/null || true
ln -sf "$DUST3R_CKPT" /opt/ViewCrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth 2>/dev/null || true

# --- Step 7: ViewCrafter + PyTorch3D ---
echo ""
echo "=== Step 7/12: ViewCrafter + PyTorch3D ==="

if [ -d /opt/ViewCrafter ]; then
    echo "ViewCrafter repo already cloned — skipping clone"
else
    echo "Cloning ViewCrafter..."
    git clone https://github.com/Drexubery/ViewCrafter.git /opt/ViewCrafter
    rm -rf /opt/ViewCrafter/.git
fi

echo "Installing ViewCrafter requirements..."
cd /opt/ViewCrafter
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -3 || true

# PyTorch3D with CUDA kernels
echo ""
echo "Building PyTorch3D from source with CUDA support..."
echo "FORCE_CUDA=1  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

EXISTING_SO=$(find /opt/conda/lib/python*/site-packages/pytorch3d -name "_C*.so" 2>/dev/null | head -1)
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
    rm -rf /opt/conda/lib/python*/site-packages/pytorch3d* 2>/dev/null || true

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

    PYTORCH3D_SO=$(find /opt/conda/lib/python*/site-packages/pytorch3d -name "_C*.so" 2>/dev/null | head -1)
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

# Download ViewCrafter sparse checkpoint
echo ""
echo "Downloading ViewCrafter sparse checkpoint..."
if [ -f "/opt/ViewCrafter/checkpoints/model_sparse.ckpt" ]; then
    echo "Sparse checkpoint already present — skipping"
else
    python -c "
from huggingface_hub import hf_hub_download
import os, shutil
ckpt_dir = '/opt/ViewCrafter/checkpoints'
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
    print('Will fall back to single-view model if sparse mode is used')
"
fi

# Download ViewCrafter 25_512 checkpoint directory (used by walkable pipeline stage 4)
echo ""
echo "Downloading ViewCrafter 25_512 checkpoint..."
if [ -d "/opt/ViewCrafter/checkpoints/ViewCrafter_25_512" ]; then
    echo "ViewCrafter_25_512 checkpoint already present — skipping"
else
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Drexubery/ViewCrafter_25_512', local_dir='/opt/ViewCrafter/checkpoints/ViewCrafter_25_512')
print('ViewCrafter_25_512 checkpoint downloaded')
" || echo "WARNING: Could not download ViewCrafter_25_512 checkpoint"
fi

# Patch ANTIALIAS -> LANCZOS for Pillow 10+
echo ""
echo "Patching ViewCrafter for Pillow 10+ (ANTIALIAS -> LANCZOS)..."
grep -rl "Image.ANTIALIAS" /opt/ViewCrafter/ 2>/dev/null | while read f; do
    sed -i "s/Image.ANTIALIAS/Image.LANCZOS/g" "$f"
    echo "  Patched $f"
done || echo "  No files needed patching"

# --- Step 8: Runtime Python packages ---
echo ""
echo "=== Step 8/12: Runtime Python packages ==="
if python -c "import open3d, kornia, transformers" 2>/dev/null; then
    echo "Runtime Python packages already installed — skipping"
else
    echo "Installing runtime Python packages..."
    pip install --no-cache-dir \
        icecream open3d trimesh "pyglet<2" evo matplotlib tensorboard imageio gdown \
        roma opencv-python transformers huggingface_hub \
        omegaconf pytorch-lightning open-clip-torch kornia decord \
        imageio-ffmpeg scikit-image moviepy av 2>&1 | tail -5
fi

# --- Step 9: Base Python packages ---
echo ""
echo "=== Step 9/12: Base Python packages ==="
pip install --no-cache-dir "numpy<2" scipy pillow tqdm einops timm boto3 awscli plyfile 2>&1 | tail -3

# --- Step 10: Pipeline scripts ---
echo ""
echo "=== Step 10/12: Pipeline scripts ==="

# Try to find scripts from the repo (if run from repo root or SCP'd alongside)
SCRIPT_DIR=""
if [ -d "$(dirname "$0")/../docker/gpu" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")/../docker/gpu" && pwd)"
elif [ -d "/root/pipeline-scripts" ]; then
    SCRIPT_DIR="/root/pipeline-scripts"
fi

if [ -n "$SCRIPT_DIR" ]; then
    echo "Copying pipeline scripts from $SCRIPT_DIR to /opt/..."
    for script in entrypoint.sh run_pipeline.sh render_descent.py enhance_with_viewcrafter.py quality_gate.py convert_to_ksplat.py; do
        if [ -f "$SCRIPT_DIR/$script" ]; then
            cp "$SCRIPT_DIR/$script" "/opt/$script"
            echo "  Copied $script"
        else
            echo "  WARNING: $script not found in $SCRIPT_DIR"
        fi
    done
    chmod +x /opt/entrypoint.sh /opt/run_pipeline.sh 2>/dev/null || true
else
    echo "WARNING: Pipeline scripts directory not found."
    echo "  Expected either repo at $(dirname "$0")/../docker/gpu"
    echo "  or SCP'd scripts at /root/pipeline-scripts/"
    echo "  Scripts will be fetched from GitHub at runtime instead."
fi

# --- Step 11: Verification ---
echo ""
echo "=== Step 11/12: Verification ==="

echo "Checking key imports..."
python -c "
import torch; print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import open3d; print(f'Open3D {open3d.__version__}')
import trimesh; print(f'Trimesh {trimesh.__version__}')
import kornia; print(f'Kornia {kornia.__version__}')
import transformers; print(f'Transformers {transformers.__version__}')
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

# --- Step 12: Cleanup ---
echo ""
echo "=== Step 12/12: Cleanup ==="
pip cache purge 2>/dev/null || true
rm -rf /root/.cache/pip /tmp/* /root/.cache/huggingface
apt-get clean 2>/dev/null || true
rm -rf /var/lib/apt/lists/*

# --- Get droplet ID for snapshot command ---
DROPLET_ID=$(curl -s http://169.254.169.254/metadata/v1/id 2>/dev/null || echo "<DROPLET_ID>")

echo ""
echo "============================================"
echo "Setup complete! (native, no Docker)"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Power off this droplet:"
echo "       shutdown -h now"
echo "  2. Create a snapshot from DO console or API:"
echo "       curl -X POST -H 'Authorization: Bearer \$DO_API_TOKEN' \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d '{\"type\":\"snapshot\",\"name\":\"splatwalk-gpu-native-v1\"}' \\"
echo "         https://api.digitalocean.com/v2/droplets/$DROPLET_ID/actions"
echo "  3. Update .env: GPU_DROPLET_IMAGE=<snapshot-id>"
echo "  4. Destroy this droplet"
echo ""
