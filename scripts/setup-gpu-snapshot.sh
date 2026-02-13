#!/bin/bash
set -euo pipefail
#
# Setup script for SplatWalk GPU snapshot.
#
# Run this on a fresh DigitalOcean GPU droplet (gpu-h100x1-80gb with base Ubuntu image).
# After it completes, create a snapshot from the DO console and update GPU_DROPLET_IMAGE in .env.
#
# Usage:
#   1. Create a GPU droplet from DO console (any base GPU image with NVIDIA drivers)
#   2. SSH in: ssh root@<droplet-ip>
#   3. Upload and run this script:
#        scp -i ~/.ssh/splatwalk_gpu scripts/setup-gpu-snapshot.sh root@<droplet-ip>:/root/
#        ssh -i ~/.ssh/splatwalk_gpu root@<droplet-ip> 'bash /root/setup-gpu-snapshot.sh'
#   4. After completion, power off and snapshot the droplet
#   5. Update GPU_DROPLET_IMAGE=<snapshot-id> in .env
#

echo "============================================"
echo "SplatWalk GPU Snapshot Setup"
echo "============================================"

# --- Env vars (override via environment or edit here) ---
GHCR_USERNAME="${GHCR_USERNAME:-KevinColten}"
GHCR_TOKEN="${GHCR_TOKEN:-}"
GPU_DOCKER_IMAGE="${GPU_DOCKER_IMAGE:-ghcr.io/austindevs/splatwalk-gpu:latest}"

# --- Step 1: Verify GPU ---
echo ""
echo "=== Step 1/5: Verify GPU ==="
if ! nvidia-smi; then
    echo "ERROR: No NVIDIA GPU detected. This script must run on a GPU droplet."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"

# --- Step 2: Configure Docker with NVIDIA runtime ---
echo ""
echo "=== Step 2/5: Configure Docker + NVIDIA runtime ==="
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
sleep 5

# Verify Docker can see the GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1 && echo "Docker GPU access OK" || {
    echo "WARNING: Docker GPU test failed, trying anyway..."
}

# --- Step 3: Pull the Docker image ---
echo ""
echo "=== Step 3/5: Pull Docker image ==="
if [ -n "$GHCR_TOKEN" ]; then
    echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin
fi

docker pull "$GPU_DOCKER_IMAGE"
echo "Docker image pulled: $(docker images --format '{{.Size}}' "$GPU_DOCKER_IMAGE")"

# --- Step 4: Build a prepared image with all deps + PyTorch3D compiled ---
echo ""
echo "=== Step 4/5: Build prepared image (install deps + compile PyTorch3D) ==="
echo "This will take ~20-30 minutes..."

CONTAINER_NAME="splatwalk-setup"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# FORCE_CUDA=1 + explicit TORCH_CUDA_ARCH_LIST ensures nvcc compiles real CUDA kernels.
# Without TORCH_CUDA_ARCH_LIST, the build silently produces a CPU-only _C.so (~10MB).
# With it, we get a proper CUDA build (~27MB _C.so).
# We skip GPU tests inside this container since Docker GPU access can be flaky after
# daemon restarts. The real GPU test happens in Step 5 with a fresh container.
docker run --gpus all --name "$CONTAINER_NAME" \
    -e FORCE_CUDA=1 \
    -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    --entrypoint bash \
    "$GPU_DOCKER_IMAGE" -c '
set -e

echo "=== Installing runtime Python dependencies ==="
pip install --no-cache-dir \
    icecream open3d trimesh "pyglet<2" evo matplotlib tensorboard imageio gdown \
    roma opencv-python transformers huggingface_hub \
    omegaconf pytorch-lightning open-clip-torch kornia decord \
    imageio-ffmpeg scikit-image moviepy 2>&1 | tail -5

echo ""
echo "=== Building PyTorch3D from source with CUDA support ==="
echo "FORCE_CUDA=$FORCE_CUDA"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# Always recompile — the pre-built wheel lacks CUDA kernels
echo "Removing pre-built PyTorch3D (no GPU kernels)..."
pip uninstall -y pytorch3d 2>/dev/null || true
rm -rf /opt/conda/lib/python*/site-packages/pytorch3d* 2>/dev/null || true

echo "Compiling PyTorch3D from source with FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST..."
echo "This takes ~15-20 minutes on H100..."
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    pip install --no-cache-dir --no-build-isolation --verbose \
    "git+https://github.com/facebookresearch/pytorch3d.git" 2>&1 | \
    tee /tmp/pytorch3d_build.log | \
    grep -E "(nvcc|Running|Building|Compiling|error|warning.*cuda|_C)" || true
echo ""

# Verify the compiled _C.so is actually large (CUDA kernels present)
echo "=== Checking PyTorch3D _C.so size ==="
PYTORCH3D_SO=$(find /opt/conda/lib/python*/site-packages/pytorch3d -name "_C*.so" 2>/dev/null | head -1)
if [ -n "$PYTORCH3D_SO" ]; then
    SO_SIZE=$(stat -c%s "$PYTORCH3D_SO" 2>/dev/null || stat -f%z "$PYTORCH3D_SO" 2>/dev/null)
    SO_SIZE_MB=$((SO_SIZE / 1024 / 1024))
    echo "_C.so path: $PYTORCH3D_SO"
    echo "_C.so size: ${SO_SIZE_MB}MB ($SO_SIZE bytes)"
    if [ "$SO_SIZE_MB" -lt 15 ]; then
        echo "ERROR: _C.so is only ${SO_SIZE_MB}MB — CUDA kernels were NOT compiled!"
        echo "Expected ~27MB for a proper CUDA build."
        echo "Last 50 lines of build log:"
        tail -50 /tmp/pytorch3d_build.log
        exit 1
    fi
    echo "Size looks good (>15MB = CUDA kernels present)"
else
    echo "ERROR: _C.so not found!"
    exit 1
fi

# Basic import test (GPU test is in Step 5)
python -c "
import pytorch3d
print(f\"PyTorch3D {pytorch3d.__version__} installed\")
from pytorch3d.renderer.points import rasterize_points
print(\"rasterize_points module loaded OK\")
"

echo ""
echo "=== Patching ViewCrafter for Pillow 10+ (ANTIALIAS -> LANCZOS) ==="
grep -rl "Image.ANTIALIAS" /opt/ViewCrafter/ 2>/dev/null | while read f; do
    sed -i "s/Image.ANTIALIAS/Image.LANCZOS/g" "$f"
    echo "  Patched $f"
done || echo "  No files needed patching"

echo ""
echo "=== Downloading ViewCrafter sparse checkpoint ==="
python -c "
from huggingface_hub import hf_hub_download
import os
ckpt_dir = \"/opt/ViewCrafter/checkpoints\"
os.makedirs(ckpt_dir, exist_ok=True)
# Download model_sparse.ckpt for sparse_view_interp mode
try:
    path = hf_hub_download(repo_id=\"Drexubery/ViewCrafter_25\", filename=\"model.ckpt\",
                           local_dir=ckpt_dir, local_dir_use_symlinks=False)
    # Rename to model_sparse.ckpt
    sparse_path = os.path.join(ckpt_dir, \"model_sparse.ckpt\")
    if not os.path.exists(sparse_path):
        os.rename(path, sparse_path)
    print(f\"Sparse checkpoint: {sparse_path} ({os.path.getsize(sparse_path) / 1e9:.1f}GB)\")
except Exception as e:
    print(f\"WARNING: Could not download sparse checkpoint: {e}\")
    print(\"Will fall back to single-view model if sparse mode is used\")
"

echo ""
echo "=== Verifying key imports ==="
python -c "
import torch; print(f\"PyTorch {torch.__version__}\")
import open3d; print(f\"Open3D {open3d.__version__}\")
import trimesh; print(f\"Trimesh {trimesh.__version__}\")
import kornia; print(f\"Kornia {kornia.__version__}\")
import transformers; print(f\"Transformers {transformers.__version__}\")
print(\"All imports OK\")
"

echo ""
echo "=== Cleanup ==="
pip cache purge 2>/dev/null || true
rm -rf /root/.cache/pip /tmp/*
echo "Done!"
'

SETUP_EXIT=$?
if [ "$SETUP_EXIT" -ne 0 ]; then
    echo "ERROR: Setup container failed (exit $SETUP_EXIT)"
    echo "Check logs above for details."
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    exit 1
fi

# Commit the container as the same image (overwrite the pulled image)
echo ""
echo "Committing container as $GPU_DOCKER_IMAGE..."
docker commit "$CONTAINER_NAME" "$GPU_DOCKER_IMAGE"
docker rm "$CONTAINER_NAME"

echo "Committed image size: $(docker images --format '{{.Size}}' "$GPU_DOCKER_IMAGE")"

# --- Step 5: Verify the committed image works with ACTUAL GPU rasterization ---
echo ""
echo "=== Step 5/5: Verify committed image with GPU rasterization test ==="
docker run --rm --gpus all --entrypoint python "$GPU_DOCKER_IMAGE" -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

import pytorch3d
print(f'PyTorch3D {pytorch3d.__version__}')

# Actual GPU rasterization test - this is the critical check
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
" && echo "Verification PASSED" || {
    echo "ERROR: Verification FAILED — PyTorch3D GPU rasterization does not work"
    echo "The compiled library may not be compatible with this GPU/CUDA version."
    exit 1
}

# --- Cleanup ---
echo ""
echo "=== Cleanup ==="
docker image prune -f
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/*

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Power off this droplet:"
echo "       shutdown -h now"
echo "  2. Create a snapshot from DO console or API:"
echo "       curl -X POST -H 'Authorization: Bearer \$DO_API_TOKEN' \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d '{\"type\":\"snapshot\",\"name\":\"splatwalk-gpu-ready\"}' \\"
echo "         https://api.digitalocean.com/v2/droplets/<DROPLET_ID>/actions"
echo "  3. Update .env: GPU_DROPLET_IMAGE=<snapshot-id>"
echo "  4. Destroy this droplet"
echo ""
