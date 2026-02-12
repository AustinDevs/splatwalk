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
sleep 3

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

docker run --gpus all --name "$CONTAINER_NAME" \
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

# Check if already has ACTUAL GPU support (not just importable)
if python -c "
import torch
from pytorch3d.renderer.points.rasterize_points import rasterize_points
# Actually test GPU rasterization, not just import
pts = torch.randn(1, 100, 3).cuda()
radius = torch.ones(1, 100).cuda() * 0.01
idx, zbuf, dists = rasterize_points(pts, (64, 64), radius, 1)
print(\"PyTorch3D GPU rasterization test PASSED\")
" 2>/dev/null; then
    echo "PyTorch3D already has working GPU support!"
else
    echo "Removing pre-built PyTorch3D (no GPU kernels)..."
    pip uninstall -y pytorch3d 2>/dev/null || true

    echo "Compiling PyTorch3D from source with FORCE_CUDA=1..."
    echo "This takes ~15-20 minutes on H100..."
    FORCE_CUDA=1 pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git" 2>&1

    # Verify with actual GPU rasterization test
    echo ""
    echo "Verifying PyTorch3D GPU support with actual rasterization test..."
    python -c "
import torch
print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")
from pytorch3d.renderer.points.rasterize_points import rasterize_points
import pytorch3d
print(f\"PyTorch3D version: {pytorch3d.__version__}\")
# Actually test GPU rasterization
pts = torch.randn(1, 100, 3).cuda()
radius = torch.ones(1, 100).cuda() * 0.01
idx, zbuf, dists = rasterize_points(pts, (64, 64), radius, 1)
print(\"PyTorch3D GPU rasterization test PASSED\")
" || {
        echo "ERROR: PyTorch3D GPU rasterization test failed!"
        exit 1
    }
fi

echo ""
echo "=== Patching ViewCrafter for Pillow 10+ (ANTIALIAS -> LANCZOS) ==="
grep -rl "Image.ANTIALIAS" /opt/ViewCrafter/ 2>/dev/null | while read f; do
    sed -i "s/Image.ANTIALIAS/Image.LANCZOS/g" "$f"
    echo "  Patched $f"
done || echo "  No files needed patching"

echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f\"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"GPU: {torch.cuda.get_device_name(0)}\")

# Test PyTorch3D
from pytorch3d.renderer.points.rasterize_points import rasterize_points
print(\"PyTorch3D GPU rasterization: OK\")

# Test key imports
import open3d; print(f\"Open3D {open3d.__version__}: OK\")
import trimesh; print(f\"Trimesh {trimesh.__version__}: OK\")
import kornia; print(f\"Kornia {kornia.__version__}: OK\")
import transformers; print(f\"Transformers {transformers.__version__}: OK\")

# Test ViewCrafter imports
import sys; sys.path.insert(0, \"/opt/ViewCrafter\")
try:
    from omegaconf import OmegaConf
    print(\"ViewCrafter deps: OK\")
except Exception as e:
    print(f\"ViewCrafter deps issue: {e}\")

print()
print(\"All checks passed!\")
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

# --- Step 5: Verify the committed image works ---
echo ""
echo "=== Step 5/5: Verify committed image ==="
docker run --rm --gpus all --entrypoint python "$GPU_DOCKER_IMAGE" -c "
import torch
from pytorch3d.renderer.points.rasterize_points import rasterize_points
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
# Actual GPU rasterization test
pts = torch.randn(1, 100, 3).cuda()
radius = torch.ones(1, 100).cuda() * 0.01
idx, zbuf, dists = rasterize_points(pts, (64, 64), radius, 1)
print('PyTorch3D GPU rasterization: PASSED')
" && echo "Verification PASSED" || {
    echo "ERROR: Verification FAILED"
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
