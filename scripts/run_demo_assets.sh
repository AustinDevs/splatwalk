#!/bin/bash
set -euo pipefail

# ============================================================================
# Launch GPU droplet for Mesh Pipeline + Gemini Quality Loop
#
# Uses the aukerman dataset to:
# 1. ODM orthomosaic + DSM/DTM generation (~8 min)
# 2. Generate aerial GLB (orthophoto draped on DSM terrain mesh) (~1 min)
# 3. Build ground-level scene GLB (PBR terrain + vegetation) (~3 min)
# 4. Gemini quality loop on ground GLB (up to 3 iterations) (~5 min)
# 5. Upload GLBs + manifest to Spaces CDN, self-destruct
#
# Total estimated time: ~18 min on RTX 6000 Ada (~$0.47)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/.env"

JOB_ID="demo-$(date +%s | tail -c 9)"
DATASET_URL="${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/datasets/aukerman.zip"

echo "=== SplatWalk Demo Asset Generator ==="
echo "Job ID:    $JOB_ID"
echo "Dataset:   $DATASET_URL"
echo "Region:    ${GPU_DROPLET_REGION}"
echo "Size:      ${GPU_DROPLET_SIZE}"
echo ""

# --- Check for existing GPU droplets (DO has a 1 GPU limit) ---
echo "Checking for existing GPU droplets..."
EXISTING=$(curl -s -H "Authorization: Bearer $DO_API_TOKEN" \
    "https://api.digitalocean.com/v2/droplets?tag_name=splatwalk" \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
droplets = data.get('droplets', [])
gpu_droplets = [d for d in droplets if 'gpu' in d.get('size_slug', '')]
for d in gpu_droplets:
    print(f\"{d['id']} {d['name']} {d['status']}\")
" 2>/dev/null || true)

# Also check by name prefix
EXISTING2=$(curl -s -H "Authorization: Bearer $DO_API_TOKEN" \
    "https://api.digitalocean.com/v2/droplets?per_page=50" \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
for d in data.get('droplets', []):
    if d['name'].startswith('splatwalk-gpu'):
        print(f\"{d['id']} {d['name']} {d['status']}\")
" 2>/dev/null || true)

ALL_GPU="$(echo -e "${EXISTING}\n${EXISTING2}" | sort -u | grep -v '^$' || true)"

if [ -n "$ALL_GPU" ]; then
    echo "Found existing GPU droplets:"
    echo "$ALL_GPU"
    echo ""
    read -p "Destroy existing droplets before creating new one? [y/N] " CONFIRM
    if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
        echo "$ALL_GPU" | while read -r DROPLET_ID REST; do
            echo "Destroying droplet $DROPLET_ID ($REST)..."
            curl -s -X DELETE \
                -H "Authorization: Bearer $DO_API_TOKEN" \
                "https://api.digitalocean.com/v2/droplets/$DROPLET_ID"
        done
        echo "Waiting 10s for cleanup..."
        sleep 10
    else
        echo "Aborting."
        exit 1
    fi
fi

# --- Build cloud-init script ---
# This is a self-contained script that runs stages 1-2 then generate_viewer_assets.py
CLOUD_INIT=$(cat <<'CLOUD_INIT_EOF'
#!/bin/bash
exec > /var/log/splatwalk-job.log 2>&1
set -x  # verbose for debugging

echo "=== SplatWalk Demo Asset Runner ==="
echo "Job: __JOB_ID__"
echo "Started: $(date -u)"

# Conda TOS env var (belt and suspenders — also set in setup-volume.sh)
export CONDA_AUTO_ACCEPT_CHANNEL_NOTICES=true

# --- Slack helper ---
notify_slack() {
  local message="$1"
  local status="${2:-info}"
  if [ -z "__SLACK_WEBHOOK_URL__" ]; then return; fi
  local emoji=":arrows_counterclockwise:"
  [ "$status" = "success" ] && emoji=":white_check_mark:"
  [ "$status" = "error" ] && emoji=":x:"
  local payload="{\"text\":\"${emoji} *[__JOB_ID_SHORT__] demo* -- ${message}\"}"
  curl -s -X POST -H 'Content-type: application/json' --data "$payload" "__SLACK_WEBHOOK_URL__" > /dev/null 2>&1 &
}

# --- Upload log to Spaces ---
upload_log() {
  local log_key="jobs/__JOB_ID__/logs/cloud-init.log"
  local date_str=$(date -u +"%a, %d %b %Y %H:%M:%S GMT")
  local content_type="text/plain"
  local acl="public-read"
  local resource="/__SPACES_BUCKET__/$log_key"
  local string_to_sign="PUT\n\n${content_type}\n${date_str}\nx-amz-acl:${acl}\n${resource}"
  local signature=$(echo -en "$string_to_sign" | openssl dgst -sha1 -hmac "__SPACES_SECRET__" -binary | base64)
  curl -s -X PUT \
    -H "Date: $date_str" \
    -H "Content-Type: $content_type" \
    -H "x-amz-acl: $acl" \
    -H "Authorization: AWS __SPACES_KEY__:$signature" \
    --data-binary @/var/log/splatwalk-job.log \
    "__SPACES_ENDPOINT__/__SPACES_BUCKET__/$log_key" || true
  notify_slack "Log: __SPACES_ENDPOINT__/__SPACES_BUCKET__/$log_key"
}

# --- Self-destruct trap --- ALWAYS destroys droplet on exit (success or failure)
PIPELINE_SUCCESS=false
self_destruct() {
  echo "EXIT trap fired (success=$PIPELINE_SUCCESS)..."
  upload_log
  if [ "$PIPELINE_SUCCESS" = "true" ]; then
    notify_slack "Pipeline complete. Self-destructing." "success"
  else
    notify_slack "Pipeline FAILED. Uploading logs + self-destructing." "error"
  fi
  sleep 2
  umount /mnt/splatwalk 2>/dev/null || true
  DROPLET_ID=$(curl -s http://169.254.169.254/metadata/v1/id)
  curl -s -X POST \
    -H "Authorization: Bearer __DO_API_TOKEN__" \
    -H "Content-Type: application/json" \
    -d '{"type":"detach","droplet_id":'$DROPLET_ID'}' \
    "https://api.digitalocean.com/v2/volumes/__DO_VOLUME_ID__/actions"
  sleep 10
  curl -s -X DELETE \
    -H "Authorization: Bearer __DO_API_TOKEN__" \
    "https://api.digitalocean.com/v2/droplets/$DROPLET_ID"
}
trap self_destruct EXIT

# Safety timeout: 3 hours
( sleep 10800; echo "Safety timeout (3h)..."; notify_slack "Safety timeout (3h) reached." "error"; kill -9 $$ 2>/dev/null ) &

notify_slack "Droplet booted, waiting for GPU driver..."

# --- Wait for GPU ---
for i in $(seq 1 30); do
  nvidia-smi && break
  echo "Waiting for GPU... (attempt $i)"
  sleep 10
done

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
notify_slack "GPU ready: $GPU_NAME. Attaching Volume..."

# --- Attach + mount Volume ---
DROPLET_ID=$(curl -s http://169.254.169.254/metadata/v1/id)
curl -s -X POST \
  -H "Authorization: Bearer __DO_API_TOKEN__" \
  -H "Content-Type: application/json" \
  -d '{"type":"attach","droplet_id":'$DROPLET_ID'}' \
  "https://api.digitalocean.com/v2/volumes/__DO_VOLUME_ID__/actions"

for i in $(seq 1 30); do
  ls /dev/disk/by-id/scsi-0DO_Volume_splatwalk-* 2>/dev/null && break
  echo "Waiting for Volume... (attempt $i)"
  sleep 5
done

mkdir -p /mnt/splatwalk
VOLUME_DEV=$(ls /dev/disk/by-id/scsi-0DO_Volume_splatwalk-* 2>/dev/null | head -1)
if [ -z "$VOLUME_DEV" ]; then
  echo "ERROR: Volume device not found after 30 attempts"
  notify_slack "Volume device never appeared" "error"
  exit 1
fi
echo "Volume device found: $VOLUME_DEV"
mount -o discard,defaults,noatime "$VOLUME_DEV" /mnt/splatwalk || {
  echo "ERROR: mount failed, trying mkfs first..."
  mkfs.ext4 -F "$VOLUME_DEV"
  mount -o discard,defaults,noatime "$VOLUME_DEV" /mnt/splatwalk || {
    notify_slack "Volume mount FAILED even after mkfs" "error"
    exit 1
  }
}
echo "Volume mounted at /mnt/splatwalk (dev=$VOLUME_DEV)"
df -h /mnt/splatwalk

# --- Run setup if volume is empty/new ---
if ! /mnt/splatwalk/conda/bin/python --version > /dev/null 2>&1; then
  notify_slack "Fresh volume detected — running setup (~30 min)..."
  echo "Fresh volume — downloading and running setup-volume.sh..."
  curl -fsSL -H "Accept: application/vnd.github.raw" \
    "https://api.github.com/repos/AustinDevs/splatwalk/contents/scripts/setup-volume.sh" \
    -o /tmp/setup-volume.sh 2>/dev/null \
    || curl -fsSL "__SPACES_ENDPOINT__/__SPACES_BUCKET__/scripts/setup-volume.sh" \
    -o /tmp/setup-volume.sh
  bash /tmp/setup-volume.sh || { notify_slack "Volume setup FAILED" "error"; exit 1; }
  notify_slack "Volume setup complete!"
fi

# --- Set up Python environment ---
export PATH="/mnt/splatwalk/conda/bin:$PATH"

# Verify basic torch + CUDA works (for Real-ESRGAN upscaling)
/mnt/splatwalk/conda/bin/python -c "
import torch
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    t = torch.randn(100, 3, device='cuda')
    print(f'CUDA tensor OK: {t.shape}')
" || notify_slack "WARNING: CUDA not available, Real-ESRGAN will use CPU" "info"

# --- Ensure trimesh + rasterio installed (mesh pipeline deps) ---
/mnt/splatwalk/conda/bin/pip install --no-cache-dir trimesh rasterio scipy 2>&1 | tail -3

# --- Ensure Real-ESRGAN is installed (handles existing volumes without it) ---
if ! /mnt/splatwalk/conda/bin/python -c "import realesrgan" 2>/dev/null; then
  notify_slack "Installing Real-ESRGAN on existing volume..."
  /mnt/splatwalk/conda/bin/pip install --no-cache-dir realesrgan 2>&1 | tail -3
  # Download model weights
  /mnt/splatwalk/conda/bin/python -c "
import os, urllib.request
model_path = '/mnt/splatwalk/models/RealESRGAN_x2plus.pth'
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model_path)
    print(f'Downloaded RealESRGAN_x2plus.pth ({os.path.getsize(model_path)/1e6:.1f}MB)')
"
fi

# --- Ensure geospatial packages are installed (handles existing volumes without them) ---
if ! /mnt/splatwalk/conda/bin/python -c "import py3dep, pyproj" 2>/dev/null; then
  notify_slack "Installing geospatial packages on existing volume..."
  /mnt/splatwalk/conda/bin/pip install --no-cache-dir py3dep rioxarray rasterio pyproj 2>&1 | tail -3
fi

# --- Ensure Stage 4 packages are installed (handles existing volumes without them) ---
if ! /mnt/splatwalk/conda/bin/python -c "import pygltflib, pvlib" 2>/dev/null; then
  notify_slack "Installing Stage 4 packages (pygltflib, pvlib, SAM2) on existing volume..."
  /mnt/splatwalk/conda/bin/pip install --no-cache-dir pygltflib pvlib 2>&1 | tail -3
  # SAM2: install with --no-deps to avoid upgrading torch (breaks compiled CUDA extensions)
  /mnt/splatwalk/conda/bin/pip install --no-cache-dir --no-deps "git+https://github.com/facebookresearch/sam2.git" 2>&1 | tail -5 || true
  # Install SAM2's non-torch deps separately
  /mnt/splatwalk/conda/bin/pip install --no-cache-dir hydra-core iopath 2>&1 | tail -3 || true
fi
# Download SAM2 model if missing
if [ ! -f "/mnt/splatwalk/models/sam2/sam2.1_hiera_large.pt" ]; then
  mkdir -p /mnt/splatwalk/models/sam2
  /mnt/splatwalk/conda/bin/python -c "
import os, urllib.request
model_path = '/mnt/splatwalk/models/sam2/sam2.1_hiera_large.pt'
if not os.path.exists(model_path):
    print('Downloading SAM2 weights (~898MB)...')
    urllib.request.urlretrieve(
        'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
        model_path)
    print(f'Saved ({os.path.getsize(model_path)/1e6:.0f}MB)')
" || echo "WARNING: SAM2 model download failed"
fi

# --- Ensure pyrender + EGL deps for headless GLB rendering (quality loop) ---
/mnt/splatwalk/conda/bin/pip install --no-cache-dir pyrender PyOpenGL 2>&1 | tail -3
export PYOPENGL_PLATFORM=egl

notify_slack "Volume ready. Downloading dataset..."

# --- Download aukerman dataset ---
mkdir -p /workspace/__JOB_ID__/input
curl -fSL -o /workspace/__JOB_ID__/dataset.zip "__DATASET_URL__"
apt-get install -y unzip || true
unzip -q -o /workspace/__JOB_ID__/dataset.zip -d /workspace/__JOB_ID__/input
rm /workspace/__JOB_ID__/dataset.zip

# Handle nested directory (unzip may create a subdirectory)
NESTED=$(find /workspace/__JOB_ID__/input -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -n "$NESTED" ] && [ -d "$NESTED/images" ]; then
    mv "$NESTED/images"/* /workspace/__JOB_ID__/input/ 2>/dev/null || true
    mv "$NESTED"/*.jpg "$NESTED"/*.jpeg "$NESTED"/*.png /workspace/__JOB_ID__/input/ 2>/dev/null || true
    # Check if images are in a nested images/ dir
    find "$NESTED" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) -exec mv {} /workspace/__JOB_ID__/input/ \;
fi
# Remove non-image files
find /workspace/__JOB_ID__/input -type d -empty -delete 2>/dev/null || true

NUM_IMAGES=$(find /workspace/__JOB_ID__/input -maxdepth 2 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.JPG" \) | wc -l)
echo "Dataset: $NUM_IMAGES images"
notify_slack "Dataset ready ($NUM_IMAGES images). Starting pipeline..."

# --- Set up environment ---
export PATH="/mnt/splatwalk/conda/bin:$PATH"
export TRANSFORMERS_CACHE=/mnt/splatwalk/models/transformers
export HF_TOKEN=__HF_TOKEN__
export GEMINI_API_KEY='__GEMINI_API_KEY__'
export GOOGLE_MAPS_API_KEY='__GOOGLE_MAPS_API_KEY__'
export SLACK_WEBHOOK_URL='__SLACK_WEBHOOK_URL__'

# Spaces credentials for upload_to_cdn in Python scripts
export SPACES_KEY=__SPACES_KEY__
export SPACES_SECRET=__SPACES_SECRET__
export SPACES_BUCKET=__SPACES_BUCKET__
export SPACES_REGION=__SPACES_REGION__
export SPACES_ENDPOINT=__SPACES_ENDPOINT__
export AWS_ACCESS_KEY_ID=$SPACES_KEY
export AWS_SECRET_ACCESS_KEY=$SPACES_SECRET
export AWS_DEFAULT_REGION=$SPACES_REGION

# --- Fetch latest scripts from GitHub ---
echo "Fetching latest pipeline scripts from GitHub..."
_CURL_ARGS=(-fsSL -H "Accept: application/vnd.github.raw")
[ -n "__GITHUB_TOKEN__" ] && _CURL_ARGS+=(-H "Authorization: token __GITHUB_TOKEN__")
_BASE_URL="https://api.github.com/repos/AustinDevs/splatwalk/contents/scripts/gpu"
for _script in generate_viewer_assets.py build_game_scene.py generate_aerial_glb.py gemini_score_scene.py; do
    if curl "${_CURL_ARGS[@]}" "$_BASE_URL/$_script" -o "/mnt/splatwalk/scripts/$_script" 2>/dev/null; then
        echo "  Updated $_script (from GitHub)"
    elif curl -fsSL "__SPACES_ENDPOINT__/__SPACES_BUCKET__/scripts/$_script" -o "/mnt/splatwalk/scripts/$_script" 2>/dev/null; then
        echo "  Updated $_script (from Spaces)"
    else
        echo "  Using baked-in $_script (fetch failed)"
    fi
done
# Also fetch run_pipeline.sh (orchestrator)
if curl "${_CURL_ARGS[@]}" "$_BASE_URL/run_pipeline.sh" -o "/mnt/splatwalk/scripts/run_pipeline.sh" 2>/dev/null; then
    chmod +x /mnt/splatwalk/scripts/run_pipeline.sh
    echo "  Updated run_pipeline.sh (from GitHub)"
elif curl -fsSL "__SPACES_ENDPOINT__/__SPACES_BUCKET__/scripts/run_pipeline.sh" -o "/mnt/splatwalk/scripts/run_pipeline.sh" 2>/dev/null; then
    chmod +x /mnt/splatwalk/scripts/run_pipeline.sh
    echo "  Updated run_pipeline.sh (from Spaces)"
fi

# --- Find images directory (handle various unzip structures) ---
INPUT_DIR=/workspace/__JOB_ID__/input
# If there's a nested images/ dir, use that
if [ -d "$INPUT_DIR/images" ]; then
    # Move images up if needed
    find "$INPUT_DIR/images" -type f \( -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.png" \) -exec mv {} "$INPUT_DIR/" \;
fi

# --- Run mesh pipeline via run_pipeline.sh (handles ODM, GLB, quality loop) ---
export INPUT_DIR=/workspace/__JOB_ID__/input
export OUTPUT_DIR=/workspace/__JOB_ID__/output
export PIPELINE_MODE="mesh"
export JOB_ID="aukerman"

notify_slack "Launching mesh pipeline (GLB aerial + quality-gated ground GLB)..."
bash /mnt/splatwalk/scripts/run_pipeline.sh

notify_slack "Demo assets complete! View at: https://splatwalk.austindevs.com/?manifest=__SPACES_ENDPOINT__/__SPACES_BUCKET__/demo/aukerman/manifest.json" "success"

echo "=== DEMO ASSET GENERATION COMPLETE ==="
PIPELINE_SUCCESS=true
# EXIT trap fires → self_destruct
CLOUD_INIT_EOF
)

# --- Substitute variables into cloud-init ---
CLOUD_INIT="${CLOUD_INIT//__JOB_ID__/$JOB_ID}"
CLOUD_INIT="${CLOUD_INIT//__JOB_ID_SHORT__/${JOB_ID:0:8}}"
CLOUD_INIT="${CLOUD_INIT//__DATASET_URL__/$DATASET_URL}"
CLOUD_INIT="${CLOUD_INIT//__DO_API_TOKEN__/$DO_API_TOKEN}"
CLOUD_INIT="${CLOUD_INIT//__DO_VOLUME_ID__/$DO_VOLUME_ID}"
CLOUD_INIT="${CLOUD_INIT//__SPACES_KEY__/$DO_SPACES_KEY}"
CLOUD_INIT="${CLOUD_INIT//__SPACES_SECRET__/$DO_SPACES_SECRET}"
CLOUD_INIT="${CLOUD_INIT//__SPACES_BUCKET__/$DO_SPACES_BUCKET}"
CLOUD_INIT="${CLOUD_INIT//__SPACES_REGION__/$DO_SPACES_REGION}"
CLOUD_INIT="${CLOUD_INIT//__SPACES_ENDPOINT__/$DO_SPACES_ENDPOINT}"
CLOUD_INIT="${CLOUD_INIT//__SLACK_WEBHOOK_URL__/$SLACK_WEBHOOK_URL}"
CLOUD_INIT="${CLOUD_INIT//__GITHUB_TOKEN__/${GHCR_TOKEN:-}}"
CLOUD_INIT="${CLOUD_INIT//__HF_TOKEN__/${HUGGINGFACE_TOKEN:-}}"
CLOUD_INIT="${CLOUD_INIT//__GEMINI_API_KEY__/$GEMINI_API_KEY}"
CLOUD_INIT="${CLOUD_INIT//__GOOGLE_MAPS_API_KEY__/$GOOGLE_MAPS_API_KEY}"

# --- Create the droplet ---
echo "Creating GPU droplet..."

# Write cloud-init to temp file, then build JSON payload via Python
CLOUD_INIT_FILE=$(mktemp)
echo "$CLOUD_INIT" > "$CLOUD_INIT_FILE"

PAYLOAD_FILE=$(mktemp)
python3 -c "
import json
with open('$CLOUD_INIT_FILE') as f:
    cloud_init = f.read()
payload = {
    'name': 'splatwalk-gpu-${JOB_ID:0:8}',
    'region': '${GPU_DROPLET_REGION}',
    'size': '${GPU_DROPLET_SIZE}',
    'image': '${GPU_DROPLET_IMAGE}',
    'ssh_keys': ['${DO_SSH_KEY_ID}'],
    'user_data': cloud_init,
    'monitoring': True,
}
with open('$PAYLOAD_FILE', 'w') as f:
    json.dump(payload, f)
"
rm "$CLOUD_INIT_FILE"

RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $DO_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d "@$PAYLOAD_FILE" \
    "https://api.digitalocean.com/v2/droplets")
rm "$PAYLOAD_FILE"

DROPLET_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['droplet']['id'])" 2>/dev/null)

if [ -z "$DROPLET_ID" ]; then
    echo "ERROR: Failed to create droplet"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    exit 1
fi

echo ""
echo "=========================================="
echo "Droplet created: $DROPLET_ID"
echo "Job ID:          $JOB_ID"
echo "=========================================="
echo ""
echo "The droplet will:"
echo "  1. Attach Volume + download aukerman dataset"
echo "  2. ODM orthomosaic + DSM/DTM generation (~8 min)"
echo "  3. Generate aerial GLB (orthophoto on DSM terrain mesh) (~1 min)"
echo "  4. Build ground-level scene GLB (PBR terrain + vegetation) (~3 min)"
echo "  5. Gemini quality loop on ground GLB (up to 3 iterations, ~5 min)"
echo "  6. Upload GLBs + manifest to CDN, then self-destruct"
echo ""
echo "Monitor progress via Slack or check logs:"
echo "  ${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/jobs/${JOB_ID}/logs/cloud-init.log"
echo ""
echo "When complete, view demo at:"
echo "  http://localhost:3000/?manifest=${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/demo/aukerman/manifest.json"
echo ""

# --- Wait for droplet to become active ---
echo "Waiting for droplet to become active..."
for i in $(seq 1 60); do
    STATUS=$(curl -s -H "Authorization: Bearer $DO_API_TOKEN" \
        "https://api.digitalocean.com/v2/droplets/$DROPLET_ID" \
        | python3 -c "import sys,json; d=json.load(sys.stdin)['droplet']; ip=[n['ip_address'] for n in d['networks']['v4'] if n['type']=='public']; print(f\"{d['status']} {ip[0] if ip else 'no-ip'}\")" 2>/dev/null)
    echo "  [$i] $STATUS"
    if echo "$STATUS" | grep -q "^active"; then
        IP=$(echo "$STATUS" | awk '{print $2}')
        echo ""
        echo "Droplet active at: $IP"
        echo "SSH: ssh root@$IP"
        echo "Logs: ssh root@$IP tail -f /var/log/splatwalk-job.log"
        break
    fi
    sleep 5
done

echo ""
echo "Launcher done. Droplet is running autonomously."
echo "It will self-destruct when finished (~18 min total)."
