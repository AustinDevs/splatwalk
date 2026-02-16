#!/bin/bash
set -euo pipefail

# ============================================================================
# Launch GPU droplet to generate Top-Down Fly-Over Demo Assets
#
# Uses the aukerman dataset to:
# 1. Run MASt3R init + 10K InstantSplat training (~10 min)
# 2. Top-down progressive zoom descent: 5 altitude levels with FLUX
#    enhancement + retraining at each level (~60 min)
# 3. Compress .splat + upload manifest.json to Spaces CDN
# 4. Self-destruct
#
# Total estimated time: ~90 min on H100 (~$10)
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

echo "=== SplatWalk Demo Asset Runner ==="
echo "Job: __JOB_ID__"
echo "Started: $(date -u)"

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

# --- Self-destruct trap ---
self_destruct() {
  echo "Self-destructing droplet..."
  upload_log
  notify_slack "Droplet self-destructing." "info"
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
mount -o discard,defaults,noatime /dev/disk/by-id/scsi-0DO_Volume_splatwalk-* /mnt/splatwalk
echo "Volume mounted at /mnt/splatwalk"
notify_slack "Volume mounted. Downloading dataset..."

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
export HF_HOME=/mnt/splatwalk/models/flux
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
for _script in render_zoom_descent.py compress_splat.py quality_gate.py generate_viewer_assets.py; do
    if curl "${_CURL_ARGS[@]}" "$_BASE_URL/$_script" -o "/mnt/splatwalk/scripts/$_script" 2>/dev/null; then
        echo "  Updated $_script (from GitHub)"
    elif curl -fsSL "__SPACES_ENDPOINT__/__SPACES_BUCKET__/scripts/$_script" -o "/mnt/splatwalk/scripts/$_script" 2>/dev/null; then
        echo "  Updated $_script (from Spaces)"
    else
        echo "  Using baked-in $_script (fetch failed)"
    fi
done

# --- Find images directory (handle various unzip structures) ---
INPUT_DIR=/workspace/__JOB_ID__/input
# If there's a nested images/ dir, use that
if [ -d "$INPUT_DIR/images" ]; then
    # Move images up if needed
    find "$INPUT_DIR/images" -type f \( -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.png" \) -exec mv {} "$INPUT_DIR/" \;
fi

# --- Preprocess images to 512x512 ---
SCENE_DIR=/workspace/__JOB_ID__/output/scene
mkdir -p "$SCENE_DIR/images"

echo "Preprocessing images to 512x512..."
python3 << 'PREPROCESS_SCRIPT'
import os, sys
from PIL import Image
from pathlib import Path

input_dir = "/workspace/__JOB_ID__/input"
output_dir = "/workspace/__JOB_ID__/output/scene/images"
target_size = (512, 512)
os.makedirs(output_dir, exist_ok=True)
count = 0
for img_file in sorted(Path(input_dir).glob('*')):
    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        try:
            img = Image.open(img_file).convert('RGB')
            w, h = img.size
            scale = max(target_size[0] / w, target_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - target_size[0]) // 2
            top = (new_h - target_size[1]) // 2
            img = img.crop((left, top, left + target_size[0], top + target_size[1]))
            out_path = os.path.join(output_dir, img_file.stem.lower() + '.jpg')
            img.save(out_path, 'JPEG', quality=95)
            count += 1
        except Exception as e:
            print(f"  Error: {img_file.name}: {e}", file=sys.stderr)
print(f"Preprocessed {count} images to 512x512")
PREPROCESS_SCRIPT

NUM_PROCESSED=$(ls -1 "$SCENE_DIR/images"/*.jpg 2>/dev/null | wc -l)
echo "Preprocessed $NUM_PROCESSED images"

# --- Stage 1: MASt3R geometry init ---
notify_slack "Stage 1: MASt3R geometry init ($NUM_PROCESSED images)..."
cd /mnt/splatwalk/InstantSplat

MAX_VIEWS=24
N_VIEWS=$NUM_PROCESSED
[ "$N_VIEWS" -gt "$MAX_VIEWS" ] && N_VIEWS=$MAX_VIEWS

python init_geo.py \
    --source_path "$SCENE_DIR" \
    --model_path "/workspace/__JOB_ID__/output/instantsplat" \
    --n_views "$N_VIEWS" \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking \
    2>&1 || { notify_slack "Failed at Stage 1" "error"; exit 1; }

notify_slack "Stage 1 complete. Starting Stage 2 (10K training)..."

# --- Stage 2: InstantSplat training (10K iterations) ---
python train.py \
    --source_path "$SCENE_DIR" \
    --model_path "/workspace/__JOB_ID__/output/instantsplat" \
    --iterations 10000 \
    --n_views "$N_VIEWS" \
    --pp_optimizer \
    --optim_pose \
    || { notify_slack "Failed at Stage 2" "error"; exit 1; }

notify_slack "Stage 2 complete (10K aerial splat trained). Starting zoom descent..."

# --- Extract EXIF GPS + altitude ---
DRONE_AGL=63  # default for aukerman
eval "$(python3 -c "
import sys
from pathlib import Path
try:
    from PIL import Image, ExifTags
    for f in sorted(Path('/workspace/__JOB_ID__/input').glob('*')):
        if f.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
        img = Image.open(f)
        exif = img._getexif()
        if not exif: continue
        gps = exif.get(ExifTags.Base.GPSInfo)
        if not gps: continue
        tags = {}
        for k,v in gps.items():
            tags[ExifTags.GPSTAGS.get(k,k)] = v
        if 'GPSLatitude' not in tags: continue
        def dms(d,r):
            v=float(d[0])+float(d[1])/60+float(d[2])/3600
            return -v if r in ('S','W') else v
        alt=float(tags['GPSAltitude']) if 'GPSAltitude' in tags else 0
        if alt > 0:
            lat=dms(tags['GPSLatitude'],tags.get('GPSLatitudeRef','N'))
            lon=dms(tags['GPSLongitude'],tags.get('GPSLongitudeRef','W'))
            # Get ground elevation
            import urllib.request, json
            url=f'https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}'
            resp=json.loads(urllib.request.urlopen(url, timeout=5).read())
            elev=resp.get('elevation',0)
            if isinstance(elev,list): elev=elev[0]
            agl=max(10, alt-elev)
            print(f'DRONE_AGL={agl:.0f}')
        break
except Exception as e:
    print(f'# EXIF extraction failed: {e}', file=sys.stderr)
" 2>&1)" || true

echo "Drone AGL: ${DRONE_AGL}m"

# --- Stage 3: Top-Down Progressive Zoom Descent ---
notify_slack "Stage 3: Top-down zoom descent (5 altitude levels)..."
python /mnt/splatwalk/scripts/render_zoom_descent.py \
    --model_path "/workspace/__JOB_ID__/output/instantsplat" \
    --scene_path "$SCENE_DIR" \
    --output_dir "/workspace/__JOB_ID__/output/descent" \
    --altitudes "1.0,0.5,0.25,0.12,0.05" \
    --drone_agl "$DRONE_AGL" \
    --retrain_iterations 2000 \
    --max_images_per_level 64 \
    --slack_webhook_url "$SLACK_WEBHOOK_URL" \
    --job_id "__JOB_ID__" \
    || { notify_slack "render_zoom_descent.py failed" "error"; exit 1; }

notify_slack "Stage 3 complete. Generating demo assets..."

# --- Generate demo viewer assets (top-down fly-over) ---
python /mnt/splatwalk/scripts/generate_viewer_assets.py \
    --model_path "/workspace/__JOB_ID__/output/descent/final" \
    --scene_path "$SCENE_DIR" \
    --output_dir "/workspace/__JOB_ID__/output/demo" \
    --job_id "aukerman" \
    --drone_agl "$DRONE_AGL" \
    --prune_ratio 0.20 \
    --scene_scale 50.0 \
    --slack_webhook_url "$SLACK_WEBHOOK_URL" \
    || { notify_slack "generate_viewer_assets.py failed" "error"; exit 1; }

notify_slack "Demo assets complete! View at: https://splatwalk.austindevs.com/?manifest=__SPACES_ENDPOINT__/__SPACES_BUCKET__/demo/aukerman/manifest.json" "success"

echo "=== DEMO ASSET GENERATION COMPLETE ==="
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
echo "  2. Run MASt3R init + 10K InstantSplat training (~10 min)"
echo "  3. Top-down zoom descent: 5 altitude levels + FLUX (~60 min)"
echo "  4. Compress .splat + upload to CDN (~5 min)"
echo "  5. Self-destruct"
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
echo "It will self-destruct when finished (~90 min total)."
