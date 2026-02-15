#!/bin/bash
set -euo pipefail
#
# SplatWalk GPU Session Orchestrator
#
# Comprehensive script that:
#   1. Sets up SSH access to the GPU droplet
#   2. Installs new dependencies (google-genai, Depth-Anything-V2, IP-Adapter)
#   3. Runs the full walkable pipeline on the Aukerman dataset
#   4. Scores renders with Gemini at multiple altitudes
#   5. Iterates with tuned parameters until ground-level reaches 9/10
#   6. Sends progress to Slack with viewer links
#   7. Destroys the GPU droplet when done
#
# Prerequisites:
#   - SSH access to GPU droplet (ssh gpu "echo ok")
#   - Environment variables set (see below)
#   - jq installed (apt-get install jq)
#
# Usage:
#   export DO_API_TOKEN=... SLACK_WEBHOOK_URL=... etc.
#   bash scripts/run-gpu-session.sh
#

# ─── Configuration ────────────────────────────────────────────────────────────

GPU_HOST="${GPU_HOST:-gpu}"
GPU_IP="${GPU_IP:-159.203.23.47}"
CONDA_PATH="/mnt/splatwalk/conda/bin"
VOLUME_ROOT="/mnt/splatwalk"

# Dataset (already on GPU)
INPUT_DIR="/root/input/aukerman/images"
OUTPUT_BASE="/root/output/aukerman-walkable"

# Spaces CDN
SPACES_CDN="https://splatwalk.nyc3.digitaloceanspaces.com"
VIEWER_URL="https://splatwalk.com/view"

# Pipeline tuning (iteration 0 defaults, adjusted per iteration)
MAX_N_VIEWS=24
TRAIN_ITERATIONS=10000
DESCENT_ALTITUDES="0.67,0.37,0.18,0.075,0.025"
DESCENT_RETRAIN_ITERS=2000
GROUND_VIEWS_PERIMETER=20
GROUND_VIEWS_GRID=12
FLUX_DENOISING=0.85
FLUX_CONTROLNET_SCALE=0.4
FLUX_OUTPUT_SIZE=384
GROUND_RETRAIN_ITERS=7000
PRUNE_RATIO=0.20
SCENE_SCALE=50.0

# Quality target
TARGET_SCORE=9
MAX_ITERATIONS=3

# ─── Environment variable checks ─────────────────────────────────────────────

required_vars=(
    DO_API_TOKEN SLACK_WEBHOOK_URL GEMINI_API_KEY
    DO_SPACES_KEY DO_SPACES_SECRET DO_SPACES_BUCKET
    DO_SPACES_ENDPOINT GOOGLE_MAPS_API_KEY
    HUGGINGFACE_TOKEN
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set"
        exit 1
    fi
done

# ─── Helpers ──────────────────────────────────────────────────────────────────

notify_slack() {
    local message="$1"
    local status="${2:-info}"
    local emoji=":arrows_counterclockwise:"
    [ "$status" = "success" ] && emoji=":white_check_mark:"
    [ "$status" = "error" ] && emoji=":x:"
    [ "$status" = "score" ] && emoji=":dart:"

    curl -s -X POST -H 'Content-type: application/json' \
        -d "{\"text\":\"${emoji} *[splatwalk session]* — ${message}\"}" \
        "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
}

ssh_gpu() {
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=60 "$GPU_HOST" "$@"
}

ssh_gpu_long() {
    # For long-running commands: no timeout, keep alive
    ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=120 "$GPU_HOST" "$@"
}

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# ─── Step 0: SSH Setup ───────────────────────────────────────────────────────

echo "=== Step 0: Setting up SSH access ==="

# Write SSH key if not already present
if [ ! -f ~/.ssh/splatwalk_gpu ]; then
    mkdir -p ~/.ssh
    cat > ~/.ssh/splatwalk_gpu << 'SSHEOF'
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACARBV2NmNNv8JnqKx3hZfa/bKRz/2OKg9cjxiQmETJtxwAAAJACWbFJAlmx
SQAAAAtzc2gtZWQyNTUxOQAAACARBV2NmNNv8JnqKx3hZfa/bKRz/2OKg9cjxiQmETJtxw
AAAEBbuMHw8ewDKIQJvq8wgjgVm7KRgep4jgshw8/kzwANXREFXY2Y02/wmeorHeFl9r9s
pHP/Y4qD1yPGJCYRMm3HAAAADXNwbGF0d2Fsay1ncHU=
-----END OPENSSH PRIVATE KEY-----
SSHEOF
    chmod 600 ~/.ssh/splatwalk_gpu
fi

# Add SSH config if not already present
if ! grep -q "Host gpu" ~/.ssh/config 2>/dev/null; then
    cat >> ~/.ssh/config << 'EOF'
Host gpu
  HostName 159.203.23.47
  User root
  IdentityFile ~/.ssh/splatwalk_gpu
  StrictHostKeyChecking no
EOF
fi

# Test SSH connection
echo "Testing SSH connection to $GPU_IP..."
if ! ssh_gpu "echo 'SSH connection OK'" 2>/dev/null; then
    echo "ERROR: Cannot connect to GPU droplet at $GPU_IP"
    echo "The droplet may not be running. Create one first with:"
    echo "  node scripts/test-gpu-pipeline.js"
    exit 1
fi

GPU_NAME=$(ssh_gpu "nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null" || echo "unknown")
echo "Connected to GPU: $GPU_NAME"
notify_slack "Session started. GPU: $GPU_NAME"

# ─── Step 1: Install new dependencies ────────────────────────────────────────

echo ""
echo "=== Step 1: Installing new dependencies on volume ==="
notify_slack "Installing new dependencies (google-genai, Depth-Anything-V2, IP-Adapter)..."

ssh_gpu_long << 'INSTALL_EOF'
export PATH="/mnt/splatwalk/conda/bin:$PATH"
export HF_HOME="/mnt/splatwalk/models/flux"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

echo "Checking existing installations..."

# google-genai SDK
if python -c "from google import genai; print('google-genai OK')" 2>/dev/null; then
    echo "google-genai already installed"
else
    echo "Installing google-genai..."
    pip install --no-cache-dir google-genai 2>&1 | tail -3
fi

# Depth-Anything-V2
if python -c "
from transformers import pipeline
p = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf')
del p
print('Depth-Anything-V2 OK')
" 2>/dev/null; then
    echo "Depth-Anything-V2 already cached"
else
    echo "Downloading Depth-Anything-V2..."
    python -c "
from transformers import pipeline
p = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf')
del p
print('Depth-Anything-V2 downloaded')
" 2>&1 | tail -5
fi

# XLabs IP-Adapter + CLIP
if python -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download('XLabs-AI/flux-ip-adapter-v2', 'ip_adapter.safetensors')
print(f'IP-Adapter cached at: {path}')
from transformers import CLIPVisionModelWithProjection
m = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
del m
print('CLIP ViT-L/14 OK')
" 2>/dev/null; then
    echo "IP-Adapter + CLIP already cached"
else
    echo "Downloading IP-Adapter + CLIP..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('XLabs-AI/flux-ip-adapter-v2', 'ip_adapter.safetensors')
from transformers import CLIPVisionModelWithProjection
CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
print('IP-Adapter + CLIP downloaded')
" 2>&1 | tail -5
fi

# FLUX ControlNet-depth
if python -c "
from diffusers import FluxControlNetModel
import torch
cn = FluxControlNetModel.from_pretrained('XLabs-AI/flux-controlnet-depth-diffusers', torch_dtype=torch.bfloat16)
del cn
print('FLUX ControlNet OK')
" 2>/dev/null; then
    echo "FLUX ControlNet already cached"
else
    echo "Downloading FLUX ControlNet (this takes a while)..."
    python -c "
from diffusers import FluxControlNetPipeline, FluxControlNetModel
import torch
cn = FluxControlNetModel.from_pretrained('XLabs-AI/flux-controlnet-depth-diffusers', torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', controlnet=cn, torch_dtype=torch.bfloat16)
del pipe, cn
print('FLUX models cached')
" 2>&1 | tail -5
fi

echo "All dependencies ready."
INSTALL_EOF

echo "Dependencies installed."
notify_slack "Dependencies installed. Starting pipeline..." "success"

# ─── Step 2: Update pipeline scripts on GPU ──────────────────────────────────

echo ""
echo "=== Step 2: Updating pipeline scripts on GPU ==="

# Push latest scripts to GPU
for script in render_descent.py generate_ground_views.py compress_splat.py quality_gate.py score_renders.py; do
    local_path="docker/gpu/$script"
    if [ -f "$local_path" ]; then
        scp -o ConnectTimeout=10 "$local_path" "$GPU_HOST:/mnt/splatwalk/scripts/$script"
        echo "  Updated $script"
    fi
done

# Also push run_pipeline.sh
if [ -f "docker/gpu/run_pipeline.sh" ]; then
    scp -o ConnectTimeout=10 "docker/gpu/run_pipeline.sh" "$GPU_HOST:/mnt/splatwalk/scripts/run_pipeline.sh"
    ssh_gpu "chmod +x /mnt/splatwalk/scripts/run_pipeline.sh"
    echo "  Updated run_pipeline.sh"
fi

echo "Scripts updated."

# ─── Step 3: Main pipeline + scoring loop ────────────────────────────────────

echo ""
echo "=== Step 3: Running pipeline with quality iteration ==="

ITERATION=0
GROUND_SCORE=0

while [ "$ITERATION" -lt "$MAX_ITERATIONS" ]; do
    ITERATION=$((ITERATION + 1))
    OUTPUT_DIR="${OUTPUT_BASE}/iter${ITERATION}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ITERATION $ITERATION / $MAX_ITERATIONS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Output: $OUTPUT_DIR"
    echo "  Train iters: $TRAIN_ITERATIONS"
    echo "  Descent altitudes: $DESCENT_ALTITUDES"
    echo "  FLUX denoising: $FLUX_DENOISING"
    echo "  Ground retrain: $GROUND_RETRAIN_ITERS iters"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    notify_slack "Iteration $ITERATION/$MAX_ITERATIONS starting (train=$TRAIN_ITERATIONS, ground_retrain=$GROUND_RETRAIN_ITERS)"

    # ── 3a: Run the walkable pipeline ─────────────────────────────────────

    echo "Running walkable pipeline..."
    notify_slack "Stage 1-2: Training aerial splat ($TRAIN_ITERATIONS iterations)..."

    ssh_gpu_long << PIPELINE_EOF
set -e
export PATH="/mnt/splatwalk/conda/bin:\$PATH"
export HF_HOME="/mnt/splatwalk/models/flux"
export TRANSFORMERS_CACHE="/mnt/splatwalk/models/transformers"
export GEMINI_API_KEY='${GEMINI_API_KEY}'
export GOOGLE_MAPS_API_KEY='${GOOGLE_MAPS_API_KEY}'
export HF_TOKEN='${HUGGINGFACE_TOKEN}'

INPUT_DIR="$INPUT_DIR"
OUTPUT_DIR="$OUTPUT_DIR"
SCENE_DIR="\$OUTPUT_DIR/scene"

mkdir -p "\$SCENE_DIR/images" "\$OUTPUT_DIR"

# ── Preprocess images (512x512) ──
echo "Preprocessing images..."
python3 << 'PREPROCESS_PY'
import os
from PIL import Image
from pathlib import Path

input_dir = "$INPUT_DIR"
output_dir = "$OUTPUT_DIR/scene/images"
target_size = (512, 512)
os.makedirs(output_dir, exist_ok=True)

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
            img.save(os.path.join(output_dir, img_file.stem + '.jpg'), 'JPEG', quality=95)
        except Exception as e:
            print(f"  Error: {img_file.name}: {e}")
PREPROCESS_PY

NUM_IMAGES=\$(ls -1 "\$SCENE_DIR/images/"*.jpg 2>/dev/null | wc -l)
echo "Prepared \$NUM_IMAGES images"

# ── Stage 1: Geometry init ──
cd /mnt/splatwalk/InstantSplat
echo "Stage 1: MASt3R geometry init..."
python init_geo.py \
    --source_path "\$SCENE_DIR" \
    --model_path "\$OUTPUT_DIR/instantsplat" \
    --n_views $MAX_N_VIEWS \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking

# ── Stage 2: Train aerial splat ──
echo "Stage 2: Training aerial splat ($TRAIN_ITERATIONS iterations)..."
python train.py \
    --source_path "\$SCENE_DIR" \
    --model_path "\$OUTPUT_DIR/instantsplat" \
    --iterations $TRAIN_ITERATIONS \
    --n_views $MAX_N_VIEWS \
    --pp_optimizer \
    --optim_pose

# ── EXIF GPS extraction ──
echo "Extracting EXIF GPS..."
EXIF_RESULT=\$(python3 -c "
from pathlib import Path
from PIL import Image, ExifTags
for f in sorted(Path('$INPUT_DIR').glob('*')):
    if f.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
    try:
        img = Image.open(f)
        exif = img._getexif()
        if not exif: continue
        gps = exif.get(ExifTags.Base.GPSInfo)
        if not gps: continue
        tags = {}
        for k,v in gps.items(): tags[ExifTags.GPSTAGS.get(k,k)] = v
        if 'GPSLatitude' not in tags: continue
        def dms(d,r):
            v=float(d[0])+float(d[1])/60+float(d[2])/3600
            return -v if r in ('S','W') else v
        lat=dms(tags['GPSLatitude'],tags.get('GPSLatitudeRef','N'))
        lon=dms(tags['GPSLongitude'],tags.get('GPSLongitudeRef','W'))
        alt=float(tags.get('GPSAltitude',0))
        print(f'{lat:.6f},{lon:.6f},{alt:.1f}')
        break
    except: continue
print('')
" 2>/dev/null || echo "")

COORDS=""
DRONE_AGL=0
if [ -n "\$EXIF_RESULT" ]; then
    IFS=',' read -r LAT LON ALT <<< "\$EXIF_RESULT"
    if [ -n "\$LAT" ] && [ -n "\$LON" ]; then
        COORDS="\$LAT,\$LON"
        echo "  GPS: \$COORDS, alt: \${ALT}m"
        # Compute AGL
        GROUND_ELEV=\$(curl -s --max-time 5 \
            "https://api.open-meteo.com/v1/elevation?latitude=\${LAT}&longitude=\${LON}" \
            | python3 -c "import sys,json;d=json.load(sys.stdin);e=d.get('elevation',0);print(e[0] if isinstance(e,list) else e)" 2>/dev/null || echo "0")
        if [ "\$GROUND_ELEV" != "0" ] && [ "\$ALT" != "0" ]; then
            DRONE_AGL=\$(python3 -c "print(max(0, \${ALT} - \${GROUND_ELEV}))")
            echo "  Ground elev: \${GROUND_ELEV}m, AGL: \${DRONE_AGL}m"
        fi
    fi
fi

# ── Stage 3: Iterative descent ──
echo "Stage 3: Iterative descent..."
DESCENT_ARGS=(
    --model_path "\$OUTPUT_DIR/instantsplat"
    --scene_path "\$SCENE_DIR"
    --output_dir "\$OUTPUT_DIR/descent"
    --altitudes "$DESCENT_ALTITUDES"
    --retrain_iterations $DESCENT_RETRAIN_ITERS
    --slack_webhook_url "$SLACK_WEBHOOK_URL"
    --job_id "iter$ITERATION"
)
[ "\$DRONE_AGL" != "0" ] && DESCENT_ARGS+=(--drone_agl "\$DRONE_AGL")

python /mnt/splatwalk/scripts/render_descent.py "\${DESCENT_ARGS[@]}"

# ── Stage 3.5: FLUX ground view generation ──
echo "Stage 3.5: FLUX ground view generation..."
GROUND_ARGS=(
    --model_path "\$OUTPUT_DIR/descent/final"
    --scene_path "\$SCENE_DIR"
    --output_dir "\$OUTPUT_DIR/ground_views"
    --num_perimeter_views $GROUND_VIEWS_PERIMETER
    --num_grid_views $GROUND_VIEWS_GRID
    --denoising_strength $FLUX_DENOISING
    --controlnet_scale $FLUX_CONTROLNET_SCALE
    --output_size $FLUX_OUTPUT_SIZE
    --retrain_iterations $GROUND_RETRAIN_ITERS
    --slack_webhook_url "$SLACK_WEBHOOK_URL"
    --job_id "iter$ITERATION"
)
[ -n "\$COORDS" ] && GROUND_ARGS+=(--coordinates "\$COORDS")

python /mnt/splatwalk/scripts/generate_ground_views.py "\${GROUND_ARGS[@]}"

# ── Stage 4: Quality gate ──
echo "Stage 4: Quality gate..."
python /mnt/splatwalk/scripts/quality_gate.py \
    --model_path "\$OUTPUT_DIR/ground_views/model" \
    --real_images "\$SCENE_DIR/images" \
    --output_dir "\$OUTPUT_DIR/walkable" \
    || {
        echo "Quality gate failed (non-fatal), using model directly"
        mkdir -p "\$OUTPUT_DIR/walkable"
        cp -r "\$OUTPUT_DIR/ground_views/model"/* "\$OUTPUT_DIR/walkable/" 2>/dev/null || true
    }

# ── Compress to .splat ──
echo "Compressing to .splat..."
PLY_PATH=""
for search_dir in "\$OUTPUT_DIR/walkable" "\$OUTPUT_DIR/ground_views/model" "\$OUTPUT_DIR/descent/final" "\$OUTPUT_DIR/instantsplat"; do
    PLY_PATH=\$(find "\$search_dir" -path "*/point_cloud/iteration_*/point_cloud.ply" -type f 2>/dev/null | sort -t_ -k2 -n -r | head -1)
    [ -n "\$PLY_PATH" ] && [ -f "\$PLY_PATH" ] && break
done

if [ -n "\$PLY_PATH" ]; then
    CONF_NPY="\$OUTPUT_DIR/walkable/per_gaussian_confidence.npy"
    python /mnt/splatwalk/scripts/compress_splat.py \
        "\$PLY_PATH" "\$OUTPUT_DIR/output.splat" \
        --prune_ratio $PRUNE_RATIO \
        --scene_scale $SCENE_SCALE \
        --confidence_npy "\$CONF_NPY"
    echo "Compressed: \$OUTPUT_DIR/output.splat"
else
    echo "ERROR: No PLY file found for compression"
    exit 1
fi

echo "Pipeline iteration $ITERATION complete."
PIPELINE_EOF

    echo "Pipeline iteration $ITERATION finished."
    notify_slack "Pipeline iteration $ITERATION complete. Uploading splat and scoring..." "success"

    # ── 3b: Upload .splat to Spaces ───────────────────────────────────────

    echo "Uploading .splat to Spaces..."

    TIMESTAMP=$(date +%s)
    SPLAT_KEY="scenes/aukerman/iter${ITERATION}/scene.splat"
    SPLAT_URL="${SPACES_CDN}/scenes/aukerman/iter${ITERATION}/scene.splat"

    ssh_gpu << UPLOAD_EOF
export AWS_ACCESS_KEY_ID="${DO_SPACES_KEY}"
export AWS_SECRET_ACCESS_KEY="${DO_SPACES_SECRET}"
aws s3 cp "$OUTPUT_DIR/output.splat" "s3://${DO_SPACES_BUCKET}/${SPLAT_KEY}" \
    --endpoint-url "${DO_SPACES_ENDPOINT}" \
    --acl public-read \
    --content-type "application/octet-stream"
echo "Uploaded: ${SPLAT_URL}"
UPLOAD_EOF

    VIEWER_LINK="${VIEWER_URL}?url=${SPLAT_URL}"
    notify_slack "Splat uploaded: <${VIEWER_LINK}|View in browser>"

    # ── 3c: Score with Gemini ─────────────────────────────────────────────

    echo "Scoring renders with Gemini..."

    # Determine the best model path for scoring
    SCORE_MODEL_PATH="$OUTPUT_DIR/ground_views/model"

    ssh_gpu_long << SCORE_EOF
set -e
export PATH="/mnt/splatwalk/conda/bin:\$PATH"
export HF_HOME="/mnt/splatwalk/models/flux"

python /mnt/splatwalk/scripts/score_renders.py \
    --model_path "$SCORE_MODEL_PATH" \
    --scene_path "$OUTPUT_DIR/scene" \
    --output_dir "$OUTPUT_DIR/scores" \
    --gemini_api_key "$GEMINI_API_KEY" \
    --slack_webhook_url "$SLACK_WEBHOOK_URL" \
    --spaces_key "$DO_SPACES_KEY" \
    --spaces_secret "$DO_SPACES_SECRET" \
    --spaces_bucket "$DO_SPACES_BUCKET" \
    --spaces_endpoint "$DO_SPACES_ENDPOINT" \
    --viewer_base_url "${VIEWER_LINK}" \
    || true  # Don't fail the whole script if scoring fails

echo "SCORING_DONE"
SCORE_EOF

    # Read the score report
    REPORT_JSON=$(ssh_gpu "cat '$OUTPUT_DIR/scores/score_report.json' 2>/dev/null" || echo '{}')
    GROUND_SCORE=$(echo "$REPORT_JSON" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('ground_average', 0))
except:
    print(0)
" 2>/dev/null || echo "0")

    AVG_SCORE=$(echo "$REPORT_JSON" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('average_score', 0))
except:
    print(0)
" 2>/dev/null || echo "0")

    echo ""
    echo "  Iteration $ITERATION scores:"
    echo "    Average: $AVG_SCORE / 10"
    echo "    Ground:  $GROUND_SCORE / 10"
    echo ""

    # Check if we hit the target
    TARGET_MET=$(python3 -c "print(1 if float('$GROUND_SCORE') >= $TARGET_SCORE else 0)" 2>/dev/null || echo "0")

    if [ "$TARGET_MET" = "1" ]; then
        echo "TARGET REACHED! Ground score: $GROUND_SCORE >= $TARGET_SCORE"
        notify_slack "TARGET REACHED! Ground-level score: $GROUND_SCORE/10 after $ITERATION iterations! <${VIEWER_LINK}|View splat>" "success"
        break
    fi

    # If not the last iteration, tune parameters for next round
    if [ "$ITERATION" -lt "$MAX_ITERATIONS" ]; then
        echo "Ground score $GROUND_SCORE < $TARGET_SCORE, tuning parameters for next iteration..."

        # Extract Gemini suggestions to inform tuning
        SUGGESTIONS=$(echo "$REPORT_JSON" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for s in d.get('scores', []):
        if s.get('fraction', 1) <= 0.05:
            g = s.get('gemini', {})
            print(g.get('suggestions', ''))
except:
    pass
" 2>/dev/null || echo "")

        notify_slack "Iteration $ITERATION: ground=$GROUND_SCORE/10. Tuning params for retry $((ITERATION+1))..."

        # Iteration 2: More training, more ground views, higher FLUX fidelity
        if [ "$ITERATION" -eq 1 ]; then
            TRAIN_ITERATIONS=12000
            DESCENT_RETRAIN_ITERS=3000
            GROUND_RETRAIN_ITERS=10000
            GROUND_VIEWS_PERIMETER=28
            GROUND_VIEWS_GRID=16
            FLUX_DENOISING=0.80
            FLUX_CONTROLNET_SCALE=0.5
            FLUX_OUTPUT_SIZE=512
            PRUNE_RATIO=0.15
            echo "  Iter 2 params: train=12K, ground_retrain=10K, views=44, denoising=0.80, prune=15%"
        fi

        # Iteration 3: Maximum quality settings
        if [ "$ITERATION" -eq 2 ]; then
            TRAIN_ITERATIONS=15000
            DESCENT_RETRAIN_ITERS=4000
            GROUND_RETRAIN_ITERS=12000
            GROUND_VIEWS_PERIMETER=32
            GROUND_VIEWS_GRID=20
            FLUX_DENOISING=0.75
            FLUX_CONTROLNET_SCALE=0.55
            FLUX_OUTPUT_SIZE=512
            PRUNE_RATIO=0.10
            DESCENT_ALTITUDES="0.75,0.50,0.30,0.15,0.08,0.04,0.02"
            echo "  Iter 3 params: train=15K, 7 descent levels, ground_retrain=12K, views=52, prune=10%"
        fi
    fi
done

# ─── Step 4: Final summary ───────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SESSION COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Iterations: $ITERATION"
echo "  Final ground score: $GROUND_SCORE / 10"
echo "  Target: $TARGET_SCORE / 10"
echo "  Splat URL: $SPLAT_URL"
echo "  Viewer: $VIEWER_LINK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$TARGET_MET" = "1" ]; then
    notify_slack "Session complete! Ground-level: $GROUND_SCORE/10 after $ITERATION iterations.\n<${VIEWER_LINK}|View final splat>" "success"
else
    notify_slack "Session ended after $ITERATION iterations. Best ground score: $GROUND_SCORE/10.\n<${VIEWER_LINK}|View splat>" "info"
fi

# ─── Step 5: Destroy GPU droplet ─────────────────────────────────────────────

echo ""
echo "=== Step 5: Destroying GPU droplet ==="

# Get droplet ID
DROPLET_ID=$(curl -s -X GET \
    -H "Authorization: Bearer ${DO_API_TOKEN}" \
    -H "Content-Type: application/json" \
    "https://api.digitalocean.com/v2/droplets" \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
for d in data.get('droplets', []):
    nets = d.get('networks', {}).get('v4', [])
    for n in nets:
        if n.get('ip_address') == '$GPU_IP':
            print(d['id'])
            break
" 2>/dev/null || echo "")

if [ -n "$DROPLET_ID" ]; then
    echo "Detaching volume..."
    curl -s -X POST \
        -H "Authorization: Bearer ${DO_API_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"type\":\"detach\",\"droplet_id\":$DROPLET_ID}" \
        "https://api.digitalocean.com/v2/volumes/${DO_VOLUME_ID:-}/actions" > /dev/null 2>&1 || true
    sleep 10

    echo "Destroying droplet $DROPLET_ID..."
    curl -s -X DELETE \
        -H "Authorization: Bearer ${DO_API_TOKEN}" \
        "https://api.digitalocean.com/v2/droplets/$DROPLET_ID" > /dev/null 2>&1
    echo "Droplet destroyed."
    notify_slack "GPU droplet $DROPLET_ID destroyed." "success"
else
    echo "Could not find droplet ID for $GPU_IP — may need manual cleanup"
    notify_slack "Could not auto-destroy droplet. Manual cleanup needed." "error"
fi

echo ""
echo "Done!"
