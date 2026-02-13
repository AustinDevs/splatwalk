#!/bin/bash
set -e
export PATH="/opt/conda/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
shopt -s nocaseglob  # case-insensitive globbing (matches .JPG, .jpg, .Jpg, etc.)

# Environment variables expected:
# INPUT_DIR - Directory containing input images
# OUTPUT_DIR - Directory for output files
# PIPELINE_MODE - viewcrafter, instantsplat, combined, or walkable
# JOB_ID - Job identifier
# SPACES_KEY - DO Spaces access key
# SPACES_SECRET - DO Spaces secret key
# SPACES_BUCKET - DO Spaces bucket name
# SPACES_REGION - DO Spaces region
# SPACES_ENDPOINT - DO Spaces endpoint
# SLACK_WEBHOOK_URL - (optional) Slack incoming webhook for progress notifications
# MAX_N_VIEWS - (optional) Max views for MASt3R, default 24 (use 16 for 20GB GPUs)
# VIEWCRAFTER_BATCH_SIZE - (optional) Frames per ViewCrafter run, default 10 (use 5 for 20GB GPUs)
# TRAIN_ITERATIONS - (optional) Stage 2 training iterations, default 7000

# --- Fetch latest Python scripts from GitHub (allows hotfixes without Docker rebuild) ---
echo "Updating pipeline scripts from GitHub..."
_CURL_ARGS=(-fsSL -H "Accept: application/vnd.github.raw")
[ -n "$GITHUB_TOKEN" ] && _CURL_ARGS+=(-H "Authorization: token $GITHUB_TOKEN")
_BASE_URL="https://api.github.com/repos/AustinDevs/splatwalk/contents/docker/gpu"
for _script in render_descent.py enhance_with_viewcrafter.py quality_gate.py convert_to_ksplat.py; do
    if curl "${_CURL_ARGS[@]}" "$_BASE_URL/$_script" -o "/opt/$_script" 2>/dev/null; then
        echo "  Updated $_script"
    else
        echo "  Using baked-in $_script (fetch failed)"
    fi
done

echo "=========================================="
echo "SplatWalk GPU Pipeline"
echo "=========================================="
echo "Pipeline: $PIPELINE_MODE"
echo "Job ID: $JOB_ID"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# --- Slack notification helper ---
notify_slack() {
    local message="$1"
    local status="${2:-info}"  # info, success, error

    if [ -z "$SLACK_WEBHOOK_URL" ]; then return; fi

    local emoji=":arrows_counterclockwise:"
    [ "$status" = "success" ] && emoji=":white_check_mark:"
    [ "$status" = "error" ] && emoji=":x:"

    local payload="{\"text\":\"${emoji} *[${JOB_ID:0:8}] ${PIPELINE_MODE}* -- ${message}\"}"
    curl -s -X POST -H 'Content-type: application/json' \
        --data "$payload" "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 &
}

# Validate required environment variables
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$PIPELINE_MODE" ] || [ -z "$JOB_ID" ]; then
    echo "Error: Missing required environment variables"
    echo "Required: INPUT_DIR, OUTPUT_DIR, PIPELINE_MODE, JOB_ID"
    exit 1
fi

if [ -z "$SPACES_KEY" ] || [ -z "$SPACES_SECRET" ] || [ -z "$SPACES_BUCKET" ]; then
    echo "Error: Missing Spaces credentials"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Configure AWS CLI for DO Spaces
export AWS_ACCESS_KEY_ID="$SPACES_KEY"
export AWS_SECRET_ACCESS_KEY="$SPACES_SECRET"
export AWS_DEFAULT_REGION="$SPACES_REGION"

# Detect video input
VIDEO_FILE=$(ls -1 "$INPUT_DIR"/*.mov "$INPUT_DIR"/*.mp4 2>/dev/null | head -1)
if [ -n "$VIDEO_FILE" ]; then
    echo "Found video input: $VIDEO_FILE"
    HAS_VIDEO=1
else
    HAS_VIDEO=0
    # Count input images
    IMAGE_COUNT=$(ls -1 "$INPUT_DIR"/*.jpg 2>/dev/null | wc -l)
    echo "Found $IMAGE_COUNT input images"

    if [ "$IMAGE_COUNT" -lt 2 ]; then
        echo "Error: Need at least 2 input images"
        exit 1
    fi
fi

# Create manifest template
create_manifest() {
    local status="$1"
    local splat_url="${2:-}"
    local thumbnail_url="${3:-}"
    local error="${4:-}"

    cat > "$OUTPUT_DIR/manifest.json" << EOF
{
    "jobId": "$JOB_ID",
    "pipeline": "$PIPELINE_MODE",
    "status": "$status",
    "splatUrl": "$splat_url",
    "thumbnailUrl": "$thumbnail_url",
    "error": "$error",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

# Upload to Spaces
upload_to_spaces() {
    local local_path="$1"
    local remote_key="$2"
    local content_type="${3:-application/octet-stream}"

    # Redirect progress output to stderr so only URL is captured in variable
    aws s3 cp "$local_path" "s3://$SPACES_BUCKET/$remote_key" \
        --endpoint-url "$SPACES_ENDPOINT" \
        --acl public-read \
        --content-type "$content_type" >&2

    echo "$SPACES_ENDPOINT/$SPACES_BUCKET/$remote_key"
}

run_viewcrafter() {
    echo "Running ViewCrafter pipeline..."

    cd /opt/ViewCrafter

    # Generate novel views from input images
    python inference.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR/viewcrafter" \
        --num_views 30 \
        --save_video \
        || return 1

    # Convert frames to 3DGS using DUSt3R
    cd /opt/dust3r
    python demo.py \
        --images "$OUTPUT_DIR/viewcrafter/frames" \
        --output "$OUTPUT_DIR/viewcrafter/pointcloud.ply" \
        || return 1

    echo "ViewCrafter pipeline completed"
    return 0
}

run_instantsplat() {
    echo "Running InstantSplat pipeline..."

    # Set up directory structure
    local scene_dir="$OUTPUT_DIR/scene"
    mkdir -p "$scene_dir/images"

    # If video input, extract frames with ffmpeg first
    if [ "$HAS_VIDEO" -eq 1 ]; then
        echo "Extracting frames from video: $VIDEO_FILE"
        local frames_dir="$INPUT_DIR/frames"
        mkdir -p "$frames_dir"
        ffmpeg -i "$VIDEO_FILE" -qscale:v 1 -vf "fps=1" "$frames_dir/frame-%04d.jpg" 2>&1
        local extracted=$(ls -1 "$frames_dir"/*.jpg 2>/dev/null | wc -l)
        echo "Extracted $extracted frames from video (1 fps)"
        # Point preprocessing at the extracted frames directory
        export INPUT_DIR="$frames_dir"
    fi

    # Preprocess images: resize all to consistent dimensions (512x512)
    # InstantSplat requires all images to have the same dimensions
    echo "Preprocessing images to consistent 512x512 size..."
    python3 << 'PREPROCESS_SCRIPT'
import os
import sys
from PIL import Image
from pathlib import Path

input_dir = os.environ.get('INPUT_DIR', '/data/input')
output_dir = os.path.join(os.environ.get('OUTPUT_DIR', '/data/output'), 'scene', 'images')
target_size = (512, 512)

os.makedirs(output_dir, exist_ok=True)

for img_file in sorted(Path(input_dir).glob('*')):
    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        try:
            img = Image.open(img_file).convert('RGB')
            # Resize with center crop to maintain aspect ratio quality
            # First resize so smaller dimension is target, then center crop
            w, h = img.size
            scale = max(target_size[0] / w, target_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            # Center crop to target size
            left = (new_w - target_size[0]) // 2
            top = (new_h - target_size[1]) // 2
            img = img.crop((left, top, left + target_size[0], top + target_size[1]))
            # Save as JPG
            out_path = os.path.join(output_dir, img_file.stem + '.jpg')
            img.save(out_path, 'JPEG', quality=95)
            print(f"  Processed: {img_file.name} -> {target_size[0]}x{target_size[1]}")
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}", file=sys.stderr)
PREPROCESS_SCRIPT

    local num_images=$(ls -1 "$scene_dir/images" | wc -l)
    echo "Prepared $num_images images (all 512x512)"

    cd /opt/InstantSplat

    # Cap views for MASt3R memory (configurable via MAX_N_VIEWS)
    local max_views=${MAX_N_VIEWS:-24}
    local n_views=$num_images
    if [ "$n_views" -gt "$max_views" ]; then
        n_views=$max_views
    fi

    # Step 1: Initialize geometry with init_geo.py (creates point cloud and camera poses)
    # This is the correct script for inference - init_test_pose.py is for evaluation with GT
    echo "Step 1/2: Running geometry initialization with MASt3R..."
    python init_geo.py \
        --source_path "$scene_dir" \
        --model_path "$OUTPUT_DIR/instantsplat" \
        --n_views "$n_views" \
        --focal_avg \
        --co_vis_dsp \
        --conf_aware_ranking \
        2>&1 || {
            echo "init_geo.py failed"
            ls -la "$scene_dir/sparse_"*/0/ 2>/dev/null || echo "No sparse output"
            return 1
        }

    # Step 2: Train Gaussian Splatting with pose optimization
    echo "Step 2/2: Training Gaussian Splatting..."
    python train.py \
        --source_path "$scene_dir" \
        --model_path "$OUTPUT_DIR/instantsplat" \
        --iterations 3000 \
        --n_views "$n_views" \
        --pp_optimizer \
        --optim_pose \
        || return 1

    echo "InstantSplat pipeline completed"
    return 0
}

run_combined() {
    echo "Running Combined (MASt3R) pipeline..."

    cd /opt/mast3r

    # Run MASt3R for dense reconstruction
    python demo.py \
        --images "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR/combined" \
        --model_name MASt3R_ViTLarge \
        || return 1

    echo "Combined pipeline completed"
    return 0
}

run_walkable() {
    echo "Running Walkable Splat pipeline (Stages 1-5)..."
    notify_slack "Starting walkable pipeline with $(ls -1 "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.jpeg "$INPUT_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ') images"

    # === STAGES 1-2: Aerial splat (reuses InstantSplat flow) ===
    local scene_dir="$OUTPUT_DIR/scene"
    mkdir -p "$scene_dir/images"

    # If video input, extract frames
    if [ "$HAS_VIDEO" -eq 1 ]; then
        echo "Extracting frames from video: $VIDEO_FILE"
        local frames_dir="$INPUT_DIR/frames"
        mkdir -p "$frames_dir"
        ffmpeg -i "$VIDEO_FILE" -qscale:v 1 -vf "fps=1" "$frames_dir/frame-%04d.jpg" 2>&1
        export INPUT_DIR="$frames_dir"
    fi

    # Preprocess images to 512x512
    echo "Preprocessing images to consistent 512x512 size..."
    python3 << 'PREPROCESS_SCRIPT'
import os, sys
from PIL import Image
from pathlib import Path

input_dir = os.environ.get('INPUT_DIR', '/data/input')
output_dir = os.path.join(os.environ.get('OUTPUT_DIR', '/data/output'), 'scene', 'images')
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
            out_path = os.path.join(output_dir, img_file.stem + '.jpg')
            img.save(out_path, 'JPEG', quality=95)
            print(f"  Processed: {img_file.name} -> {target_size[0]}x{target_size[1]}")
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}", file=sys.stderr)
PREPROCESS_SCRIPT

    local num_images=$(ls -1 "$scene_dir/images" | wc -l)
    echo "Prepared $num_images images (all 512x512)"

    cd /opt/InstantSplat

    # Cap views for MASt3R memory (24 for H100 80GB, 16 for RTX 4000 20GB)
    local max_views=${MAX_N_VIEWS:-24}
    local n_views=$num_images
    if [ "$n_views" -gt "$max_views" ]; then
        n_views=$max_views
    fi
    local train_iters=${TRAIN_ITERATIONS:-7000}

    # Stage 1: Geometry init
    notify_slack "Stage 1: Geometry init (MASt3R, $n_views views)..."
    echo "Stage 1/5: Running geometry initialization with MASt3R ($n_views views)..."
    python init_geo.py \
        --source_path "$scene_dir" \
        --model_path "$OUTPUT_DIR/instantsplat" \
        --n_views "$n_views" \
        --focal_avg \
        --co_vis_dsp \
        --conf_aware_ranking \
        2>&1 || {
            notify_slack "Failed at Stage 1: init_geo.py" "error"
            return 1
        }
    notify_slack "Stage 1 complete: point cloud generated"

    # Stage 2: Train aerial splat
    notify_slack "Stage 2: Training aerial splat ($train_iters iterations)..."
    echo "Stage 2/5: Training Gaussian Splatting ($train_iters iterations)..."
    python train.py \
        --source_path "$scene_dir" \
        --model_path "$OUTPUT_DIR/instantsplat" \
        --iterations "$train_iters" \
        --n_views "$n_views" \
        --pp_optimizer \
        --optim_pose \
        || {
            notify_slack "Failed at Stage 2: train.py" "error"
            return 1
        }
    notify_slack "Stage 2 complete: aerial splat trained"

    # Stage 3: Iterative Virtual Descent
    notify_slack "Stage 3: Virtual descent (5 altitude levels)..."
    echo "Stage 3/5: Iterative virtual descent..."
    python /opt/render_descent.py \
        --model_path "$OUTPUT_DIR/instantsplat" \
        --scene_path "$scene_dir" \
        --output_dir "$OUTPUT_DIR/descent" \
        --altitudes "0.75,0.5,0.25,0.1,0.025" \
        --retrain_iterations 2000 \
        --slack_webhook_url "${SLACK_WEBHOOK_URL:-}" \
        --job_id "${JOB_ID:-}" \
        || {
            notify_slack "Failed at Stage 3: render_descent.py" "error"
            return 1
        }
    notify_slack "Stage 3 complete: descended to ground level"

    # Stage 4: Diffusion Enhancement (ViewCrafter) — optional, non-fatal
    notify_slack "Stage 4: Diffusion enhancement (ViewCrafter)..."
    echo "Stage 4/5: ViewCrafter diffusion enhancement..."
    local final_model_path="$OUTPUT_DIR/descent/final"
    if python /opt/enhance_with_viewcrafter.py \
        --model_path "$OUTPUT_DIR/descent/final" \
        --scene_path "$scene_dir" \
        --output_dir "$OUTPUT_DIR/enhanced" \
        --viewcrafter_ckpt "/opt/ViewCrafter/checkpoints/ViewCrafter_25_512" \
        --batch_size "${VIEWCRAFTER_BATCH_SIZE:-10}" 2>&1; then
        notify_slack "Stage 4 complete: ground views enhanced"
        final_model_path="$OUTPUT_DIR/enhanced"
    else
        notify_slack "Stage 4 skipped: ViewCrafter enhancement failed (non-fatal), using Stage 3 output" "info"
        echo "WARNING: ViewCrafter enhancement failed, continuing with Stage 3 descent output"
    fi

    # Stage 5: Quality Gating
    echo "Stage 5/5: Quality gate confidence scoring..."
    python /opt/quality_gate.py \
        --model_path "$final_model_path" \
        --real_images "$scene_dir/images" \
        --output_dir "$OUTPUT_DIR/walkable" \
        || {
            notify_slack "Stage 5 failed (non-fatal), using model directly" "info"
            echo "WARNING: Quality gate failed, copying model directly"
            mkdir -p "$OUTPUT_DIR/walkable"
            cp -r "$final_model_path"/* "$OUTPUT_DIR/walkable/" 2>/dev/null || true
        }
    notify_slack "Stage 5 complete"

    echo "Walkable Splat pipeline completed"
    return 0
}

convert_and_upload() {
    local ply_path="$1"
    local pipeline_name="$2"

    echo "Converting PLY to .splat..."

    # Find the PLY file
    if [ ! -f "$ply_path" ]; then
        # Try to find it
        ply_path=$(find "$OUTPUT_DIR" -name "*.ply" -type f | head -1)
    fi

    if [ -z "$ply_path" ] || [ ! -f "$ply_path" ]; then
        echo "Error: No PLY file found"
        return 1
    fi

    local splat_path="$OUTPUT_DIR/output.splat"

    python /opt/convert_to_ksplat.py "$ply_path" "$splat_path" || return 1

    if [ ! -f "$splat_path" ]; then
        echo "Error: .splat conversion failed"
        return 1
    fi

    echo "Uploading results to Spaces..."

    # Upload KSPLAT
    local splat_key="jobs/$JOB_ID/output/$pipeline_name/scene.splat"
    local splat_url=$(upload_to_spaces "$splat_path" "$splat_key" "application/octet-stream")

    # Upload thumbnail if available
    local thumb_url=""
    local thumb_path=$(find "$OUTPUT_DIR" -name "thumbnail.jpg" -o -name "preview.jpg" | head -1)
    if [ -f "$thumb_path" ]; then
        local thumb_key="jobs/$JOB_ID/output/$pipeline_name/thumbnail.jpg"
        thumb_url=$(upload_to_spaces "$thumb_path" "$thumb_key" "image/jpeg")
    fi

    # Create and upload manifest
    create_manifest "success" "$splat_url" "$thumb_url"
    local manifest_key="jobs/$JOB_ID/output/$pipeline_name/manifest.json"
    upload_to_spaces "$OUTPUT_DIR/manifest.json" "$manifest_key" "application/json"

    echo "Upload complete: $splat_url"
    return 0
}

# Main execution
echo "Starting $PIPELINE_MODE pipeline..."

case "$PIPELINE_MODE" in
    "viewcrafter")
        if run_viewcrafter; then
            convert_and_upload "$OUTPUT_DIR/viewcrafter/pointcloud.ply" "viewcrafter"
        else
            create_manifest "failed" "" "" "ViewCrafter processing failed"
            upload_to_spaces "$OUTPUT_DIR/manifest.json" "jobs/$JOB_ID/output/viewcrafter/manifest.json" "application/json"
            exit 1
        fi
        ;;
    "instantsplat")
        if run_instantsplat; then
            convert_and_upload "$OUTPUT_DIR/instantsplat/point_cloud/iteration_3000/point_cloud.ply" "instantsplat"
        else
            create_manifest "failed" "" "" "InstantSplat processing failed"
            upload_to_spaces "$OUTPUT_DIR/manifest.json" "jobs/$JOB_ID/output/instantsplat/manifest.json" "application/json"
            exit 1
        fi
        ;;
    "combined")
        if run_combined; then
            convert_and_upload "$OUTPUT_DIR/combined/pointcloud.ply" "combined"
        else
            create_manifest "failed" "" "" "Combined pipeline processing failed"
            upload_to_spaces "$OUTPUT_DIR/manifest.json" "jobs/$JOB_ID/output/combined/manifest.json" "application/json"
            exit 1
        fi
        ;;
    "walkable")
        if run_walkable; then
            notify_slack "Uploading KSPLAT to Spaces..."
            # Quality gate outputs the final model; find the PLY
            local_ply="$OUTPUT_DIR/walkable/point_cloud/point_cloud.ply"
            if [ ! -f "$local_ply" ]; then
                local_ply=$(find "$OUTPUT_DIR/walkable" "$OUTPUT_DIR/enhanced" "$OUTPUT_DIR/descent/final" "$OUTPUT_DIR/instantsplat" -name "*.ply" -type f 2>/dev/null | head -1)
            fi
            if convert_and_upload "$local_ply" "walkable"; then
                # Append confidence metadata to manifest if available
                if [ -f "$OUTPUT_DIR/walkable/confidence.json" ]; then
                    python3 -c "
import json
with open('$OUTPUT_DIR/manifest.json') as f: m = json.load(f)
with open('$OUTPUT_DIR/walkable/confidence.json') as f: c = json.load(f)
m['confidence'] = c
with open('$OUTPUT_DIR/manifest.json', 'w') as f: json.dump(m, f, indent=2)
"
                    upload_to_spaces "$OUTPUT_DIR/manifest.json" "jobs/$JOB_ID/output/walkable/manifest.json" "application/json"
                fi
                notify_slack "Pipeline complete! Splat URL: $SPACES_ENDPOINT/$SPACES_BUCKET/jobs/$JOB_ID/output/walkable/scene.splat" "success"
            else
                create_manifest "failed" "" "" "Walkable pipeline conversion/upload failed"
                upload_to_spaces "$OUTPUT_DIR/manifest.json" "jobs/$JOB_ID/output/walkable/manifest.json" "application/json"
                notify_slack "Pipeline failed at upload/conversion stage" "error"
                exit 1
            fi
        else
            create_manifest "failed" "" "" "Walkable pipeline processing failed"
            upload_to_spaces "$OUTPUT_DIR/manifest.json" "jobs/$JOB_ID/output/walkable/manifest.json" "application/json"
            notify_slack "Pipeline failed" "error"
            exit 1
        fi
        ;;
    *)
        echo "Error: Unknown pipeline mode: $PIPELINE_MODE"
        exit 1
        ;;
esac

notify_slack "Droplet self-destructing"
echo "Pipeline $PIPELINE_MODE completed successfully!"
exit 0
