#!/bin/bash
set -e

# Environment variables expected:
# INPUT_DIR - Directory containing input images
# OUTPUT_DIR - Directory for output files
# PIPELINE_MODE - viewcrafter, instantsplat, or combined
# JOB_ID - Job identifier
# SPACES_KEY - DO Spaces access key
# SPACES_SECRET - DO Spaces secret key
# SPACES_BUCKET - DO Spaces bucket name
# SPACES_REGION - DO Spaces region
# SPACES_ENDPOINT - DO Spaces endpoint

echo "=========================================="
echo "SplatWalk GPU Pipeline"
echo "=========================================="
echo "Pipeline: $PIPELINE_MODE"
echo "Job ID: $JOB_ID"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

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

# Count input images
IMAGE_COUNT=$(ls -1 "$INPUT_DIR"/*.jpg 2>/dev/null | wc -l)
echo "Found $IMAGE_COUNT input images"

if [ "$IMAGE_COUNT" -lt 2 ]; then
    echo "Error: Need at least 2 input images"
    exit 1
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

    aws s3 cp "$local_path" "s3://$SPACES_BUCKET/$remote_key" \
        --endpoint-url "$SPACES_ENDPOINT" \
        --acl public-read \
        --content-type "$content_type"

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

    cd /opt/InstantSplat

    # Determine number of views (cap at 6 for stability - larger values can cause silent failures)
    local n_views=$IMAGE_COUNT
    if [ "$n_views" -gt 6 ]; then
        n_views=6
    fi

    # Set up directory structure expected by InstantSplat
    # InstantSplat expects: <source_path>/images/*.jpg
    local scene_dir="$OUTPUT_DIR/scene"
    mkdir -p "$scene_dir/images"
    cp "$INPUT_DIR"/*.jpg "$scene_dir/images/" 2>/dev/null || \
    cp "$INPUT_DIR"/*.png "$scene_dir/images/" 2>/dev/null || \
    cp "$INPUT_DIR"/* "$scene_dir/images/"

    echo "Prepared $(ls -1 "$scene_dir/images" | wc -l) images"

    # Step 1: Initialize pose estimation with DUSt3R/MASt3R
    # This creates sparse_N/ directory with camera poses and point clouds
    echo "Step 1/2: Running pose estimation with MASt3R..."
    python init_test_pose.py \
        --source_path "$scene_dir" \
        --model_path "$OUTPUT_DIR/instantsplat" \
        --n_views "$n_views" \
        --niter 500 \
        --focal_avg \
        --min_conf_thr 1.5 \
        2>&1 || {
            echo "init_test_pose.py failed, checking output..."
            echo "Scene dir contents:"
            ls -la "$scene_dir/" || true
            echo "Sparse dir contents:"
            ls -la "$scene_dir/sparse_"* 2>/dev/null || echo "No sparse directories created"
            echo "Sparse_*/0 contents:"
            ls -la "$scene_dir/sparse_"*/0/ 2>/dev/null || echo "No 0 subdirectory"
            echo "All .npy files:"
            find "$scene_dir" -name "*.npy" -ls 2>/dev/null || echo "No .npy files found"
            return 1
        }

    # Debug: Show what was created
    echo "Checking created files..."
    ls -la "$scene_dir/sparse_"* 2>/dev/null || echo "Warning: No sparse directories found"
    find "$scene_dir" -name "*.npy" 2>/dev/null | head -10 || echo "No .npy files found"

    # Step 2: Train Gaussian Splatting
    echo "Step 2/2: Training Gaussian Splatting..."
    python train.py \
        --source_path "$scene_dir" \
        --model_path "$OUTPUT_DIR/instantsplat" \
        --iterations 3000 \
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

convert_and_upload() {
    local ply_path="$1"
    local pipeline_name="$2"

    echo "Converting PLY to KSPLAT..."

    # Find the PLY file
    if [ ! -f "$ply_path" ]; then
        # Try to find it
        ply_path=$(find "$OUTPUT_DIR" -name "*.ply" -type f | head -1)
    fi

    if [ -z "$ply_path" ] || [ ! -f "$ply_path" ]; then
        echo "Error: No PLY file found"
        return 1
    fi

    local ksplat_path="$OUTPUT_DIR/output.ksplat"

    # Convert using 3dgsconverter
    python /opt/convert_to_ksplat.py "$ply_path" "$ksplat_path" || {
        # Fallback: try 3dgsconverter CLI
        3dgsconverter -i "$ply_path" -o "$ksplat_path" -f ksplat || return 1
    }

    if [ ! -f "$ksplat_path" ]; then
        echo "Error: KSPLAT conversion failed"
        return 1
    fi

    echo "Uploading results to Spaces..."

    # Upload KSPLAT
    local splat_key="jobs/$JOB_ID/output/$pipeline_name/scene.ksplat"
    local splat_url=$(upload_to_spaces "$ksplat_path" "$splat_key" "application/octet-stream")

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
    *)
        echo "Error: Unknown pipeline mode: $PIPELINE_MODE"
        exit 1
        ;;
esac

echo "Pipeline $PIPELINE_MODE completed successfully!"
exit 0
