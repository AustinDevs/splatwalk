#!/bin/bash
set -e
export PATH="/opt/conda/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"

# Install runtime Python deps if not already present (snapshot images have them pre-installed)
if python -c "import open3d, kornia, transformers" 2>/dev/null; then
    echo "Runtime Python deps already installed (snapshot image)"
else
    echo "Installing runtime Python dependencies..."
    pip install --no-cache-dir \
        icecream open3d trimesh "pyglet<2" evo matplotlib tensorboard imageio gdown \
        roma opencv-python transformers huggingface_hub \
        omegaconf pytorch-lightning open-clip-torch kornia decord \
        imageio-ffmpeg scikit-image moviepy 2>&1 | tail -5
fi

# Fetch the latest pipeline scripts from GitHub at runtime
# so we don't have to rebuild the Docker image for script changes.
# Uses the GitHub API (works with fine-grained PATs, unlike raw.githubusercontent.com).
BASE_URL="${PIPELINE_SCRIPTS_BASE:-https://api.github.com/repos/AustinDevs/splatwalk/contents/docker/gpu}"

echo "Fetching pipeline scripts from GitHub..."
CURL_ARGS=(-fsSL -H "Accept: application/vnd.github.raw")
if [ -n "$GITHUB_TOKEN" ]; then
    CURL_ARGS+=(-H "Authorization: token $GITHUB_TOKEN")
fi

for script in run_pipeline.sh render_descent.py generate_ground_views.py enhance_with_viewcrafter.py quality_gate.py convert_to_ksplat.py compress_splat.py; do
    if curl "${CURL_ARGS[@]}" "$BASE_URL/$script" -o "/opt/$script" 2>/dev/null; then
        echo "  Updated $script"
    else
        echo "  Using baked-in $script (fetch failed)"
    fi
done
chmod +x /opt/run_pipeline.sh

exec /opt/run_pipeline.sh
