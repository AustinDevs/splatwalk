#!/bin/bash
set -e
export PATH="/mnt/splatwalk/conda/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"

# Symlink model weights to expected locations
ln -sf /mnt/splatwalk/models/dust3r/*.pth /mnt/splatwalk/InstantSplat/dust3r/checkpoints/ 2>/dev/null || true
ln -sf /mnt/splatwalk/models/mast3r/*.pth /mnt/splatwalk/InstantSplat/mast3r/checkpoints/ 2>/dev/null || true

# Fetch the latest pipeline scripts from GitHub at runtime
# so we don't have to rebuild the Volume for script changes.
# Uses the GitHub API (works with fine-grained PATs, unlike raw.githubusercontent.com).
BASE_URL="${PIPELINE_SCRIPTS_BASE:-https://api.github.com/repos/AustinDevs/splatwalk/contents/docker/gpu}"

echo "Fetching pipeline scripts from GitHub..."
CURL_ARGS=(-fsSL -H "Accept: application/vnd.github.raw")
if [ -n "$GITHUB_TOKEN" ]; then
    CURL_ARGS+=(-H "Authorization: token $GITHUB_TOKEN")
fi

for script in run_pipeline.sh render_descent.py generate_ground_views.py enhance_with_viewcrafter.py quality_gate.py convert_to_ksplat.py compress_splat.py; do
    if curl "${CURL_ARGS[@]}" "$BASE_URL/$script" -o "/mnt/splatwalk/scripts/$script" 2>/dev/null; then
        echo "  Updated $script"
    else
        echo "  Using baked-in $script (fetch failed)"
    fi
done
chmod +x /mnt/splatwalk/scripts/run_pipeline.sh

# Symlink scripts to /opt for backward compatibility
ln -sf /mnt/splatwalk/scripts/* /opt/ 2>/dev/null || true
ln -sf /mnt/splatwalk/InstantSplat /opt/InstantSplat 2>/dev/null || true
ln -sf /mnt/splatwalk/ViewCrafter /opt/ViewCrafter 2>/dev/null || true

exec /mnt/splatwalk/scripts/run_pipeline.sh
