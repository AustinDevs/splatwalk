#!/bin/bash
set -e

# Fetch the latest pipeline script from GitHub at runtime
# so we don't have to rebuild this image for script changes.
SCRIPT_URL="${PIPELINE_SCRIPT_URL:-https://raw.githubusercontent.com/AustinDevs/splatwalk/master/docker/gpu/run_pipeline.sh}"

echo "Fetching pipeline script from: $SCRIPT_URL"
if [ -n "$GITHUB_TOKEN" ]; then
    curl -fsSL -H "Authorization: token $GITHUB_TOKEN" "$SCRIPT_URL" -o /opt/run_pipeline.sh
else
    curl -fsSL "$SCRIPT_URL" -o /opt/run_pipeline.sh
fi
chmod +x /opt/run_pipeline.sh

exec /opt/run_pipeline.sh
