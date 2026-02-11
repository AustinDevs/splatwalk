#!/bin/bash
set -e

# Fetch the latest pipeline script from GitHub at runtime
# so we don't have to rebuild this image for script changes.
# Uses the GitHub API (works with fine-grained PATs, unlike raw.githubusercontent.com).
SCRIPT_URL="${PIPELINE_SCRIPT_URL:-https://api.github.com/repos/AustinDevs/splatwalk/contents/docker/gpu/run_pipeline.sh}"

echo "Fetching pipeline script from: $SCRIPT_URL"
CURL_ARGS=(-fsSL -H "Accept: application/vnd.github.raw")
if [ -n "$GITHUB_TOKEN" ]; then
    CURL_ARGS+=(-H "Authorization: token $GITHUB_TOKEN")
fi
curl "${CURL_ARGS[@]}" "$SCRIPT_URL" -o /opt/run_pipeline.sh
chmod +x /opt/run_pipeline.sh

exec /opt/run_pipeline.sh
