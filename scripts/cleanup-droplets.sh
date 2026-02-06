#!/bin/bash
#
# SplatWalk Stale Droplet Cleanup Script
# Run via cron every 15 minutes: */15 * * * * /path/to/cleanup-droplets.sh
#

set -e

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

DO_API_TOKEN="${DO_API_TOKEN:-}"
MAX_AGE_MINUTES="${MAX_AGE_MINUTES:-65}"

if [ -z "$DO_API_TOKEN" ]; then
    echo "Error: DO_API_TOKEN not set"
    exit 1
fi

echo "$(date): Starting droplet cleanup..."

# Get all droplets with splatwalk tag
DROPLETS=$(curl -s -X GET \
    -H "Authorization: Bearer $DO_API_TOKEN" \
    -H "Content-Type: application/json" \
    "https://api.digitalocean.com/v2/droplets?tag_name=splatwalk")

# Check if we got a valid response
if [ "$(echo "$DROPLETS" | jq -r '.droplets')" == "null" ]; then
    echo "Error: Failed to fetch droplets"
    echo "$DROPLETS"
    exit 1
fi

# Get current timestamp
NOW=$(date +%s)
MAX_AGE_SECONDS=$((MAX_AGE_MINUTES * 60))

# Process each droplet
echo "$DROPLETS" | jq -c '.droplets[]' | while read -r droplet; do
    ID=$(echo "$droplet" | jq -r '.id')
    NAME=$(echo "$droplet" | jq -r '.name')
    CREATED_AT=$(echo "$droplet" | jq -r '.created_at')

    # Convert created_at to timestamp
    CREATED_TS=$(date -d "$CREATED_AT" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$CREATED_AT" +%s 2>/dev/null || echo "0")

    if [ "$CREATED_TS" == "0" ]; then
        echo "Warning: Could not parse timestamp for droplet $ID ($NAME)"
        continue
    fi

    AGE=$((NOW - CREATED_TS))
    AGE_MINUTES=$((AGE / 60))

    echo "Droplet $ID ($NAME): age ${AGE_MINUTES}min"

    if [ $AGE -gt $MAX_AGE_SECONDS ]; then
        echo "  -> Stale! Destroying..."

        RESPONSE=$(curl -s -X DELETE \
            -H "Authorization: Bearer $DO_API_TOKEN" \
            -H "Content-Type: application/json" \
            "https://api.digitalocean.com/v2/droplets/$ID")

        if [ -z "$RESPONSE" ]; then
            echo "  -> Destroyed successfully"
        else
            echo "  -> Error: $RESPONSE"
        fi
    fi
done

echo "$(date): Cleanup complete"
