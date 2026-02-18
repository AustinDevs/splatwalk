#!/usr/bin/env python3
"""
Generate Viewer Assets for Top-Down Fly-Over Demo

Generates all assets needed by the single-mode splat fly-over viewer:
  1. Compress .splat (prune + uniform scene scaling)
  2. Compute scene bounds (XY extents + altitude range)
  3. Upload .splat + manifest.json to Spaces CDN

The manifest includes:
  - splat_url: URL to the compressed .splat file
  - scene_bounds: {min_xyz, max_xyz, center_xyz, ground_z}
  - camera_defaults: {altitude, look_at, fov}

The splat IS the product — just compress and upload.

Usage:
    python generate_viewer_assets.py \
        --model_path /root/output/aukerman/model \
        --scene_path /root/output/aukerman/scene \
        --output_dir /root/output/aukerman-demo \
        --job_id aukerman \
        --drone_agl 63
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData


# ---------------------------------------------------------------------------
# CDN upload
# ---------------------------------------------------------------------------

def upload_to_cdn(local_path, remote_key):
    """Upload a file to DO Spaces CDN. Returns the public URL."""
    endpoint = os.environ.get("SPACES_ENDPOINT", "")
    bucket = os.environ.get("SPACES_BUCKET", "")

    if not endpoint or not bucket:
        print(f"  SKIP upload (no Spaces credentials): {local_path}")
        return f"file://{os.path.abspath(local_path)}"

    content_types = {
        ".splat": "application/octet-stream",
        ".json": "application/json",
        ".jpg": "image/jpeg",
        ".png": "image/png",
    }
    ext = Path(local_path).suffix.lower()
    content_type = content_types.get(ext, "application/octet-stream")

    cmd = [
        "aws", "s3", "cp", local_path,
        f"s3://{bucket}/{remote_key}",
        "--endpoint-url", endpoint,
        "--acl", "public-read",
        "--content-type", content_type,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Upload failed: {result.stderr[:200]}")
        return f"file://{os.path.abspath(local_path)}"

    url = f"{endpoint}/{bucket}/{remote_key}"
    print(f"  Uploaded: {remote_key}")
    return url


def notify_slack(message, webhook_url, job_id, status="info"):
    """Non-blocking Slack notification."""
    if not webhook_url:
        return
    import urllib.request
    emoji = {"info": "\U0001f504", "success": "\u2705", "error": "\u274c"}.get(status, "\U0001f504")
    payload = json.dumps({"text": f"{emoji} *[{job_id[:8]}] demo* \u2014 {message}"})
    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scene bounds computation
# ---------------------------------------------------------------------------

def compute_scene_bounds(model_path, scene_scale=50.0):
    """Compute scene bounds from the trained point cloud.

    Returns bounds in the SCALED coordinate system (matching .splat output).
    """
    # Find latest checkpoint PLY
    ckpt_dirs = sorted(Path(model_path).glob("point_cloud/iteration_*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {model_path}")

    ply_path = str(ckpt_dirs[-1] / "point_cloud.ply")
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]

    positions = np.stack([
        np.array(vertex["x"], dtype=np.float32),
        np.array(vertex["y"], dtype=np.float32),
        np.array(vertex["z"], dtype=np.float32),
    ], axis=-1)

    # Apply same centering + scaling as compress_splat.py
    centroid = positions.mean(axis=0)
    positions_scaled = (positions - centroid) * scene_scale

    ground_z_raw = float(np.percentile(positions[:, 2], 5))
    ground_z_scaled = float((ground_z_raw - centroid[2]) * scene_scale)

    bounds = {
        "min": [float(v) for v in positions_scaled.min(axis=0)],
        "max": [float(v) for v in positions_scaled.max(axis=0)],
        "center": [0.0, 0.0, 0.0],  # centered at origin by compress_splat
        "size": [float(v) for v in (positions_scaled.max(axis=0) - positions_scaled.min(axis=0))],
        "ground_z": ground_z_scaled,
        "centroid_raw": [float(v) for v in centroid],
        "scene_scale": float(scene_scale),
    }

    print(f"  Scene bounds (scaled {scene_scale}x):")
    print(f"    X: [{bounds['min'][0]:.1f}, {bounds['max'][0]:.1f}]")
    print(f"    Y: [{bounds['min'][1]:.1f}, {bounds['max'][1]:.1f}]")
    print(f"    Z: [{bounds['min'][2]:.1f}, {bounds['max'][2]:.1f}]")
    print(f"    Ground Z: {bounds['ground_z']:.1f}")

    return bounds


# ---------------------------------------------------------------------------
# Compress .splat
# ---------------------------------------------------------------------------

def compress_splat(model_path, output_splat, prune_ratio=0.20, scene_scale=50.0):
    """Invoke compress_splat.py to create .splat file."""
    ckpt_dirs = sorted(Path(model_path).glob("point_cloud/iteration_*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {model_path}")
    input_ply = str(ckpt_dirs[-1] / "point_cloud.ply")

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compress_splat.py")
    cmd = [
        sys.executable, script,
        input_ply, output_splat,
        "--prune_ratio", str(prune_ratio),
        "--scene_scale", str(scene_scale),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  compress_splat stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"compress_splat.py failed with code {result.returncode}")
    print(f"  Compressed splat: {output_splat}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Top-Down Fly-Over Viewer Assets")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--job_id", default="aukerman", help="Job ID for CDN paths")
    parser.add_argument("--drone_agl", type=float, default=63, help="Drone AGL in meters")
    parser.add_argument("--prune_ratio", type=float, default=0.20, help="Splat prune ratio")
    parser.add_argument("--scene_scale", type=float, default=50.0, help="Uniform scene scale")
    parser.add_argument("--slack_webhook_url", default="")
    args = parser.parse_args()

    if not args.slack_webhook_url:
        args.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")

    os.makedirs(args.output_dir, exist_ok=True)

    notify_slack("Demo assets: starting generation...",
                 args.slack_webhook_url, args.job_id)

    # --- 1. Compute scene bounds ---
    print("Computing scene bounds...")
    bounds = compute_scene_bounds(args.model_path, scene_scale=args.scene_scale)

    # --- 2. Compress .splat ---
    splat_path = os.path.join(args.output_dir, "scene.splat")
    print("Compressing .splat...")
    notify_slack("Demo assets: compressing .splat...",
                 args.slack_webhook_url, args.job_id)
    compress_splat(
        args.model_path, splat_path,
        prune_ratio=args.prune_ratio,
        scene_scale=args.scene_scale,
    )

    splat_size_mb = Path(splat_path).stat().st_size / 1024 / 1024
    print(f"  Splat size: {splat_size_mb:.1f} MB")

    # --- 3. Upload to CDN ---
    print("Uploading assets to CDN...")
    notify_slack("Demo assets: uploading to CDN...",
                 args.slack_webhook_url, args.job_id)

    remote_prefix = f"demo/{args.job_id}"
    splat_url = upload_to_cdn(splat_path, f"{remote_prefix}/scene.splat")

    # --- 4. Generate manifest ---
    # Camera defaults for the web viewer (Z-up scene from COLMAP)
    # Position camera above the scene max Z, looking down at ground
    scene_height = bounds["max"][2] - bounds["min"][2]
    viewing_altitude = float(bounds["max"][2] + scene_height * 0.5)

    manifest = {
        "splat_url": splat_url,
        "viewer_mode": "topdown",
        "scene_bounds": {
            "min": bounds["min"],
            "max": bounds["max"],
            "center": bounds["center"],
            "size": bounds["size"],
            "ground_z": bounds["ground_z"],
        },
        "camera_defaults": {
            "position": [0.0, 0.0, viewing_altitude],
            "look_at": [0.0, 0.0, bounds["ground_z"]],
            "up": [0.0, 1.0, 0.0],
        },
        "metadata": {
            "drone_agl_m": args.drone_agl,
            "scene_scale": args.scene_scale,
            "splat_size_mb": round(splat_size_mb, 1),
        },
    }

    # Check if ground views were generated — add walk mode
    ground_meta_path = os.path.join(os.path.dirname(args.output_dir), "ground_views", "ground_views_metadata.json")
    if os.path.exists(ground_meta_path):
        try:
            with open(ground_meta_path) as f:
                ground_meta = json.load(f)
            manifest["viewer_modes"] = ["topdown", "walk"]
            # Walk camera at ground level, looking horizontal
            ground_z = bounds["ground_z"]
            walk_height = ground_z + 3.0  # slightly above ground in scaled coords
            manifest["walk_camera_defaults"] = {
                "position": [0.0, 0.0, walk_height],
                "look_at": [5.0, 0.0, walk_height],
                "up": [0.0, 0.0, 1.0],
            }
            manifest["metadata"]["ground_views"] = ground_meta.get("num_cameras", 0)
            print(f"  Walk mode enabled ({ground_meta.get('num_cameras', 0)} ground views)")
        except Exception as e:
            print(f"  Walk mode metadata read failed: {e}")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    upload_to_cdn(manifest_path, f"{remote_prefix}/manifest.json")

    # Cross-link: check if ortho manifest exists and add reference
    ortho_manifest_path = os.path.join(os.path.dirname(args.output_dir), "ortho", "ortho_manifest.json")
    if os.path.exists(ortho_manifest_path):
        endpoint = os.environ.get("SPACES_ENDPOINT", "")
        bucket = os.environ.get("SPACES_BUCKET", "")
        if endpoint and bucket:
            manifest["ortho_manifest_url"] = f"{endpoint}/{bucket}/demo/{args.job_id}/ortho_manifest.json"
        else:
            manifest["ortho_manifest_url"] = f"file://{os.path.abspath(ortho_manifest_path)}"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        upload_to_cdn(manifest_path, f"{remote_prefix}/manifest.json")
        print(f"  Cross-linked ortho manifest: {manifest['ortho_manifest_url']}")

    notify_slack(
        f"Demo assets complete! {splat_size_mb:.1f}MB splat, "
        f"scene {bounds['size'][0]:.0f}x{bounds['size'][1]:.0f} units",
        args.slack_webhook_url, args.job_id, "success",
    )

    print(f"\nDone. Manifest: {json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main()
