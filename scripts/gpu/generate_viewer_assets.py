#!/usr/bin/env python3
"""
Generate Viewer Assets for Mesh-Based Fly-Over Demo

Generates all assets needed by the mesh-based fly-over viewer:
  1. Upload aerial.glb + scene.glb to CDN
  2. Compute scene bounds from aerial GLB or scene_transform.json
  3. Generate manifest.json (v3 — mesh mode)

Manifest v3 format:
  {
    "aerial_glb_url": "...",
    "scene_glb_url": "...",
    "viewer_mode": "mesh",
    "viewer_modes": ["topdown", "walk"],
    "crossfade_altitude": 30.0,
    "scene_bounds": {...},
    "camera_defaults": {...},
    "walk_camera_defaults": {...},
    "metadata": {...}
  }

Backward compat: if --model_path is provided with PLY data, also generates
.splat and includes splat_url for legacy viewers.

Usage:
    python generate_viewer_assets.py \\
        --output_dir /root/output/demo \\
        --aerial_glb /root/output/aerial.glb \\
        --scene_glb /root/output/scene.glb \\
        --scene_transform /root/output/scene_transform.json \\
        --job_id aukerman \\
        --drone_agl 63
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


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
        ".glb": "model/gltf-binary",
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
# Scene bounds from scene_transform.json or GLB
# ---------------------------------------------------------------------------

def load_scene_bounds(scene_transform_path):
    """Load scene bounds from scene_transform.json (output of generate_aerial_glb.py)."""
    with open(scene_transform_path) as f:
        data = json.load(f)
    return data["scene_bounds"], data


def compute_bounds_from_glb(glb_path, scene_scale=50.0):
    """Compute scene bounds by loading the GLB and measuring bounding box."""
    try:
        import trimesh
        scene = trimesh.load(glb_path)
        if hasattr(scene, 'bounds'):
            bounds_arr = scene.bounds  # (2, 3) array: [min, max]
        else:
            # Single mesh
            bounds_arr = np.array([scene.vertices.min(axis=0), scene.vertices.max(axis=0)])

        min_xyz = bounds_arr[0].tolist()
        max_xyz = bounds_arr[1].tolist()
        size = (bounds_arr[1] - bounds_arr[0]).tolist()
        center = ((bounds_arr[0] + bounds_arr[1]) / 2).tolist()
        ground_z = float(np.percentile(bounds_arr[:, 2], 5)) if bounds_arr.shape[0] > 2 else float(min_xyz[2])

        return {
            "min": [float(v) for v in min_xyz],
            "max": [float(v) for v in max_xyz],
            "center": [float(v) for v in center],
            "size": [float(v) for v in size],
            "ground_z": ground_z,
        }
    except Exception as e:
        print(f"  WARNING: Could not compute bounds from GLB: {e}")
        return None


# ---------------------------------------------------------------------------
# Legacy splat support
# ---------------------------------------------------------------------------

def compute_scene_bounds_from_ply(model_path, scene_scale=50.0):
    """Compute scene bounds from PLY (legacy path for backward compat)."""
    try:
        from plyfile import PlyData
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
        from plyfile import PlyData

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

    centroid = positions.mean(axis=0)
    positions_scaled = (positions - centroid) * scene_scale
    ground_z_raw = float(np.percentile(positions[:, 2], 5))
    ground_z_scaled = float((ground_z_raw - centroid[2]) * scene_scale)

    return {
        "min": [float(v) for v in positions_scaled.min(axis=0)],
        "max": [float(v) for v in positions_scaled.max(axis=0)],
        "center": [0.0, 0.0, 0.0],
        "size": [float(v) for v in (positions_scaled.max(axis=0) - positions_scaled.min(axis=0))],
        "ground_z": ground_z_scaled,
    }


def compress_splat(model_path, output_splat, prune_ratio=0.20, scene_scale=50.0):
    """Invoke compress_splat.py to create .splat file (legacy path)."""
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
    parser = argparse.ArgumentParser(description="Generate Viewer Assets (Mesh or Splat)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--job_id", default="aukerman", help="Job ID for CDN paths")
    parser.add_argument("--drone_agl", type=float, default=63, help="Drone AGL in meters")
    parser.add_argument("--scene_scale", type=float, default=50.0, help="Uniform scene scale")
    parser.add_argument("--slack_webhook_url", default="")

    # Mesh mode (new pipeline)
    parser.add_argument("--aerial_glb", default="", help="Aerial GLB path")
    parser.add_argument("--scene_glb", default="", help="Ground scene GLB path")
    parser.add_argument("--scene_transform", default="", help="scene_transform.json from generate_aerial_glb.py")

    # Splat mode
    parser.add_argument("--model_path", default="", help="Path to trained model (PLY, triggers splat compression)")
    parser.add_argument("--scene_path", default="", help="Path to scene directory (legacy)")
    parser.add_argument("--prune_ratio", type=float, default=0.20, help="Splat prune ratio")
    args = parser.parse_args()

    if not args.slack_webhook_url:
        args.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")

    os.makedirs(args.output_dir, exist_ok=True)

    notify_slack("Demo assets: starting generation...",
                 args.slack_webhook_url, args.job_id)

    remote_prefix = f"demo/{args.job_id}"
    is_mesh_mode = bool(args.aerial_glb or args.scene_glb)

    # --- Compute scene bounds ---
    bounds = None

    if args.scene_transform and os.path.exists(args.scene_transform):
        print("Loading scene bounds from scene_transform.json...")
        bounds, transform_data = load_scene_bounds(args.scene_transform)
    elif args.aerial_glb and os.path.exists(args.aerial_glb):
        print("Computing scene bounds from aerial GLB...")
        bounds = compute_bounds_from_glb(args.aerial_glb, args.scene_scale)
    elif args.model_path:
        print("Computing scene bounds from PLY (legacy)...")
        bounds = compute_scene_bounds_from_ply(args.model_path, args.scene_scale)

    if not bounds:
        # Default bounds
        bounds = {
            "min": [-50, -50, -10], "max": [50, 50, 10],
            "center": [0, 0, 0], "size": [100, 100, 20], "ground_z": -5,
        }
        print("  Using default bounds (no data source available)")

    print(f"  Scene bounds:")
    print(f"    X: [{bounds['min'][0]:.1f}, {bounds['max'][0]:.1f}]")
    print(f"    Y: [{bounds['min'][1]:.1f}, {bounds['max'][1]:.1f}]")
    print(f"    Z: [{bounds['min'][2]:.1f}, {bounds['max'][2]:.1f}]")
    print(f"    Ground Z: {bounds['ground_z']:.1f}")

    # --- Upload assets ---
    print("Uploading assets to CDN...")
    notify_slack("Demo assets: uploading to CDN...",
                 args.slack_webhook_url, args.job_id)

    aerial_glb_url = None
    scene_glb_url = None
    splat_url = None

    if args.aerial_glb and os.path.exists(args.aerial_glb):
        aerial_glb_url = upload_to_cdn(args.aerial_glb, f"{remote_prefix}/aerial.glb")
        aerial_size_mb = Path(args.aerial_glb).stat().st_size / 1024 / 1024
        print(f"  Aerial GLB: {aerial_size_mb:.1f}MB")

    if args.scene_glb and os.path.exists(args.scene_glb):
        scene_glb_url = upload_to_cdn(args.scene_glb, f"{remote_prefix}/scene.glb")
        scene_size_mb = Path(args.scene_glb).stat().st_size / 1024 / 1024
        print(f"  Scene GLB: {scene_size_mb:.1f}MB")
    else:
        # Check common location
        glb_path = os.path.join(os.path.dirname(args.output_dir), "scene.glb")
        if os.path.exists(glb_path):
            scene_glb_url = upload_to_cdn(glb_path, f"{remote_prefix}/scene.glb")
            scene_size_mb = Path(glb_path).stat().st_size / 1024 / 1024
            print(f"  Scene GLB (auto-found): {scene_size_mb:.1f}MB")

    # Upload splat (compress from PLY — legacy path)
    if args.model_path and not is_mesh_mode:
        splat_path = os.path.join(args.output_dir, "scene.splat")
        print("Compressing .splat from PLY...")
        compress_splat(args.model_path, splat_path,
                       prune_ratio=args.prune_ratio, scene_scale=args.scene_scale)
        splat_url = upload_to_cdn(splat_path, f"{remote_prefix}/scene.splat")
        splat_size_mb = Path(splat_path).stat().st_size / 1024 / 1024
        print(f"  Splat: {splat_size_mb:.1f}MB")

    # --- Generate manifest ---
    scene_height = bounds["max"][2] - bounds["min"][2]
    viewing_altitude = float(bounds["max"][2] + scene_height * 0.5)
    ground_z = bounds["ground_z"]

    # viewer_mode: mesh if GLBs present, splat if legacy, topdown fallback
    if is_mesh_mode:
        viewer_mode = "mesh"
    elif splat_url:
        viewer_mode = "splat"
    else:
        viewer_mode = "topdown"

    manifest = {
        "viewer_mode": viewer_mode,
        "viewer_modes": ["topdown", "walk"],
        "scene_bounds": {
            "min": bounds["min"],
            "max": bounds["max"],
            "center": bounds["center"],
            "size": bounds["size"],
            "ground_z": ground_z,
        },
        "camera_defaults": {
            "position": [0.0, 0.0, viewing_altitude],
            "look_at": [0.0, 0.0, ground_z],
            "up": [0.0, 1.0, 0.0],
        },
        "walk_camera_defaults": {
            "position": [0.0, 0.0, ground_z + 3.0],
            "look_at": [5.0, 0.0, ground_z + 3.0],
            "up": [0.0, 0.0, 1.0],
        },
        "crossfade_altitude": 30.0,
        "metadata": {
            "drone_agl_m": args.drone_agl,
            "scene_scale": args.scene_scale,
        },
    }

    if aerial_glb_url:
        manifest["aerial_glb_url"] = aerial_glb_url
    if scene_glb_url:
        manifest["scene_glb_url"] = scene_glb_url
    if splat_url:
        manifest["splat_url"] = splat_url

    # Cross-link: check if ortho manifest exists
    ortho_manifest_path = os.path.join(os.path.dirname(args.output_dir), "ortho", "ortho_manifest.json")
    if os.path.exists(ortho_manifest_path):
        endpoint = os.environ.get("SPACES_ENDPOINT", "")
        bucket = os.environ.get("SPACES_BUCKET", "")
        if endpoint and bucket:
            manifest["ortho_manifest_url"] = f"{endpoint}/{bucket}/demo/{args.job_id}/ortho_manifest.json"

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    upload_to_cdn(manifest_path, f"{remote_prefix}/manifest.json")

    total_size = sum(
        Path(p).stat().st_size / 1024 / 1024
        for p in [args.aerial_glb, args.scene_glb] if p and os.path.exists(p)
    )
    notify_slack(
        f"Demo assets complete! {total_size:.1f}MB total, "
        f"scene {bounds['size'][0]:.0f}x{bounds['size'][1]:.0f} units",
        args.slack_webhook_url, args.job_id, "success",
    )

    print(f"\nDone. Manifest: {json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main()
