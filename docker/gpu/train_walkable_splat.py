#!/usr/bin/env python3
"""
Train Walkable Splat

Combines original drone images with FLUX-generated tiered images,
creates COLMAP reconstruction from known camera poses, and trains
an InstantSplat model that covers ground-level viewpoints.

Usage:
    python train_walkable_splat.py \
        --aerial_model /root/output/aukerman-v12/model \
        --scene_path /root/output/aukerman-v9/scene \
        --tiers_dir /root/output/aukerman-tiers \
        --output_dir /root/output/aukerman-walkable \
        --drone_agl 63 \
        --iterations 10000
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

# Add generate_tiered_views for camera generation functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_tiered_views import (
    generate_tier2_cameras,
    generate_tier3_cameras,
    generate_tier4_cameras,
)

# Add InstantSplat to path
sys.path.insert(0, "/mnt/splatwalk/InstantSplat")

from generate_ground_views import load_point_cloud, load_camera_poses


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


def camera_to_colmap(camera, camera_id, image_id, image_name):
    """Convert camera dict to COLMAP images.txt format.

    camera dict has 'rotation' (R_c2w 3x3) and 'center' (position).
    COLMAP needs R_w2c quaternion and T_w2c = -R_w2c @ position.
    """
    R_c2w = np.array(camera["rotation"])
    position = np.array(camera["center"])

    R_w2c = R_c2w.T
    T_w2c = -R_w2c @ position

    qw, qx, qy, qz = rotation_matrix_to_quaternion(R_w2c)

    return f"{image_id} {qw} {qx} {qy} {qz} {T_w2c[0]} {T_w2c[1]} {T_w2c[2]} {camera_id} {image_name}\n\n"


def collect_best_tier_images(tiers_dir, tier_num, best_iteration):
    """Collect image paths from the best iteration of a tier."""
    iter_dir = os.path.join(tiers_dir, f"tier{tier_num}", f"iteration_{best_iteration}", "images")
    if not os.path.isdir(iter_dir):
        print(f"  WARNING: No images at {iter_dir}")
        return []
    images = sorted([
        os.path.join(iter_dir, f)
        for f in os.listdir(iter_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return images


def notify_slack(message, webhook_url, job_id="walkable"):
    """Send a simple Slack notification."""
    if not webhook_url:
        return
    import urllib.request
    payload = json.dumps({
        "text": f"[{job_id}] {message}",
    })
    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  Slack notification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Walkable Splat from Tiered Images")
    parser.add_argument("--aerial_model", required=True, help="Path to aerial splat model (v12)")
    parser.add_argument("--scene_path", required=True, help="Original scene with drone images")
    parser.add_argument("--tiers_dir", required=True, help="Tiered images output directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for walkable model")
    parser.add_argument("--drone_agl", type=float, default=63)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--tier2_best", type=int, default=2, help="Best iteration for tier 2")
    parser.add_argument("--tier3_best", type=int, default=1, help="Best iteration for tier 3")
    parser.add_argument("--tier4_best", type=int, default=2, help="Best iteration for tier 4")
    parser.add_argument("--skip_tiers", default="", help="Comma-separated tier numbers to skip (e.g. '3,4')")
    parser.add_argument("--job_id", default="walkable-v1")
    args = parser.parse_args()

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    skip_tiers = set()
    if args.skip_tiers:
        skip_tiers = {int(t.strip()) for t in args.skip_tiers.split(",")}
        print(f"Skipping tiers: {skip_tiers}")

    os.makedirs(args.output_dir, exist_ok=True)
    scene_dir = os.path.join(args.output_dir, "scene")
    images_dir = os.path.join(scene_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    notify_slack("Starting walkable splat training", webhook_url, args.job_id)

    # ===== 1. Load scene geometry =====
    print("Loading point cloud...")
    positions, ply_path = load_point_cloud(args.aerial_model)
    print(f"  {len(positions)} points from {ply_path}")

    ground_z = np.percentile(positions[:, 2], 5)
    drone_z = np.percentile(positions[:, 2], 95)

    print("Loading camera poses...")
    poses = load_camera_poses(args.scene_path)
    print(f"  {len(poses)} original camera poses")

    # Compute scene scale
    avg_cam_z = np.mean([p["center"][2] for p in poses])
    scene_height = abs(avg_cam_z - ground_z)
    meters_per_unit = args.drone_agl / max(scene_height, 1e-6)
    z_sign = 1.0 if avg_cam_z > ground_z else -1.0
    print(f"  Scene scale: {meters_per_unit:.2f} m/unit, z_sign={z_sign:+.0f}")

    # ===== 2. Re-generate camera poses (deterministic) =====
    print("\nRe-generating tier camera poses...")
    t2_cameras, t3_cameras, t4_cameras = [], [], []
    if 2 not in skip_tiers:
        t2_cameras = generate_tier2_cameras(
            poses, positions, args.drone_agl, ground_z,
            n_target=40, z_sign=z_sign, meters_per_unit=meters_per_unit,
        )
    if 3 not in skip_tiers:
        t3_cameras = generate_tier3_cameras(
            poses, positions, ground_z, args.drone_agl,
            n_target=32, z_sign=z_sign, meters_per_unit=meters_per_unit,
        )
    if 4 not in skip_tiers:
        t4_cameras = generate_tier4_cameras(
            positions, ground_z, args.drone_agl,
            n_pois=8, cams_per_poi=2,
            z_sign=z_sign, meters_per_unit=meters_per_unit,
        )
    print(f"  Tier 2: {len(t2_cameras)}, Tier 3: {len(t3_cameras)}, Tier 4: {len(t4_cameras)}")

    # ===== 3. Collect best images =====
    print("\nCollecting best-iteration images...")
    t2_images = collect_best_tier_images(args.tiers_dir, 2, args.tier2_best) if 2 not in skip_tiers else []
    t3_images = collect_best_tier_images(args.tiers_dir, 3, args.tier3_best) if 3 not in skip_tiers else []
    t4_images = collect_best_tier_images(args.tiers_dir, 4, args.tier4_best) if 4 not in skip_tiers else []
    print(f"  Tier 2: {len(t2_images)}, Tier 3: {len(t3_images)}, Tier 4: {len(t4_images)}")

    # ===== 4. Copy images + build COLMAP =====
    print("\nBuilding combined scene...")

    # Read original COLMAP camera intrinsics
    orig_cameras_txt = os.path.join(args.scene_path, "sparse_24", "0", "cameras.txt")
    with open(orig_cameras_txt) as f:
        cam_lines = [l for l in f.readlines() if not l.startswith("#")]

    # Parse original camera intrinsic (all same — PINHOLE 512 512 fx fy cx cy)
    parts = cam_lines[0].strip().split()
    orig_model = parts[1]  # PINHOLE
    orig_w, orig_h = int(parts[2]), int(parts[3])
    orig_fx, orig_fy = float(parts[4]), float(parts[5])
    orig_cx, orig_cy = float(parts[6]), float(parts[7])
    print(f"  Original camera: {orig_model} {orig_w}x{orig_h} fx={orig_fx:.1f} fy={orig_fy:.1f}")

    # We'll use same intrinsics for all images (all are 512x512)
    # For FLUX images at 384x384, we'll resize them to 512x512

    all_cameras_txt = []
    all_images_txt = []
    image_id = 1
    cam_id = 1

    # Camera model: shared PINHOLE for all 512x512 images
    all_cameras_txt.append(f"{cam_id} PINHOLE 512 512 {orig_fx} {orig_fy} {orig_cx} {orig_cy}\n")

    # --- Tier 1: Original drone images ---
    print("  Copying tier 1 (original drone images)...")
    orig_images_dir = os.path.join(args.scene_path, "images")
    orig_images_txt = os.path.join(args.scene_path, "sparse_24", "0", "images.txt")

    with open(orig_images_txt) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # images.txt has pairs: pose line, then points2D line (empty for us)
        parts = line.split()
        if len(parts) >= 9 and not parts[0].startswith("#"):
            # This is a pose line
            img_name = parts[-1]
            src = os.path.join(orig_images_dir, img_name)
            dst = os.path.join(images_dir, img_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

            # Write with new sequential ID but same pose
            qw, qx, qy, qz = parts[1], parts[2], parts[3], parts[4]
            tx, ty, tz = parts[5], parts[6], parts[7]
            all_images_txt.append(
                f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {img_name}\n\n"
            )
            image_id += 1

    n_tier1 = image_id - 1
    print(f"    {n_tier1} drone images")

    # --- Helper: match generated images to camera poses ---
    def add_tier_images(tier_cameras, tier_images, tier_name):
        nonlocal image_id
        count = 0
        # Map image filenames to camera poses by index
        # Images are named tier{N}_000.jpg, tier{N}_001.jpg etc (sorted)
        # But some might be skipped (dark renders), so match by filename index
        img_by_idx = {}
        for img_path in tier_images:
            fname = os.path.basename(img_path)
            # Extract index from filename like "tier2_003.jpg"
            idx_str = fname.split("_")[-1].split(".")[0]
            try:
                idx = int(idx_str)
                img_by_idx[idx] = img_path
            except ValueError:
                continue

        for idx, cam in enumerate(tier_cameras):
            if idx not in img_by_idx:
                continue
            img_path = img_by_idx[idx]
            fname = os.path.basename(img_path)
            dst = os.path.join(images_dir, fname)

            # Copy image (resize to 512x512 if needed)
            if not os.path.exists(dst):
                from PIL import Image
                img = Image.open(img_path)
                if img.size != (512, 512):
                    img = img.resize((512, 512), Image.LANCZOS)
                img.save(dst, quality=95)

            # Write COLMAP entry
            all_images_txt.append(camera_to_colmap(cam, cam_id, image_id, fname))
            image_id += 1
            count += 1

        print(f"    {count} {tier_name} images")
        return count

    # --- Tier 2: Oblique aerials ---
    print("  Adding tier 2 (oblique aerial)...")
    n_tier2 = add_tier_images(t2_cameras, t2_images, "tier 2")

    # --- Tier 3: Ground 360s ---
    print("  Adding tier 3 (ground 360s)...")
    n_tier3 = add_tier_images(t3_cameras, t3_images, "tier 3")

    # --- Tier 4: Detail close-ups ---
    print("  Adding tier 4 (detail close-ups)...")
    n_tier4 = add_tier_images(t4_cameras, t4_images, "tier 4")

    total_images = n_tier1 + n_tier2 + n_tier3 + n_tier4
    print(f"\n  Total: {total_images} images ({n_tier1} drone + {n_tier2} oblique + {n_tier3} ground + {n_tier4} detail)")

    # ===== 5. Write COLMAP files =====
    n_views = total_images
    sparse_dir = os.path.join(scene_dir, f"sparse_{n_views}", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    # cameras.txt
    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_path, "w") as f:
        f.write(f"# Camera list with one line of data per camera:\n")
        f.write(f"# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        for line in all_cameras_txt:
            f.write(line)
    print(f"  Wrote {cameras_path}")

    # images.txt
    images_path = os.path.join(sparse_dir, "images.txt")
    with open(images_path, "w") as f:
        f.write(f"# Image list with two lines of data per image:\n")
        f.write(f"# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write(f"# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {total_images}\n")
        for line in all_images_txt:
            f.write(line)
    print(f"  Wrote {images_path}")

    # Copy point cloud + required files
    src_sparse = os.path.join(args.scene_path, "sparse_24", "0")
    for fname in ["points3D.ply", "points3D_all.npy", "confidence_dsp.npy"]:
        src = os.path.join(src_sparse, fname)
        dst = os.path.join(sparse_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

    # ===== 6. Train InstantSplat =====
    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    notify_slack(
        f"Training walkable splat: {total_images} images, {args.iterations} iters",
        webhook_url, args.job_id,
    )

    print(f"\n{'='*60}")
    print(f"TRAINING: {total_images} images, {args.iterations} iterations")
    print(f"{'='*60}")

    train_cmd = [
        "python", "train.py",
        "--source_path", scene_dir,
        "--model_path", model_dir,
        "--iterations", str(args.iterations),
        "--n_views", str(n_views),
        "--pp_optimizer",
        "--optim_pose",
    ]

    print(f"  Command: {' '.join(train_cmd)}")
    result = subprocess.run(
        train_cmd,
        cwd="/mnt/splatwalk/InstantSplat",
        capture_output=False,
    )

    if result.returncode != 0:
        notify_slack(f"Training FAILED (exit {result.returncode})", webhook_url, args.job_id)
        print(f"ERROR: Training failed with exit code {result.returncode}")
        sys.exit(1)

    print("Training complete!")

    # ===== 7. Compress to .splat =====
    print(f"\n{'='*60}")
    print("COMPRESSING to .splat")
    print(f"{'='*60}")

    # compress_splat.py uses positional args: input_ply output_splat [--prune_ratio] [--scene_scale]
    ply_path = os.path.join(model_dir, "point_cloud", f"iteration_{args.iterations}", "point_cloud.ply")
    splat_output = os.path.join(args.output_dir, "walkable.splat")
    compress_cmd = [
        "python", "/mnt/splatwalk/scripts/compress_splat.py",
        ply_path, splat_output,
        "--scene_scale", "50",
        "--prune_ratio", "0.20",
    ]

    print(f"  Command: {' '.join(compress_cmd)}")
    result = subprocess.run(compress_cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Compression failed (exit {result.returncode})")
    else:
        splat_path = os.path.join(args.output_dir, "walkable.splat")
        if os.path.exists(splat_path):
            size_mb = os.path.getsize(splat_path) / (1024 * 1024)
            print(f"  Output: {splat_path} ({size_mb:.1f} MB)")

    # ===== 8. Upload to CDN =====
    splat_path = os.path.join(args.output_dir, "walkable.splat")
    if os.path.exists(splat_path):
        try:
            import boto3
            s3 = boto3.client(
                "s3",
                region_name=os.environ.get("SPACES_REGION", "nyc3"),
                endpoint_url=os.environ.get("SPACES_ENDPOINT", "https://nyc3.digitaloceanspaces.com"),
                aws_access_key_id=os.environ.get("SPACES_KEY"),
                aws_secret_access_key=os.environ.get("SPACES_SECRET"),
            )
            bucket = os.environ.get("SPACES_BUCKET", "splatwalk")
            remote_key = f"splats/{args.job_id}/walkable.splat"
            print(f"\n  Uploading to CDN: {remote_key}...")
            s3.upload_file(
                splat_path, bucket, remote_key,
                ExtraArgs={"ACL": "public-read", "ContentType": "application/octet-stream"},
            )
            cdn_url = f"https://{bucket}.nyc3.digitaloceanspaces.com/{remote_key}"
            print(f"  CDN URL: {cdn_url}")
            notify_slack(f"Walkable splat uploaded: {cdn_url}", webhook_url, args.job_id)
        except Exception as e:
            print(f"  Upload failed: {e}")

    print(f"\n{'='*60}")
    print("WALKABLE SPLAT TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model: {model_dir}")
    print(f"  Splat: {splat_path}")
    print(f"  Images: {total_images} ({n_tier1}+{n_tier2}+{n_tier3}+{n_tier4})")


if __name__ == "__main__":
    main()
