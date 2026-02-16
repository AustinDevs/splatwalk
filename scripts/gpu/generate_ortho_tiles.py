#!/usr/bin/env python3
"""
Generate 2D Orthomosaic Deep-Zoom Tile Pyramid

Takes drone photos + COLMAP poses from Stage 1, stitches an orthomosaic,
then progressively enhances tiles with FLUX to create multiple zoom levels.
Uploads tile pyramid to CDN in {z}/{x}/{y}.jpg format for Leaflet viewer.

Pipeline:
  1. Load geometry (reuse from render_zoom_descent.py)
  2. Load camera intrinsics from COLMAP cameras.bin/txt
  3. Stitch orthomosaic (Level 0) via homography onto ground plane
  4. Stitch-first FLUX Img2Img refinement (Levels 1-2): reconstruct full
     parent image, upscale 2x, enhance tiles with low-strength Img2Img
  5. Bicubic upscale + sharpen (Levels 3-4)
  6. Upload tile pyramid to CDN + generate ortho_manifest.json

Usage:
    python generate_ortho_tiles.py \
        --scene_path /workspace/job/output/scene \
        --model_path /workspace/job/output/instantsplat \
        --output_dir /workspace/job/output/ortho \
        --job_id aukerman
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData

# Import geometry helpers from render_zoom_descent.py (both live on the volume)
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from render_zoom_descent import (
    load_camera_poses,
    load_point_cloud,
    estimate_ground_level,
    notify_slack,
    caption_scene,
)


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def load_camera_intrinsics(scene_path):
    """Load camera intrinsics (focal length, image size) from COLMAP cameras.bin/txt.

    Returns dict with fx, fy, cx, cy, width, height.
    """
    import struct

    for sparse_dir in sorted(Path(scene_path).glob("sparse_*/0")):
        cam_bin = sparse_dir / "cameras.bin"
        if cam_bin.exists():
            return _load_intrinsics_bin(cam_bin)
        cam_txt = sparse_dir / "cameras.txt"
        if cam_txt.exists():
            return _load_intrinsics_txt(cam_txt)

    for sparse_dir in [Path(scene_path) / "sparse" / "0", Path(scene_path) / "sparse"]:
        for fmt in ["cameras.bin", "cameras.txt"]:
            cam_file = sparse_dir / fmt
            if cam_file.exists():
                if fmt.endswith(".bin"):
                    return _load_intrinsics_bin(cam_file)
                else:
                    return _load_intrinsics_txt(cam_file)

    # Fallback: estimate from 512x512 images with ~60deg FOV
    print("  No COLMAP cameras file found, using default intrinsics")
    w, h = 512, 512
    fx = w / (2.0 * math.tan(math.radians(30)))
    return {"fx": fx, "fy": fx, "cx": w / 2.0, "cy": h / 2.0, "width": w, "height": h}


def _load_intrinsics_bin(cameras_path):
    """Load intrinsics from COLMAP binary cameras.bin."""
    import struct

    # COLMAP camera model IDs
    MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}  # SIMPLE_PINHOLE, PINHOLE, etc.

    with open(str(cameras_path), "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = MODEL_PARAMS.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(num_params * 8))

            if model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 1:  # PINHOLE: fx, fy, cx, cy
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            else:  # Default: treat first as focal
                fx = fy = params[0]
                cx, cy = width / 2.0, height / 2.0

            return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": int(width), "height": int(height)}

    raise ValueError("No cameras found in cameras.bin")


def _load_intrinsics_txt(cameras_path):
    """Load intrinsics from COLMAP text cameras.txt."""
    with open(str(cameras_path), "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]

            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = params[0]
                cx = params[1] if len(params) > 1 else width / 2.0
                cy = params[2] if len(params) > 2 else height / 2.0
            elif model in ("PINHOLE", "OPENCV"):
                fx, fy = params[0], params[1]
                cx = params[2] if len(params) > 2 else width / 2.0
                cy = params[3] if len(params) > 3 else height / 2.0
            else:
                fx = fy = params[0] if params else width
                cx, cy = width / 2.0, height / 2.0

            return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}

    raise ValueError("No cameras found in cameras.txt")


# ---------------------------------------------------------------------------
# Orthomosaic stitching
# ---------------------------------------------------------------------------

def stitch_orthomosaic(poses, intrinsics, scene_images_dir, ground_z, canvas_size=2048):
    """Stitch drone images into a nadir orthomosaic on the Z=ground_z plane.

    For each image:
      - Back-project 4 corners through K^-1 to rays
      - Intersect rays with Z=ground_z plane -> world XY
      - cv2.getPerspectiveTransform maps corners to ortho pixel coords
      - Center-weighted blending for seamless overlap

    Returns (ortho_image, world_bounds) where world_bounds is (xy_min, xy_max).
    """
    # Build intrinsic matrix
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1],
    ])
    K_inv = np.linalg.inv(K)
    img_w, img_h = intrinsics["width"], intrinsics["height"]

    # Image corners in pixel coords
    corners_px = np.array([
        [0, 0],
        [img_w, 0],
        [img_w, img_h],
        [0, img_h],
    ], dtype=np.float64)

    # --- Pass 1: Back-project all image corners onto ground plane to find bounds ---
    pose_corners = []  # list of (pose, world_corners_4x2) for valid poses
    all_ground_pts = []

    for pose in poses:
        # Check image exists
        img_name = pose["image_name"]
        img_path = os.path.join(scene_images_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(scene_images_dir, img_name.lower())
        if not os.path.exists(img_path):
            continue

        R_w2c = pose["rotation"]
        cam_center = pose["center"]

        world_corners = []
        valid = True
        for cx, cy in corners_px:
            ray_cam = K_inv @ np.array([cx, cy, 1.0])
            ray_world = R_w2c.T @ ray_cam
            dz = ray_world[2]
            if abs(dz) < 1e-8:
                valid = False
                break
            t_param = (ground_z - cam_center[2]) / dz
            if t_param < 0:
                valid = False
                break
            world_pt = cam_center + t_param * ray_world
            world_corners.append(world_pt[:2])

        if not valid or len(world_corners) != 4:
            continue

        wc = np.array(world_corners)
        pose_corners.append((pose, img_path, wc))
        all_ground_pts.append(wc)

    if not all_ground_pts:
        print("  WARNING: No valid poses for stitching")
        return np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8), (np.zeros(2), np.zeros(2))

    all_ground_pts = np.concatenate(all_ground_pts, axis=0)  # (N*4, 2)
    xy_min = all_ground_pts.min(axis=0)
    xy_max = all_ground_pts.max(axis=0)
    xy_range = xy_max - xy_min

    # Add 5% padding (small since bounds already include full footprints)
    padding = 0.05 * xy_range
    xy_min -= padding
    xy_max += padding
    xy_range = xy_max - xy_min

    # World-to-pixel scale
    pixels_per_unit = canvas_size / max(xy_range[0], xy_range[1])

    # --- Pass 2: Warp and blend each image using cached corners ---
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.float64)
    weight_map = np.zeros((canvas_size, canvas_size), dtype=np.float64)

    for pose, img_path, world_corners in pose_corners:
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
        except Exception:
            continue

        # World XY to canvas pixel coordinates
        # Y-axis flip: world Y increases upward, pixel Y increases downward
        canvas_corners = np.array([
            [(wc[0] - xy_min[0]) * pixels_per_unit,
             (xy_max[1] - wc[1]) * pixels_per_unit]
            for wc in world_corners
        ], dtype=np.float32)

        # Source corners in image
        src_corners = corners_px.astype(np.float32)

        # Perspective transform
        M = cv2.getPerspectiveTransform(src_corners, canvas_corners)
        warped = cv2.warpPerspective(
            img.astype(np.float64), M, (canvas_size, canvas_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )

        # Create distance-from-center weight mask for blending
        mask = cv2.warpPerspective(
            np.ones(img.shape[:2], dtype=np.float64), M, (canvas_size, canvas_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )

        # Distance transform: higher weight at center, falls off to edges
        mask_uint8 = (mask * 255).astype(np.uint8)
        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        dist_max = dist.max()
        if dist_max > 0:
            dist = dist / dist_max

        # Blend
        canvas += warped * dist[:, :, np.newaxis]
        weight_map += dist

    # Normalize
    valid_mask = weight_map > 0
    for c in range(3):
        canvas[:, :, c][valid_mask] /= weight_map[valid_mask]

    ortho = canvas.clip(0, 255).astype(np.uint8)
    print(f"  Stitched orthomosaic: {canvas_size}x{canvas_size}")

    return ortho, (xy_min, xy_max)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_image(image, tile_size=512):
    """Split image into tile_size x tile_size tiles. Returns list of (x, y, tile)."""
    h, w = image.shape[:2]
    tiles = []
    nx = math.ceil(w / tile_size)
    ny = math.ceil(h / tile_size)
    for ty in range(ny):
        for tx in range(nx):
            x0 = tx * tile_size
            y0 = ty * tile_size
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            tile = image[y0:y1, x0:x1]
            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            tiles.append((tx, ty, tile))
    return tiles


def save_tiles(tiles, zoom_level, output_dir):
    """Save tiles in {z}/{x}/{y}.jpg format."""
    level_dir = os.path.join(output_dir, str(zoom_level))
    os.makedirs(level_dir, exist_ok=True)
    for tx, ty, tile in tiles:
        col_dir = os.path.join(level_dir, str(tx))
        os.makedirs(col_dir, exist_ok=True)
        tile_path = os.path.join(col_dir, f"{ty}.jpg")
        cv2.imwrite(tile_path, tile, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Level {zoom_level}: saved {len(tiles)} tiles")


# ---------------------------------------------------------------------------
# FLUX enhancement for tiles
# ---------------------------------------------------------------------------

def reconstruct_image_from_tiles(tiles, tile_size=512):
    """Reconstruct a full image from a list of (x, y, tile) tuples."""
    if not tiles:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    max_x = max(tx for tx, ty, _ in tiles)
    max_y = max(ty for tx, ty, _ in tiles)
    w = (max_x + 1) * tile_size
    h = (max_y + 1) * tile_size
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for tx, ty, tile in tiles:
        x0 = tx * tile_size
        y0 = ty * tile_size
        image[y0:y0 + tile_size, x0:x0 + tile_size] = tile
    return image


def enhance_tiles_with_flux(parent_tiles, zoom_level, output_dir, scene_images_dir,
                            prompt_base="", skip_flux=False, flux_strength=0.25,
                            tile_size=512):
    """Stitch-first FLUX Img2Img enhancement for a zoom level.

    1. Reconstruct full parent image from tiles
    2. Upscale 2x with Lanczos
    3. Slice into grid of tile_size patches (with overlap for seamless edges)
    4. Run FluxImg2ImgPipeline on each patch at low strength
    5. Stitch enhanced patches back and slice into final tiles

    Returns list of (x, y, tile) for the child level.
    """
    if skip_flux:
        return _upscale_tiles_bicubic(parent_tiles, zoom_level, output_dir)

    import torch
    from PIL import Image as PILImage

    # Reconstruct parent-level full image
    parent_image = reconstruct_image_from_tiles(parent_tiles, tile_size)
    ph, pw = parent_image.shape[:2]
    print(f"  Reconstructed parent image: {pw}x{ph}")

    # Upscale 2x with Lanczos
    new_w, new_h = pw * 2, ph * 2
    upscaled = cv2.resize(parent_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    print(f"  Upscaled to {new_w}x{new_h}")

    # Monkey-patch: PyTorch 2.4 doesn't support enable_gqa
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _patched_sdpa(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

    from diffusers import FluxImg2ImgPipeline

    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    prompt = f"aerial orthophoto, nadir view, {prompt_base}, sharp, photorealistic"

    # Process tiles with overlap for seamless edges
    overlap = 64
    nx = math.ceil(new_w / tile_size)
    ny = math.ceil(new_h / tile_size)

    # Output canvas for enhanced image
    enhanced = np.zeros((new_h, new_w, 3), dtype=np.float64)
    weight_map = np.zeros((new_h, new_w), dtype=np.float64)

    total_patches = nx * ny
    patch_idx = 0

    for ty in range(ny):
        for tx in range(nx):
            patch_idx += 1

            # Tile region in the upscaled image
            x0 = tx * tile_size
            y0 = ty * tile_size
            x1 = min(x0 + tile_size, new_w)
            y1 = min(y0 + tile_size, new_h)

            # Expand with overlap (clamped to image bounds)
            ex0 = max(x0 - overlap, 0)
            ey0 = max(y0 - overlap, 0)
            ex1 = min(x1 + overlap, new_w)
            ey1 = min(y1 + overlap, new_h)

            patch = upscaled[ey0:ey1, ex0:ex1]

            # Skip very dark patches (background)
            if patch.mean() < 15:
                # Just use the upscaled data directly
                enhanced[y0:y1, x0:x1] += upscaled[y0:y1, x0:x1].astype(np.float64)
                weight_map[y0:y1, x0:x1] += 1.0
                continue

            # Resize patch to 512x512 for FLUX input
            patch_pil = PILImage.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            patch_pil = patch_pil.resize((512, 512), PILImage.LANCZOS)

            result = pipe(
                prompt=prompt,
                image=patch_pil,
                strength=flux_strength,
                num_inference_steps=20,
                guidance_scale=3.5,
                height=512,
                width=512,
            ).images[0]

            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

            # Resize result back to the expanded patch size
            result_resized = cv2.resize(result_np, (ex1 - ex0, ey1 - ey0),
                                        interpolation=cv2.INTER_LANCZOS4)

            # Compute offset of the core tile within the expanded patch
            core_x0 = x0 - ex0
            core_y0 = y0 - ey0
            core_x1 = core_x0 + (x1 - x0)
            core_y1 = core_y0 + (y1 - y0)

            # Only use the core region (discard overlap margins)
            core = result_resized[core_y0:core_y1, core_x0:core_x1]
            enhanced[y0:y1, x0:x1] += core.astype(np.float64)
            weight_map[y0:y1, x0:x1] += 1.0

            if patch_idx % 10 == 0 or patch_idx == total_patches:
                print(f"  FLUX Img2Img: {patch_idx}/{total_patches} patches "
                      f"(strength={flux_strength})")

    # Normalize
    valid = weight_map > 0
    for c in range(3):
        enhanced[:, :, c][valid] /= weight_map[valid]
    enhanced = enhanced.clip(0, 255).astype(np.uint8)

    del pipe
    torch.cuda.empty_cache()

    # Slice final enhanced image into tiles
    child_tiles = tile_image(enhanced, tile_size)

    print(f"  FLUX enhanced level {zoom_level}: {len(child_tiles)} tiles "
          f"({new_w}x{new_h})")
    return child_tiles


def _upscale_tiles_bicubic(tiles, zoom_level, output_dir):
    """Bicubic upscale: each parent tile becomes 4 child tiles."""
    child_tiles = []
    for tx, ty, tile in tiles:
        upscaled = cv2.resize(tile, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        # Unsharp mask for sharpening
        blurred = cv2.GaussianBlur(upscaled, (0, 0), 3)
        upscaled = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

        for dy in range(2):
            for dx in range(2):
                child = upscaled[dy*512:(dy+1)*512, dx*512:(dx+1)*512]
                child_tiles.append((tx*2 + dx, ty*2 + dy, child))

    print(f"  Bicubic upscaled {len(tiles)} -> {len(child_tiles)} tiles")
    return child_tiles


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
        ".jpg": "image/jpeg", ".json": "application/json",
        ".png": "image/png", ".splat": "application/octet-stream",
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

    return f"{endpoint}/{bucket}/{remote_key}"


def upload_tile_pyramid(output_dir, job_id, total_levels):
    """Upload all tiles to CDN as jobs/{job_id}/ortho/{z}/{x}/{y}.jpg."""
    endpoint = os.environ.get("SPACES_ENDPOINT", "")
    bucket = os.environ.get("SPACES_BUCKET", "")
    if not endpoint or not bucket:
        print("  SKIP tile upload (no Spaces credentials)")
        return f"file://{os.path.abspath(output_dir)}/{{z}}/{{x}}/{{y}}.jpg"

    remote_prefix = f"jobs/{job_id}/ortho"

    # Use aws s3 sync for bulk upload
    cmd = [
        "aws", "s3", "sync", output_dir,
        f"s3://{bucket}/{remote_prefix}",
        "--endpoint-url", endpoint,
        "--acl", "public-read",
        "--content-type", "image/jpeg",
        "--exclude", "*.json",
        "--exclude", "*.py",
    ]
    print(f"  Uploading tile pyramid to CDN...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Sync failed: {result.stderr[:300]}")

    return f"{endpoint}/{bucket}/{remote_prefix}/{{z}}/{{x}}/{{y}}.jpg"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate 2D Orthomosaic Deep-Zoom Tiles")
    parser.add_argument("--scene_path", required=True, help="Scene dir (images/ + sparse_*/0/)")
    parser.add_argument("--model_path", required=True, help="Stage 2 InstantSplat model")
    parser.add_argument("--output_dir", required=True, help="Output directory for tiles")
    parser.add_argument("--job_id", default="aukerman", help="Job ID for CDN paths")
    parser.add_argument("--drone_agl", type=float, default=63, help="Drone AGL in meters")
    parser.add_argument("--max_enhance_level", type=int, default=2,
                        help="Max zoom level to FLUX-enhance (default 2)")
    parser.add_argument("--total_levels", type=int, default=5,
                        help="Total zoom levels 0-N (default 5)")
    parser.add_argument("--canvas_size", type=int, default=2048,
                        help="Base orthomosaic canvas size (default 2048)")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size (default 512)")
    parser.add_argument("--flux_strength", type=float, default=0.25,
                        help="FLUX Img2Img strength (default 0.25, lower=subtler)")
    parser.add_argument("--skip_flux", action="store_true",
                        help="Skip FLUX enhancement (bicubic only)")
    parser.add_argument("--slack_webhook_url", default="")
    args = parser.parse_args()

    if not args.slack_webhook_url:
        args.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Orthomosaic Deep-Zoom Tile Generator ===")
    print(f"Scene: {args.scene_path}")
    print(f"Model: {args.model_path}")
    print(f"Levels: 0-{args.total_levels - 1} (FLUX through level {args.max_enhance_level})")

    notify_slack("Ortho: starting tile generation...",
                 args.slack_webhook_url, args.job_id)

    # --- Step 1: Load geometry ---
    print("\nStep 1: Loading geometry...")
    positions, ply_path = load_point_cloud(args.model_path)
    ground_z = estimate_ground_level(positions)
    print(f"  {len(positions)} points, ground_z={ground_z:.4f}")

    poses = load_camera_poses(args.scene_path)
    print(f"  {len(poses)} camera poses")

    # --- Step 2: Load intrinsics ---
    print("\nStep 2: Loading camera intrinsics...")
    intrinsics = load_camera_intrinsics(args.scene_path)
    print(f"  fx={intrinsics['fx']:.1f}, image={intrinsics['width']}x{intrinsics['height']}")

    # --- Step 3: Stitch orthomosaic (Level 0) ---
    print("\nStep 3: Stitching orthomosaic...")
    notify_slack("Ortho: stitching orthomosaic...",
                 args.slack_webhook_url, args.job_id)

    scene_images_dir = os.path.join(args.scene_path, "images")
    ortho, world_bounds = stitch_orthomosaic(
        poses, intrinsics, scene_images_dir, ground_z,
        canvas_size=args.canvas_size,
    )

    # Save full orthomosaic for reference
    ortho_path = os.path.join(args.output_dir, "orthomosaic.jpg")
    cv2.imwrite(ortho_path, ortho, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Tile Level 0
    tiles_dir = os.path.join(args.output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    current_tiles = tile_image(ortho, args.tile_size)
    save_tiles(current_tiles, 0, tiles_dir)

    # Caption scene for FLUX prompts
    prompt_base = caption_scene(scene_images_dir)

    # --- Step 4: Progressive zoom levels ---
    image_dim = args.canvas_size
    for level in range(1, args.total_levels):
        image_dim *= 2
        print(f"\nStep 4.{level}: Zoom level {level} ({image_dim}x{image_dim})...")

        if level <= args.max_enhance_level and not args.skip_flux:
            notify_slack(f"Ortho: FLUX enhancing level {level} ({len(current_tiles)} tiles)...",
                         args.slack_webhook_url, args.job_id)
            child_tiles = enhance_tiles_with_flux(
                current_tiles, level, tiles_dir, scene_images_dir,
                prompt_base=prompt_base, skip_flux=False,
                flux_strength=args.flux_strength, tile_size=args.tile_size,
            )
        else:
            # Bicubic upscale + sharpen
            child_tiles = _upscale_tiles_bicubic(current_tiles, level, tiles_dir)

        save_tiles(child_tiles, level, tiles_dir)
        current_tiles = child_tiles

    # --- Step 5: Upload ---
    print("\nStep 5: Uploading tile pyramid...")
    notify_slack("Ortho: uploading tiles to CDN...",
                 args.slack_webhook_url, args.job_id)

    tile_url_template = upload_tile_pyramid(tiles_dir, args.job_id, args.total_levels)

    # Count total tiles
    total_tiles = 0
    for level in range(args.total_levels):
        level_dir = os.path.join(tiles_dir, str(level))
        if os.path.exists(level_dir):
            for root, dirs, files in os.walk(level_dir):
                total_tiles += len([f for f in files if f.endswith(".jpg")])

    # Final image dimensions
    final_dim = args.canvas_size * (2 ** (args.total_levels - 1))

    # --- Step 6: Generate manifest ---
    # Check if splat manifest exists for cross-linking
    splat_manifest_url = None
    endpoint = os.environ.get("SPACES_ENDPOINT", "")
    bucket = os.environ.get("SPACES_BUCKET", "")
    if endpoint and bucket:
        splat_manifest_url = f"{endpoint}/{bucket}/demo/{args.job_id}/manifest.json"

    manifest = {
        "viewer_mode": "ortho",
        "tile_url_template": tile_url_template,
        "tile_size": args.tile_size,
        "min_zoom": 0,
        "max_zoom": args.total_levels - 1,
        "max_enhanced_zoom": min(args.max_enhance_level, args.total_levels - 1),
        "image_width": final_dim,
        "image_height": final_dim,
        "scene_caption": prompt_base,
        "total_tiles": total_tiles,
    }
    if splat_manifest_url:
        manifest["splat_manifest_url"] = splat_manifest_url

    manifest_path = os.path.join(args.output_dir, "ortho_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Upload manifest
    if endpoint and bucket:
        upload_to_cdn(manifest_path, f"jobs/{args.job_id}/ortho/ortho_manifest.json")
        # Also upload to demo path
        upload_to_cdn(manifest_path, f"demo/{args.job_id}/ortho_manifest.json")

    notify_slack(
        f"Ortho complete! {total_tiles} tiles, {args.total_levels} levels, "
        f"FLUX through level {args.max_enhance_level}",
        args.slack_webhook_url, args.job_id, "success",
    )

    print(f"\n=== Orthomosaic tile generation complete ===")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Levels: 0-{args.total_levels - 1}")
    print(f"  FLUX enhanced: 0-{args.max_enhance_level}")
    print(f"  Final resolution: {final_dim}x{final_dim}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
