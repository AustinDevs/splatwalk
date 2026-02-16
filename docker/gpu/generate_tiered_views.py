#!/usr/bin/env python3
"""
Tiered Training Image Generation

Generates multi-altitude training images for walkable ground-level Gaussian Splats.
Each tier is scored by Gemini and reported to Slack with spatially-coherent mosaics.
Iterates until every tier scores 9/10 or max retries exhausted.

Tiers:
  1. Existing nadir drone photos (stitch + score only)
  2. Oblique aerial (45deg pitch, 30m + 15m AGL) — 40 images via FLUX
  3. Ground-level 360s (1.7m AGL, 8 directions) — 32 images via FLUX
  4. Detail close-ups (1.2m, at dense point clusters) — 16 images via FLUX

Usage:
    python generate_tiered_views.py \
        --model_path /root/output/aukerman-v12/model \
        --scene_path <scene_dir> \
        --output_dir /root/output/aukerman-tiers \
        --drone_agl 63
"""

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

try:
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from plyfile import PlyData
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData

# Import reusable functions from generate_ground_views.py (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_ground_views import (
    load_point_cloud,
    load_camera_poses,
    extract_gps_from_images,
    get_geographic_context,
    caption_aerial_scene,
    _render_rgb_from_splat,
    _estimate_depth_maps,
    extract_nearest_aerial_crop,
)


# ---------------------------------------------------------------------------
# FLUX parameters per tier
# ---------------------------------------------------------------------------

TIER_PARAMS = {
    2: {
        "controlnet_scale": 0.5,
        "guidance_scale": 4.0,
        "ip_adapter_scale": 0.7,
        "steps": 28,
        "denoising": 0.80,
        "output_size": 512,
    },
    3: {
        "controlnet_scale": 0.4,
        "guidance_scale": 4.5,
        "ip_adapter_scale": 0.6,
        "steps": 28,
        "denoising": 0.85,
        "output_size": 384,
    },
    4: {
        "controlnet_scale": 0.3,
        "guidance_scale": 5.0,
        "ip_adapter_scale": 0.5,
        "steps": 35,
        "denoising": 0.90,
        "output_size": 512,
    },
}

# Gemini feedback keyword -> parameter adjustments
FEEDBACK_ADJUSTMENTS = {
    "artifact": {"controlnet_scale": 0.15},
    "distort": {"controlnet_scale": 0.15},
    "flat": {"steps": 7, "guidance_scale": 0.5},
    "blurry": {"steps": 7, "guidance_scale": 0.5},
    "unrealistic": {"ip_adapter_scale": 0.15, "denoising": -0.05},
    "ai": {"ip_adapter_scale": 0.15, "denoising": -0.05},
    "inconsistent": {"denoising": -0.1},
}


# ---------------------------------------------------------------------------
# Farthest-point sampling
# ---------------------------------------------------------------------------

def farthest_point_sample(points, n_samples):
    """Select n_samples points maximizing minimum distance between selected points.

    Args:
        points: (N, D) array of point positions
        n_samples: number of points to select

    Returns:
        indices of selected points
    """
    if len(points) <= n_samples:
        return list(range(len(points)))

    selected = [0]
    min_dists = np.full(len(points), np.inf)

    for _ in range(n_samples - 1):
        last = points[selected[-1]]
        dists = np.linalg.norm(points - last, axis=-1)
        min_dists = np.minimum(min_dists, dists)
        next_idx = np.argmax(min_dists)
        selected.append(int(next_idx))

    return selected


# ---------------------------------------------------------------------------
# Tier-specific camera generators
# ---------------------------------------------------------------------------

def generate_tier2_cameras(poses, positions, drone_agl, ground_z, n_target=40, z_sign=1.0, meters_per_unit=None):
    """Tier 2: Oblique aerial cameras at 45deg pitch, 30m and 15m AGL.

    For each drone XY, create 4 cameras (N/S/E/W) at 45deg pitch.
    Two altitude bands: 30m and 15m AGL.
    Farthest-point sample to n_target.
    """
    from scipy.spatial.transform import Rotation

    drone_centers = np.array([p["center"] for p in poses])
    drone_xy = drone_centers[:, :2]

    if meters_per_unit is None:
        avg_drone_z = np.mean(drone_centers[:, 2])
        scene_height = abs(avg_drone_z - ground_z)
        meters_per_unit = drone_agl / max(scene_height, 1e-6)

    # Target altitudes in scene units
    alt_30m = ground_z + z_sign * 30.0 / meters_per_unit
    alt_15m = ground_z + z_sign * 15.0 / meters_per_unit

    yaw_angles = [0, 90, 180, 270]  # N, E, S, W
    candidates = []

    for xy in drone_xy:
        for alt in [alt_30m, alt_15m]:
            for yaw_deg in yaw_angles:
                yaw_rad = math.radians(yaw_deg)
                # 45deg pitch = looking halfway between down and horizon
                R = Rotation.from_euler("zyx", [yaw_rad, math.radians(-45), 0]).as_matrix()
                center = np.array([xy[0], xy[1], alt])
                t_vec = -R @ center
                candidates.append({
                    "center": center,
                    "rotation": R,
                    "translation": t_vec,
                    "yaw_deg": yaw_deg,
                    "alt_m": 30.0 if alt == alt_30m else 15.0,
                    "camera_type": "oblique_aerial",
                })

    # Farthest-point sample to n_target
    candidate_positions = np.array([c["center"] for c in candidates])
    indices = farthest_point_sample(candidate_positions, n_target)
    sampled = [candidates[i] for i in indices]

    print(f"  Tier 2: {len(candidates)} candidates -> {len(sampled)} sampled oblique aerials")
    return sampled


def generate_tier3_cameras(poses, positions, ground_z, drone_agl, n_target=32, z_sign=1.0, meters_per_unit=None):
    """Tier 3: Ground-level 360s at 1.7m AGL, 8 yaw angles per grid point.

    Regular grid ~10m apart, 8 directions, -5deg pitch.
    Farthest-point sample to n_target.
    """
    from scipy.spatial.transform import Rotation

    drone_centers = np.array([p["center"] for p in poses])
    drone_xy = drone_centers[:, :2]

    if meters_per_unit is None:
        avg_drone_z = np.mean(drone_centers[:, 2])
        scene_height = abs(avg_drone_z - ground_z)
        meters_per_unit = drone_agl / max(scene_height, 1e-6)

    eye_z = ground_z + z_sign * 1.7 / meters_per_unit

    # Grid spacing: ~10m in scene units
    grid_spacing = 10.0 / meters_per_unit

    xy_min = drone_xy.min(axis=0)
    xy_max = drone_xy.max(axis=0)
    xy_range = xy_max - xy_min

    # Create grid (cap at 20x20 to prevent runaway)
    n_x = min(20, max(2, int(xy_range[0] / grid_spacing)))
    n_y = min(20, max(2, int(xy_range[1] / grid_spacing)))
    grid_x = np.linspace(xy_min[0] + 0.1 * xy_range[0], xy_max[0] - 0.1 * xy_range[0], n_x)
    grid_y = np.linspace(xy_min[1] + 0.1 * xy_range[1], xy_max[1] - 0.1 * xy_range[1], n_y)

    yaw_angles = np.linspace(0, 360, 8, endpoint=False)  # 0, 45, 90, ...315
    candidates = []

    for gx in grid_x:
        for gy in grid_y:
            for yaw_deg in yaw_angles:
                yaw_rad = math.radians(yaw_deg)
                R = Rotation.from_euler("zyx", [yaw_rad, math.radians(-5), 0]).as_matrix()
                center = np.array([gx, gy, eye_z])
                t_vec = -R @ center
                candidates.append({
                    "center": center,
                    "rotation": R,
                    "translation": t_vec,
                    "yaw_deg": yaw_deg,
                    "grid_x": gx,
                    "grid_y": gy,
                    "camera_type": "ground_360",
                })

    candidate_positions = np.array([c["center"] for c in candidates])
    indices = farthest_point_sample(candidate_positions, n_target)
    sampled = [candidates[i] for i in indices]

    print(f"  Tier 3: {len(candidates)} candidates -> {len(sampled)} sampled ground views")
    return sampled


def generate_tier4_cameras(positions, ground_z, drone_agl, n_pois=8, cams_per_poi=2, z_sign=1.0, meters_per_unit=None):
    """Tier 4: Detail close-ups at 1.2m AGL, at dense point clusters.

    K-means cluster point cloud XY to find POIs, then place cameras
    on opposite sides of each cluster center.
    """
    from scipy.spatial.transform import Rotation

    if meters_per_unit is None:
        drone_z = np.percentile(positions[:, 2], 95)
        scene_height = abs(drone_z - ground_z)
        meters_per_unit = drone_agl / max(scene_height, 1e-6)
    else:
        drone_z = np.percentile(positions[:, 2], 95)

    eye_z = ground_z + z_sign * 1.2 / meters_per_unit

    # Use only lower-half points (ground-level features)
    z_mid = (ground_z + drone_z) / 2
    ground_points = positions[positions[:, 2] < z_mid]
    if len(ground_points) < n_pois:
        ground_points = positions

    # K-means clustering for POI detection
    from scipy.cluster.vq import kmeans2
    xy = ground_points[:, :2].astype(np.float64)
    centroids, labels = kmeans2(xy, n_pois, minit="points")

    # Score clusters by density (more points = more interesting)
    cluster_counts = np.bincount(labels, minlength=n_pois)
    top_clusters = np.argsort(-cluster_counts)[:n_pois]

    cameras = []
    for ci in top_clusters:
        cx, cy = centroids[ci]
        # Offset distance: ~2m from center
        offset = 2.0 / meters_per_unit

        for angle_idx in range(cams_per_poi):
            # Opposite sides
            angle = math.radians(angle_idx * 180)
            cam_x = cx + offset * math.cos(angle)
            cam_y = cy + offset * math.sin(angle)

            # Look at cluster center
            look_dir = np.array([cx - cam_x, cy - cam_y])
            look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)
            yaw = math.atan2(look_dir[1], look_dir[0])

            R = Rotation.from_euler("zyx", [yaw, math.radians(-10), 0]).as_matrix()
            center = np.array([cam_x, cam_y, eye_z])
            t_vec = -R @ center

            cameras.append({
                "center": center,
                "rotation": R,
                "translation": t_vec,
                "poi_idx": int(ci),
                "poi_center": (float(cx), float(cy)),
                "yaw_deg": math.degrees(yaw),
                "camera_type": "detail_closeup",
            })

    print(f"  Tier 4: {len(cameras)} detail cameras at {n_pois} POIs")
    return cameras


# ---------------------------------------------------------------------------
# Rendering + FLUX generation
# ---------------------------------------------------------------------------

def render_reference_views(model_path, cameras, output_dir, scene_path):
    """Render RGB + depth from the aerial splat for FLUX conditioning."""
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print("  Rendering reference RGB from splat...")
    _render_rgb_from_splat(model_path, cameras, rgb_dir, scene_path)

    print("  Estimating depth maps with Depth-Anything-V2...")
    _estimate_depth_maps(rgb_dir, depth_dir)

    return rgb_dir, depth_dir


def find_nearest_reference_image(camera, ref_images, ref_cameras):
    """Find the nearest image from a reference tier by XY distance.

    Args:
        camera: dict with 'center' key
        ref_images: list of image paths from the reference tier
        ref_cameras: list of camera dicts from the reference tier (or poses)

    Returns:
        PIL Image of the nearest reference, or None
    """
    cam_xy = camera["center"][:2]
    min_dist = float("inf")
    nearest_path = None

    for ref_cam, ref_path in zip(ref_cameras, ref_images):
        ref_xy = ref_cam["center"][:2]
        dist = np.linalg.norm(ref_xy - cam_xy)
        if dist < min_dist:
            min_dist = dist
            nearest_path = ref_path

    if nearest_path and os.path.exists(nearest_path):
        try:
            img = Image.open(nearest_path).convert("RGB")
            w, h = img.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            img = img.crop((left, top, left + crop_size, top + crop_size))
            img = img.resize((512, 512), Image.LANCZOS)
            return img
        except Exception:
            pass
    return None


def generate_tier_images(tier, cameras, model_path, scene_path, output_dir,
                         prompt, params, ref_images=None, ref_cameras=None,
                         poses=None, scene_images_dir=None):
    """Generate images for a tier using FLUX + ControlNet + IP-Adapter.

    For tier 1, this is a no-op (real images only).
    """
    import torch
    from PIL import Image as PILImage
    from diffusers import FluxControlNetPipeline, FluxControlNetModel

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Render reference views
    rgb_dir, depth_dir = render_reference_views(model_path, cameras, output_dir, scene_path)

    # Monkey-patch: PyTorch 2.4 doesn't support enable_gqa
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _patched_sdpa(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

    print(f"  Loading FLUX.1-dev + ControlNet-depth pipeline...")
    controlnet = FluxControlNetModel.from_pretrained(
        "XLabs-AI/flux-controlnet-depth-diffusers",
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxControlNetPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # Load IP-Adapter
    ip_adapter_loaded = False
    try:
        print(f"  Loading XLabs IP-Adapter v2...")
        pipe.load_ip_adapter(
            "XLabs-AI/flux-ip-adapter-v2",
            weight_name="ip_adapter.safetensors",
            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
        )
        pipe.set_ip_adapter_scale(params["ip_adapter_scale"])
        ip_adapter_loaded = True
        print(f"  IP-Adapter loaded (scale={params['ip_adapter_scale']})")
    except Exception as e:
        print(f"  IP-Adapter loading failed ({e}), continuing without it")

    depth_files = sorted(Path(depth_dir).glob("*.jpg"))
    rgb_files = sorted(Path(rgb_dir).glob("*.jpg"))

    generated_paths = []
    generated_cameras = []

    for idx, (depth_file, rgb_file) in enumerate(zip(depth_files, rgb_files)):
        if idx >= len(cameras):
            break

        blurry_render = PILImage.open(rgb_file).convert("RGB").resize((1024, 1024), PILImage.LANCZOS)

        # Skip very dark renders
        avg_brightness = np.array(blurry_render).mean()
        if avg_brightness < 20:
            print(f"  Skipping view {idx} (too dark, brightness={avg_brightness:.1f})")
            continue

        depth_map = PILImage.open(depth_file).convert("RGB").resize((1024, 1024), PILImage.LANCZOS)

        gen_kwargs = dict(
            prompt=prompt,
            control_image=depth_map,
            controlnet_conditioning_scale=params["controlnet_scale"],
            num_inference_steps=params["steps"],
            guidance_scale=params["guidance_scale"],
            height=1024,
            width=1024,
        )

        # IP-Adapter reference from the tier above
        if ip_adapter_loaded:
            ip_ref = None
            if ref_images and ref_cameras:
                ip_ref = find_nearest_reference_image(cameras[idx], ref_images, ref_cameras)
            elif poses and scene_images_dir:
                # Tier 2 uses drone images as reference
                ip_ref = extract_nearest_aerial_crop(cameras[idx], poses, scene_images_dir)

            if ip_ref is not None:
                gen_kwargs["ip_adapter_image"] = ip_ref

        result = pipe(**gen_kwargs).images[0]
        result = result.resize(
            (params["output_size"], params["output_size"]), PILImage.LANCZOS
        )

        out_name = f"tier{tier}_{idx:03d}.jpg"
        out_path = os.path.join(images_dir, out_name)
        result.save(out_path, "JPEG", quality=95)
        generated_paths.append(out_path)
        generated_cameras.append(cameras[idx])

        if (idx + 1) % 5 == 0:
            print(f"  Generated {idx + 1}/{len(cameras)} tier {tier} views")

    print(f"  Generated {len(generated_paths)} tier {tier} images")

    del pipe, controlnet
    torch.cuda.empty_cache()

    return generated_paths, generated_cameras


# ---------------------------------------------------------------------------
# Mosaic stitching
# ---------------------------------------------------------------------------

def _get_font(size=14):
    """Get a PIL font, falling back to default if needed."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _draw_compass(draw, x, y, size=40):
    """Draw a compass rose at (x, y)."""
    # N arrow
    draw.line([(x, y + size), (x, y - size)], fill="white", width=2)
    draw.polygon([(x, y - size), (x - 6, y - size + 12), (x + 6, y - size + 12)], fill="white")
    font = _get_font(12)
    draw.text((x - 3, y - size - 16), "N", fill="white", font=font)
    # E/W/S labels
    draw.text((x + size + 4, y - 5), "E", fill="gray", font=font)
    draw.text((x - size - 12, y - 5), "W", fill="gray", font=font)
    draw.text((x - 3, y + size + 4), "S", fill="gray", font=font)


def stitch_tier1_mosaic(image_paths, cameras_or_poses, drone_agl):
    """Tier 1: Orthomosaic — place drone photos by GPS XY on canvas."""
    if not image_paths:
        return None

    # Get XY positions
    positions = []
    for cam in cameras_or_poses:
        positions.append(cam["center"][:2])
    positions = np.array(positions)

    # Normalize positions to canvas coords
    xy_min = positions.min(axis=0)
    xy_max = positions.max(axis=0)
    xy_range = xy_max - xy_min
    xy_range[xy_range == 0] = 1

    thumb_size = 128
    padding = 80
    canvas_w = int(xy_range[0] / xy_range.max() * 1200) + 2 * padding + thumb_size
    canvas_h = int(xy_range[1] / xy_range.max() * 1200) + 2 * padding + thumb_size
    canvas_w = max(canvas_w, 400)
    canvas_h = max(canvas_h, 400)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for img_path, pos in zip(image_paths, positions):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        except Exception:
            continue

        # Map position to canvas
        norm = (pos - xy_min) / xy_range
        cx = int(norm[0] * (canvas_w - 2 * padding - thumb_size)) + padding
        cy = int((1 - norm[1]) * (canvas_h - 2 * padding - thumb_size)) + padding
        canvas.paste(img, (cx, cy))

    # Annotations
    font = _get_font(16)
    font_sm = _get_font(12)
    draw.text((10, 10), f"Tier 1: Nadir Drone ({len(image_paths)} images)", fill="white", font=font)
    draw.text((10, canvas_h - 25), f"Drone AGL: ~{drone_agl:.0f}m", fill="gray", font=font_sm)
    _draw_compass(draw, canvas_w - 50, 60)

    return canvas


def stitch_tier2_mosaic(image_paths, cameras, drone_agl):
    """Tier 2: Position-grouped oblique grid, sorted by altitude and XY."""
    if not image_paths:
        return None

    # Separate by altitude
    high_imgs, high_cams = [], []
    low_imgs, low_cams = [], []
    for path, cam in zip(image_paths, cameras):
        if cam.get("alt_m", 30) >= 20:
            high_imgs.append(path)
            high_cams.append(cam)
        else:
            low_imgs.append(path)
            low_cams.append(cam)

    def _sort_by_xy(imgs, cams):
        """Sort images west-to-east (primary) then north-to-south."""
        if not cams:
            return imgs, cams
        pairs = list(zip(imgs, cams))
        pairs.sort(key=lambda p: (p[1]["center"][0], -p[1]["center"][1]))
        return [p[0] for p in pairs], [p[1] for p in pairs]

    high_imgs, high_cams = _sort_by_xy(high_imgs, high_cams)
    low_imgs, low_cams = _sort_by_xy(low_imgs, low_cams)

    thumb_size = 160
    cols = max(1, int(math.ceil(math.sqrt(max(len(high_imgs), len(low_imgs))))))
    padding = 60
    row_label_w = 80

    all_rows = []
    for label, imgs in [("30m", high_imgs), ("15m", low_imgs)]:
        if not imgs:
            continue
        n_rows = max(1, int(math.ceil(len(imgs) / cols)))
        all_rows.append((label, imgs, n_rows))

    total_rows = sum(r[2] for r in all_rows)
    canvas_w = cols * thumb_size + 2 * padding + row_label_w
    canvas_h = total_rows * thumb_size + (len(all_rows) + 1) * 40 + 2 * padding

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    font = _get_font(16)
    font_sm = _get_font(12)
    draw.text((10, 10), f"Tier 2: Oblique Aerial ({len(image_paths)} images)", fill="white", font=font)

    y_offset = padding + 10
    for label, imgs, n_rows in all_rows:
        draw.text((10, y_offset), f"{label} AGL", fill="cyan", font=font_sm)
        y_offset += 20
        for i, img_path in enumerate(imgs):
            row = i // cols
            col = i % cols
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
                canvas.paste(img, (row_label_w + col * thumb_size + padding, y_offset + row * thumb_size))
            except Exception:
                continue
        y_offset += n_rows * thumb_size + 20

    _draw_compass(draw, canvas_w - 50, 60)
    return canvas


def stitch_tier3_mosaic(image_paths, cameras):
    """Tier 3: Spatial panorama grid — by grid position with yaw sub-tiles."""
    if not image_paths:
        return None

    # Group cameras by approximate grid position
    positions = np.array([c["center"][:2] for c in cameras])
    xy_min = positions.min(axis=0)
    xy_max = positions.max(axis=0)
    xy_range = xy_max - xy_min
    xy_range[xy_range == 0] = 1

    # Quantize to grid cells
    grid_res = 5  # approximate grid resolution
    cell_size = xy_range / grid_res

    cells = {}
    for path, cam, pos in zip(image_paths, cameras, positions):
        cx = int((pos[0] - xy_min[0]) / cell_size[0])
        cy = int((pos[1] - xy_min[1]) / cell_size[1])
        key = (cx, cy)
        if key not in cells:
            cells[key] = []
        cells[key].append((path, cam))

    sub_tile = 96
    cell_cols = max(k[0] for k in cells) + 1 if cells else 1
    cell_rows = max(k[1] for k in cells) + 1 if cells else 1
    max_per_cell = max(len(v) for v in cells.values()) if cells else 1
    sub_cols = int(math.ceil(math.sqrt(max_per_cell)))
    sub_rows = int(math.ceil(max_per_cell / sub_cols))

    cell_w = sub_cols * sub_tile + 4
    cell_h = sub_rows * sub_tile + 4
    padding = 80

    canvas_w = cell_cols * cell_w + 2 * padding
    canvas_h = cell_rows * cell_h + 2 * padding

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    font = _get_font(16)
    draw.text((10, 10), f"Tier 3: Ground 360s ({len(image_paths)} images)", fill="white", font=font)

    for (cx, cy), items in cells.items():
        base_x = padding + cx * cell_w
        base_y = padding + (cell_rows - 1 - cy) * cell_h  # flip Y for north-up
        draw.rectangle(
            [(base_x, base_y), (base_x + cell_w - 2, base_y + cell_h - 2)],
            outline="gray",
        )
        for si, (path, cam) in enumerate(items):
            sr = si // sub_cols
            sc = si % sub_cols
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((sub_tile, sub_tile), Image.LANCZOS)
                canvas.paste(img, (base_x + 2 + sc * sub_tile, base_y + 2 + sr * sub_tile))
            except Exception:
                continue

    _draw_compass(draw, canvas_w - 50, 60)
    return canvas


def stitch_tier4_mosaic(image_paths, cameras):
    """Tier 4: POI cluster layout — group by POI with labels."""
    if not image_paths:
        return None

    # Group by POI
    pois = {}
    for path, cam in zip(image_paths, cameras):
        poi_idx = cam.get("poi_idx", 0)
        if poi_idx not in pois:
            pois[poi_idx] = {"images": [], "center": cam.get("poi_center", (0, 0))}
        pois[poi_idx]["images"].append(path)

    thumb_size = 200
    padding = 60
    poi_spacing = 30

    # Sort POIs by XY position
    sorted_pois = sorted(pois.items(), key=lambda kv: (kv[1]["center"][0], kv[1]["center"][1]))

    cols = 2  # POIs per row
    n_poi_rows = int(math.ceil(len(sorted_pois) / cols))
    poi_w = 2 * thumb_size + 20  # 2 images side by side
    poi_h = thumb_size + 30  # image + label

    canvas_w = cols * poi_w + (cols + 1) * poi_spacing + 2 * padding
    canvas_h = n_poi_rows * poi_h + (n_poi_rows + 1) * poi_spacing + 2 * padding

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    font = _get_font(16)
    font_sm = _get_font(12)
    draw.text((10, 10), f"Tier 4: Detail Close-ups ({len(image_paths)} images)", fill="white", font=font)

    for si, (poi_idx, poi_data) in enumerate(sorted_pois):
        row = si // cols
        col = si % cols
        base_x = padding + col * (poi_w + poi_spacing) + poi_spacing
        base_y = padding + 30 + row * (poi_h + poi_spacing) + poi_spacing

        cx, cy = poi_data["center"]
        draw.text((base_x, base_y - 18), f"POI {poi_idx} ({cx:.2f}, {cy:.2f})",
                  fill="yellow", font=font_sm)

        for ii, img_path in enumerate(poi_data["images"][:2]):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
                canvas.paste(img, (base_x + ii * (thumb_size + 10), base_y))
            except Exception:
                continue

    _draw_compass(draw, canvas_w - 50, 60)
    return canvas


def stitch_mosaic(tier, image_paths, cameras, drone_agl=63):
    """Dispatch to tier-specific mosaic stitcher."""
    if tier == 1:
        return stitch_tier1_mosaic(image_paths, cameras, drone_agl)
    elif tier == 2:
        return stitch_tier2_mosaic(image_paths, cameras, drone_agl)
    elif tier == 3:
        return stitch_tier3_mosaic(image_paths, cameras)
    elif tier == 4:
        return stitch_tier4_mosaic(image_paths, cameras)
    return None


# ---------------------------------------------------------------------------
# CDN upload
# ---------------------------------------------------------------------------

def upload_to_cdn(local_path, remote_key):
    """Upload a file to DO Spaces via boto3 and return the public URL."""
    try:
        import boto3
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
        import boto3

    endpoint = os.environ.get("SPACES_ENDPOINT", "https://nyc3.digitaloceanspaces.com")
    region = os.environ.get("SPACES_REGION", "nyc3")
    bucket = os.environ.get("SPACES_BUCKET", "splatwalk")
    key = os.environ.get("SPACES_KEY", os.environ.get("AWS_ACCESS_KEY_ID", ""))
    secret = os.environ.get("SPACES_SECRET", os.environ.get("AWS_SECRET_ACCESS_KEY", ""))

    if not key or not secret:
        print(f"  WARNING: No Spaces credentials, skipping upload of {local_path}")
        return None

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )

    content_type = "image/jpeg" if local_path.endswith(".jpg") else "application/octet-stream"
    client.upload_file(
        local_path, bucket, remote_key,
        ExtraArgs={"ACL": "public-read", "ContentType": content_type},
    )

    url = f"{endpoint}/{bucket}/{remote_key}"
    print(f"  Uploaded to CDN: {url}")
    return url


# ---------------------------------------------------------------------------
# Gemini scoring
# ---------------------------------------------------------------------------

TIER_SCORE_PROMPTS = {
    1: """Score this aerial drone image mosaic of undeveloped land. The images show nadir (straight-down) views arranged by their geographic position.

Evaluate:
- Image clarity and sharpness
- Coverage completeness (gaps in the mosaic?)
- Exposure consistency across images
- Sufficient overlap between adjacent images for 3D reconstruction

Reply with SCORE: N/10 followed by specific feedback.""",

    2: """Score this set of oblique aerial training images for 3D reconstruction. These are AI-generated views at 45-degree pitch angles from 30m and 15m above ground level, conditioned on an existing aerial splat model.

Evaluate:
- Photorealism (do they look like real drone photos?)
- Consistency with aerial perspective (correct oblique angle, proper scale)
- Ground texture detail (vegetation, terrain, paths visible?)
- Color/lighting consistency across images
- Any obvious AI artifacts (repeated patterns, distortions, unrealistic elements)

Reply with SCORE: N/10 followed by specific feedback.""",

    3: """Score this set of ground-level training images for 3D reconstruction. These are AI-generated views at eye height (1.7m) showing 360-degree panoramic coverage, conditioned on aerial imagery.

Evaluate:
- Photorealism (do they look like actual ground-level photographs?)
- Vegetation quality (realistic grass, trees, leaves, bark textures)
- Spatial consistency (do adjacent views look like the same location?)
- Depth and perspective (correct ground-level vanishing points?)
- Lighting and shadows (consistent with outdoor natural light?)
- Any AI artifacts (blurriness, distortions, flat textures, unrealistic elements)

Reply with SCORE: N/10 followed by specific feedback.""",

    4: """Score this set of close-up detail training images for 3D reconstruction. These are AI-generated views at 1.2m height showing ground-level details like plants, soil, rocks, and terrain features.

Evaluate:
- Fine detail quality (leaf veins, bark texture, soil granularity)
- Photorealism at close range (most demanding test)
- Consistency with the overall scene (colors match aerial views?)
- Depth of field (appropriate for close-up outdoor photography)
- Any AI artifacts (smearing, repeated textures, impossible geometry)

Reply with SCORE: N/10 followed by specific feedback.""",
}


def score_with_gemini(mosaic_path, tier):
    """Send mosaic to Gemini 2.5 Flash for scoring. Returns (score, feedback)."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print("  No GEMINI_API_KEY, returning default score")
        return 7, "No Gemini API key available for scoring."

    try:
        from google import genai
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
        from google import genai

    # Load and resize mosaic for API
    img = Image.open(mosaic_path).convert("RGB")
    max_dim = max(img.size)
    if max_dim > 2048:
        scale = 2048 / max_dim
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    image_bytes = buf.getvalue()

    prompt = TIER_SCORE_PROMPTS.get(tier, TIER_SCORE_PROMPTS[1])

    try:
        client = genai.Client(api_key=gemini_key)
        contents = [
            prompt,
            genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        text = response.text
        print(f"  Gemini response ({len(text)} chars)")

        # Extract score
        score_match = re.search(r"SCORE:\s*(\d+)/10", text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 5
        feedback = text

        return score, feedback

    except Exception as e:
        print(f"  Gemini scoring failed: {e}")
        return 5, f"Gemini scoring failed: {e}"


# ---------------------------------------------------------------------------
# Slack reporting (Blocks API with image)
# ---------------------------------------------------------------------------

def notify_slack_with_mosaic(tier, iteration, score, feedback, mosaic_url,
                             webhook_url, job_id="tiered"):
    """Send Slack blocks message with mosaic image and Gemini score."""
    if not webhook_url:
        return

    # Truncate feedback for Slack
    feedback_short = feedback[:500].replace('"', "'")
    if len(feedback) > 500:
        feedback_short += "..."

    emoji = ":white_check_mark:" if score >= 9 else ":bar_chart:"
    header = f"{emoji} Tier {tier} | Score: {score}/10 | Iteration {iteration}"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": header[:150]},
        },
    ]

    # Add mosaic image if URL available
    if mosaic_url:
        blocks.append({
            "type": "image",
            "image_url": mosaic_url,
            "alt_text": f"Tier {tier} mosaic (iteration {iteration})",
        })

    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*Gemini Feedback:*\n{feedback_short}",
        },
    })

    payload = json.dumps({
        "text": header,  # fallback for notifications
        "blocks": blocks,
    })

    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  Slack notification failed: {e}")


def notify_slack_text(message, webhook_url, job_id="tiered", status="info"):
    """Simple text Slack notification."""
    if not webhook_url:
        return
    emoji = {"info": "\U0001f504", "success": "\u2705", "error": "\u274c"}.get(status, "\U0001f504")
    payload = json.dumps({"text": f"{emoji} *[{job_id[:8]}] tiered* \u2014 {message}"})
    try:
        req = urllib.request.Request(
            webhook_url, data=payload.encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Iteration adjustment
# ---------------------------------------------------------------------------

def adjust_params(params, feedback):
    """Adjust FLUX params based on Gemini feedback keywords."""
    feedback_lower = feedback.lower()
    adjusted = dict(params)
    adjustments_made = []

    for keyword, deltas in FEEDBACK_ADJUSTMENTS.items():
        if keyword in feedback_lower:
            for param, delta in deltas.items():
                old = adjusted.get(param, 0)
                adjusted[param] = round(old + delta, 3)
                adjustments_made.append(f"{param}: {old} -> {adjusted[param]}")

    # Clamp values to reasonable ranges
    adjusted["controlnet_scale"] = max(0.1, min(1.0, adjusted.get("controlnet_scale", 0.5)))
    adjusted["guidance_scale"] = max(2.0, min(10.0, adjusted.get("guidance_scale", 4.0)))
    adjusted["ip_adapter_scale"] = max(0.2, min(1.0, adjusted.get("ip_adapter_scale", 0.6)))
    adjusted["steps"] = max(20, min(50, adjusted.get("steps", 28)))
    adjusted["denoising"] = max(0.5, min(0.95, adjusted.get("denoising", 0.85)))

    if adjustments_made:
        print(f"  Parameter adjustments: {', '.join(adjustments_made)}")

    return adjusted


# ---------------------------------------------------------------------------
# Tier 1: Existing drone photos
# ---------------------------------------------------------------------------

def process_tier1(poses, scene_images_dir, output_dir, drone_agl, webhook_url, job_id):
    """Tier 1: Stitch + score existing nadir drone photos."""
    tier_dir = os.path.join(output_dir, "tier1")
    os.makedirs(tier_dir, exist_ok=True)

    # Collect drone image paths + their camera poses
    image_paths = []
    image_cameras = []
    for pose in poses:
        img_path = os.path.join(scene_images_dir, pose["image_name"])
        if not os.path.exists(img_path):
            # Try lowercase
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(scene_images_dir, Path(pose["image_name"]).stem.lower() + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
        if os.path.exists(img_path):
            image_paths.append(img_path)
            image_cameras.append(pose)

    print(f"\n{'='*60}")
    print(f"TIER 1: Nadir Drone Photos ({len(image_paths)} images)")
    print(f"{'='*60}")

    # Stitch mosaic
    mosaic = stitch_mosaic(1, image_paths, image_cameras, drone_agl)
    mosaic_path = os.path.join(tier_dir, "mosaic.jpg")
    if mosaic:
        mosaic.save(mosaic_path, "JPEG", quality=90)
        print(f"  Mosaic saved: {mosaic_path} ({mosaic.size[0]}x{mosaic.size[1]})")

    # Upload
    mosaic_url = upload_to_cdn(mosaic_path, f"tiered/{job_id}/tier1/mosaic.jpg")

    # Score
    score, feedback = score_with_gemini(mosaic_path, 1)
    print(f"  Tier 1 Score: {score}/10")

    # Save score
    score_data = {"tier": 1, "score": score, "feedback": feedback, "image_count": len(image_paths)}
    with open(os.path.join(tier_dir, "gemini_score.json"), "w") as f:
        json.dump(score_data, f, indent=2)

    # Notify Slack
    notify_slack_with_mosaic(1, 1, score, feedback, mosaic_url, webhook_url, job_id)

    return image_paths, image_cameras, score


# ---------------------------------------------------------------------------
# Process a FLUX-generated tier (2, 3, or 4)
# ---------------------------------------------------------------------------

def process_flux_tier(tier, cameras, model_path, scene_path, output_dir,
                      prompt, drone_agl, ref_images, ref_cameras,
                      poses, scene_images_dir, webhook_url, job_id,
                      max_iterations=3, target_score=9):
    """Process a FLUX-generated tier with iterative refinement."""
    tier_dir = os.path.join(output_dir, f"tier{tier}")
    os.makedirs(tier_dir, exist_ok=True)

    params = dict(TIER_PARAMS[tier])
    best_score = 0
    best_images = []
    best_cameras = []

    print(f"\n{'='*60}")
    print(f"TIER {tier}: {len(cameras)} cameras, target score {target_score}/10")
    print(f"{'='*60}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Tier {tier}, Iteration {iteration}/{max_iterations} ---")
        print(f"  Params: {params}")

        iter_dir = os.path.join(tier_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        notify_slack_text(
            f"Tier {tier} iteration {iteration}: generating {len(cameras)} images...",
            webhook_url, job_id,
        )

        # Generate images
        gen_images, gen_cameras = generate_tier_images(
            tier, cameras, model_path, scene_path, iter_dir,
            prompt, params,
            ref_images=ref_images, ref_cameras=ref_cameras,
            poses=poses, scene_images_dir=scene_images_dir,
        )

        if not gen_images:
            print(f"  No images generated for tier {tier}, skipping scoring")
            continue

        # Stitch mosaic
        mosaic = stitch_mosaic(tier, gen_images, gen_cameras, drone_agl)
        mosaic_path = os.path.join(iter_dir, "mosaic.jpg")
        if mosaic:
            mosaic.save(mosaic_path, "JPEG", quality=90)

        # Upload mosaic
        mosaic_url = upload_to_cdn(
            mosaic_path, f"tiered/{job_id}/tier{tier}/iter{iteration}_mosaic.jpg"
        )

        # Score
        score, feedback = score_with_gemini(mosaic_path, tier)
        print(f"  Tier {tier} Iteration {iteration} Score: {score}/10")

        # Save score
        score_data = {
            "tier": tier, "iteration": iteration, "score": score,
            "feedback": feedback, "params": params,
            "image_count": len(gen_images),
        }
        with open(os.path.join(iter_dir, "gemini_score.json"), "w") as f:
            json.dump(score_data, f, indent=2)

        # Notify Slack
        notify_slack_with_mosaic(tier, iteration, score, feedback, mosaic_url, webhook_url, job_id)

        if score > best_score:
            best_score = score
            best_images = gen_images
            best_cameras = gen_cameras

        if score >= target_score:
            print(f"  Target score reached! ({score}/{target_score})")
            break

        if iteration < max_iterations:
            params = adjust_params(params, feedback)

    return best_images, best_cameras, best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tiered Training Image Generation")
    parser.add_argument("--model_path", required=True, help="Path to aerial splat model (e.g. v12)")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory with images/")
    parser.add_argument("--output_dir", required=True, help="Output directory for tiered images")
    parser.add_argument("--drone_agl", type=float, default=63, help="Drone altitude AGL in meters")
    parser.add_argument("--target_score", type=int, default=9, help="Target Gemini score (1-10)")
    parser.add_argument("--max_iterations", type=int, default=3, help="Max iterations per tier")
    parser.add_argument("--tier2_count", type=int, default=40, help="Number of tier 2 images")
    parser.add_argument("--tier3_count", type=int, default=32, help="Number of tier 3 images")
    parser.add_argument("--tier4_count", type=int, default=16, help="Number of tier 4 images")
    parser.add_argument("--slack_webhook_url", default="", help="Slack webhook URL")
    parser.add_argument("--job_id", default="tiered", help="Job ID for Slack + CDN paths")
    parser.add_argument("--skip_tiers", default="", help="Comma-separated tier numbers to skip")
    # T2 FLUX param overrides
    parser.add_argument("--t2_steps", type=int, default=None, help="Override T2 FLUX steps")
    parser.add_argument("--t2_guidance", type=float, default=None, help="Override T2 guidance_scale")
    parser.add_argument("--t2_ip_scale", type=float, default=None, help="Override T2 ip_adapter_scale")
    parser.add_argument("--t2_denoising", type=float, default=None, help="Override T2 denoising")
    parser.add_argument("--t2_cn_scale", type=float, default=None, help="Override T2 controlnet_scale")
    args = parser.parse_args()

    if not args.slack_webhook_url:
        args.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")

    # Apply T2 FLUX param overrides
    if args.t2_steps is not None:
        TIER_PARAMS[2]["steps"] = args.t2_steps
    if args.t2_guidance is not None:
        TIER_PARAMS[2]["guidance_scale"] = args.t2_guidance
    if args.t2_ip_scale is not None:
        TIER_PARAMS[2]["ip_adapter_scale"] = args.t2_ip_scale
    if args.t2_denoising is not None:
        TIER_PARAMS[2]["denoising"] = args.t2_denoising
    if args.t2_cn_scale is not None:
        TIER_PARAMS[2]["controlnet_scale"] = args.t2_cn_scale

    skip_tiers = set()
    if args.skip_tiers:
        skip_tiers = {int(t) for t in args.skip_tiers.split(",")}

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load scene geometry ---
    print("Loading point cloud...")
    positions, ply_path = load_point_cloud(args.model_path)
    print(f"  {len(positions)} points from {ply_path}")

    ground_z = np.percentile(positions[:, 2], 5)
    drone_z = np.percentile(positions[:, 2], 95)
    print(f"  Ground Z: {ground_z:.4f}, Drone Z: {drone_z:.4f}")

    # --- Load camera poses ---
    print("Loading camera poses...")
    poses = load_camera_poses(args.scene_path)
    print(f"  {len(poses)} camera poses")

    # --- Compute scene scale (Z may be inverted) ---
    avg_cam_z = np.mean([p["center"][2] for p in poses])
    scene_height = abs(avg_cam_z - ground_z)
    meters_per_unit = args.drone_agl / max(scene_height, 1e-6)
    z_sign = 1.0 if avg_cam_z > ground_z else -1.0
    print(f"  Scene scale: {meters_per_unit:.2f} m/unit, z_sign={z_sign:+.0f} (cam_z={avg_cam_z:.4f})")

    scene_images_dir = os.path.join(args.scene_path, "images")

    # --- Extract GPS + caption scene ---
    print("Extracting GPS + captioning scene...")
    lat, lon, _alt = extract_gps_from_images(scene_images_dir)
    google_maps_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    prompt = caption_aerial_scene(scene_images_dir, lat=lat, lon=lon, google_maps_key=google_maps_key)

    notify_slack_text(
        f"Starting tiered image generation (AGL={args.drone_agl}m, {len(poses)} drone images)",
        args.slack_webhook_url, args.job_id,
    )

    report = {"tiers": {}, "drone_agl": args.drone_agl, "prompt": prompt[:200]}

    # ===== TIER 1: Existing drone photos =====
    if 1 not in skip_tiers:
        t1_images, t1_cameras, t1_score = process_tier1(
            poses, scene_images_dir, args.output_dir,
            args.drone_agl, args.slack_webhook_url, args.job_id,
        )
        report["tiers"]["1"] = {"score": t1_score, "count": len(t1_images)}
    else:
        # Still need reference data from tier 1
        t1_images = []
        t1_cameras = []
        for pose in poses:
            img_path = os.path.join(scene_images_dir, pose["image_name"])
            if os.path.exists(img_path):
                t1_images.append(img_path)
                t1_cameras.append(pose)
        print(f"  Tier 1 skipped, {len(t1_images)} reference images loaded")

    # ===== TIER 2: Oblique aerial =====
    if 2 not in skip_tiers:
        t2_cameras = generate_tier2_cameras(
            poses, positions, args.drone_agl, ground_z, n_target=args.tier2_count,
            z_sign=z_sign, meters_per_unit=meters_per_unit,
        )
        t2_images, t2_ref_cameras, t2_score = process_flux_tier(
            tier=2, cameras=t2_cameras, model_path=args.model_path,
            scene_path=args.scene_path, output_dir=args.output_dir,
            prompt=prompt, drone_agl=args.drone_agl,
            ref_images=t1_images, ref_cameras=t1_cameras,
            poses=poses, scene_images_dir=scene_images_dir,
            webhook_url=args.slack_webhook_url, job_id=args.job_id,
            max_iterations=args.max_iterations, target_score=args.target_score,
        )
        report["tiers"]["2"] = {"score": t2_score, "count": len(t2_images)}
    else:
        t2_images, t2_ref_cameras = [], []
        print("  Tier 2 skipped")

    # ===== TIER 3: Ground-level 360s =====
    if 3 not in skip_tiers:
        t3_cameras = generate_tier3_cameras(
            poses, positions, ground_z, args.drone_agl, n_target=args.tier3_count,
            z_sign=z_sign, meters_per_unit=meters_per_unit,
        )
        # Reference = tier 2 if available, else tier 1
        ref_imgs = t2_images if t2_images else t1_images
        ref_cams = t2_ref_cameras if t2_ref_cameras else t1_cameras
        t3_images, t3_ref_cameras, t3_score = process_flux_tier(
            tier=3, cameras=t3_cameras, model_path=args.model_path,
            scene_path=args.scene_path, output_dir=args.output_dir,
            prompt=prompt, drone_agl=args.drone_agl,
            ref_images=ref_imgs, ref_cameras=ref_cams,
            poses=poses, scene_images_dir=scene_images_dir,
            webhook_url=args.slack_webhook_url, job_id=args.job_id,
            max_iterations=args.max_iterations, target_score=args.target_score,
        )
        report["tiers"]["3"] = {"score": t3_score, "count": len(t3_images)}
    else:
        t3_images, t3_ref_cameras = [], []
        print("  Tier 3 skipped")

    # ===== TIER 4: Detail close-ups =====
    if 4 not in skip_tiers:
        t4_cameras = generate_tier4_cameras(
            positions, ground_z, args.drone_agl, n_pois=8, cams_per_poi=2,
            z_sign=z_sign, meters_per_unit=meters_per_unit,
        )
        ref_imgs = t3_images if t3_images else (t2_images if t2_images else t1_images)
        ref_cams = t3_ref_cameras if t3_ref_cameras else (t2_ref_cameras if t2_ref_cameras else t1_cameras)
        t4_images, t4_ref_cameras, t4_score = process_flux_tier(
            tier=4, cameras=t4_cameras, model_path=args.model_path,
            scene_path=args.scene_path, output_dir=args.output_dir,
            prompt=prompt, drone_agl=args.drone_agl,
            ref_images=ref_imgs, ref_cameras=ref_cams,
            poses=poses, scene_images_dir=scene_images_dir,
            webhook_url=args.slack_webhook_url, job_id=args.job_id,
            max_iterations=args.max_iterations, target_score=args.target_score,
        )
        report["tiers"]["4"] = {"score": t4_score, "count": len(t4_images)}
    else:
        print("  Tier 4 skipped")

    # ===== Save report =====
    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("TIERED IMAGE GENERATION COMPLETE")
    print(f"{'='*60}")
    for tier_num, tier_data in sorted(report.get("tiers", {}).items()):
        print(f"  Tier {tier_num}: score={tier_data['score']}/10, {tier_data['count']} images")
    print(f"  Report: {report_path}")
    print(f"  Output: {args.output_dir}")

    all_pass = all(t["score"] >= args.target_score for t in report.get("tiers", {}).values())
    scores_str = ", ".join(
        f"T{k}={v['score']}" for k, v in sorted(report.get("tiers", {}).items())
    )
    if all_pass:
        notify_slack_text(
            f"All tiers passed! Scores: {scores_str}",
            args.slack_webhook_url, args.job_id, "success",
        )
    else:
        notify_slack_text(
            f"Generation complete. Scores: {scores_str}",
            args.slack_webhook_url, args.job_id, "info",
        )


if __name__ == "__main__":
    main()
