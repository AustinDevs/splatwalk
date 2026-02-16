#!/usr/bin/env python3
"""
Coverage Analysis + Gap-Filling Image Generation

Analyzes camera coverage of an existing aerial splat model, identifies gaps
in overlap/angular diversity/scale diversity, generates images to fill those
gaps using FLUX + ControlNet + IP-Adapter, and uploads everything to CDN.

Designed to run on a GPU droplet with the existing Volume at /mnt/splatwalk/.

Usage:
    python analyze_and_generate.py \
        --model_path /root/output/aerial_model \
        --scene_path /root/output/scene \
        --output_dir /root/output/coverage \
        --drone_agl 63 \
        --job_id coverage-v1 \
        --max_iterations 2
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

# Import reusable functions from sibling scripts (same directory)
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
from generate_tiered_views import (
    farthest_point_sample,
    upload_to_cdn,
    score_with_gemini,
    notify_slack_with_mosaic,
    notify_slack_text,
    adjust_params,
    FEEDBACK_ADJUSTMENTS,
)
from render_descent import estimate_ground_level


# ---------------------------------------------------------------------------
# FLUX parameters per altitude band
# ---------------------------------------------------------------------------

BAND_PARAMS = {
    2: {  # Oblique aerial (15-40m)
        "controlnet_scale": 0.6,
        "guidance_scale": 5.5,
        "ip_adapter_scale": 0.8,
        "steps": 40,
        "denoising": 0.70,
        "output_size": 1024,
    },
    3: {  # Elevated ground (3-15m)
        "controlnet_scale": 0.5,
        "guidance_scale": 5.0,
        "ip_adapter_scale": 0.7,
        "steps": 35,
        "denoising": 0.75,
        "output_size": 768,
    },
    4: {  # Ground level (0-3m)
        "controlnet_scale": 0.4,
        "guidance_scale": 4.5,
        "ip_adapter_scale": 0.6,
        "steps": 30,
        "denoising": 0.80,
        "output_size": 512,
    },
    "detail": {  # Detail close-ups (1.2m at POIs)
        "controlnet_scale": 0.3,
        "guidance_scale": 5.0,
        "ip_adapter_scale": 0.5,
        "steps": 35,
        "denoising": 0.85,
        "output_size": 512,
    },
}

# Gemini scoring prompts per band
BAND_SCORE_PROMPTS = {
    2: """Score this set of oblique aerial images (15-40m AGL, 45-degree pitch) for 3D reconstruction training.
Evaluate: photorealism, correct oblique perspective, ground detail, color consistency, AI artifacts.
Reply with SCORE: N/10 followed by specific feedback.""",
    3: """Score this set of elevated ground-level images (3-15m AGL) for 3D reconstruction training.
Evaluate: photorealism, vegetation quality, spatial consistency, depth perspective, lighting.
Reply with SCORE: N/10 followed by specific feedback.""",
    4: """Score this set of ground-level images (eye height, 1.7m AGL) for 3D reconstruction training.
Evaluate: photorealism, vegetation detail, ground textures, spatial coherence, natural lighting.
Reply with SCORE: N/10 followed by specific feedback.""",
    "detail": """Score this set of close-up detail images (1.2m AGL) for 3D reconstruction training.
Evaluate: fine detail quality, photorealism at close range, texture consistency, depth of field.
Reply with SCORE: N/10 followed by specific feedback.""",
}


# =========================================================================
# Phase 1: Coverage Analysis
# =========================================================================

def load_colmap_intrinsics(scene_path):
    """Load camera intrinsics from COLMAP to get FOV.

    Returns (fov_x_rad, fov_y_rad) or defaults if not found.
    """
    import struct

    for sparse_dir in sorted(Path(scene_path).glob("sparse_*/0")) + [
        Path(scene_path) / "sparse" / "0",
        Path(scene_path) / "sparse",
    ]:
        cameras_bin = sparse_dir / "cameras.bin"
        cameras_txt = sparse_dir / "cameras.txt"

        if cameras_bin.exists():
            with open(str(cameras_bin), "rb") as f:
                num_cameras = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num_cameras):
                    cam_id = struct.unpack("<I", f.read(4))[0]
                    model_id = struct.unpack("<i", f.read(4))[0]
                    width = struct.unpack("<Q", f.read(8))[0]
                    height = struct.unpack("<Q", f.read(8))[0]
                    num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
                    params = struct.unpack(f"<{num_params}d", f.read(num_params * 8))
                    # First param is focal length (fx)
                    fx = params[0]
                    fov_x = 2.0 * math.atan(width / (2.0 * fx))
                    fov_y = 2.0 * math.atan(height / (2.0 * fx))
                    return fov_x, fov_y

        if cameras_txt.exists():
            with open(str(cameras_txt), "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        width = int(parts[2])
                        height = int(parts[3])
                        fx = float(parts[4])
                        fov_x = 2.0 * math.atan(width / (2.0 * fx))
                        fov_y = 2.0 * math.atan(height / (2.0 * fx))
                        return fov_x, fov_y

    # Default: ~60 degrees
    return math.radians(60), math.radians(60)


def compute_bev_coverage(cameras, scene_bounds, grid_resolution=0.5, fov_rad=None,
                         ground_z=0.0, meters_per_unit=1.0):
    """Compute Bird's Eye View coverage grid from camera frustum projections.

    For each camera, projects a viewing cone onto the XY ground plane and marks
    which grid cells are covered.

    Args:
        cameras: list of camera dicts with 'center' and 'rotation' keys
        scene_bounds: (xy_min, xy_max) in scene units
        grid_resolution: grid cell size in meters
        fov_rad: camera field of view in radians (default ~60 deg)
        ground_z: estimated ground Z in scene units
        meters_per_unit: scale factor from scene units to meters

    Returns:
        overlap_grid: (H, W) int array — number of cameras covering each cell
        camera_indices_grid: (H, W) list of lists — which cameras cover each cell
        grid_info: dict with grid metadata (xy_min, xy_max, cell_size, shape)
    """
    if fov_rad is None:
        fov_rad = math.radians(60)

    xy_min, xy_max = scene_bounds
    cell_size = grid_resolution / meters_per_unit  # in scene units
    nx = max(1, int((xy_max[0] - xy_min[0]) / cell_size))
    ny = max(1, int((xy_max[1] - xy_min[1]) / cell_size))

    # Cap grid size to prevent memory issues
    nx = min(nx, 500)
    ny = min(ny, 500)

    # Recompute cell_size after capping
    cell_size_x = (xy_max[0] - xy_min[0]) / nx
    cell_size_y = (xy_max[1] - xy_min[1]) / ny

    overlap_grid = np.zeros((ny, nx), dtype=np.int32)
    camera_indices_grid = [[[] for _ in range(nx)] for _ in range(ny)]

    # Precompute grid cell centers
    cell_centers_x = np.linspace(xy_min[0] + cell_size_x / 2, xy_max[0] - cell_size_x / 2, nx)
    cell_centers_y = np.linspace(xy_min[1] + cell_size_y / 2, xy_max[1] - cell_size_y / 2, ny)
    grid_xx, grid_yy = np.meshgrid(cell_centers_x, cell_centers_y)

    for cam_idx, cam in enumerate(cameras):
        center = cam["center"]
        cam_xy = center[:2]
        cam_z = center[2]
        altitude = abs(cam_z - ground_z)

        if altitude < 1e-6:
            altitude = 0.1 / meters_per_unit  # minimum altitude

        # Cone base radius on ground plane
        cone_radius = altitude * math.tan(fov_rad / 2)

        # For oblique cameras, shift the cone center based on look direction
        R = cam["rotation"]
        # Camera looks along -Z in camera frame; in world frame that's -R.T @ [0,0,1]
        look_dir_world = -R.T @ np.array([0, 0, 1])
        look_xy = look_dir_world[:2]
        look_xy_norm = np.linalg.norm(look_xy)

        if look_xy_norm > 0.1:
            # Oblique camera: shift cone center along look direction
            look_xy_unit = look_xy / look_xy_norm
            pitch = math.asin(np.clip(-look_dir_world[2], -1, 1))
            shift = altitude * math.tan(max(0, math.pi / 2 - abs(pitch)))
            shift = min(shift, cone_radius * 3)  # cap shift
            cone_center = cam_xy + look_xy_unit * shift * 0.5
        else:
            cone_center = cam_xy

        # Mark cells within cone radius
        dists = np.sqrt((grid_xx - cone_center[0])**2 + (grid_yy - cone_center[1])**2)
        mask = dists <= cone_radius

        overlap_grid[mask] += 1
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            camera_indices_grid[y][x].append(cam_idx)

    grid_info = {
        "xy_min": xy_min.tolist(),
        "xy_max": xy_max.tolist(),
        "cell_size_x": float(cell_size_x),
        "cell_size_y": float(cell_size_y),
        "shape": (ny, nx),
        "meters_per_unit": float(meters_per_unit),
    }

    return overlap_grid, camera_indices_grid, grid_info


def compute_angular_diversity(cameras, camera_indices_grid, grid_info):
    """Compute angular diversity per grid cell (8-octant binning).

    For each cell, bins the viewing angles from cameras into 8 octants
    (N/NE/E/SE/S/SW/W/NW) and counts unique octants with >=1 camera.

    Returns:
        angular_grid: (H, W) int array — number of unique octants per cell (0-8)
    """
    ny, nx = grid_info["shape"]
    angular_grid = np.zeros((ny, nx), dtype=np.int32)

    xy_min = np.array(grid_info["xy_min"])
    cell_size_x = grid_info["cell_size_x"]
    cell_size_y = grid_info["cell_size_y"]

    for yi in range(ny):
        for xi in range(nx):
            cam_indices = camera_indices_grid[yi][xi]
            if not cam_indices:
                continue

            cell_x = xy_min[0] + (xi + 0.5) * cell_size_x
            cell_y = xy_min[1] + (yi + 0.5) * cell_size_y

            octants = set()
            for ci in cam_indices:
                cam_xy = cameras[ci]["center"][:2]
                dx = cam_xy[0] - cell_x
                dy = cam_xy[1] - cell_y
                angle = math.atan2(dy, dx)  # -pi to pi
                # Map to 8 octants (0-7)
                octant = int((angle + math.pi) / (2 * math.pi / 8)) % 8
                octants.add(octant)

            angular_grid[yi, xi] = len(octants)

    return angular_grid


def compute_scale_diversity(cameras, camera_indices_grid, grid_info, ground_z, meters_per_unit):
    """Compute altitude band diversity per grid cell.

    Bins cameras into 4 altitude bands:
        Band 1: >40m (aerial nadir)
        Band 2: 15-40m (oblique aerial)
        Band 3: 3-15m (elevated ground)
        Band 4: 0-3m (ground level)

    Returns:
        scale_grid: (H, W) int array — number of unique bands per cell (0-4)
    """
    ny, nx = grid_info["shape"]
    scale_grid = np.zeros((ny, nx), dtype=np.int32)

    def _altitude_band(cam):
        alt_m = abs(cam["center"][2] - ground_z) * meters_per_unit
        if alt_m > 40:
            return 1
        elif alt_m > 15:
            return 2
        elif alt_m > 3:
            return 3
        else:
            return 4

    for yi in range(ny):
        for xi in range(nx):
            cam_indices = camera_indices_grid[yi][xi]
            if not cam_indices:
                continue

            bands = set()
            for ci in cam_indices:
                bands.add(_altitude_band(cameras[ci]))

            scale_grid[yi, xi] = len(bands)

    return scale_grid


def compute_combined_score(overlap_grid, angular_grid, scale_grid):
    """Compute combined readiness score per cell.

    combined = overlap_norm * angle_norm * scale_norm, each normalized 0-1.
    Overlap target: 3+, angular target: 3+ octants, scale target: 2+ bands.

    Returns:
        combined_grid: (H, W) float array, 0-1
    """
    overlap_norm = np.clip(overlap_grid / 3.0, 0, 1)
    angle_norm = np.clip(angular_grid / 3.0, 0, 1)
    scale_norm = np.clip(scale_grid / 2.0, 0, 1)

    combined = overlap_norm * angle_norm * scale_norm
    return combined.astype(np.float32)


def run_coverage_analysis(cameras, positions, scene_bounds, ground_z, meters_per_unit,
                          fov_rad=None, grid_resolution=0.5):
    """Run full coverage analysis and return all grids + metadata.

    Returns:
        dict with keys: overlap, angular, scale, combined, grid_info, stats
    """
    overlap_grid, cam_idx_grid, grid_info = compute_bev_coverage(
        cameras, scene_bounds, grid_resolution, fov_rad, ground_z, meters_per_unit
    )
    angular_grid = compute_angular_diversity(cameras, cam_idx_grid, grid_info)
    scale_grid = compute_scale_diversity(cameras, cam_idx_grid, grid_info, ground_z, meters_per_unit)
    combined_grid = compute_combined_score(overlap_grid, angular_grid, scale_grid)

    # Stats
    total_cells = overlap_grid.size
    covered = (overlap_grid > 0).sum()
    well_covered = (combined_grid >= 0.5).sum()
    critical_gaps = (combined_grid < 0.3).sum()

    stats = {
        "total_cells": int(total_cells),
        "covered_cells": int(covered),
        "coverage_pct": float(100 * covered / max(total_cells, 1)),
        "well_covered_cells": int(well_covered),
        "well_covered_pct": float(100 * well_covered / max(total_cells, 1)),
        "critical_gap_cells": int(critical_gaps),
        "critical_gap_pct": float(100 * critical_gaps / max(total_cells, 1)),
        "mean_overlap": float(overlap_grid.mean()),
        "mean_angular": float(angular_grid.mean()),
        "mean_scale": float(scale_grid.mean()),
        "mean_combined": float(combined_grid.mean()),
    }

    print(f"  Coverage: {stats['coverage_pct']:.1f}% cells covered, "
          f"{stats['well_covered_pct']:.1f}% well-covered, "
          f"{stats['critical_gap_pct']:.1f}% critical gaps")
    print(f"  Means: overlap={stats['mean_overlap']:.2f}, "
          f"angular={stats['mean_angular']:.2f}, "
          f"scale={stats['mean_scale']:.2f}, "
          f"combined={stats['mean_combined']:.3f}")

    return {
        "overlap": overlap_grid,
        "angular": angular_grid,
        "scale": scale_grid,
        "combined": combined_grid,
        "camera_indices": cam_idx_grid,
        "grid_info": grid_info,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def _get_font(size=14):
    """Get a PIL font, falling back to default if needed."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_heatmaps(analysis, output_dir):
    """Generate matplotlib heatmap PNGs for each coverage metric.

    Returns dict of {name: path} for the saved images.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    grid_info = analysis["grid_info"]
    extent = [grid_info["xy_min"][0], grid_info["xy_max"][0],
              grid_info["xy_min"][1], grid_info["xy_max"][1]]

    heatmaps = [
        ("overlap", analysis["overlap"], "Overlap Count", "RdYlGn",
         "Number of cameras covering each cell (target: 3+)", None),
        ("angular", analysis["angular"], "Angular Diversity", "RdYlGn",
         "Unique viewing octants per cell (target: 3+ of 8)", 8),
        ("scale", analysis["scale"], "Scale Band Diversity", "RdYlGn",
         "Unique altitude bands per cell (target: 2+ of 4)", 4),
        ("combined", analysis["combined"], "Combined Readiness", "RdYlGn",
         "Overlap x Angular x Scale (normalized 0-1)", 1.0),
    ]

    for name, grid, title, cmap, subtitle, vmax in heatmaps:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if vmax is not None:
            im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap,
                           vmin=0, vmax=vmax, aspect="equal")
        else:
            im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap,
                           aspect="equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
        ax.set_xlabel("X (scene units)")
        ax.set_ylabel("Y (scene units)")

        path = os.path.join(output_dir, f"{name}_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[name] = path
        print(f"  Saved heatmap: {path}")

    return paths


# ---------------------------------------------------------------------------
# Gap identification
# ---------------------------------------------------------------------------

def identify_gaps(combined_grid, grid_info, threshold=0.3, min_cluster_size=4):
    """Identify spatial clusters of low-coverage cells.

    Uses simple grid-based flood-fill clustering (no sklearn dependency).

    Returns list of gap cluster dicts:
        {center_xy, radius, cell_count, deficiency, mean_overlap, mean_angular, mean_scale}
    """
    ny, nx = combined_grid.shape
    low_mask = combined_grid < threshold

    # Flood-fill clustering
    visited = np.zeros_like(low_mask, dtype=bool)
    clusters = []

    def _flood_fill(start_y, start_x):
        """BFS flood fill to find connected low-score cells."""
        queue = [(start_y, start_x)]
        cells = []
        visited[start_y, start_x] = True
        while queue:
            cy, cx = queue.pop(0)
            cells.append((cy, cx))
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny2, nx2 = cy + dy, cx + dx
                if 0 <= ny2 < ny and 0 <= nx2 < nx:
                    if low_mask[ny2, nx2] and not visited[ny2, nx2]:
                        visited[ny2, nx2] = True
                        queue.append((ny2, nx2))
        return cells

    for yi in range(ny):
        for xi in range(nx):
            if low_mask[yi, xi] and not visited[yi, xi]:
                cells = _flood_fill(yi, xi)
                if len(cells) >= min_cluster_size:
                    clusters.append(cells)

    # Convert clusters to gap descriptors
    xy_min = np.array(grid_info["xy_min"])
    cell_size_x = grid_info["cell_size_x"]
    cell_size_y = grid_info["cell_size_y"]

    gaps = []
    for cells in clusters:
        ys = np.array([c[0] for c in cells])
        xs = np.array([c[1] for c in cells])

        center_x = xy_min[0] + (xs.mean() + 0.5) * cell_size_x
        center_y = xy_min[1] + (ys.mean() + 0.5) * cell_size_y

        # Radius = max distance from center to any cell
        cell_xs = xy_min[0] + (xs + 0.5) * cell_size_x
        cell_ys = xy_min[1] + (ys + 0.5) * cell_size_y
        dists = np.sqrt((cell_xs - center_x)**2 + (cell_ys - center_y)**2)
        radius = float(dists.max()) + max(cell_size_x, cell_size_y)

        gaps.append({
            "center_xy": [float(center_x), float(center_y)],
            "radius": float(radius),
            "cell_count": len(cells),
        })

    # Sort by size (largest gaps first)
    gaps.sort(key=lambda g: -g["cell_count"])

    print(f"  Identified {len(gaps)} gap clusters ({sum(g['cell_count'] for g in gaps)} total cells)")
    return gaps


# =========================================================================
# Phase 2: Gap-Filling Camera Placement + Image Generation
# =========================================================================

def plan_gap_cameras(gaps, existing_cameras, ground_z, drone_agl, meters_per_unit, z_sign=1.0):
    """Plan cameras to fill coverage gaps across all altitude bands.

    Returns dict of {band_name: [camera_dicts]}.
    """
    from scipy.spatial.transform import Rotation

    drone_centers = np.array([c["center"] for c in existing_cameras])
    drone_xy = drone_centers[:, :2]
    xy_min = drone_xy.min(axis=0)
    xy_max = drone_xy.max(axis=0)
    xy_range = xy_max - xy_min

    bands = {}

    # --- Band 2: Oblique aerial (15-30m AGL), 45deg pitch, N/S/E/W ---
    alt_30 = ground_z + z_sign * 30.0 / meters_per_unit
    alt_15 = ground_z + z_sign * 15.0 / meters_per_unit
    yaw_angles = [0, 90, 180, 270]

    candidates_b2 = []
    for gap in gaps:
        cx, cy = gap["center_xy"]
        for alt in [alt_30, alt_15]:
            for yaw_deg in yaw_angles:
                yaw_rad = math.radians(yaw_deg)
                R = Rotation.from_euler("zyx", [yaw_rad, math.radians(-45), 0]).as_matrix()
                center = np.array([cx, cy, alt])
                t_vec = -R @ center
                candidates_b2.append({
                    "center": center,
                    "rotation": R,
                    "translation": t_vec,
                    "yaw_deg": yaw_deg,
                    "alt_m": round(abs(alt - ground_z) * meters_per_unit, 1),
                    "camera_type": "oblique_aerial",
                    "band": 2,
                })

    # Also add cameras at regular grid for broader coverage
    grid_spacing = 15.0 / meters_per_unit
    gx_vals = np.arange(xy_min[0] + 0.1 * xy_range[0], xy_max[0] - 0.1 * xy_range[0], grid_spacing)
    gy_vals = np.arange(xy_min[1] + 0.1 * xy_range[1], xy_max[1] - 0.1 * xy_range[1], grid_spacing)
    for gx in gx_vals:
        for gy in gy_vals:
            for alt in [alt_30, alt_15]:
                for yaw_deg in yaw_angles:
                    yaw_rad = math.radians(yaw_deg)
                    R = Rotation.from_euler("zyx", [yaw_rad, math.radians(-45), 0]).as_matrix()
                    center = np.array([gx, gy, alt])
                    t_vec = -R @ center
                    candidates_b2.append({
                        "center": center,
                        "rotation": R,
                        "translation": t_vec,
                        "yaw_deg": yaw_deg,
                        "alt_m": round(abs(alt - ground_z) * meters_per_unit, 1),
                        "camera_type": "oblique_aerial",
                        "band": 2,
                    })

    if candidates_b2:
        positions_b2 = np.array([c["center"] for c in candidates_b2])
        indices = farthest_point_sample(positions_b2, 40)
        bands["band2_oblique"] = [candidates_b2[i] for i in indices]
        print(f"  Band 2 (oblique): {len(candidates_b2)} candidates -> {len(bands['band2_oblique'])} sampled")

    # --- Band 3: Elevated ground (5m AGL), 30deg pitch, 8 directions ---
    alt_5m = ground_z + z_sign * 5.0 / meters_per_unit
    yaw_angles_8 = np.linspace(0, 360, 8, endpoint=False)
    grid_spacing_b3 = 10.0 / meters_per_unit

    candidates_b3 = []
    gx_vals = np.arange(xy_min[0] + 0.1 * xy_range[0], xy_max[0] - 0.1 * xy_range[0], grid_spacing_b3)
    gy_vals = np.arange(xy_min[1] + 0.1 * xy_range[1], xy_max[1] - 0.1 * xy_range[1], grid_spacing_b3)
    for gx in gx_vals:
        for gy in gy_vals:
            for yaw_deg in yaw_angles_8:
                yaw_rad = math.radians(yaw_deg)
                R = Rotation.from_euler("zyx", [yaw_rad, math.radians(-30), 0]).as_matrix()
                center = np.array([gx, gy, alt_5m])
                t_vec = -R @ center
                candidates_b3.append({
                    "center": center,
                    "rotation": R,
                    "translation": t_vec,
                    "yaw_deg": float(yaw_deg),
                    "camera_type": "elevated_ground",
                    "band": 3,
                })

    if candidates_b3:
        positions_b3 = np.array([c["center"] for c in candidates_b3])
        indices = farthest_point_sample(positions_b3, 32)
        bands["band3_elevated"] = [candidates_b3[i] for i in indices]
        print(f"  Band 3 (elevated): {len(candidates_b3)} candidates -> {len(bands['band3_elevated'])} sampled")

    # --- Band 4: Ground level (1.7m AGL), 0deg pitch (horizon), 8 directions ---
    alt_eye = ground_z + z_sign * 1.7 / meters_per_unit
    grid_spacing_b4 = 8.0 / meters_per_unit

    candidates_b4 = []
    gx_vals = np.arange(xy_min[0] + 0.1 * xy_range[0], xy_max[0] - 0.1 * xy_range[0], grid_spacing_b4)
    gy_vals = np.arange(xy_min[1] + 0.1 * xy_range[1], xy_max[1] - 0.1 * xy_range[1], grid_spacing_b4)
    for gx in gx_vals:
        for gy in gy_vals:
            for yaw_deg in yaw_angles_8:
                yaw_rad = math.radians(yaw_deg)
                R = Rotation.from_euler("zyx", [yaw_rad, math.radians(-5), 0]).as_matrix()
                center = np.array([gx, gy, alt_eye])
                t_vec = -R @ center
                candidates_b4.append({
                    "center": center,
                    "rotation": R,
                    "translation": t_vec,
                    "yaw_deg": float(yaw_deg),
                    "camera_type": "ground_level",
                    "band": 4,
                })

    if candidates_b4:
        positions_b4 = np.array([c["center"] for c in candidates_b4])
        indices = farthest_point_sample(positions_b4, 32)
        bands["band4_ground"] = [candidates_b4[i] for i in indices]
        print(f"  Band 4 (ground): {len(candidates_b4)} candidates -> {len(bands['band4_ground'])} sampled")

    # --- Detail: 1.2m at dense point clusters, opposing camera pairs ---
    alt_detail = ground_z + z_sign * 1.2 / meters_per_unit

    try:
        from scipy.cluster.vq import kmeans2

        # Use gap centers as POIs, supplemented by point cloud clusters
        n_pois = min(8, max(len(gaps), 4))
        if gaps:
            poi_centers = np.array([g["center_xy"] for g in gaps[:n_pois]])
        else:
            # Fall back to k-means on drone positions
            poi_centers, _ = kmeans2(drone_xy.astype(np.float64), n_pois, minit="points")

        detail_cams = []
        offset = 2.0 / meters_per_unit
        for cx, cy in poi_centers:
            for angle_idx in range(2):
                angle = math.radians(angle_idx * 180)
                cam_x = cx + offset * math.cos(angle)
                cam_y = cy + offset * math.sin(angle)
                look_dir = np.array([cx - cam_x, cy - cam_y])
                look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)
                yaw = math.atan2(look_dir[1], look_dir[0])
                R = Rotation.from_euler("zyx", [yaw, math.radians(-10), 0]).as_matrix()
                center = np.array([cam_x, cam_y, alt_detail])
                t_vec = -R @ center
                detail_cams.append({
                    "center": center,
                    "rotation": R,
                    "translation": t_vec,
                    "yaw_deg": float(math.degrees(yaw)),
                    "poi_center": (float(cx), float(cy)),
                    "camera_type": "detail_closeup",
                    "band": "detail",
                })

        if detail_cams:
            bands["detail"] = detail_cams
            print(f"  Detail: {len(detail_cams)} cameras at {len(poi_centers)} POIs")

    except Exception as e:
        print(f"  Detail camera planning failed: {e}")

    total = sum(len(v) for v in bands.values())
    print(f"  Total planned cameras: {total}")
    return bands


# ---------------------------------------------------------------------------
# FLUX image generation per band
# ---------------------------------------------------------------------------

def generate_band_images(band_name, cameras, model_path, scene_path, output_dir,
                         prompt, params, poses, scene_images_dir):
    """Generate images for one altitude band using FLUX + ControlNet + IP-Adapter.

    Returns (image_paths, cameras_used).
    """
    import torch
    from PIL import Image as PILImage
    from diffusers import FluxControlNetPipeline, FluxControlNetModel

    band_dir = os.path.join(output_dir, band_name)
    images_dir = os.path.join(band_dir, "images")
    rgb_dir = os.path.join(band_dir, "rgb")
    depth_dir = os.path.join(band_dir, "depth")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Render reference RGB from splat
    print(f"  Rendering reference RGB for {band_name}...")
    _render_rgb_from_splat(model_path, cameras, rgb_dir, scene_path)

    print(f"  Estimating depth maps for {band_name}...")
    _estimate_depth_maps(rgb_dir, depth_dir)

    # Monkey-patch: PyTorch 2.4 doesn't support enable_gqa
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _patched_sdpa(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

    print(f"  Loading FLUX.1-dev + ControlNet-depth for {band_name}...")
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
            print(f"  Skipping {band_name} view {idx} (too dark, brightness={avg_brightness:.1f})")
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

        # IP-Adapter reference from nearest aerial image
        if ip_adapter_loaded:
            ip_ref = extract_nearest_aerial_crop(cameras[idx], poses, scene_images_dir)
            if ip_ref is not None:
                gen_kwargs["ip_adapter_image"] = ip_ref

        result = pipe(**gen_kwargs).images[0]
        result = result.resize(
            (params["output_size"], params["output_size"]), PILImage.LANCZOS
        )

        out_name = f"{band_name}_{idx:03d}.jpg"
        out_path = os.path.join(images_dir, out_name)
        result.save(out_path, "JPEG", quality=95)
        generated_paths.append(out_path)
        generated_cameras.append(cameras[idx])

        if (idx + 1) % 5 == 0:
            print(f"  Generated {idx + 1}/{len(cameras)} {band_name} views")

    print(f"  Generated {len(generated_paths)} {band_name} images")

    del pipe, controlnet
    torch.cuda.empty_cache()

    return generated_paths, generated_cameras


# ---------------------------------------------------------------------------
# Mosaic stitching (spatial layout)
# ---------------------------------------------------------------------------

def mosaic_by_band(image_paths, cameras, band_name, drone_agl=63):
    """Create a spatially-arranged mosaic for a band of images.

    Places thumbnails according to their XY position on a canvas.
    """
    if not image_paths:
        return None

    positions = np.array([c["center"][:2] for c in cameras])
    xy_min = positions.min(axis=0)
    xy_max = positions.max(axis=0)
    xy_range = xy_max - xy_min
    xy_range[xy_range == 0] = 1

    thumb_size = 128
    padding = 60
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

        norm = (pos - xy_min) / xy_range
        cx = int(norm[0] * (canvas_w - 2 * padding - thumb_size)) + padding
        cy = int((1 - norm[1]) * (canvas_h - 2 * padding - thumb_size)) + padding
        canvas.paste(img, (cx, cy))

    font = _get_font(16)
    band_labels = {
        "band2_oblique": "Band 2: Oblique Aerial (15-30m)",
        "band3_elevated": "Band 3: Elevated Ground (5m)",
        "band4_ground": "Band 4: Ground Level (1.7m)",
        "detail": "Detail: Close-ups (1.2m)",
    }
    label = band_labels.get(band_name, band_name)
    draw.text((10, 10), f"{label} ({len(image_paths)} images)", fill="white", font=font)

    return canvas


# ---------------------------------------------------------------------------
# Gemini scoring for bands
# ---------------------------------------------------------------------------

def score_band_with_gemini(mosaic_path, band_key):
    """Score a band mosaic with Gemini. Returns (score, feedback)."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return 7, "No Gemini API key available."

    try:
        from google import genai
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
        from google import genai

    img = Image.open(mosaic_path).convert("RGB")
    max_dim = max(img.size)
    if max_dim > 2048:
        scale = 2048 / max_dim
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    image_bytes = buf.getvalue()

    prompt = BAND_SCORE_PROMPTS.get(band_key, BAND_SCORE_PROMPTS.get(2))

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
        score_match = re.search(r"SCORE:\s*(\d+)/10", text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 5
        return score, text
    except Exception as e:
        print(f"  Gemini scoring failed: {e}")
        return 5, f"Scoring failed: {e}"


# =========================================================================
# Phase 3: Upload + Report
# =========================================================================

def upload_analysis(analysis, heatmap_paths, gaps, output_dir, job_id):
    """Upload analysis artifacts to CDN.

    Returns dict of CDN URLs.
    """
    urls = {}

    # Upload heatmaps
    for name, path in heatmap_paths.items():
        remote_key = f"coverage/{job_id}/analysis/{name}_heatmap.png"
        url = upload_to_cdn(path, remote_key)
        if url:
            urls[f"{name}_heatmap"] = url

    # Save and upload gaps.json
    gaps_path = os.path.join(output_dir, "analysis", "gaps.json")
    os.makedirs(os.path.dirname(gaps_path), exist_ok=True)
    with open(gaps_path, "w") as f:
        json.dump(gaps, f, indent=2)

    remote_key = f"coverage/{job_id}/analysis/gaps.json"
    url = upload_to_cdn(gaps_path, remote_key)
    if url:
        urls["gaps"] = url

    return urls


def upload_generated_images(band_images, band_cameras, mosaics, output_dir, job_id):
    """Upload generated images and mosaics to CDN.

    Returns dict of CDN URLs.
    """
    urls = {}

    for band_name, mosaic in mosaics.items():
        if mosaic is None:
            continue
        mosaic_path = os.path.join(output_dir, "mosaics", f"{band_name}_mosaic.jpg")
        os.makedirs(os.path.dirname(mosaic_path), exist_ok=True)
        mosaic.save(mosaic_path, "JPEG", quality=90)

        remote_key = f"coverage/{job_id}/mosaics/{band_name}_mosaic.jpg"
        url = upload_to_cdn(mosaic_path, remote_key)
        if url:
            urls[f"{band_name}_mosaic"] = url

    return urls


def build_cameras_json(original_poses, planned_cameras, output_dir):
    """Write combined cameras.json (original + generated) for future reference."""
    all_cameras = []

    for pose in original_poses:
        all_cameras.append({
            "image_name": pose.get("image_name", ""),
            "center": pose["center"].tolist() if hasattr(pose["center"], "tolist") else pose["center"],
            "type": "original_drone",
        })

    for band_name, cameras in planned_cameras.items():
        for cam in cameras:
            all_cameras.append({
                "center": cam["center"].tolist() if hasattr(cam["center"], "tolist") else list(cam["center"]),
                "type": cam.get("camera_type", band_name),
                "band": str(cam.get("band", band_name)),
                "yaw_deg": cam.get("yaw_deg", 0),
            })

    cameras_path = os.path.join(output_dir, "cameras.json")
    with open(cameras_path, "w") as f:
        json.dump(all_cameras, f, indent=2)
    print(f"  Wrote {len(all_cameras)} cameras to {cameras_path}")
    return cameras_path


def send_final_report(report, before_stats, after_stats, heatmap_urls, mosaic_urls,
                      webhook_url, job_id):
    """Send final Slack report with before/after comparison."""
    if not webhook_url:
        return

    before_pct = before_stats.get("well_covered_pct", 0)
    after_pct = after_stats.get("well_covered_pct", 0) if after_stats else before_pct

    total_original = report.get("original_count", 0)
    total_generated = report.get("generated_count", 0)

    ready = after_pct >= 80
    status_emoji = ":white_check_mark:" if ready else ":bar_chart:"
    ready_text = "READY for splat training" if ready else "MORE images needed"

    scores_text = ""
    for band, data in report.get("band_scores", {}).items():
        scores_text += f"\n  {band}: {data.get('score', '?')}/10"

    header = f"{status_emoji} Coverage Analysis Complete | {ready_text}"

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": header[:150]}},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Before:* {before_pct:.1f}% well-covered | "
                    f"*After:* {after_pct:.1f}% well-covered\n"
                    f"*Images:* {total_original} original + {total_generated} generated = "
                    f"{total_original + total_generated} total\n"
                    f"*Per-band Gemini scores:*{scores_text}"
                ),
            },
        },
    ]

    # Add heatmap images if available
    for key in ["combined_heatmap"]:
        url = heatmap_urls.get(key)
        if url:
            blocks.append({
                "type": "image",
                "image_url": url,
                "alt_text": f"Combined readiness heatmap",
            })

    payload = json.dumps({"text": header, "blocks": blocks})

    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  Slack report failed: {e}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Coverage Analysis + Gap-Filling Image Generation")
    parser.add_argument("--model_path", required=True, help="Path to aerial splat model")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory with images/")
    parser.add_argument("--output_dir", required=True, help="Output directory for analysis + generated images")
    parser.add_argument("--drone_agl", type=float, default=63, help="Drone altitude AGL in meters")
    parser.add_argument("--job_id", default="coverage-v1", help="Job ID for CDN paths + Slack")
    parser.add_argument("--max_iterations", type=int, default=2, help="Max FLUX refinement iterations per band")
    parser.add_argument("--grid_resolution", type=float, default=0.5, help="Coverage grid resolution in meters")
    parser.add_argument("--gap_threshold", type=float, default=0.3, help="Combined score threshold for gaps")
    parser.add_argument("--target_score", type=int, default=7, help="Target Gemini score per band")
    parser.add_argument("--skip_generation", action="store_true", help="Analysis only, skip FLUX generation")
    parser.add_argument("--slack_webhook_url", default="", help="Slack webhook URL")
    args = parser.parse_args()

    if not args.slack_webhook_url:
        args.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")

    os.makedirs(args.output_dir, exist_ok=True)

    report = {
        "job_id": args.job_id,
        "drone_agl": args.drone_agl,
        "band_scores": {},
    }

    # =================================================================
    # Load scene data
    # =================================================================
    print("Loading point cloud...")
    positions, ply_path = load_point_cloud(args.model_path)
    print(f"  {len(positions)} Gaussians from {ply_path}")

    ground_z = estimate_ground_level(positions)
    drone_z = np.percentile(positions[:, 2], 95)
    print(f"  Ground Z: {ground_z:.4f}, Drone Z: {drone_z:.4f}")

    print("Loading camera poses...")
    poses = load_camera_poses(args.scene_path)
    print(f"  {len(poses)} camera poses")

    # Compute scene scale
    avg_cam_z = np.mean([p["center"][2] for p in poses])
    scene_height = abs(avg_cam_z - ground_z)
    meters_per_unit = args.drone_agl / max(scene_height, 1e-6)
    z_sign = 1.0 if avg_cam_z > ground_z else -1.0
    print(f"  Scale: {meters_per_unit:.2f} m/unit, z_sign={z_sign:+.0f}")

    # Scene bounds
    drone_centers = np.array([p["center"] for p in poses])
    drone_xy = drone_centers[:, :2]
    xy_min = drone_xy.min(axis=0) - 5.0 / meters_per_unit  # 5m padding
    xy_max = drone_xy.max(axis=0) + 5.0 / meters_per_unit
    scene_bounds = (xy_min, xy_max)

    # Load FOV from COLMAP intrinsics
    fov_x, fov_y = load_colmap_intrinsics(args.scene_path)
    print(f"  FOV: {math.degrees(fov_x):.1f} x {math.degrees(fov_y):.1f} degrees")

    scene_images_dir = os.path.join(args.scene_path, "images")

    # Extract GPS + caption
    print("Extracting GPS + captioning scene...")
    lat, lon, _alt = extract_gps_from_images(scene_images_dir)
    google_maps_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    prompt = caption_aerial_scene(scene_images_dir, lat=lat, lon=lon, google_maps_key=google_maps_key)
    report["prompt"] = prompt[:200]

    notify_slack_text(
        f"Starting coverage analysis ({len(poses)} cameras, AGL={args.drone_agl}m)",
        args.slack_webhook_url, args.job_id,
    )

    # =================================================================
    # Phase 1: Coverage Analysis (before generation)
    # =================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: COVERAGE ANALYSIS (BEFORE)")
    print(f"{'='*60}")

    # Build camera list (original poses)
    all_cameras_before = []
    for pose in poses:
        all_cameras_before.append({
            "center": pose["center"],
            "rotation": pose["rotation"],
            "image_name": pose.get("image_name", ""),
        })

    analysis_before = run_coverage_analysis(
        all_cameras_before, positions, scene_bounds, ground_z, meters_per_unit,
        fov_rad=fov_x, grid_resolution=args.grid_resolution,
    )

    # Generate heatmaps
    heatmap_dir = os.path.join(args.output_dir, "analysis")
    heatmap_paths = generate_heatmaps(analysis_before, heatmap_dir)

    # Identify gaps
    gaps = identify_gaps(
        analysis_before["combined"], analysis_before["grid_info"],
        threshold=args.gap_threshold,
    )

    # Upload analysis
    heatmap_urls = upload_analysis(analysis_before, heatmap_paths, gaps, args.output_dir, args.job_id)

    report["before_stats"] = analysis_before["stats"]
    report["gap_count"] = len(gaps)
    report["original_count"] = len(poses)

    # Notify Slack with combined heatmap
    combined_url = heatmap_urls.get("combined_heatmap")
    if combined_url and args.slack_webhook_url:
        stats = analysis_before["stats"]
        notify_slack_with_mosaic(
            tier=0, iteration=0,
            score=int(stats["well_covered_pct"] / 10),
            feedback=(
                f"Coverage: {stats['coverage_pct']:.0f}% covered, "
                f"{stats['well_covered_pct']:.0f}% well-covered, "
                f"{stats['critical_gap_pct']:.0f}% critical gaps. "
                f"{len(gaps)} gap clusters found."
            ),
            mosaic_url=combined_url,
            webhook_url=args.slack_webhook_url,
            job_id=args.job_id,
        )

    if args.skip_generation:
        print("\n--skip_generation set, exiting after analysis.")
        report_path = os.path.join(args.output_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_path}")
        return

    # =================================================================
    # Phase 2: Gap-Filling Camera Placement + Image Generation
    # =================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: GAP-FILLING IMAGE GENERATION")
    print(f"{'='*60}")

    # Plan cameras
    print("Planning gap-filling cameras...")
    planned_cameras = plan_gap_cameras(
        gaps, all_cameras_before, ground_z, args.drone_agl, meters_per_unit, z_sign,
    )

    # Write cameras.json
    build_cameras_json(poses, planned_cameras, args.output_dir)

    # Generate images per band with iterative refinement
    all_generated_images = {}
    all_generated_cameras = {}
    mosaics = {}
    total_generated = 0

    band_order = ["band2_oblique", "band3_elevated", "band4_ground", "detail"]
    for band_name in band_order:
        cameras = planned_cameras.get(band_name)
        if not cameras:
            continue

        band_key = cameras[0].get("band", 2)
        params = dict(BAND_PARAMS.get(band_key, BAND_PARAMS[2]))

        print(f"\n--- {band_name}: {len(cameras)} cameras ---")
        notify_slack_text(
            f"Generating {len(cameras)} {band_name} images...",
            args.slack_webhook_url, args.job_id,
        )

        best_score = 0
        best_images = []
        best_cameras_used = []

        for iteration in range(1, args.max_iterations + 1):
            print(f"\n  Iteration {iteration}/{args.max_iterations}, params: {params}")

            iter_dir = os.path.join(args.output_dir, "generated", band_name, f"iter_{iteration}")

            gen_images, gen_cameras = generate_band_images(
                band_name, cameras, args.model_path, args.scene_path,
                iter_dir, prompt, params, poses, scene_images_dir,
            )

            if not gen_images:
                print(f"  No images generated for {band_name}")
                continue

            # Stitch mosaic
            mosaic = mosaic_by_band(gen_images, gen_cameras, band_name, args.drone_agl)
            mosaic_path = os.path.join(iter_dir, f"{band_name}_mosaic.jpg")
            if mosaic:
                mosaic.save(mosaic_path, "JPEG", quality=90)

            # Upload mosaic
            remote_key = f"coverage/{args.job_id}/mosaics/{band_name}_iter{iteration}_mosaic.jpg"
            mosaic_url = upload_to_cdn(mosaic_path, remote_key)

            # Score with Gemini
            score, feedback = score_band_with_gemini(mosaic_path, band_key)
            print(f"  {band_name} iter {iteration} score: {score}/10")

            # Notify Slack
            if args.slack_webhook_url:
                notify_slack_with_mosaic(
                    tier=band_key if isinstance(band_key, int) else 5,
                    iteration=iteration, score=score, feedback=feedback,
                    mosaic_url=mosaic_url,
                    webhook_url=args.slack_webhook_url, job_id=args.job_id,
                )

            if score > best_score:
                best_score = score
                best_images = gen_images
                best_cameras_used = gen_cameras
                if mosaic:
                    mosaics[band_name] = mosaic

            if score >= args.target_score:
                print(f"  Target score reached ({score}>={args.target_score})")
                break

            if iteration < args.max_iterations:
                params = adjust_params(params, feedback)

        all_generated_images[band_name] = best_images
        all_generated_cameras[band_name] = best_cameras_used
        total_generated += len(best_images)

        report["band_scores"][band_name] = {
            "score": best_score,
            "count": len(best_images),
        }

    report["generated_count"] = total_generated

    # Upload mosaics
    mosaic_urls = upload_generated_images(
        all_generated_images, all_generated_cameras, mosaics,
        args.output_dir, args.job_id,
    )

    # =================================================================
    # Phase 3: Re-Analysis + Final Report
    # =================================================================
    print(f"\n{'='*60}")
    print("PHASE 3: RE-ANALYSIS (AFTER GENERATION)")
    print(f"{'='*60}")

    # Build combined camera list
    all_cameras_after = list(all_cameras_before)
    for band_name, cameras in all_generated_cameras.items():
        for cam in cameras:
            all_cameras_after.append({
                "center": cam["center"],
                "rotation": cam["rotation"],
            })

    analysis_after = run_coverage_analysis(
        all_cameras_after, positions, scene_bounds, ground_z, meters_per_unit,
        fov_rad=fov_x, grid_resolution=args.grid_resolution,
    )

    # Generate after-heatmaps
    after_heatmap_dir = os.path.join(args.output_dir, "analysis_after")
    after_heatmap_paths = generate_heatmaps(analysis_after, after_heatmap_dir)

    # Upload after-heatmaps
    after_urls = {}
    for name, path in after_heatmap_paths.items():
        remote_key = f"coverage/{args.job_id}/analysis_after/{name}_heatmap.png"
        url = upload_to_cdn(path, remote_key)
        if url:
            after_urls[f"{name}_heatmap_after"] = url

    report["after_stats"] = analysis_after["stats"]

    # Print comparison
    before_wc = analysis_before["stats"]["well_covered_pct"]
    after_wc = analysis_after["stats"]["well_covered_pct"]
    print(f"\n  Well-covered: {before_wc:.1f}% -> {after_wc:.1f}% ({after_wc - before_wc:+.1f}%)")
    print(f"  Cameras: {len(all_cameras_before)} -> {len(all_cameras_after)}")

    readiness = "READY" if after_wc >= 80 else "NOT READY"
    print(f"  Readiness: {readiness} for splat training")
    report["readiness"] = readiness

    # Save report
    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")

    # Upload report
    upload_to_cdn(report_path, f"coverage/{args.job_id}/report.json")

    # Send final Slack report
    all_urls = {**heatmap_urls, **mosaic_urls, **after_urls}
    send_final_report(
        report,
        analysis_before["stats"],
        analysis_after["stats"],
        all_urls, mosaic_urls,
        args.slack_webhook_url, args.job_id,
    )

    # Summary
    print(f"\n{'='*60}")
    print("COVERAGE ANALYSIS + GAP FILLING COMPLETE")
    print(f"{'='*60}")
    print(f"  Original images: {report['original_count']}")
    print(f"  Generated images: {report['generated_count']}")
    print(f"  Coverage: {before_wc:.1f}% -> {after_wc:.1f}%")
    for band, data in report.get("band_scores", {}).items():
        print(f"  {band}: score={data['score']}/10, {data['count']} images")
    print(f"  Readiness: {readiness}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
