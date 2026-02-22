#!/usr/bin/env python3
"""
Stage 4: Game-Level Scene Construction

Builds a textured 3D scene (.glb) from ODM outputs (orthophoto + DSM/DTM) for
ground-level viewing. The GLB crossfades with the Gaussian splat at altitude.

Seven sequential steps, individually runnable via --step:
  1. terrain   — DSM → mesh with orthophoto UVs
  2. superres  — Real-ESRGAN 4x upscale of orthophoto
  3. segment   — SAM2 + Gemini material classification
  4. materials — PBR texture assignment + color-shifting
  5. assets    — Procedural trees/fences/shrubs from segmentation
  6. export    — Assemble + export GLB with PBR materials
  7. lighting  — Solar position from EXIF for directional light metadata

Usage:
    python build_game_scene.py \
        --orthophoto outputs/odm/odm_orthophoto/odm_orthophoto.tif \
        --dsm outputs/odm/odm_dem/dsm.tif \
        --dtm outputs/odm/odm_dem/dtm.tif \
        --images inputs/ \
        --scene_path outputs/scene \
        --model_path outputs/descent/final \
        --output outputs/scene.glb \
        --texture_library /mnt/splatwalk/textures/ \
        --scene_scale 50.0 \
        --gemini_api_key $GEMINI_API_KEY \
        --step all
"""

import argparse
import json
import os
import struct
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

# Add script directory to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_intermediate(data, name, intermediates_dir):
    """Save intermediate result for debugging."""
    os.makedirs(intermediates_dir, exist_ok=True)
    path = os.path.join(intermediates_dir, name)
    if isinstance(data, np.ndarray):
        if data.dtype == np.uint8 and data.ndim in (2, 3):
            cv2.imwrite(path, data)
        else:
            np.save(path, data)
    elif isinstance(data, dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    else:
        with open(path, "w") as f:
            f.write(str(data))
    print(f"  Saved intermediate: {path}")


def load_centroid_from_transform(scene_transform_path):
    """Load centroid from scene_transform.json (from generate_aerial_glb.py)."""
    with open(scene_transform_path) as f:
        data = json.load(f)
    centroid = np.array(data["centroid_utm"], dtype=np.float32)
    print(f"  Centroid from scene_transform: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    return centroid


def load_ply_centroid(model_path):
    """Load centroid from the PLY file, matching compress_splat.py line 237."""
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

    # Filter same as compress_splat: opacity >= 0.10 + position < 3σ
    field_names = set(vertex.data.dtype.names)
    if "opacity" in field_names:
        opacity_logit = np.array(vertex["opacity"], dtype=np.float32)
        opacity = 1 / (1 + np.exp(-np.clip(opacity_logit, -20, 20)))
        keep = opacity >= 0.10
    else:
        keep = np.ones(len(positions), dtype=bool)

    if "scale_0" in field_names:
        scales = np.exp(np.stack([
            np.array(vertex["scale_0"], dtype=np.float32),
            np.array(vertex["scale_1"], dtype=np.float32),
            np.array(vertex["scale_2"], dtype=np.float32),
        ], axis=-1))
        max_scale = scales.max(axis=-1)
        scale_cap = np.percentile(max_scale[keep], 97.0)
        keep &= max_scale <= scale_cap
        min_scale = scales.min(axis=-1)
        min_scale[min_scale == 0] = 1e-10
        keep &= (max_scale / min_scale) <= 25.0

    centroid_pos = positions[keep]
    centroid = centroid_pos.mean(axis=0)
    dists = np.linalg.norm(centroid_pos - centroid, axis=-1)
    dist_std = dists.std()
    pos_mask = dists <= 3.0 * dist_std
    centroid = centroid_pos[pos_mask].mean(axis=0)

    print(f"  PLY centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    return centroid


def get_colmap_poses(scene_path):
    """Read COLMAP camera poses from images.txt or images.bin."""
    sparse_dirs = sorted(Path(scene_path).glob("sparse_*/0"))
    if not sparse_dirs:
        sparse_dirs = sorted(Path(scene_path).glob("sparse/0"))
    if not sparse_dirs:
        raise FileNotFoundError(f"No COLMAP sparse dir in {scene_path}")

    sparse_dir = sparse_dirs[0]

    # Try images.txt first
    images_txt = sparse_dir / "images.txt"
    if images_txt.exists():
        poses = []
        with open(images_txt) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    name = parts[9]
                    # Convert R_w2c, T_w2c to camera center
                    from scipy.spatial.transform import Rotation
                    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                    center = -R.T @ np.array([tx, ty, tz])
                    poses.append({"image_name": name, "center": center})
        return poses

    # Try images.bin
    images_bin = sparse_dir / "images.bin"
    if images_bin.exists():
        poses = []
        with open(images_bin, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                tx, ty, tz = struct.unpack("<3d", f.read(24))
                camera_id = struct.unpack("<I", f.read(4))[0]
                name = b""
                while True:
                    ch = f.read(1)
                    if ch == b"\x00":
                        break
                    name += ch
                name = name.decode("utf-8")
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.read(num_points2d * 24)  # skip point2D data

                from scipy.spatial.transform import Rotation
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                center = -R.T @ np.array([tx, ty, tz])
                poses.append({"image_name": name, "center": center})
        return poses

    raise FileNotFoundError(f"No images.txt or images.bin in {sparse_dir}")


# ---------------------------------------------------------------------------
# Step 1: Terrain mesh from DSM (with flat fallback)
# ---------------------------------------------------------------------------

def _build_flat_terrain(orthophoto_path, model_path, scene_path,
                        scene_scale, intermediates_dir, scene_transform_path=""):
    """Fallback: build a flat terrain mesh from orthophoto bounds when DSM is unavailable.

    Uses the orthophoto pixel dimensions + COLMAP/PLY centroid to create a flat
    grid at ground Z, with UVs mapping to the orthophoto.
    """
    import rasterio

    print("  Building flat terrain (DSM fallback)...")

    # Get orthophoto bounds for terrain extent
    try:
        with rasterio.open(orthophoto_path) as src:
            ortho_bounds = src.bounds
            ortho_crs = src.crs
        x_extent = ortho_bounds.right - ortho_bounds.left
        y_extent = ortho_bounds.top - ortho_bounds.bottom
        print(f"  Orthophoto bounds: {x_extent:.1f}m x {y_extent:.1f}m")
    except Exception as e:
        print(f"  Cannot read orthophoto bounds: {e}")
        # Fallback: use PLY extent
        x_extent = 100.0
        y_extent = 100.0

    # Load centroid
    if scene_transform_path and os.path.exists(scene_transform_path):
        centroid = load_centroid_from_transform(scene_transform_path)
        with open(scene_transform_path) as f:
            st_data = json.load(f)
        ground_z_scaled = st_data.get("ground_z_scaled", 0.0)
    elif model_path:
        centroid = load_ply_centroid(model_path)
        from plyfile import PlyData
        ckpt_dirs = sorted(Path(model_path).glob("point_cloud/iteration_*"))
        ply_path = str(ckpt_dirs[-1] / "point_cloud.ply")
        ply_data = PlyData.read(ply_path)
        vertex = ply_data["vertex"]
        z_vals = np.array(vertex["z"], dtype=np.float32)
        ground_z_raw = float(np.percentile(z_vals, 5))
        ground_z_scaled = (ground_z_raw - centroid[2]) * scene_scale
    else:
        # Auto-center from orthophoto bounds
        centroid = np.array([
            (ortho_bounds.left + ortho_bounds.right) / 2,
            (ortho_bounds.bottom + ortho_bounds.top) / 2,
            0.0,
        ], dtype=np.float32)
        ground_z_scaled = 0.0

    # Try geo transform to get proper COLMAP alignment (requires model_path + scene_path)
    has_geo_transform = False
    if model_path and scene_path:
        try:
            from align_geo_colmap import extract_exif_gps, compute_geo_to_colmap_transform, project_gps_to_utm
            input_dir = os.path.join(scene_path, "images")
            if not os.path.isdir(input_dir):
                for candidate in [
                    os.path.join(os.path.dirname(model_path), "..", "scene", "images"),
                ]:
                    if os.path.isdir(candidate):
                        input_dir = candidate
                        break

            exif_gps = extract_exif_gps(input_dir)
            colmap_poses = get_colmap_poses(scene_path)
            if len(exif_gps) >= 4 and len(colmap_poses) >= 4:
                utm_coords = project_gps_to_utm(exif_gps)
                s, R, t = compute_geo_to_colmap_transform(utm_coords, colmap_poses)
                has_geo_transform = True
                # Need ground Z in UTM for corner placement
                gz_utm = centroid[2]  # use centroid Z as ground proxy
                corners_utm = np.array([
                    [ortho_bounds.left, ortho_bounds.bottom, gz_utm],
                    [ortho_bounds.right, ortho_bounds.bottom, gz_utm],
                    [ortho_bounds.right, ortho_bounds.top, gz_utm],
                    [ortho_bounds.left, ortho_bounds.top, gz_utm],
                ])
                corners_colmap = (s * (R @ corners_utm.T).T + t)
                print(f"  Geo transform: scale={s:.6f}")
        except Exception as e:
            print(f"  Geo transform failed: {e}, using centered flat grid")

    # Grid size: ~200x200 for flat terrain
    rows, cols = 100, 100

    if has_geo_transform:
        # Interpolate between corners to make the grid
        xs_min = corners_colmap[:, 0].min()
        xs_max = corners_colmap[:, 0].max()
        ys_min = corners_colmap[:, 1].min()
        ys_max = corners_colmap[:, 1].max()
        xs = np.linspace(xs_min, xs_max, cols)
        ys = np.linspace(ys_min, ys_max, rows)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, corners_colmap[:, 2].mean())
        colmap_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        web_points = (colmap_points - centroid) * scene_scale
    elif model_path:
        # Use PLY positions to define grid extent
        from plyfile import PlyData as _PD  # noqa
        ckpt_dirs = sorted(Path(model_path).glob("point_cloud/iteration_*"))
        ply_path = str(ckpt_dirs[-1] / "point_cloud.ply")
        ply_data = _PD.read(ply_path)
        vertex = ply_data["vertex"]
        positions = np.stack([
            np.array(vertex["x"], dtype=np.float32),
            np.array(vertex["y"], dtype=np.float32),
            np.array(vertex["z"], dtype=np.float32),
        ], axis=-1)
        pos_scaled = (positions - centroid) * scene_scale
        xs = np.linspace(pos_scaled[:, 0].min(), pos_scaled[:, 0].max(), cols)
        ys = np.linspace(pos_scaled[:, 1].min(), pos_scaled[:, 1].max(), rows)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, ground_z_scaled)
        web_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    else:
        # Use orthophoto UTM bounds centered + scaled
        half_x = (ortho_bounds.right - ortho_bounds.left) / 2 * scene_scale
        half_y = (ortho_bounds.top - ortho_bounds.bottom) / 2 * scene_scale
        xs = np.linspace(-half_x, half_x, cols)
        ys = np.linspace(-half_y, half_y, rows)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, ground_z_scaled)
        web_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Generate triangle mesh
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            idx00 = r * cols + c
            idx01 = r * cols + (c + 1)
            idx10 = (r + 1) * cols + c
            idx11 = (r + 1) * cols + (c + 1)
            faces.append([idx00, idx10, idx01])
            faces.append([idx01, idx10, idx11])

    vertices = web_points.astype(np.float32)
    faces = np.array(faces, dtype=np.int32)

    # UVs
    us = np.linspace(0, 1, cols)
    vs = np.linspace(0, 1, rows)
    uu, vv = np.meshgrid(us, vs)
    uvs = np.stack([uu.ravel(), vv.ravel()], axis=-1).astype(np.float32)

    terrain = {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "grid_shape": (rows, cols),
        "has_geo_transform": has_geo_transform,
        "centroid": centroid.tolist(),
        "scene_scale": scene_scale,
        "is_flat_fallback": True,
    }

    print(f"  Flat terrain mesh: {len(vertices)} vertices, {len(faces)} faces")
    save_intermediate({"vertices": len(vertices), "faces": len(faces),
                       "grid": [rows, cols], "flat_fallback": True},
                      "terrain_stats.json", intermediates_dir)

    return terrain


def step_terrain(dsm_path, dtm_path, orthophoto_path, model_path, scene_path,
                 scene_scale, intermediates_dir, scene_transform_path="",
                 dsm_smooth_sigma=1.5):
    """Load DSM, generate terrain mesh with orthophoto UVs.

    Returns dict with: vertices, faces, uvs, transform metadata.
    """
    import rasterio
    from scipy.ndimage import gaussian_filter, distance_transform_edt

    print("Step 1: Building terrain mesh from DSM...")

    # Check if DSM exists; if not, create flat terrain from orthophoto bounds
    if not os.path.exists(dsm_path):
        print(f"  WARNING: DSM not found at {dsm_path}")
        print("  Falling back to flat terrain from orthophoto bounds...")
        return _build_flat_terrain(orthophoto_path, model_path, scene_path,
                                   scene_scale, intermediates_dir, scene_transform_path)

    # Load DSM
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        dsm_transform = src.transform
        dsm_crs = src.crs
        dsm_bounds = src.bounds
        dsm_res = src.res  # (y_res, x_res) in meters

    print(f"  DSM: {dsm.shape[1]}x{dsm.shape[0]} px, res={dsm_res[0]:.2f}m, "
          f"bounds=({dsm_bounds.left:.1f},{dsm_bounds.bottom:.1f})-({dsm_bounds.right:.1f},{dsm_bounds.top:.1f})")

    # Fill NaN holes with distance transform interpolation
    nan_mask = np.isnan(dsm) | (dsm < -1000) | (dsm > 10000)
    if nan_mask.any():
        nan_count = nan_mask.sum()
        print(f"  Filling {nan_count} NaN/invalid pixels ({100*nan_count/dsm.size:.1f}%)")
        valid_mask = ~nan_mask
        _, nearest_idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
        dsm[nan_mask] = dsm[nearest_idx[0][nan_mask], nearest_idx[1][nan_mask]]

    # Gaussian blur for smoother terrain
    dsm = gaussian_filter(dsm, sigma=dsm_smooth_sigma)

    # Load DTM if available (for ground height estimation)
    dtm = None
    if dtm_path and os.path.exists(dtm_path):
        try:
            with rasterio.open(dtm_path) as src:
                dtm = src.read(1).astype(np.float32)
                dtm_nan = np.isnan(dtm) | (dtm < -1000)
                if dtm_nan.any():
                    valid_mask = ~dtm_nan
                    _, nearest_idx = distance_transform_edt(dtm_nan, return_distances=True, return_indices=True)
                    dtm[dtm_nan] = dtm[nearest_idx[0][dtm_nan], nearest_idx[1][dtm_nan]]
            print(f"  DTM loaded: {dtm.shape}")
        except Exception as e:
            print(f"  DTM load failed: {e}, estimating ground from DSM 5th percentile")
            dtm = None

    # Subsample to ~200x300 grid (~1 vertex per 15cm)
    target_verts = 200 * 300
    total_pixels = dsm.shape[0] * dsm.shape[1]
    subsample = max(1, int(np.sqrt(total_pixels / target_verts)))
    dsm_sub = dsm[::subsample, ::subsample]
    if dtm is not None:
        dtm_sub = dtm[::subsample, ::subsample]
    else:
        dtm_sub = None

    rows, cols = dsm_sub.shape
    print(f"  Subsampled: {cols}x{rows} (every {subsample} pixels)")

    # Generate UTM coordinates for each vertex
    row_indices, col_indices = np.mgrid[0:rows, 0:cols]
    # Map subsampled indices back to original pixel coordinates
    orig_rows = row_indices * subsample
    orig_cols = col_indices * subsample

    # Pixel to UTM using rasterio transform
    xs = dsm_transform.c + orig_cols * dsm_transform.a + orig_rows * dsm_transform.b
    ys = dsm_transform.f + orig_cols * dsm_transform.d + orig_rows * dsm_transform.e
    zs = dsm_sub

    # Build UTM point cloud: (rows*cols, 3)
    utm_points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=-1)

    # Compute UTM → COLMAP transform using EXIF GPS + COLMAP poses
    try:
        from align_geo_colmap import extract_exif_gps, compute_geo_to_colmap_transform, project_gps_to_utm
        # Get EXIF GPS from images
        input_dir = os.path.join(os.path.dirname(model_path), "..", "scene", "images")
        if not os.path.isdir(input_dir):
            # Try alternative paths
            for candidate in [
                os.path.join(scene_path, "images"),
                os.path.join(os.path.dirname(model_path), "..", "..", "scene", "images"),
            ]:
                if os.path.isdir(candidate):
                    input_dir = candidate
                    break

        exif_gps = extract_exif_gps(input_dir)
        colmap_poses = get_colmap_poses(scene_path)

        if len(exif_gps) >= 4 and len(colmap_poses) >= 4:
            s, R, T, inliers = compute_geo_to_colmap_transform(exif_gps, colmap_poses)
            print(f"  UTM→COLMAP transform: scale={s:.6f}, {inliers.sum()} inliers")

            # Transform UTM points to COLMAP space
            colmap_points = s * (R @ utm_points.T).T + T
            has_geo_transform = True
        else:
            raise ValueError(f"Insufficient correspondences: {len(exif_gps)} GPS, {len(colmap_poses)} COLMAP")
    except Exception as e:
        print(f"  WARNING: Geo transform failed ({e}), using ODM coords directly")
        # Fall back: use ODM coordinates directly (won't align perfectly with splat)
        colmap_points = utm_points.copy()
        has_geo_transform = False

    # COLMAP → web coords: (point - centroid) * scene_scale
    if scene_transform_path and os.path.exists(scene_transform_path):
        centroid = load_centroid_from_transform(scene_transform_path)
        # With scene_transform, skip COLMAP transform — use UTM coords directly
        web_points = (utm_points - centroid) * scene_scale
    elif model_path:
        centroid = load_ply_centroid(model_path)
        web_points = (colmap_points - centroid) * scene_scale
    else:
        # Fallback: center UTM points at their own centroid
        centroid = utm_points.mean(axis=0)
        web_points = (utm_points - centroid) * scene_scale
        print(f"  Using auto-centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")

    # Reshape to grid
    verts_grid = web_points.reshape(rows, cols, 3)

    # Generate triangle mesh (2 triangles per cell)
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            idx00 = r * cols + c
            idx01 = r * cols + (c + 1)
            idx10 = (r + 1) * cols + c
            idx11 = (r + 1) * cols + (c + 1)
            faces.append([idx00, idx10, idx01])
            faces.append([idx01, idx10, idx11])

    vertices = web_points.astype(np.float32)
    faces = np.array(faces, dtype=np.int32)

    # UVs: map to orthophoto [0,1] range
    # Column maps to U (0→1 left→right), Row maps to V (0→1 top→bottom)
    us = np.linspace(0, 1, cols)
    vs = np.linspace(0, 1, rows)
    uu, vv = np.meshgrid(us, vs)
    uvs = np.stack([uu.ravel(), vv.ravel()], axis=-1).astype(np.float32)

    terrain = {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "grid_shape": (rows, cols),
        "has_geo_transform": has_geo_transform,
        "centroid": centroid.tolist(),
        "scene_scale": scene_scale,
    }

    # Save DTM heights for asset placement (step 5)
    if dtm_sub is not None:
        terrain["dtm_grid"] = dtm_sub
    terrain["dsm_grid"] = dsm_sub

    print(f"  Terrain mesh: {len(vertices)} vertices, {len(faces)} faces")
    save_intermediate({"vertices": len(vertices), "faces": len(faces), "grid": [rows, cols]},
                      "terrain_stats.json", intermediates_dir)

    return terrain


# ---------------------------------------------------------------------------
# Step 2: Super-resolution
# ---------------------------------------------------------------------------

def step_superres(orthophoto_path, output_path, intermediates_dir):
    """Real-ESRGAN 4x upscale of orthophoto.

    Tiles large images with overlap to avoid seam artifacts.
    Falls back to Lanczos if Real-ESRGAN fails.
    """
    print("Step 2: Super-resolution upscale of orthophoto...")

    img = cv2.imread(orthophoto_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read orthophoto: {orthophoto_path}")

    h, w = img.shape[:2]
    print(f"  Input: {w}x{h}")

    try:
        import torch
        # Patch for torchvision 0.19+ (removed functional_tensor module)
        try:
            from torchvision.transforms import functional_tensor  # noqa: F401
        except ImportError:
            import torchvision.transforms.functional as _F
            import types
            _mod = types.ModuleType("torchvision.transforms.functional_tensor")
            _mod.rgb_to_grayscale = _F.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = _mod
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )

        model_path = "/mnt/splatwalk/models/RealESRGAN_x4plus.pth"
        if not os.path.exists(model_path):
            # Try downloading
            import urllib.request
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            print(f"  Downloading RealESRGAN_x4plus.pth...")
            urllib.request.urlretrieve(url, model_path)

        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=512,
            tile_pad=32,
            pre_pad=0,
            half=True,
        )

        output, _ = upsampler.enhance(img, outscale=4)
        print(f"  Real-ESRGAN 4x: {w}x{h} → {output.shape[1]}x{output.shape[0]}")

    except Exception as e:
        print(f"  Real-ESRGAN failed ({e}), falling back to Lanczos 4x")
        output = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
        print(f"  Lanczos 4x: {w}x{h} → {output.shape[1]}x{output.shape[0]}")

    # Save as JPEG q90
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 90])
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({size_mb:.1f}MB)")

    save_intermediate({"input_size": [w, h], "output_size": [output.shape[1], output.shape[0]]},
                      "superres_stats.json", intermediates_dir)

    return output_path


# ---------------------------------------------------------------------------
# Step 3: Segmentation (SAM2 + Gemini classification)
# ---------------------------------------------------------------------------

MATERIAL_CLASSES = [
    "grass", "concrete", "asphalt", "gravel", "bare_soil", "mulch",
    "wood_deck", "roof_shingle", "water", "tree_canopy", "shrub",
    "fence", "wall", "driveway", "sidewalk", "garden_bed",
    "sand", "other",
]


def _segment_with_sam2(orthophoto_path, intermediates_dir, points_per_side=32):
    """Run SAM2 automatic mask generation."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    import torch

    print("  Running SAM2 automatic mask generation...")

    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "/mnt/splatwalk/models/sam2/sam2.1_hiera_large.pt"

    sam2 = build_sam2(model_cfg, checkpoint, device="cuda")
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )

    img = cv2.imread(orthophoto_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(img_rgb)
    print(f"  SAM2 generated {len(masks)} masks")

    # Sort by area (largest first)
    masks.sort(key=lambda x: x["area"], reverse=True)

    # Merge tiny masks (<0.5% area) into nearest neighbor
    total_area = img.shape[0] * img.shape[1]
    min_area = total_area * 0.005
    merged_masks = []
    tiny_masks = []

    for m in masks:
        if m["area"] >= min_area:
            merged_masks.append(m)
        else:
            tiny_masks.append(m)

    if tiny_masks:
        print(f"  Merging {len(tiny_masks)} tiny masks into nearest neighbors")
        # For each tiny mask, find nearest large mask by centroid distance
        large_centroids = []
        for m in merged_masks:
            ys, xs = np.where(m["segmentation"])
            large_centroids.append((xs.mean(), ys.mean()))

        for tm in tiny_masks:
            ys, xs = np.where(tm["segmentation"])
            tc = (xs.mean(), ys.mean())
            # Find closest large mask
            best_idx = 0
            best_dist = float("inf")
            for i, lc in enumerate(large_centroids):
                d = (tc[0] - lc[0])**2 + (tc[1] - lc[1])**2
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            # Merge into that mask
            merged_masks[best_idx]["segmentation"] |= tm["segmentation"]
            merged_masks[best_idx]["area"] += tm["area"]

    # Create uint8 mask image (each pixel = mask index)
    h, w = img.shape[:2]
    seg_mask = np.zeros((h, w), dtype=np.uint8)
    for i, m in enumerate(merged_masks):
        seg_mask[m["segmentation"]] = i + 1  # 0 = background

    print(f"  Final: {len(merged_masks)} segments")
    save_intermediate(seg_mask, "sam2_segments.png", intermediates_dir)

    return seg_mask, merged_masks


def _segment_grid_fallback(orthophoto_path, intermediates_dir):
    """Fallback: 8x8 grid segmentation when SAM2 fails."""
    print("  Using 8x8 grid fallback segmentation...")

    img = cv2.imread(orthophoto_path)
    h, w = img.shape[:2]

    grid_rows, grid_cols = 8, 8
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    seg_mask = np.zeros((h, w), dtype=np.uint8)
    segments = []

    idx = 1
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, y1 = r * cell_h, min((r + 1) * cell_h, h)
            x0, x1 = c * cell_w, min((c + 1) * cell_w, w)
            seg_mask[y0:y1, x0:x1] = idx
            mask = np.zeros((h, w), dtype=bool)
            mask[y0:y1, x0:x1] = True
            segments.append({
                "segmentation": mask,
                "area": int((y1 - y0) * (x1 - x0)),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
            })
            idx += 1

    print(f"  Grid segmentation: {len(segments)} cells")
    return seg_mask, segments


def _classify_with_gemini(orthophoto_path, seg_mask, segments, gemini_api_key, intermediates_dir):
    """Classify segments using Gemini 2.5 Flash."""
    import base64

    print("  Classifying segments with Gemini 2.5 Flash...")

    img = cv2.imread(orthophoto_path)

    # Draw numbered mask outlines on orthophoto
    overlay = img.copy()
    for i, seg in enumerate(segments):
        mask = seg["segmentation"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        # Number label at centroid
        ys, xs = np.where(seg["segmentation"])
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(overlay, str(i + 1), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    overlay_path = os.path.join(intermediates_dir, "segmented_overlay.jpg")
    cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Encode image for API
    with open(overlay_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    classes_str = ", ".join(MATERIAL_CLASSES)
    prompt = (
        f"This is a top-down aerial/drone orthophoto with {len(segments)} numbered segments outlined in green. "
        f"For each numbered segment (1-{len(segments)}), classify the ground material as one of: {classes_str}. "
        f"Respond with ONLY a JSON object mapping segment number (as string) to material class. "
        f'Example: {{"1": "grass", "2": "concrete", "3": "tree_canopy"}}'
    )

    try:
        from google import genai
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                prompt,
            ],
        )
        text = response.text.strip()
        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        classifications = json.loads(text)
        print(f"  Gemini classified {len(classifications)} segments")
        return classifications
    except Exception as e:
        print(f"  Gemini classification failed: {e}")
        return None


def _classify_hsv_fallback(orthophoto_path, seg_mask, segments):
    """Fallback: HSV color heuristic classification."""
    print("  Using HSV color heuristic classification...")

    img = cv2.imread(orthophoto_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    classifications = {}
    for i, seg in enumerate(segments):
        mask = seg["segmentation"]
        region_hsv = hsv[mask]
        if len(region_hsv) == 0:
            classifications[str(i + 1)] = "other"
            continue

        mean_h = region_hsv[:, 0].mean()
        mean_s = region_hsv[:, 1].mean()
        mean_v = region_hsv[:, 2].mean()

        # Simple heuristic classification
        if mean_s < 30 and mean_v > 180:
            classifications[str(i + 1)] = "concrete"
        elif mean_s < 30 and mean_v < 80:
            classifications[str(i + 1)] = "asphalt"
        elif 30 < mean_h < 85 and mean_s > 50:
            if mean_v > 100:
                classifications[str(i + 1)] = "grass"
            else:
                classifications[str(i + 1)] = "tree_canopy"
        elif 10 < mean_h < 30 and mean_s > 40:
            classifications[str(i + 1)] = "bare_soil"
        elif mean_s < 40 and 80 < mean_v < 180:
            classifications[str(i + 1)] = "gravel"
        else:
            classifications[str(i + 1)] = "other"

    print(f"  HSV classified {len(classifications)} segments")
    return classifications


def step_segment(orthophoto_path, gemini_api_key, intermediates_dir,
                 sam2_points_per_side=32):
    """Segment orthophoto and classify materials.

    Returns (seg_mask, manifest) where manifest maps segment_id → material class.
    """
    print("Step 3: Segmentation + material classification...")

    # Step 3a: SAM2 segmentation
    try:
        seg_mask, segments = _segment_with_sam2(orthophoto_path, intermediates_dir,
                                                points_per_side=sam2_points_per_side)
    except Exception as e:
        print(f"  SAM2 failed: {e}")
        seg_mask, segments = _segment_grid_fallback(orthophoto_path, intermediates_dir)

    # Step 3b: Classification
    classifications = None
    if gemini_api_key:
        classifications = _classify_with_gemini(
            orthophoto_path, seg_mask, segments, gemini_api_key, intermediates_dir
        )

    if classifications is None:
        classifications = _classify_hsv_fallback(orthophoto_path, seg_mask, segments)

    # Step 3c: Build manifest
    manifest = {
        "num_segments": len(segments),
        "classifications": classifications,
        "material_classes": MATERIAL_CLASSES,
    }

    save_intermediate(manifest, "segment_manifest.json", intermediates_dir)
    save_intermediate(seg_mask, "segment_mask.png", intermediates_dir)

    # Print summary
    class_counts = {}
    for cls in classifications.values():
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {count} segments")

    return seg_mask, manifest


# ---------------------------------------------------------------------------
# Step 4: PBR Materials
# ---------------------------------------------------------------------------

def step_materials(seg_mask, manifest, superres_path, texture_library_path, intermediates_dir):
    """Assign PBR materials to each segment, color-shifting to match orthophoto."""
    print("Step 4: Assigning PBR materials...")

    classifications = manifest["classifications"]

    # Load super-res orthophoto for color matching
    ortho = cv2.imread(superres_path)
    if ortho is None:
        print(f"  WARNING: Cannot load super-res image, using placeholder materials")
        ortho = np.zeros((256, 256, 3), dtype=np.uint8)

    materials = {}

    for seg_id_str, material_class in classifications.items():
        seg_id = int(seg_id_str)

        # Get mean color of this segment in the orthophoto
        # Scale seg_mask to match orthophoto size
        if seg_mask.shape[:2] != ortho.shape[:2]:
            seg_scaled = cv2.resize(seg_mask, (ortho.shape[1], ortho.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        else:
            seg_scaled = seg_mask

        region_mask = seg_scaled == seg_id
        if not region_mask.any():
            continue

        region_pixels = ortho[region_mask]
        mean_bgr = region_pixels.mean(axis=0)
        mean_rgb = mean_bgr[::-1]  # BGR → RGB

        # Load PBR textures from library
        mat_dir = os.path.join(texture_library_path, material_class)
        albedo = None
        normal = None
        roughness = None

        if os.path.isdir(mat_dir):
            albedo_path = os.path.join(mat_dir, "albedo.jpg")
            normal_path = os.path.join(mat_dir, "normal.jpg")
            roughness_path = os.path.join(mat_dir, "roughness.jpg")

            if os.path.exists(albedo_path):
                albedo = cv2.imread(albedo_path)
                # Color-shift albedo to match orthophoto mean
                albedo_mean = albedo.mean(axis=(0, 1))
                if albedo_mean.max() > 0:
                    shift = mean_bgr / np.maximum(albedo_mean, 1)
                    shift = np.clip(shift, 0.5, 2.0)  # Don't shift too aggressively
                    albedo = np.clip(albedo * shift, 0, 255).astype(np.uint8)

            if os.path.exists(normal_path):
                normal = cv2.imread(normal_path)
            if os.path.exists(roughness_path):
                roughness = cv2.imread(roughness_path, cv2.IMREAD_GRAYSCALE)

        if albedo is None:
            # Fallback: solid color from orthophoto mean
            albedo = np.full((256, 256, 3), mean_bgr, dtype=np.uint8)

        materials[seg_id] = {
            "material_class": material_class,
            "albedo": albedo,
            "normal": normal,
            "roughness": roughness,
            "mean_rgb": mean_rgb.tolist(),
        }

    print(f"  Assigned materials to {len(materials)} segments")
    has_pbr = sum(1 for m in materials.values() if m["normal"] is not None)
    print(f"  {has_pbr} segments have full PBR textures")

    return materials


# ---------------------------------------------------------------------------
# Step 5: Procedural assets (trees, fences, shrubs)
# ---------------------------------------------------------------------------

def _make_tree_broadleaf(height, canopy_diameter):
    """Procedural broadleaf tree: cylinder trunk + multi-tier canopy with color variation."""
    import trimesh

    rng = np.random.default_rng()

    trunk_radius = max(0.05, height * 0.04)
    trunk_height = height * 0.4

    # Slight random lean on the trunk
    lean_angle = rng.uniform(-0.08, 0.08)
    lean_axis = [rng.normal(), rng.normal(), 0]
    lean_axis = lean_axis / (np.linalg.norm(lean_axis) + 1e-8)

    trunk = trimesh.creation.cylinder(
        radius=trunk_radius, height=trunk_height, sections=8
    )
    trunk.apply_translation([0, 0, trunk_height / 2])
    if abs(lean_angle) > 0.01:
        rot = trimesh.transformations.rotation_matrix(lean_angle, lean_axis, point=[0, 0, 0])
        trunk.apply_transform(rot)
    trunk_color = [
        int(rng.integers(75, 100)),
        int(rng.integers(45, 65)),
        int(rng.integers(20, 40)),
        255,
    ]
    trunk.visual.vertex_colors = np.full((len(trunk.vertices), 4), trunk_color, dtype=np.uint8)

    # Multi-tier canopy: 2-3 overlapping spheres at varying sizes and offsets
    meshes = [trunk]
    n_tiers = rng.integers(2, 4)
    for i in range(n_tiers):
        tier_scale = 1.0 - i * 0.2 + rng.uniform(-0.05, 0.05)
        tier_radius = canopy_diameter / 2 * max(0.4, tier_scale)
        tier_offset_x = rng.uniform(-canopy_diameter * 0.15, canopy_diameter * 0.15)
        tier_offset_y = rng.uniform(-canopy_diameter * 0.15, canopy_diameter * 0.15)
        tier_z = trunk_height + canopy_diameter * (0.2 + i * 0.25)

        sphere = trimesh.creation.icosphere(subdivisions=1, radius=tier_radius)
        sphere.apply_translation([tier_offset_x, tier_offset_y, tier_z])

        # Vary green within a natural range
        green_val = int(rng.integers(90, 145))
        red_val = int(rng.integers(25, 55))
        blue_val = int(rng.integers(15, 40))
        sphere.visual.vertex_colors = np.full(
            (len(sphere.vertices), 4), [red_val, green_val, blue_val, 255], dtype=np.uint8
        )
        meshes.append(sphere)

    return trimesh.util.concatenate(meshes)


def _make_tree_conifer(height, canopy_diameter):
    """Procedural conifer: cylinder trunk + stacked cones with color variation."""
    import trimesh

    rng = np.random.default_rng()

    trunk_radius = max(0.05, height * 0.03)
    trunk_height = height * 0.3

    trunk = trimesh.creation.cylinder(
        radius=trunk_radius, height=trunk_height, sections=8
    )
    trunk.apply_translation([0, 0, trunk_height / 2])
    trunk.visual.vertex_colors = np.full(
        (len(trunk.vertices), 4),
        [int(rng.integers(70, 90)), int(rng.integers(40, 55)), int(rng.integers(18, 30)), 255],
        dtype=np.uint8,
    )

    # 3-4 stacked cones with slight random offsets
    meshes = [trunk]
    n_cones = rng.integers(3, 5)
    cone_base = canopy_diameter / 2
    for i in range(n_cones):
        frac = i / n_cones
        cone_r = cone_base * (1 - frac * 0.45) + rng.uniform(-0.05, 0.05) * cone_base
        cone_r = max(0.1, cone_r)
        cone_h = (height - trunk_height) / 2.5
        cone = trimesh.creation.cone(radius=cone_r, height=cone_h, sections=8)
        z_offset = trunk_height + i * cone_h * 0.65 + cone_h / 2
        x_off = rng.uniform(-0.05, 0.05) * canopy_diameter
        y_off = rng.uniform(-0.05, 0.05) * canopy_diameter
        cone.apply_translation([x_off, y_off, z_offset])
        green_val = int(rng.integers(65, 100)) + i * 10
        cone.visual.vertex_colors = np.full(
            (len(cone.vertices), 4),
            [int(rng.integers(10, 30)), min(green_val, 140), int(rng.integers(10, 30)), 255],
            dtype=np.uint8,
        )
        meshes.append(cone)

    return trimesh.util.concatenate(meshes)


def _make_shrub(width, height):
    """Procedural shrub: flattened UV sphere."""
    import trimesh

    shrub = trimesh.creation.icosphere(subdivisions=1, radius=width / 2)
    # Flatten vertically
    shrub.vertices[:, 2] *= height / width
    shrub.apply_translation([0, 0, height / 2])
    shrub.visual.vertex_colors = np.full((len(shrub.vertices), 4), [30, 90, 25, 255], dtype=np.uint8)
    return shrub


def _make_fence_section(length, height=1.8):
    """Procedural fence section: posts + horizontal rails + vertical slats."""
    import trimesh

    post_radius = 0.04
    meshes = []

    # Two posts
    for x in [0, length]:
        post = trimesh.creation.cylinder(radius=post_radius, height=height, sections=6)
        post.apply_translation([x, 0, height / 2])
        post.visual.vertex_colors = np.full((len(post.vertices), 4), [120, 80, 40, 255], dtype=np.uint8)
        meshes.append(post)

    # Two horizontal rails
    for z in [height * 0.3, height * 0.8]:
        rail = trimesh.creation.box(extents=[length, 0.05, 0.08])
        rail.apply_translation([length / 2, 0, z])
        rail.visual.vertex_colors = np.full((len(rail.vertices), 4), [110, 75, 35, 255], dtype=np.uint8)
        meshes.append(rail)

    # Vertical slats
    n_slats = max(2, int(length / 0.15))
    for i in range(n_slats):
        x = (i + 0.5) * length / n_slats
        slat = trimesh.creation.box(extents=[0.07, 0.02, height * 0.5])
        slat.apply_translation([x, 0, height * 0.55])
        slat.visual.vertex_colors = np.full((len(slat.vertices), 4), [100, 70, 35, 255], dtype=np.uint8)
        meshes.append(slat)

    return trimesh.util.concatenate(meshes)


def _make_grass_tuft(width=0.3, height=0.15):
    """Procedural grass tuft: small scaled box cluster."""
    import trimesh

    rng = np.random.default_rng()
    meshes = []
    n_blades = rng.integers(3, 7)
    for _ in range(n_blades):
        blade_w = width * rng.uniform(0.15, 0.3)
        blade_h = height * rng.uniform(0.6, 1.4)
        blade = trimesh.creation.box(extents=[blade_w, blade_w * 0.3, blade_h])
        blade.apply_translation([
            rng.uniform(-width * 0.4, width * 0.4),
            rng.uniform(-width * 0.4, width * 0.4),
            blade_h / 2,
        ])
        green = int(rng.integers(80, 140))
        blade.visual.vertex_colors = np.full(
            (len(blade.vertices), 4),
            [int(rng.integers(30, 60)), green, int(rng.integers(15, 35)), 255],
            dtype=np.uint8,
        )
        meshes.append(blade)
    return trimesh.util.concatenate(meshes)


def step_assets(seg_mask, manifest, terrain, intermediates_dir,
                tree_height_scale=1.0, canopy_diameter_scale=1.0, grass_density=200):
    """Generate procedural 3D assets from segmentation masks."""
    import trimesh

    print("Step 5: Generating procedural assets...")

    classifications = manifest["classifications"]
    vertices = terrain["vertices"]
    grid_shape = terrain["grid_shape"]
    rows, cols = grid_shape
    dsm_grid = terrain.get("dsm_grid")
    dtm_grid = terrain.get("dtm_grid")

    assets = {"trees": [], "shrubs": [], "fences": [], "grass": []}

    # Scale seg_mask to terrain grid size for placement lookups
    seg_grid = cv2.resize(seg_mask, (cols, rows), interpolation=cv2.INTER_NEAREST)

    # --- Trees ---
    tree_segments = [int(k) for k, v in classifications.items() if v == "tree_canopy"]
    if tree_segments:
        print(f"  Processing {len(tree_segments)} tree canopy segments...")

        for seg_id in tree_segments:
            # Find connected components for this segment
            seg_binary = (seg_mask == seg_id).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_binary)

            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < 50:  # Skip tiny blobs
                    continue

                cx_px, cy_px = centroids[label_id]
                bbox_w = stats[label_id, cv2.CC_STAT_WIDTH]
                bbox_h = stats[label_id, cv2.CC_STAT_HEIGHT]

                # Map pixel position to terrain grid
                gx = int(cx_px / seg_mask.shape[1] * (cols - 1))
                gy = int(cy_px / seg_mask.shape[0] * (rows - 1))
                gx = np.clip(gx, 0, cols - 1)
                gy = np.clip(gy, 0, rows - 1)

                # Get position from terrain vertices
                vert_idx = gy * cols + gx
                if vert_idx >= len(vertices):
                    continue
                pos = vertices[vert_idx].copy()

                # Canopy diameter from bbox (in web coords)
                canopy_diameter_px = max(bbox_w, bbox_h)
                # Convert pixel extent to web coord extent
                canopy_diameter = canopy_diameter_px / max(seg_mask.shape) * (
                    max(vertices[:, 0].max() - vertices[:, 0].min(),
                        vertices[:, 1].max() - vertices[:, 1].min())
                )
                canopy_diameter = max(1.0, min(canopy_diameter, 30.0))
                canopy_diameter *= canopy_diameter_scale

                # Height from DSM - DTM
                if dsm_grid is not None and dtm_grid is not None:
                    dsm_val = dsm_grid[min(gy, dsm_grid.shape[0]-1), min(gx, dsm_grid.shape[1]-1)]
                    dtm_val = dtm_grid[min(gy, dtm_grid.shape[0]-1), min(gx, dtm_grid.shape[1]-1)]
                    tree_height = max(2.0, (dsm_val - dtm_val) * terrain["scene_scale"])
                else:
                    tree_height = canopy_diameter * 1.2
                tree_height *= tree_height_scale

                # Shape classification by aspect ratio
                aspect = bbox_h / max(bbox_w, 1)
                if aspect > 2.0:
                    tree_mesh = _make_tree_conifer(tree_height, canopy_diameter)
                else:
                    tree_mesh = _make_tree_broadleaf(tree_height, canopy_diameter)

                tree_mesh.apply_translation(pos)
                assets["trees"].append(tree_mesh)

        print(f"  Generated {len(assets['trees'])} trees")

    # --- Shrubs ---
    shrub_segments = [int(k) for k, v in classifications.items() if v == "shrub"]
    if shrub_segments:
        print(f"  Processing {len(shrub_segments)} shrub segments...")

        for seg_id in shrub_segments:
            seg_binary = (seg_mask == seg_id).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_binary)

            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < 30:
                    continue

                cx_px, cy_px = centroids[label_id]
                bbox_w = stats[label_id, cv2.CC_STAT_WIDTH]
                bbox_h = stats[label_id, cv2.CC_STAT_HEIGHT]

                gx = int(cx_px / seg_mask.shape[1] * (cols - 1))
                gy = int(cy_px / seg_mask.shape[0] * (rows - 1))
                gx = np.clip(gx, 0, cols - 1)
                gy = np.clip(gy, 0, rows - 1)

                vert_idx = gy * cols + gx
                if vert_idx >= len(vertices):
                    continue
                pos = vertices[vert_idx].copy()

                # Shrub dimensions
                shrub_width = max(bbox_w, bbox_h) / max(seg_mask.shape) * (
                    max(vertices[:, 0].max() - vertices[:, 0].min(),
                        vertices[:, 1].max() - vertices[:, 1].min())
                )
                shrub_width = max(0.5, min(shrub_width, 10.0))
                shrub_height = shrub_width * 0.6

                shrub_mesh = _make_shrub(shrub_width, shrub_height)
                shrub_mesh.apply_translation(pos)
                assets["shrubs"].append(shrub_mesh)

        print(f"  Generated {len(assets['shrubs'])} shrubs")

    # --- Fences ---
    fence_segments = [int(k) for k, v in classifications.items() if v in ("fence", "wall")]
    if fence_segments:
        print(f"  Processing {len(fence_segments)} fence/wall segments...")
        from skimage.morphology import skeletonize

        for seg_id in fence_segments:
            seg_binary = (seg_mask == seg_id).astype(np.uint8)

            # Skeletonize to get centerline
            try:
                skeleton = skeletonize(seg_binary > 0)
            except Exception:
                continue

            # Sample points along skeleton
            ys, xs = np.where(skeleton)
            if len(xs) < 2:
                continue

            # Sort points by following the skeleton path
            points = np.column_stack([xs, ys]).astype(float)
            # Simple sort by x then y
            order = np.lexsort((points[:, 1], points[:, 0]))
            points = points[order]

            # Subsample to reasonable density
            n_samples = min(len(points), 20)
            indices = np.linspace(0, len(points) - 1, n_samples, dtype=int)
            sampled = points[indices]

            # Place fence sections between consecutive points
            for j in range(len(sampled) - 1):
                p1_px = sampled[j]
                p2_px = sampled[j + 1]

                # Map to terrain grid
                gx1 = int(p1_px[0] / seg_mask.shape[1] * (cols - 1))
                gy1 = int(p1_px[1] / seg_mask.shape[0] * (rows - 1))
                gx2 = int(p2_px[0] / seg_mask.shape[1] * (cols - 1))
                gy2 = int(p2_px[1] / seg_mask.shape[0] * (rows - 1))

                gx1, gx2 = np.clip([gx1, gx2], 0, cols - 1)
                gy1, gy2 = np.clip([gy1, gy2], 0, rows - 1)

                idx1 = gy1 * cols + gx1
                idx2 = gy2 * cols + gx2
                if idx1 >= len(vertices) or idx2 >= len(vertices):
                    continue

                pos1 = vertices[idx1]
                pos2 = vertices[idx2]
                length = np.linalg.norm(pos2[:2] - pos1[:2])

                if length < 0.5:
                    continue

                fence = _make_fence_section(length)

                # Rotate fence to align with direction
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                angle = np.arctan2(dy, dx)
                rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
                fence.apply_transform(rot)
                fence.apply_translation(pos1)
                assets["fences"].append(fence)

        print(f"  Generated {len(assets['fences'])} fence sections")

    # --- Ground cover (grass tufts in grass-classified regions) ---
    grass_segments = [int(k) for k, v in classifications.items() if v == "grass"]
    if grass_segments:
        print(f"  Scattering grass tufts in {len(grass_segments)} grass segments...")
        rng = np.random.default_rng(42)
        max_grass = grass_density  # cap for performance
        grass_count = 0

        for seg_id in grass_segments:
            if grass_count >= max_grass:
                break
            # Find grid cells belonging to this segment
            seg_cells = np.argwhere(seg_grid == seg_id)
            if len(seg_cells) == 0:
                continue
            # Random sample positions
            n_tufts = min(len(seg_cells) // 4, 30, max_grass - grass_count)
            if n_tufts < 1:
                continue
            indices = rng.choice(len(seg_cells), size=n_tufts, replace=False)
            for idx in indices:
                gy, gx = seg_cells[idx]
                vert_idx = gy * cols + gx
                if vert_idx >= len(vertices):
                    continue
                pos = vertices[vert_idx].copy()
                # Add small random offset to avoid grid pattern
                pos[0] += rng.uniform(-0.5, 0.5)
                pos[1] += rng.uniform(-0.5, 0.5)
                tuft = _make_grass_tuft()
                tuft.apply_translation(pos)
                assets["grass"].append(tuft)
                grass_count += 1

        print(f"  Generated {len(assets['grass'])} grass tufts")

    total = len(assets["trees"]) + len(assets["shrubs"]) + len(assets["fences"]) + len(assets["grass"])
    print(f"  Total assets: {total}")

    save_intermediate(
        {"trees": len(assets["trees"]), "shrubs": len(assets["shrubs"]),
         "fences": len(assets["fences"]), "grass": len(assets["grass"])},
        "assets_stats.json", intermediates_dir,
    )

    return assets


# ---------------------------------------------------------------------------
# Step 6: Export GLB
# ---------------------------------------------------------------------------

def step_export(terrain, materials, assets, metadata, output_path, intermediates_dir):
    """Assemble scene and export as GLB with PBR materials."""
    import trimesh

    print("Step 6: Exporting GLB...")

    scene = trimesh.Scene()

    # --- Terrain mesh with materials ---
    vertices = terrain["vertices"]
    faces = terrain["faces"]
    uvs = terrain["uvs"]

    if materials:
        # Split terrain faces by material class based on segment mask
        # For now, create a single textured terrain mesh using the super-res orthophoto
        # (per-segment PBR is applied as sub-meshes)

        # Create base terrain mesh with orthophoto texture
        terrain_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
        )

        # Find the albedo from the first material or use a default
        # Use the full orthophoto as the terrain texture
        first_mat = next(iter(materials.values()), None)
        if first_mat and first_mat["albedo"] is not None:
            # For a single terrain mesh, we use the orthophoto directly as texture
            # The UV mapping handles the correspondence
            pass

        # Build per-face material assignment from segmentation
        grid_shape = terrain["grid_shape"]
        rows, cols = grid_shape

        # Create material objects for each unique class
        mat_objects = {}
        for seg_id, mat_info in materials.items():
            cls = mat_info["material_class"]
            if cls in mat_objects:
                continue

            albedo_img = mat_info["albedo"]
            # Convert BGR → RGB for trimesh
            if albedo_img is not None and len(albedo_img.shape) == 3:
                albedo_rgb = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
            else:
                albedo_rgb = np.full((256, 256, 3), 128, dtype=np.uint8)

            from PIL import Image as PILImage

            pbr_kwargs = {
                "baseColorTexture": PILImage.fromarray(albedo_rgb),
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            }

            if mat_info["normal"] is not None:
                normal_rgb = cv2.cvtColor(mat_info["normal"], cv2.COLOR_BGR2RGB)
                pbr_kwargs["normalTexture"] = PILImage.fromarray(normal_rgb)

            if mat_info["roughness"] is not None:
                # Pack roughness into metallic-roughness texture (G=roughness, B=metallic=0)
                rough = mat_info["roughness"]
                if len(rough.shape) == 2:
                    mr_tex = np.zeros((*rough.shape, 3), dtype=np.uint8)
                    mr_tex[:, :, 1] = rough  # green = roughness
                    pbr_kwargs["metallicRoughnessTexture"] = PILImage.fromarray(mr_tex)

            mat_obj = trimesh.visual.material.PBRMaterial(name=cls, **pbr_kwargs)
            mat_objects[cls] = mat_obj

        # For simplicity, assign the most common material to the terrain
        # (Individual sub-meshes per material would be ideal but complex)
        from collections import Counter
        class_counts = Counter(m["material_class"] for m in materials.values())
        dominant_class = class_counts.most_common(1)[0][0] if class_counts else "grass"

        if dominant_class in mat_objects:
            from trimesh.visual import TextureVisuals
            terrain_mesh.visual = TextureVisuals(
                uv=uvs,
                material=mat_objects[dominant_class],
            )
        else:
            terrain_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

        scene.add_geometry(terrain_mesh, node_name="terrain")
    else:
        # No materials — plain mesh
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        terrain_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        scene.add_geometry(terrain_mesh, node_name="terrain")

    # --- Add procedural assets ---
    for i, tree in enumerate(assets.get("trees", [])):
        scene.add_geometry(tree, node_name=f"tree_{i}")

    for i, shrub in enumerate(assets.get("shrubs", [])):
        scene.add_geometry(shrub, node_name=f"shrub_{i}")

    for i, fence in enumerate(assets.get("fences", [])):
        scene.add_geometry(fence, node_name=f"fence_{i}")

    for i, tuft in enumerate(assets.get("grass", [])):
        scene.add_geometry(tuft, node_name=f"grass_{i}")

    # --- Metadata as glTF extras ---
    scene.metadata.update({
        "generator": "splatwalk-build-game-scene",
        "scene_scale": terrain.get("scene_scale", 50.0),
        "has_geo_transform": terrain.get("has_geo_transform", False),
        **metadata,
    })

    # --- Export GLB ---
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb_data = scene.export(file_type="glb")
    with open(output_path, "wb") as f:
        f.write(glb_data)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Exported: {output_path} ({size_mb:.1f}MB)")

    # Validate
    n_meshes = len(scene.geometry)
    if size_mb > 50:
        print(f"  WARNING: GLB is {size_mb:.1f}MB (>50MB limit)")
    if n_meshes == 0:
        print(f"  WARNING: GLB has no meshes")

    print(f"  Scene: {n_meshes} meshes, {size_mb:.1f}MB")
    save_intermediate(
        {"size_mb": round(size_mb, 1), "num_meshes": n_meshes},
        "export_stats.json", intermediates_dir,
    )

    return output_path


# ---------------------------------------------------------------------------
# Step 7: Lighting (solar position from EXIF)
# ---------------------------------------------------------------------------

def step_lighting(images_dir, intermediates_dir):
    """Extract solar position from first drone photo's EXIF timestamp + GPS."""
    print("Step 7: Computing solar position from EXIF...")

    from PIL import Image, ExifTags
    from datetime import datetime

    metadata = {}

    # Find first image with EXIF
    img_files = sorted(Path(images_dir).glob("*"))
    for img_path in img_files:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            img = Image.open(img_path)
            exif = img._getexif()
            if not exif:
                continue

            # Get timestamp
            timestamp_str = None
            for tag_id in (ExifTags.Base.DateTimeOriginal, ExifTags.Base.DateTime):
                if tag_id in exif:
                    timestamp_str = exif[tag_id]
                    break

            # Get GPS
            gps_info = exif.get(ExifTags.Base.GPSInfo)
            if not gps_info or not timestamp_str:
                continue

            tags = {}
            for k, v in gps_info.items():
                tags[ExifTags.GPSTAGS.get(k, k)] = v

            if "GPSLatitude" not in tags:
                continue

            def dms_to_decimal(dms, ref):
                val = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
                return -val if ref in ("S", "W") else val

            lat = dms_to_decimal(tags["GPSLatitude"], tags.get("GPSLatitudeRef", "N"))
            lon = dms_to_decimal(tags["GPSLongitude"], tags.get("GPSLongitudeRef", "W"))

            # Parse timestamp
            dt = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")

            print(f"  Photo: {img_path.name}, time={dt}, lat={lat:.4f}, lon={lon:.4f}")

            # Compute solar position
            try:
                import pvlib
                import pandas as pd

                times = pd.DatetimeIndex([dt], tz="UTC")
                solpos = pvlib.solarposition.get_solarposition(times, lat, lon)

                elevation = float(solpos["apparent_elevation"].iloc[0])
                azimuth = float(solpos["azimuth"].iloc[0])

                print(f"  Solar: elevation={elevation:.1f}°, azimuth={azimuth:.1f}°")

                # Convert to direction vector (for Three.js directional light)
                elev_rad = np.radians(elevation)
                az_rad = np.radians(azimuth)
                sun_dir = [
                    float(-np.sin(az_rad) * np.cos(elev_rad)),
                    float(-np.cos(az_rad) * np.cos(elev_rad)),
                    float(np.sin(elev_rad)),
                ]

                metadata = {
                    "sun_elevation_deg": round(elevation, 1),
                    "sun_azimuth_deg": round(azimuth, 1),
                    "sun_direction": [round(v, 4) for v in sun_dir],
                    "capture_time": dt.isoformat(),
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                }

            except Exception as e:
                print(f"  pvlib solar position failed: {e}")
                # Default: sun roughly overhead
                metadata = {
                    "sun_direction": [0.2, -0.3, 0.9],
                    "capture_time": dt.isoformat(),
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                }

            break

        except Exception as e:
            continue

    if not metadata:
        print("  No EXIF data found, using default lighting")
        metadata = {"sun_direction": [0.2, -0.3, 0.9]}

    save_intermediate(metadata, "lighting_metadata.json", intermediates_dir)
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Game-Level Scene Construction")
    parser.add_argument("--orthophoto", required=True, help="ODM orthophoto GeoTIFF")
    parser.add_argument("--dsm", required=True, help="ODM DSM GeoTIFF")
    parser.add_argument("--dtm", default=None, help="ODM DTM GeoTIFF (optional)")
    parser.add_argument("--images", required=True, help="Input drone images directory")
    parser.add_argument("--scene_path", default="", help="Scene directory with COLMAP sparse data (optional)")
    parser.add_argument("--model_path", default="", help="Trained model path for PLY centroid (optional)")
    parser.add_argument("--scene_transform", default="", help="scene_transform.json from generate_aerial_glb.py (replaces model_path)")
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument("--texture_library", default="/mnt/splatwalk/textures/",
                        help="PBR texture library directory")
    parser.add_argument("--scene_scale", type=float, default=50.0,
                        help="Scene scale (must match compress_splat.py)")
    parser.add_argument("--gemini_api_key", default="",
                        help="Gemini API key for material classification")
    parser.add_argument("--step", default="all",
                        help="Run specific step: terrain, superres, segment, materials, assets, export, lighting, all")
    # Tunable quality parameters (overridable by gemini_score_scene.py per iteration)
    parser.add_argument("--tree_height_scale", type=float, default=1.0,
                        help="Multiplier on computed tree height (default 1.0)")
    parser.add_argument("--canopy_diameter_scale", type=float, default=1.0,
                        help="Multiplier on canopy diameter (default 1.0)")
    parser.add_argument("--grass_density", type=int, default=200,
                        help="Max grass tufts (default 200)")
    parser.add_argument("--dsm_smooth_sigma", type=float, default=1.5,
                        help="Gaussian blur sigma for DSM smoothing (default 1.5)")
    parser.add_argument("--sam2_points_per_side", type=int, default=32,
                        help="SAM2 grid density (default 32)")
    args = parser.parse_args()

    if not args.gemini_api_key:
        args.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    intermediates_dir = args.output.replace(".glb", "") + "_intermediates"
    os.makedirs(intermediates_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 4: Game-Level Scene Construction")
    print("=" * 60)
    print(f"  Orthophoto: {args.orthophoto}")
    print(f"  DSM: {args.dsm}")
    print(f"  DTM: {args.dtm or '(not provided)'}")
    print(f"  Model path: {args.model_path}")
    print(f"  Output: {args.output}")
    print(f"  Scene scale: {args.scene_scale}")
    print(f"  Step: {args.step}")
    print()

    steps_to_run = args.step.split(",") if args.step != "all" else [
        "terrain", "superres", "segment", "materials", "assets", "export", "lighting"
    ]

    # Shared state between steps
    terrain = None
    superres_path = None
    seg_mask = None
    seg_manifest = None
    materials = None
    assets_dict = None
    lighting_metadata = {}

    for step_name in steps_to_run:
        step_name = step_name.strip()
        try:
            if step_name == "terrain":
                terrain = step_terrain(
                    args.dsm, args.dtm, args.orthophoto,
                    args.model_path, args.scene_path,
                    args.scene_scale, intermediates_dir,
                    scene_transform_path=args.scene_transform,
                    dsm_smooth_sigma=args.dsm_smooth_sigma,
                )

            elif step_name == "superres":
                superres_path = os.path.join(intermediates_dir, "orthophoto_4x.jpg")
                step_superres(args.orthophoto, superres_path, intermediates_dir)

            elif step_name == "segment":
                seg_mask, seg_manifest = step_segment(
                    args.orthophoto, args.gemini_api_key, intermediates_dir,
                    sam2_points_per_side=args.sam2_points_per_side,
                )

            elif step_name == "materials":
                if seg_mask is None or seg_manifest is None:
                    print("  Loading cached segmentation...")
                    seg_mask_path = os.path.join(intermediates_dir, "segment_mask.png")
                    manifest_path = os.path.join(intermediates_dir, "segment_manifest.json")
                    if os.path.exists(seg_mask_path) and os.path.exists(manifest_path):
                        seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
                        with open(manifest_path) as f:
                            seg_manifest = json.load(f)
                    else:
                        print("  ERROR: No segmentation data available, skipping materials")
                        continue

                if superres_path is None:
                    superres_path = os.path.join(intermediates_dir, "orthophoto_4x.jpg")
                    if not os.path.exists(superres_path):
                        superres_path = args.orthophoto

                materials = step_materials(
                    seg_mask, seg_manifest, superres_path,
                    args.texture_library, intermediates_dir,
                )

            elif step_name == "assets":
                if terrain is None:
                    print("  ERROR: Terrain data required for assets, skipping")
                    continue
                if seg_mask is None or seg_manifest is None:
                    seg_mask_path = os.path.join(intermediates_dir, "segment_mask.png")
                    manifest_path = os.path.join(intermediates_dir, "segment_manifest.json")
                    if os.path.exists(seg_mask_path) and os.path.exists(manifest_path):
                        seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
                        with open(manifest_path) as f:
                            seg_manifest = json.load(f)
                    else:
                        seg_manifest = {"classifications": {}}
                        seg_mask = np.zeros((100, 100), dtype=np.uint8)

                assets_dict = step_assets(seg_mask, seg_manifest, terrain, intermediates_dir,
                                          tree_height_scale=args.tree_height_scale,
                                          canopy_diameter_scale=args.canopy_diameter_scale,
                                          grass_density=args.grass_density)

            elif step_name == "export":
                if terrain is None:
                    print("  ERROR: Terrain data required for export, skipping")
                    continue

                if materials is None:
                    materials = {}
                if assets_dict is None:
                    assets_dict = {"trees": [], "shrubs": [], "fences": [], "grass": []}

                step_export(
                    terrain, materials, assets_dict, lighting_metadata,
                    args.output, intermediates_dir,
                )

            elif step_name == "lighting":
                lighting_metadata = step_lighting(args.images, intermediates_dir)

            else:
                print(f"  Unknown step: {step_name}")

        except Exception as e:
            print(f"  STEP '{step_name}' FAILED: {e}")
            traceback.print_exc()
            # Continue with remaining steps

    # Validate output
    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"\nStage 4 complete: {args.output} ({size_mb:.1f}MB)")
    else:
        print(f"\nStage 4: GLB not generated (some steps may have failed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
