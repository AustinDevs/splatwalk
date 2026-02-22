#!/usr/bin/env python3
"""
Generate Aerial GLB — Orthophoto-textured DSM terrain mesh for aerial viewing.

Creates a lightweight GLB with the ODM orthophoto draped over the DSM terrain.
Used as the aerial view layer in the mesh-based viewer (replaces Gaussian Splat).

Coordinate system:
  - DSM is in UTM/projected coordinates from ODM
  - Output is centered at DSM centroid and scaled by scene_scale (default 50x)
  - Same transform is used by build_game_scene.py for alignment

Usage:
    python generate_aerial_glb.py \\
        --orthophoto /path/to/odm_orthophoto.tif \\
        --dsm /path/to/dsm.tif \\
        --output /path/to/aerial.glb \\
        --scene_scale 50.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import rasterio
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio"])
    import rasterio

try:
    import trimesh
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh"])
    import trimesh


# ---------------------------------------------------------------------------
# DSM → Terrain Mesh
# ---------------------------------------------------------------------------

def load_dsm(dsm_path):
    """Load and clean DSM GeoTIFF."""
    from scipy.ndimage import gaussian_filter, distance_transform_edt

    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

    print(f"  DSM: {dsm.shape[1]}x{dsm.shape[0]} px, "
          f"bounds=({bounds.left:.1f},{bounds.bottom:.1f})-({bounds.right:.1f},{bounds.top:.1f})")

    # Fill NaN/invalid pixels
    nan_mask = np.isnan(dsm) | (dsm < -1000) | (dsm > 10000)
    if nan_mask.any():
        print(f"  Filling {nan_mask.sum()} invalid pixels")
        _, nearest_idx = distance_transform_edt(
            nan_mask, return_distances=True, return_indices=True
        )
        dsm[nan_mask] = dsm[nearest_idx[0][nan_mask], nearest_idx[1][nan_mask]]

    # --- Supplement with 3DEP bare-earth DEM ---
    dsm = _blend_3dep(dsm, bounds, crs, transform)

    # Light smoothing
    dsm = gaussian_filter(dsm, sigma=1.0)

    return dsm, transform, bounds


def _blend_3dep(dsm, bounds, crs, dsm_transform):
    """Fetch USGS 3DEP bare-earth DEM and blend with ODM DSM.

    Uses 3DEP to:
    - Fill holes where ODM DSM has noise/gaps
    - Improve ground-level estimation (3DEP is LiDAR bare-earth)
    - Weighted blend: keep ODM surface detail, anchor to 3DEP ground truth
    """
    try:
        import py3dep
        from pyproj import Transformer
    except ImportError:
        print("  3DEP: py3dep/pyproj not available, using ODM DSM only")
        return dsm

    try:
        # Convert DSM bounds from projected CRS to WGS84 (EPSG:4326) for 3DEP query
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west, south = transformer.transform(bounds.left, bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)

        print(f"  3DEP: fetching DEM for ({west:.5f},{south:.5f})-({east:.5f},{north:.5f})")

        # Fetch 3DEP DEM at 10m resolution (widely available across US)
        dem_3dep = py3dep.get_dem((west, south, east, north), resolution=10, crs="EPSG:4326")

        if dem_3dep is None or dem_3dep.size == 0:
            print("  3DEP: no data available for this area")
            return dsm

        # Reproject 3DEP to match ODM DSM grid
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds

        dem_values = dem_3dep.values.astype(np.float32)
        dem_bounds = (
            float(dem_3dep.x.min()), float(dem_3dep.y.min()),
            float(dem_3dep.x.max()), float(dem_3dep.y.max()),
        )
        dem_transform = from_bounds(*dem_bounds, dem_values.shape[1], dem_values.shape[0])

        # Reproject 3DEP into ODM DSM space
        dep_resampled = np.zeros_like(dsm)
        reproject(
            source=dem_values,
            destination=dep_resampled,
            src_transform=dem_transform,
            src_crs="EPSG:4326",
            dst_transform=dsm_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
        )

        # Where 3DEP has valid data, use weighted blend
        valid_3dep = (dep_resampled > -1000) & (dep_resampled < 10000) & ~np.isnan(dep_resampled)
        overlap_pct = 100 * valid_3dep.sum() / dsm.size

        if overlap_pct < 5:
            print(f"  3DEP: only {overlap_pct:.1f}% overlap, using ODM DSM only")
            return dsm

        # Blend: 70% ODM (local detail) + 30% 3DEP (regional accuracy)
        # This preserves ODM's fine structure while anchoring to 3DEP ground truth
        blend_weight = 0.3
        dsm[valid_3dep] = (
            (1 - blend_weight) * dsm[valid_3dep] +
            blend_weight * dep_resampled[valid_3dep]
        )

        elev_diff = np.abs(dsm[valid_3dep] - dep_resampled[valid_3dep])
        print(f"  3DEP: blended ({overlap_pct:.0f}% coverage, "
              f"mean diff={elev_diff.mean():.1f}m, max={elev_diff.max():.1f}m)")

    except Exception as e:
        print(f"  3DEP: fetch failed ({e}), using ODM DSM only")

    return dsm


def build_terrain_mesh(dsm, dsm_transform, dsm_bounds, scene_scale, grid_size=500):
    """Build terrain mesh from DSM, centered and scaled for web viewer.

    Returns (vertices, faces, uvs, centroid_utm, ground_z_scaled).
    """
    rows_full, cols_full = dsm.shape

    # Subsample to target grid size
    subsample = max(1, int(max(rows_full, cols_full) / grid_size))
    dsm_sub = dsm[::subsample, ::subsample]
    rows, cols = dsm_sub.shape
    print(f"  Grid: {cols}x{rows} (subsample={subsample})")

    # Generate UTM coordinates for each vertex
    row_indices, col_indices = np.mgrid[0:rows, 0:cols]
    orig_rows = row_indices * subsample
    orig_cols = col_indices * subsample

    xs = dsm_transform.c + orig_cols * dsm_transform.a + orig_rows * dsm_transform.b
    ys = dsm_transform.f + orig_cols * dsm_transform.d + orig_rows * dsm_transform.e
    zs = dsm_sub

    # Compute UTM centroid
    centroid_utm = np.array([
        (dsm_bounds.left + dsm_bounds.right) / 2.0,
        (dsm_bounds.bottom + dsm_bounds.top) / 2.0,
        float(np.median(dsm)),
    ])
    print(f"  UTM centroid: [{centroid_utm[0]:.2f}, {centroid_utm[1]:.2f}, {centroid_utm[2]:.2f}]")

    # Build point cloud and center + scale
    utm_points = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=-1)
    web_points = (utm_points - centroid_utm) * scene_scale

    # Ground Z (5th percentile)
    ground_z_scaled = float((np.percentile(dsm, 5) - centroid_utm[2]) * scene_scale)

    # Generate triangle faces (2 per cell)
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

    # UVs: [0,1] mapping to orthophoto
    us = np.linspace(0, 1, cols)
    vs = np.linspace(0, 1, rows)
    uu, vv = np.meshgrid(us, vs)
    uvs = np.stack([uu.ravel(), vv.ravel()], axis=-1).astype(np.float32)

    print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces")

    return vertices, faces, uvs, centroid_utm, ground_z_scaled


# ---------------------------------------------------------------------------
# Orthophoto texture (with optional upscale)
# ---------------------------------------------------------------------------

def prepare_texture(orthophoto_path, max_texture_size=4096, upscale=False):
    """Load orthophoto and optionally upscale with Real-ESRGAN.

    Returns PIL Image in RGB.
    """
    from PIL import Image

    img = cv2.imread(orthophoto_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read orthophoto: {orthophoto_path}")

    h, w = img.shape[:2]
    print(f"  Orthophoto: {w}x{h}")

    if upscale:
        try:
            import torch
            # Patch for torchvision 0.19+
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
                num_block=23, num_grow_ch=32, scale=2,
            )
            model_path = "/mnt/splatwalk/models/RealESRGAN_x2plus.pth"
            if os.path.exists(model_path):
                upsampler = RealESRGANer(
                    scale=2, model_path=model_path, model=model,
                    tile=512, tile_pad=32, pre_pad=0, half=True,
                )
                img, _ = upsampler.enhance(img, outscale=2)
                h, w = img.shape[:2]
                print(f"  Real-ESRGAN 2x: {w}x{h}")
            else:
                print(f"  Real-ESRGAN model not found, skipping upscale")
        except Exception as e:
            print(f"  Real-ESRGAN failed ({e}), using original")

    # Resize to max texture size if needed
    if max(w, h) > max_texture_size:
        scale_factor = max_texture_size / max(w, h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        print(f"  Resized to {new_w}x{new_h} (max {max_texture_size})")

    # BGR → RGB → PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ---------------------------------------------------------------------------
# Export GLB
# ---------------------------------------------------------------------------

def export_glb(vertices, faces, uvs, texture_img, output_path):
    """Export textured terrain mesh as GLB."""
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    material = PBRMaterial(
        name="aerial_orthophoto",
        baseColorTexture=texture_img,
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )
    mesh.visual = TextureVisuals(uv=uvs, material=material)

    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="terrain")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb_data = scene.export(file_type="glb")
    with open(output_path, "wb") as f:
        f.write(glb_data)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Exported: {output_path} ({size_mb:.1f}MB)")
    return size_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Aerial GLB from ODM outputs")
    parser.add_argument("--orthophoto", required=True, help="ODM orthophoto GeoTIFF")
    parser.add_argument("--dsm", required=True, help="ODM DSM GeoTIFF")
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument("--scene_scale", type=float, default=50.0, help="Uniform scene scale")
    parser.add_argument("--grid_size", type=int, default=500, help="Terrain grid size (vertices per side)")
    parser.add_argument("--max_texture_size", type=int, default=4096, help="Max texture dimension")
    parser.add_argument("--upscale", action="store_true", help="Apply Real-ESRGAN 2x upscale")
    parser.add_argument("--scene_transform_out", default="", help="Output path for scene_transform.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Generate Aerial GLB")
    print("=" * 60)
    print(f"  Orthophoto: {args.orthophoto}")
    print(f"  DSM: {args.dsm}")
    print(f"  Output: {args.output}")
    print(f"  Scene scale: {args.scene_scale}")
    print(f"  Grid size: {args.grid_size}")
    print()

    # Step 1: Load DSM
    print("Loading DSM...")
    dsm, dsm_transform, dsm_bounds = load_dsm(args.dsm)

    # Step 2: Build terrain mesh
    print("Building terrain mesh...")
    vertices, faces, uvs, centroid_utm, ground_z_scaled = build_terrain_mesh(
        dsm, dsm_transform, dsm_bounds, args.scene_scale, args.grid_size
    )

    # Step 3: Prepare texture
    print("Preparing orthophoto texture...")
    texture_img = prepare_texture(
        args.orthophoto,
        max_texture_size=args.max_texture_size,
        upscale=args.upscale,
    )

    # Step 4: Export GLB
    print("Exporting GLB...")
    size_mb = export_glb(vertices, faces, uvs, texture_img, args.output)

    # Step 5: Save scene transform for alignment with ground GLB
    scene_transform = {
        "centroid_utm": [float(v) for v in centroid_utm],
        "scene_scale": float(args.scene_scale),
        "ground_z_scaled": float(ground_z_scaled),
        "dsm_bounds": {
            "left": float(dsm_bounds.left),
            "right": float(dsm_bounds.right),
            "bottom": float(dsm_bounds.bottom),
            "top": float(dsm_bounds.top),
        },
        "scene_bounds": {
            "min": [float(v) for v in vertices.min(axis=0)],
            "max": [float(v) for v in vertices.max(axis=0)],
            "center": [0.0, 0.0, 0.0],
            "size": [float(v) for v in (vertices.max(axis=0) - vertices.min(axis=0))],
            "ground_z": float(ground_z_scaled),
        },
    }

    transform_path = args.scene_transform_out or os.path.join(
        os.path.dirname(args.output), "scene_transform.json"
    )
    with open(transform_path, "w") as f:
        json.dump(scene_transform, f, indent=2)
    print(f"  Scene transform: {transform_path}")

    print(f"\nDone. Aerial GLB: {size_mb:.1f}MB")


if __name__ == "__main__":
    main()
