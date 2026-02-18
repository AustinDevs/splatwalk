#!/usr/bin/env python3
"""
Stage 3b: Generate Ground-Level Views via 3DEP DEM + ODM Orthophoto

Generates synthetic ground-level camera views by:
  1. Fetching USGS 3DEP elevation data for the scene footprint
  2. Building a terrain mesh from the DEM
  3. Texturing it with the ODM orthophoto (or uniform color fallback)
  4. Placing 64 cameras at 1.7m above terrain (eye height)
  5. Rendering ground-level views via Open3D offscreen
  6. Writing synthetic COLMAP entries for retraining

Fatal on failure — if something fails, the pipeline aborts.

Usage:
    python generate_ground_views.py \
        --scene_path /root/output/scene \
        --model_path /root/output/instantsplat \
        --input_dir /root/input \
        --output_dir /root/output/ground_views \
        --odm_orthophoto /root/output/odm/odm_orthophoto/odm_orthophoto.tif \
        --slack_webhook_url ... \
        --job_id ...
"""

import argparse
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    pass


def notify_slack(message, webhook_url, job_id, status="info"):
    """Non-blocking Slack notification."""
    if not webhook_url:
        return
    import urllib.request
    emoji = {"info": "\U0001f504", "success": "\u2705", "error": "\u274c"}.get(status, "\U0001f504")
    payload = json.dumps({"text": f"{emoji} *[{job_id[:8]}] ground* \u2014 {message}"})
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
# DEM fetching
# ---------------------------------------------------------------------------

def fetch_dem(bbox_latlon, resolution=1):
    """Fetch USGS 3DEP DEM for the given bounding box.

    Args:
        bbox_latlon: (min_lon, min_lat, max_lon, max_lat) in WGS84
        resolution: target resolution in meters (1, 3, 10, or 30)

    Returns:
        dem_array: 2D numpy array of elevations (meters)
        transform: rasterio-style affine transform
        crs: coordinate reference system string

    Falls back to coarser resolution if finer is unavailable.
    """
    import py3dep

    resolutions = [resolution, 3, 10, 30]
    # Remove duplicates while preserving order
    seen = set()
    res_list = []
    for r in resolutions:
        if r not in seen:
            seen.add(r)
            res_list.append(r)

    for res in res_list:
        try:
            print(f"  Fetching 3DEP DEM at {res}m resolution...")
            dem = py3dep.get_dem(bbox_latlon, resolution=res, crs="EPSG:4326")
            dem_array = dem.values
            if np.isnan(dem_array).all():
                print(f"  {res}m DEM returned all NaN, trying next resolution")
                continue

            # Fill NaN with nearest neighbor
            from scipy.ndimage import distance_transform_edt
            nan_mask = np.isnan(dem_array)
            if nan_mask.any():
                indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
                dem_array = dem_array[tuple(indices)]

            print(f"  DEM shape: {dem_array.shape}, range: [{dem_array.min():.1f}, {dem_array.max():.1f}]m")
            return dem_array, dem.rio.transform(), str(dem.rio.crs)
        except Exception as e:
            print(f"  {res}m DEM fetch failed: {e}")
            continue

    return None, None, None


def create_flat_dem(median_alt, bbox_utm, grid_size=50):
    """Create a flat DEM at median GPS altitude as fallback.

    Returns dem_array and UTM bounds.
    """
    dem_array = np.full((grid_size, grid_size), median_alt, dtype=np.float32)
    return dem_array


# ---------------------------------------------------------------------------
# Terrain mesh
# ---------------------------------------------------------------------------

def build_terrain_mesh(dem_array, bbox_utm, odm_ortho_path=None, scene_caption=""):
    """Build an Open3D TriangleMesh from DEM grid, optionally textured.

    Args:
        dem_array: 2D elevation array
        bbox_utm: (min_e, min_n, max_e, max_n) in UTM meters
        odm_ortho_path: path to ODM orthophoto TIF (optional)
        scene_caption: fallback color description if no orthophoto

    Returns:
        open3d.geometry.TriangleMesh
    """
    import open3d as o3d

    rows, cols = dem_array.shape
    min_e, min_n, max_e, max_n = bbox_utm

    # Create grid vertices
    es = np.linspace(min_e, max_e, cols)
    ns = np.linspace(max_n, min_n, rows)  # top-to-bottom
    ee, nn = np.meshgrid(es, ns)

    vertices = np.stack([ee.ravel(), nn.ravel(), dem_array.ravel()], axis=-1)

    # Create triangles (two per grid cell)
    triangles = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i = r * cols + c
            # Triangle 1
            triangles.append([i, i + cols, i + 1])
            # Triangle 2
            triangles.append([i + 1, i + cols, i + cols + 1])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()

    # Apply texture / color
    if odm_ortho_path and os.path.exists(odm_ortho_path):
        try:
            ortho = PILImage.open(odm_ortho_path).convert("RGB")
            ortho_resized = ortho.resize((cols, rows), PILImage.LANCZOS)
            ortho_arr = np.array(ortho_resized, dtype=np.float64) / 255.0
            colors = ortho_arr.reshape(-1, 3)
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            print(f"  Terrain mesh textured from ODM orthophoto")
        except Exception as e:
            print(f"  Orthophoto texturing failed: {e}, using uniform color")
            _apply_uniform_color(mesh, scene_caption)
    else:
        _apply_uniform_color(mesh, scene_caption)

    return mesh


def _apply_uniform_color(mesh, caption=""):
    """Apply a uniform color based on scene description."""
    import open3d as o3d

    caption_lower = caption.lower()
    if "grass" in caption_lower or "green" in caption_lower:
        color = [0.35, 0.55, 0.25]
    elif "sand" in caption_lower or "desert" in caption_lower:
        color = [0.76, 0.70, 0.50]
    elif "water" in caption_lower or "lake" in caption_lower:
        color = [0.25, 0.40, 0.60]
    else:
        color = [0.45, 0.50, 0.40]  # generic terrain

    n_verts = len(mesh.vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(color, (n_verts, 1))
    )
    print(f"  Terrain mesh: uniform color {color}")


# ---------------------------------------------------------------------------
# Camera placement
# ---------------------------------------------------------------------------

def generate_ground_cameras(bbox_utm, dem_array, eye_height=1.7, total_cameras=64):
    """Generate ground-level camera positions at eye_height above terrain.

    Camera placement:
      - Perimeter ring (24): scene boundary at 10% inset, face inward, 15deg spacing
      - Interior grid (24): 4x3 grid, 2 orientations each
      - Path cameras (16): 2 random walk paths through interior

    Args:
        bbox_utm: (min_e, min_n, max_e, max_n)
        dem_array: 2D elevation grid
        eye_height: camera height above terrain in meters
        total_cameras: target number of cameras

    Returns:
        list of dicts: {position, look_at, up, image_name}
    """
    from scipy.spatial.transform import Rotation

    min_e, min_n, max_e, max_n = bbox_utm
    range_e = max_e - min_e
    range_n = max_n - min_n
    rows, cols = dem_array.shape

    def get_elevation(e, n):
        """Get elevation at UTM coordinates by bilinear interpolation."""
        # Map to grid indices
        ci = (e - min_e) / range_e * (cols - 1)
        ri = (max_n - n) / range_n * (rows - 1)
        ci = np.clip(ci, 0, cols - 1)
        ri = np.clip(ri, 0, rows - 1)
        r0, c0 = int(ri), int(ci)
        r1, c1 = min(r0 + 1, rows - 1), min(c0 + 1, cols - 1)
        fr, fc = ri - r0, ci - c0
        z = (dem_array[r0, c0] * (1 - fr) * (1 - fc) +
             dem_array[r0, c1] * (1 - fr) * fc +
             dem_array[r1, c0] * fr * (1 - fc) +
             dem_array[r1, c1] * fr * fc)
        return float(z)

    cameras = []
    idx = 0
    margin = 0.10
    inner_min_e = min_e + margin * range_e
    inner_max_e = max_e - margin * range_e
    inner_min_n = min_n + margin * range_n
    inner_max_n = max_n - margin * range_n
    center_e = (min_e + max_e) / 2
    center_n = (min_n + max_n) / 2

    # --- Perimeter ring (24 cameras) ---
    n_perimeter = 24
    for i in range(n_perimeter):
        angle = 2 * math.pi * i / n_perimeter
        # Point on perimeter ellipse
        e = center_e + (inner_max_e - inner_min_e) / 2 * math.cos(angle)
        n = center_n + (inner_max_n - inner_min_n) / 2 * math.sin(angle)
        z = get_elevation(e, n) + eye_height

        # Look inward toward center
        look_e = center_e
        look_n = center_n
        look_z = get_elevation(look_e, look_n) + eye_height * 0.5  # slight down

        cameras.append({
            "position": [e, n, z],
            "look_at": [look_e, look_n, look_z],
            "up": [0, 0, 1],
            "image_name": f"groundgen_{idx:03d}.jpg",
        })
        idx += 1

    # --- Interior grid (24 cameras: 4x3 grid, 2 orientations each) ---
    grid_nx, grid_ny = 4, 3
    for gi in range(grid_nx):
        for gj in range(grid_ny):
            e = inner_min_e + (gi + 0.5) / grid_nx * (inner_max_e - inner_min_e)
            n = inner_min_n + (gj + 0.5) / grid_ny * (inner_max_n - inner_min_n)
            z = get_elevation(e, n) + eye_height

            # Two orientations: 0 and 90 degrees
            for orient in [0, math.pi / 2]:
                look_dist = max(range_e, range_n) * 0.1
                look_e = e + look_dist * math.cos(orient)
                look_n = n + look_dist * math.sin(orient)
                # Clamp look_at within bounds
                look_e = np.clip(look_e, min_e, max_e)
                look_n = np.clip(look_n, min_n, max_n)
                look_z = get_elevation(look_e, look_n) + eye_height * 0.3  # slight down (-15deg)

                cameras.append({
                    "position": [e, n, z],
                    "look_at": [look_e, look_n, look_z],
                    "up": [0, 0, 1],
                    "image_name": f"groundgen_{idx:03d}.jpg",
                })
                idx += 1

    # --- Path cameras (16: 2 paths of 8) ---
    rng = np.random.RandomState(42)
    for path_i in range(2):
        # Start at random interior point
        e = rng.uniform(inner_min_e, inner_max_e)
        n = rng.uniform(inner_min_n, inner_max_n)
        heading = rng.uniform(0, 2 * math.pi)
        step_size = max(range_e, range_n) * 0.08

        for step in range(8):
            z = get_elevation(e, n) + eye_height

            # Look in heading direction
            look_e = e + step_size * 0.5 * math.cos(heading)
            look_n = n + step_size * 0.5 * math.sin(heading)
            look_e = np.clip(look_e, min_e, max_e)
            look_n = np.clip(look_n, min_n, max_n)
            look_z = get_elevation(look_e, look_n) + eye_height * 0.3

            cameras.append({
                "position": [e, n, z],
                "look_at": [look_e, look_n, look_z],
                "up": [0, 0, 1],
                "image_name": f"groundgen_{idx:03d}.jpg",
            })
            idx += 1

            # Walk forward with slight random turn
            e += step_size * math.cos(heading)
            n += step_size * math.sin(heading)
            heading += rng.uniform(-0.5, 0.5)

            # Bounce off boundaries
            if e < inner_min_e or e > inner_max_e:
                heading = math.pi - heading
                e = np.clip(e, inner_min_e, inner_max_e)
            if n < inner_min_n or n > inner_max_n:
                heading = -heading
                n = np.clip(n, inner_min_n, inner_max_n)

    print(f"  Generated {len(cameras)} ground-level cameras (eye_height={eye_height}m)")
    return cameras


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_ground_views(mesh, cameras, output_dir, image_size=512):
    """Render ground-level views using Open3D offscreen renderer.

    Args:
        mesh: Open3D TriangleMesh (textured terrain)
        cameras: list of camera dicts from generate_ground_cameras()
        output_dir: directory to save rendered JPEGs
        image_size: output image dimensions

    Returns:
        list of saved image paths
    """
    import open3d as o3d

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    renderer = o3d.visualization.rendering.OffscreenRenderer(image_size, image_size)
    renderer.scene.set_background(np.array([0.6, 0.75, 0.9, 1.0]))  # sky blue

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("terrain", mesh, mat)

    # Add ambient light
    renderer.scene.scene.set_sun_light([0.5, -0.5, -1.0], [1.0, 1.0, 0.95], 60000)
    renderer.scene.scene.enable_sun_light(True)

    for cam in cameras:
        pos = cam["position"]
        look = cam["look_at"]
        up = cam["up"]

        renderer.setup_camera(
            60.0,  # FOV
            np.array(look, dtype=np.float64),
            np.array(pos, dtype=np.float64),
            np.array(up, dtype=np.float64),
        )

        img = renderer.render_to_image()
        out_path = os.path.join(output_dir, cam["image_name"])
        o3d.io.write_image(out_path, img)
        saved.append(out_path)

    print(f"  Rendered {len(saved)} ground-level views")
    return saved


# ---------------------------------------------------------------------------
# COLMAP entry writing
# ---------------------------------------------------------------------------

def write_synthetic_colmap_entries(cameras, scene_path, geo_to_colmap, image_size=512):
    """Append synthetic ground-view cameras to COLMAP images.txt and cameras.txt.

    Also copies rendered images to scene/images/.

    Args:
        cameras: list of camera dicts (in UTM coordinates)
        scene_path: path to scene directory with sparse_*/0/
        geo_to_colmap: tuple (s, R, T) from align_geo_colmap
        image_size: rendered image dimensions
    """
    from scipy.spatial.transform import Rotation

    s, R_align, T_align = geo_to_colmap

    # Find existing COLMAP sparse directory
    sparse_dir = None
    for sd in sorted(Path(scene_path).glob("sparse_*/0")):
        if (sd / "images.txt").exists() or (sd / "images.bin").exists():
            sparse_dir = sd
            break
    if sparse_dir is None:
        for sd in [Path(scene_path) / "sparse" / "0", Path(scene_path) / "sparse"]:
            if sd.exists():
                sparse_dir = sd
                break

    if sparse_dir is None:
        raise FileNotFoundError(f"No COLMAP sparse directory found in {scene_path}")

    # Read existing camera IDs to avoid conflicts
    images_txt = sparse_dir / "images.txt"
    cameras_txt = sparse_dir / "cameras.txt"

    max_image_id = 0
    max_camera_id = 0
    if images_txt.exists():
        with open(images_txt) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    max_image_id = max(max_image_id, int(parts[0]))
                    max_camera_id = max(max_camera_id, int(parts[8]))

    # Add a new camera model for ground views
    ground_camera_id = max_camera_id + 1
    with open(cameras_txt, "a") as f:
        # PINHOLE model: fx fy cx cy
        focal = image_size * 0.8  # ~50mm equivalent at 512px
        f.write(f"\n{ground_camera_id} PINHOLE {image_size} {image_size} "
                f"{focal} {focal} {image_size/2} {image_size/2}\n")

    # Append ground camera entries to images.txt
    scene_images_dir = os.path.join(scene_path, "images")
    os.makedirs(scene_images_dir, exist_ok=True)

    with open(images_txt, "a") as f:
        f.write("\n# Ground-level synthetic cameras (Stage 3b)\n")

        for i, cam in enumerate(cameras):
            image_id = max_image_id + i + 1

            # Transform UTM position to COLMAP coordinates
            pos_utm = np.array(cam["position"])
            pos_colmap = s * R_align @ pos_utm + T_align

            look_utm = np.array(cam["look_at"])
            look_colmap = s * R_align @ look_utm + T_align

            # Build camera rotation (look-at to rotation matrix)
            forward = look_colmap - pos_colmap
            forward = forward / np.linalg.norm(forward)
            up_world = np.array([0, 0, 1.0])
            right = np.cross(forward, up_world)
            if np.linalg.norm(right) < 1e-6:
                up_world = np.array([0, 1, 0.0])
                right = np.cross(forward, up_world)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # Camera-to-world rotation (columns = right, up, -forward for OpenGL)
            R_c2w = np.column_stack([right, up, -forward])
            R_w2c = R_c2w.T
            T_w2c = -R_w2c @ pos_colmap

            # Convert to quaternion (COLMAP format: qw qx qy qz)
            quat = Rotation.from_matrix(R_w2c).as_quat()  # [x,y,z,w] scipy
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]

            # Write image entry (2 lines: header + empty 2D points)
            f.write(f"{image_id} {qw} {qx} {qy} {qz} "
                    f"{T_w2c[0]} {T_w2c[1]} {T_w2c[2]} "
                    f"{ground_camera_id} {cam['image_name']}\n")
            f.write("\n")  # empty 2D points line

    print(f"  Wrote {len(cameras)} synthetic COLMAP entries")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3b: Ground-Level View Generation")
    parser.add_argument("--scene_path", required=True, help="Scene directory with COLMAP data")
    parser.add_argument("--model_path", required=True, help="Trained model path")
    parser.add_argument("--input_dir", required=True, help="Original input images (for EXIF GPS)")
    parser.add_argument("--output_dir", required=True, help="Output directory for ground views")
    parser.add_argument("--odm_orthophoto", default="", help="ODM orthophoto TIF for texturing")
    parser.add_argument("--exif_gps_json", default="", help="Pre-extracted EXIF GPS JSON")
    parser.add_argument("--eye_height", type=float, default=1.7, help="Camera height above ground")
    parser.add_argument("--num_cameras", type=int, default=64, help="Number of ground cameras")
    parser.add_argument("--slack_webhook_url", default="")
    parser.add_argument("--job_id", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    notify_slack("Stage 3b: generating ground-level views...",
                 args.slack_webhook_url, args.job_id)

    # --- 1. Get EXIF GPS data ---
    print("=== Stage 3b: Ground-Level View Generation ===")
    print("Loading EXIF GPS data...")

    from align_geo_colmap import (
        extract_exif_gps, compute_geo_to_colmap_transform,
        project_gps_to_utm, save_exif_gps_json,
    )

    if args.exif_gps_json and os.path.exists(args.exif_gps_json):
        with open(args.exif_gps_json) as f:
            exif_gps = json.load(f)
        print(f"  Loaded {len(exif_gps)} GPS entries from {args.exif_gps_json}")
    else:
        exif_gps = extract_exif_gps(args.input_dir)
        if exif_gps:
            save_exif_gps_json(exif_gps, os.path.join(args.output_dir, "exif_gps.json"))

    if len(exif_gps) < 4:
        print(f"ERROR: Only {len(exif_gps)} GPS points found (need >= 4)")
        notify_slack(f"ABORT: only {len(exif_gps)} GPS points (need 4+)",
                     args.slack_webhook_url, args.job_id, "error")
        sys.exit(1)

    # --- 2. Compute GPS-COLMAP alignment ---
    print("Computing GPS-COLMAP coordinate alignment...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from render_zoom_descent import load_camera_poses

    colmap_poses = load_camera_poses(args.scene_path)

    try:
        s, R, T, inliers = compute_geo_to_colmap_transform(exif_gps, colmap_poses)
    except ValueError as e:
        print(f"ERROR: Alignment failed: {e}")
        notify_slack(f"ABORT: GPS-COLMAP alignment failed: {e}",
                     args.slack_webhook_url, args.job_id, "error")
        sys.exit(1)

    print(f"  Alignment: scale={s:.6f}, {inliers.sum()}/{len(inliers)} inliers")

    # --- 3. Compute scene bounding box in lat/lon ---
    lats = [g["lat"] for g in exif_gps]
    lons = [g["lon"] for g in exif_gps]
    margin_deg = 0.001  # ~100m margin
    bbox_latlon = (
        min(lons) - margin_deg,
        min(lats) - margin_deg,
        max(lons) + margin_deg,
        max(lats) + margin_deg,
    )
    print(f"  Scene bbox (lat/lon): {bbox_latlon}")

    # --- 4. Fetch DEM ---
    print("Fetching 3DEP elevation data...")
    notify_slack("Fetching 3DEP DEM...", args.slack_webhook_url, args.job_id)
    dem_array, dem_transform, dem_crs = fetch_dem(bbox_latlon, resolution=1)

    # Compute UTM bounding box
    utm_points = [project_gps_to_utm(g["lat"], g["lon"], g["alt"]) for g in exif_gps]
    utm_arr = np.array(utm_points)
    utm_margin = 50.0  # 50m margin
    bbox_utm = (
        utm_arr[:, 0].min() - utm_margin,
        utm_arr[:, 1].min() - utm_margin,
        utm_arr[:, 0].max() + utm_margin,
        utm_arr[:, 1].max() + utm_margin,
    )

    if dem_array is None:
        print("  WARNING: All DEM fetches failed, using flat terrain")
        notify_slack("DEM fetch failed, using flat terrain",
                     args.slack_webhook_url, args.job_id, "info")
        median_alt = np.median([g["alt"] for g in exif_gps])
        dem_array = create_flat_dem(median_alt, bbox_utm)

    # --- 5. Build terrain mesh ---
    print("Building terrain mesh...")
    scene_caption = ""
    try:
        from render_zoom_descent import caption_scene
        scene_images_dir = os.path.join(args.scene_path, "images")
        if os.path.isdir(scene_images_dir):
            scene_caption = caption_scene(scene_images_dir)
    except Exception:
        pass

    mesh = build_terrain_mesh(
        dem_array, bbox_utm,
        odm_ortho_path=args.odm_orthophoto if args.odm_orthophoto else None,
        scene_caption=scene_caption,
    )

    # --- 6. Generate ground cameras ---
    print("Generating ground-level cameras...")
    cameras = generate_ground_cameras(
        bbox_utm, dem_array,
        eye_height=args.eye_height,
        total_cameras=args.num_cameras,
    )

    # --- 7. Render views ---
    print("Rendering ground-level views...")
    notify_slack(f"Rendering {len(cameras)} ground-level views...",
                 args.slack_webhook_url, args.job_id)
    render_dir = os.path.join(args.output_dir, "renders")
    rendered = render_ground_views(mesh, cameras, render_dir)

    # --- 8. Copy renders to scene/images and write COLMAP entries ---
    print("Writing synthetic COLMAP entries...")
    scene_images_dir = os.path.join(args.scene_path, "images")
    for img_path in rendered:
        shutil.copy2(img_path, os.path.join(scene_images_dir, os.path.basename(img_path)))

    write_synthetic_colmap_entries(
        cameras, args.scene_path,
        geo_to_colmap=(s, R, T),
    )

    # --- 9. Create a preview mosaic for Slack ---
    try:
        n_cols = 8
        n_rows = min(8, math.ceil(len(rendered) / n_cols))
        tile_size = 128
        mosaic = PILImage.new("RGB", (n_cols * tile_size, n_rows * tile_size))
        for i, img_path in enumerate(rendered[:n_cols * n_rows]):
            r, c = divmod(i, n_cols)
            tile = PILImage.open(img_path).resize((tile_size, tile_size), PILImage.LANCZOS)
            mosaic.paste(tile, (c * tile_size, r * tile_size))
        mosaic_path = os.path.join(args.output_dir, "ground_mosaic.jpg")
        mosaic.save(mosaic_path, "JPEG", quality=85)
        print(f"  Saved preview mosaic: {mosaic_path}")
    except Exception as e:
        print(f"  Mosaic creation failed: {e}")

    # Save metadata
    metadata = {
        "num_cameras": len(cameras),
        "eye_height": args.eye_height,
        "bbox_utm": list(bbox_utm),
        "bbox_latlon": list(bbox_latlon),
        "dem_shape": list(dem_array.shape),
        "alignment_scale": float(s),
        "alignment_inliers": int(inliers.sum()),
    }
    with open(os.path.join(args.output_dir, "ground_views_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    notify_slack(
        f"Stage 3b complete: {len(cameras)} ground views rendered + COLMAP entries written",
        args.slack_webhook_url, args.job_id, "success",
    )

    print(f"\n=== Stage 3b complete ===")
    print(f"  {len(cameras)} ground-level views at {args.eye_height}m height")
    print(f"  Renders: {render_dir}")


if __name__ == "__main__":
    main()
