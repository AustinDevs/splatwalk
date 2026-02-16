#!/usr/bin/env python3
"""
Generate Viewer Assets for Three-Mode Ground Demo

Generates all assets needed by the Splat/Skybox/Mesh comparison viewer:
  1. Retrain aerial model (MASt3R init + 10K InstantSplat training)
  2. Select 6 ground positions via farthest-point sampling
  3. Render 6 cubemap faces per position from trained splat
  4. Estimate depth maps with Depth-Anything-V2
  5. Generate FLUX images for all 36 cubemap faces
  6. Stitch cubemaps -> equirectangular panoramas (color + depth)
  7. Compress .splat (prune + uniform scene scaling)
  8. Upload all assets + manifest.json to Spaces CDN

Output: manifest.json with splat_url + per-position panorama/depth URLs

Usage:
    python generate_viewer_assets.py \
        --scene_path /root/output/aukerman/scene \
        --model_path /root/output/aukerman/model \
        --output_dir /root/output/aukerman-demo \
        --job_id aukerman \
        --drone_agl 63
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from PIL import Image
    from plyfile import PlyData
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData

# Import reusable functions from sibling scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_ground_views import (
    load_point_cloud,
    load_camera_poses,
    extract_gps_from_images,
    caption_aerial_scene,
    _render_rgb_from_splat,
    _estimate_depth_maps,
    extract_nearest_aerial_crop,
)
from generate_tiered_views import (
    farthest_point_sample,
    upload_to_cdn,
    notify_slack_text,
)


# ---------------------------------------------------------------------------
# Ground position selection
# ---------------------------------------------------------------------------

def select_ground_positions(poses, positions, ground_z, drone_agl, n_positions=6):
    """Select N well-spaced ground positions via farthest-point sampling.

    Places cameras at 1.7m AGL on the XY grid covered by the drone survey.
    Returns list of [x, y, z] world coordinates.
    """
    drone_centers = np.array([p["center"] for p in poses])
    drone_xy = drone_centers[:, :2]

    avg_cam_z = np.mean(drone_centers[:, 2])
    scene_height = abs(avg_cam_z - ground_z)
    meters_per_unit = drone_agl / max(scene_height, 1e-6)
    z_sign = 1.0 if avg_cam_z > ground_z else -1.0

    eye_z = ground_z + z_sign * 1.7 / meters_per_unit

    # Create candidate grid at ~5m spacing
    xy_min = drone_xy.min(axis=0)
    xy_max = drone_xy.max(axis=0)
    xy_range = xy_max - xy_min

    grid_spacing = 5.0 / meters_per_unit
    n_x = max(3, int(xy_range[0] / grid_spacing))
    n_y = max(3, int(xy_range[1] / grid_spacing))
    # Cap grid to prevent excess
    n_x = min(n_x, 30)
    n_y = min(n_y, 30)

    grid_x = np.linspace(xy_min[0] + 0.15 * xy_range[0],
                          xy_max[0] - 0.15 * xy_range[0], n_x)
    grid_y = np.linspace(xy_min[1] + 0.15 * xy_range[1],
                          xy_max[1] - 0.15 * xy_range[1], n_y)

    candidates = []
    for gx in grid_x:
        for gy in grid_y:
            candidates.append([gx, gy])

    candidates = np.array(candidates)
    indices = farthest_point_sample(candidates, n_positions)
    selected_xy = candidates[indices]

    world_positions = []
    for xy in selected_xy:
        world_positions.append([float(xy[0]), float(xy[1]), float(eye_z)])

    print(f"  Selected {len(world_positions)} ground positions at eye_z={eye_z:.4f}")
    return world_positions


# ---------------------------------------------------------------------------
# Cubemap camera generation
# ---------------------------------------------------------------------------

# Cubemap face definitions: (name, yaw_deg, pitch_deg)
CUBEMAP_FACES = [
    ("px", 0, 0),      # +X (right)
    ("nx", 180, 0),    # -X (left)
    ("py", 0, 90),     # +Y (up)
    ("ny", 0, -90),    # -Y (down)
    ("pz", 90, 0),     # +Z (front)
    ("nz", 270, 0),    # -Z (back)
]


def generate_cubemap_cameras(world_positions):
    """Generate 6 cubemap face cameras for each ground position.

    Each face is 90 deg FOV. Returns list of camera dicts compatible with
    _render_rgb_from_splat().
    """
    from scipy.spatial.transform import Rotation

    cameras = []
    for pos_idx, pos in enumerate(world_positions):
        center = np.array(pos)
        for face_name, yaw_deg, pitch_deg in CUBEMAP_FACES:
            yaw_rad = math.radians(yaw_deg)
            pitch_rad = math.radians(pitch_deg)
            R = Rotation.from_euler("zyx", [yaw_rad, pitch_rad, 0]).as_matrix()
            t_vec = -R @ center

            cameras.append({
                "center": center,
                "rotation": R,
                "translation": t_vec,
                "pos_idx": pos_idx,
                "face": face_name,
                "camera_type": "cubemap",
            })

    print(f"  Generated {len(cameras)} cubemap cameras "
          f"({len(world_positions)} positions x 6 faces)")
    return cameras


# ---------------------------------------------------------------------------
# Cubemap -> Equirectangular stitching
# ---------------------------------------------------------------------------

def cubemap_to_equirectangular(face_images, width=2048, height=1024):
    """Convert 6 cubemap face images to equirectangular panorama.

    Args:
        face_images: dict mapping face name -> PIL Image (512x512)
                     Keys: 'px', 'nx', 'py', 'ny', 'pz', 'nz'
        width: output panorama width
        height: output panorama height

    Returns:
        PIL Image of equirectangular panorama
    """
    # Convert face images to numpy arrays
    faces = {}
    face_size = None
    for name, img in face_images.items():
        arr = np.array(img)
        faces[name] = arr
        face_size = arr.shape[0]

    # Create output coordinate grid
    u = np.linspace(0, 1, width, endpoint=False) + 0.5 / width
    v = np.linspace(0, 1, height, endpoint=False) + 0.5 / height

    # Convert to spherical coordinates
    theta = u * 2 * np.pi       # longitude: 0 to 2pi
    phi = v * np.pi              # latitude: 0 (top) to pi (bottom)

    uu, vv = np.meshgrid(u, v)
    theta = uu * 2 * np.pi
    phi = vv * np.pi

    # Spherical to cartesian (OpenGL convention: Y up)
    x = np.sin(phi) * np.sin(theta)
    y = np.cos(phi)
    z = np.sin(phi) * np.cos(theta)

    # Determine which cubemap face each pixel maps to
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z)

    max_axis = np.maximum(abs_x, np.maximum(abs_y, abs_z))

    output = np.zeros((height, width, 3), dtype=np.uint8)

    def _sample_face(face_arr, face_u, face_v, mask):
        """Sample pixels from a cubemap face using bilinear coordinates."""
        # Map [-1, 1] to [0, face_size-1]
        px = ((face_u + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        py = ((face_v + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        px = np.clip(px, 0, face_size - 1)
        py = np.clip(py, 0, face_size - 1)
        output[mask] = face_arr[py[mask], px[mask]]

    # +X face (right): x is max positive
    mask = (x == max_axis) & (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
    if "px" in faces and np.any(mask):
        fu = -z[mask] / abs_x[mask]
        fv = -y[mask] / abs_x[mask]
        px_coord = ((fu + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        py_coord = ((fv + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        px_coord = np.clip(px_coord, 0, face_size - 1)
        py_coord = np.clip(py_coord, 0, face_size - 1)
        output[mask] = faces["px"][py_coord, px_coord]

    # -X face (left): x is max negative
    mask = (-x == max_axis) & (abs_x >= abs_y) & (abs_x >= abs_z) & (x <= 0)
    if "nx" in faces and np.any(mask):
        fu = z[mask] / abs_x[mask]
        fv = -y[mask] / abs_x[mask]
        px_coord = ((fu + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        py_coord = ((fv + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        px_coord = np.clip(px_coord, 0, face_size - 1)
        py_coord = np.clip(py_coord, 0, face_size - 1)
        output[mask] = faces["nx"][py_coord, px_coord]

    # +Y face (up): y is max positive
    mask = (y == max_axis) & (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
    if "py" in faces and np.any(mask):
        fu = x[mask] / abs_y[mask]
        fv = z[mask] / abs_y[mask]
        px_coord = ((fu + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        py_coord = ((fv + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        px_coord = np.clip(px_coord, 0, face_size - 1)
        py_coord = np.clip(py_coord, 0, face_size - 1)
        output[mask] = faces["py"][py_coord, px_coord]

    # -Y face (down): y is max negative
    mask = (-y == max_axis) & (abs_y >= abs_x) & (abs_y >= abs_z) & (y <= 0)
    if "ny" in faces and np.any(mask):
        fu = x[mask] / abs_y[mask]
        fv = -z[mask] / abs_y[mask]
        px_coord = ((fu + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        py_coord = ((fv + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        px_coord = np.clip(px_coord, 0, face_size - 1)
        py_coord = np.clip(py_coord, 0, face_size - 1)
        output[mask] = faces["ny"][py_coord, px_coord]

    # +Z face (front): z is max positive
    mask = (z == max_axis) & (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
    if "pz" in faces and np.any(mask):
        fu = x[mask] / abs_z[mask]
        fv = -y[mask] / abs_z[mask]
        px_coord = ((fu + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        py_coord = ((fv + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        px_coord = np.clip(px_coord, 0, face_size - 1)
        py_coord = np.clip(py_coord, 0, face_size - 1)
        output[mask] = faces["pz"][py_coord, px_coord]

    # -Z face (back): z is max negative
    mask = (-z == max_axis) & (abs_z >= abs_x) & (abs_z >= abs_y) & (z <= 0)
    if "nz" in faces and np.any(mask):
        fu = -x[mask] / abs_z[mask]
        fv = -y[mask] / abs_z[mask]
        px_coord = ((fu + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        py_coord = ((fv + 1) * 0.5 * (face_size - 1)).astype(np.int32)
        px_coord = np.clip(px_coord, 0, face_size - 1)
        py_coord = np.clip(py_coord, 0, face_size - 1)
        output[mask] = faces["nz"][py_coord, px_coord]

    return Image.fromarray(output)


# ---------------------------------------------------------------------------
# FLUX generation for cubemap faces
# ---------------------------------------------------------------------------

def generate_cubemap_flux(depth_dir, rgb_dir, prompt, output_dir,
                          cameras, poses, scene_images_dir):
    """Generate FLUX images for all cubemap faces.

    Reuses generate_ground_views.py FLUX pipeline pattern.
    """
    import torch
    from PIL import Image as PILImage
    from diffusers import FluxControlNetPipeline, FluxControlNetModel

    # Monkey-patch: PyTorch 2.4 doesn't support enable_gqa
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _patched_sdpa(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

    os.makedirs(output_dir, exist_ok=True)

    print("  Loading FLUX.1-dev + ControlNet-depth pipeline...")
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
        print("  Loading XLabs IP-Adapter v2...")
        pipe.load_ip_adapter(
            "XLabs-AI/flux-ip-adapter-v2",
            weight_name="ip_adapter.safetensors",
            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
        )
        pipe.set_ip_adapter_scale(0.6)
        ip_adapter_loaded = True
        print("  IP-Adapter loaded (scale=0.6)")
    except Exception as e:
        print(f"  IP-Adapter loading failed ({e}), continuing without it")

    depth_files = sorted(Path(depth_dir).glob("*.jpg"))
    rgb_files = sorted(Path(rgb_dir).glob("*.jpg"))

    generated_count = 0
    for idx, (depth_file, rgb_file) in enumerate(zip(depth_files, rgb_files)):
        blurry_render = PILImage.open(rgb_file).convert("RGB").resize(
            (1024, 1024), PILImage.LANCZOS
        )

        # Skip very dark renders
        avg_brightness = np.array(blurry_render).mean()
        if avg_brightness < 15:
            print(f"  Skipping face {idx} (too dark, brightness={avg_brightness:.1f})")
            # Still save a black placeholder to keep indexing aligned
            black = PILImage.new("RGB", (512, 512), (0, 0, 0))
            black.save(os.path.join(output_dir, f"groundgen_{idx:03d}.jpg"), "JPEG")
            generated_count += 1
            continue

        depth_map = PILImage.open(depth_file).convert("RGB").resize(
            (1024, 1024), PILImage.LANCZOS
        )

        gen_kwargs = dict(
            prompt=prompt,
            control_image=depth_map,
            controlnet_conditioning_scale=0.4,
            num_inference_steps=28,
            guidance_scale=4.0,
            height=1024,
            width=1024,
        )

        # IP-Adapter reference from nearest aerial image
        if ip_adapter_loaded and idx < len(cameras):
            aerial_crop = extract_nearest_aerial_crop(
                cameras[idx], poses, scene_images_dir
            )
            if aerial_crop is not None:
                gen_kwargs["ip_adapter_image"] = aerial_crop

        result = pipe(**gen_kwargs).images[0]
        result = result.resize((512, 512), PILImage.LANCZOS)
        result.save(os.path.join(output_dir, f"groundgen_{idx:03d}.jpg"),
                    "JPEG", quality=95)
        generated_count += 1

        if generated_count % 6 == 0:
            print(f"  Generated {generated_count}/{len(depth_files)} cubemap faces")

    print(f"  Generated {generated_count} FLUX cubemap face images")

    del pipe, controlnet
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stitch panoramas from cubemap faces
# ---------------------------------------------------------------------------

def stitch_panoramas(flux_dir, n_positions, output_dir):
    """Stitch cubemap faces into equirectangular panoramas for each position.

    Returns list of panorama file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    panorama_paths = []

    face_names = [f[0] for f in CUBEMAP_FACES]

    for pos_idx in range(n_positions):
        face_images = {}
        for face_idx, face_name in enumerate(face_names):
            img_idx = pos_idx * 6 + face_idx
            img_path = os.path.join(flux_dir, f"groundgen_{img_idx:03d}.jpg")
            if os.path.exists(img_path):
                face_images[face_name] = Image.open(img_path).convert("RGB")

        if len(face_images) < 6:
            print(f"  WARNING: Position {pos_idx} has only {len(face_images)}/6 faces")
            # Fill missing faces with black
            for fn in face_names:
                if fn not in face_images:
                    face_images[fn] = Image.new("RGB", (512, 512), (0, 0, 0))

        panorama = cubemap_to_equirectangular(face_images)
        pano_path = os.path.join(output_dir, f"pos_{pos_idx}_panorama.jpg")
        panorama.save(pano_path, "JPEG", quality=92)
        panorama_paths.append(pano_path)
        print(f"  Stitched panorama for position {pos_idx}: {panorama.size}")

    return panorama_paths


def stitch_depth_panoramas(depth_dir, n_positions, output_dir):
    """Stitch depth cubemap faces into equirectangular depth panoramas."""
    os.makedirs(output_dir, exist_ok=True)
    depth_pano_paths = []

    face_names = [f[0] for f in CUBEMAP_FACES]

    for pos_idx in range(n_positions):
        face_images = {}
        for face_idx, face_name in enumerate(face_names):
            img_idx = pos_idx * 6 + face_idx
            img_path = os.path.join(depth_dir, f"groundgen_{img_idx:03d}.jpg")
            if os.path.exists(img_path):
                face_images[face_name] = Image.open(img_path).convert("RGB")

        if len(face_images) < 6:
            for fn in face_names:
                if fn not in face_images:
                    face_images[fn] = Image.new("RGB", (512, 512), (128, 128, 128))

        panorama = cubemap_to_equirectangular(face_images)
        pano_path = os.path.join(output_dir, f"pos_{pos_idx}_depth_panorama.jpg")
        panorama.save(pano_path, "JPEG", quality=92)
        depth_pano_paths.append(pano_path)

    print(f"  Stitched {len(depth_pano_paths)} depth panoramas")
    return depth_pano_paths


# ---------------------------------------------------------------------------
# Compress .splat
# ---------------------------------------------------------------------------

def compress_splat(model_path, output_splat, prune_ratio=0.20, scene_scale=50.0):
    """Invoke compress_splat.py to create .splat file."""
    # Find latest checkpoint PLY
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
# Upload assets + manifest
# ---------------------------------------------------------------------------

def upload_assets(output_dir, panorama_paths, depth_pano_paths,
                  splat_path, world_positions, job_id):
    """Upload all assets to Spaces CDN and generate manifest.json."""
    remote_prefix = f"demo/{job_id}"

    # Upload .splat
    splat_url = upload_to_cdn(splat_path, f"{remote_prefix}/scene.splat")

    # Upload panoramas + depth panoramas
    positions_manifest = []
    for idx, (pano_path, depth_path, world_xyz) in enumerate(
        zip(panorama_paths, depth_pano_paths, world_positions)
    ):
        pano_url = upload_to_cdn(pano_path, f"{remote_prefix}/pos_{idx}_panorama.jpg")
        depth_url = upload_to_cdn(depth_path, f"{remote_prefix}/pos_{idx}_depth_panorama.jpg")
        positions_manifest.append({
            "id": f"pos_{idx}",
            "world_xyz": world_xyz,
            "panorama_url": pano_url,
            "depth_panorama_url": depth_url,
        })

    manifest = {
        "splat_url": splat_url,
        "positions": positions_manifest,
    }

    # Save manifest locally
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Upload manifest
    upload_to_cdn(manifest_path, f"{remote_prefix}/manifest.json")

    print(f"  Manifest: {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Viewer Assets for Three-Mode Demo")
    parser.add_argument("--model_path", required=True, help="Path to trained aerial model")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--job_id", default="aukerman", help="Job ID for CDN paths")
    parser.add_argument("--drone_agl", type=float, default=63, help="Drone AGL in meters")
    parser.add_argument("--n_positions", type=int, default=6, help="Number of ground positions")
    parser.add_argument("--prune_ratio", type=float, default=0.20, help="Splat prune ratio")
    parser.add_argument("--scene_scale", type=float, default=50.0, help="Uniform scene scale")
    parser.add_argument("--slack_webhook_url", default="")
    args = parser.parse_args()

    if not args.slack_webhook_url:
        args.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")

    os.makedirs(args.output_dir, exist_ok=True)
    scene_images_dir = os.path.join(args.scene_path, "images")

    notify_slack_text("Demo assets: starting generation...",
                      args.slack_webhook_url, args.job_id)

    # --- 1. Load scene geometry ---
    print("Loading point cloud...")
    positions, ply_path = load_point_cloud(args.model_path)
    print(f"  {len(positions)} points from {ply_path}")

    ground_z = np.percentile(positions[:, 2], 5)
    print(f"  Ground Z: {ground_z:.4f}")

    # --- 2. Load camera poses ---
    print("Loading camera poses...")
    poses = load_camera_poses(args.scene_path)
    print(f"  {len(poses)} camera poses")

    # --- 3. Select ground positions ---
    print("Selecting ground positions...")
    world_positions = select_ground_positions(
        poses, positions, ground_z, args.drone_agl, n_positions=args.n_positions
    )

    # --- 4. Generate cubemap cameras ---
    print("Generating cubemap cameras...")
    cameras = generate_cubemap_cameras(world_positions)

    # Override FOV for cubemap faces (90 deg)
    for cam in cameras:
        cam["fov"] = math.pi / 2  # 90 degrees

    # --- 5. Render RGB from splat ---
    rgb_dir = os.path.join(args.output_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    print("Rendering cubemap RGB from splat...")
    notify_slack_text(f"Demo assets: rendering {len(cameras)} cubemap faces...",
                      args.slack_webhook_url, args.job_id)
    _render_rgb_from_splat(args.model_path, cameras, rgb_dir, args.scene_path)

    # --- 6. Estimate depth maps ---
    depth_dir = os.path.join(args.output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    print("Estimating depth maps...")
    _estimate_depth_maps(rgb_dir, depth_dir)

    # --- 7. Caption scene for FLUX prompt ---
    print("Captioning aerial scene...")
    lat, lon, _alt = extract_gps_from_images(scene_images_dir)
    google_maps_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    prompt = caption_aerial_scene(
        scene_images_dir, lat=lat, lon=lon, google_maps_key=google_maps_key
    )

    # --- 8. Generate FLUX cubemap faces ---
    flux_dir = os.path.join(args.output_dir, "flux")
    print(f"Generating {len(cameras)} FLUX cubemap faces...")
    notify_slack_text(f"Demo assets: generating {len(cameras)} FLUX faces (~25 min)...",
                      args.slack_webhook_url, args.job_id)
    generate_cubemap_flux(
        depth_dir, rgb_dir, prompt, flux_dir,
        cameras, poses, scene_images_dir,
    )

    # --- 9. Stitch panoramas ---
    pano_dir = os.path.join(args.output_dir, "panoramas")
    print("Stitching color panoramas...")
    panorama_paths = stitch_panoramas(flux_dir, args.n_positions, pano_dir)

    print("Stitching depth panoramas...")
    depth_pano_paths = stitch_depth_panoramas(
        depth_dir, args.n_positions, pano_dir
    )

    # --- 10. Compress .splat ---
    splat_path = os.path.join(args.output_dir, "scene.splat")
    print("Compressing .splat...")
    notify_slack_text("Demo assets: compressing .splat...",
                      args.slack_webhook_url, args.job_id)
    compress_splat(
        args.model_path, splat_path,
        prune_ratio=args.prune_ratio,
        scene_scale=args.scene_scale,
    )

    # --- 11. Upload to CDN ---
    print("Uploading assets to CDN...")
    notify_slack_text("Demo assets: uploading to CDN...",
                      args.slack_webhook_url, args.job_id)
    manifest = upload_assets(
        args.output_dir, panorama_paths, depth_pano_paths,
        splat_path, world_positions, args.job_id,
    )

    notify_slack_text(
        f"Demo assets complete! {len(world_positions)} positions, "
        f"{len(panorama_paths)} panoramas",
        args.slack_webhook_url, args.job_id, "success",
    )

    print(f"\nDone. Manifest: {json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main()
