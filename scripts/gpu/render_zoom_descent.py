#!/usr/bin/env python3
"""
Stage 3: Top-Down Progressive Zoom Descent

Keeps ALL cameras looking straight down (nadir) and progressively
zooms from drone altitude to ~3m AGL, generating a high-res overhead splat.

Algorithm:
  1. Load aerial splat from Stage 2 (10K training)
  2. Estimate ground Z from point cloud
  3. For each altitude level (e.g. 5 levels: 60m -> 30m -> 15m -> 7m -> 3m AGL):
     a. Generate nadir camera grid tiling the scene footprint at that altitude
     b. Narrow FOV as altitude decreases (simulates zoom / tighter crop)
     c. Render from current splat at those poses
     d. Add renders to training image set
     e. Retrain splat for N iterations with densification enabled
  4. Final model ready for compression

Key design:
  - Cameras ALWAYS look straight down (no slerp toward horizon)
  - Camera grid tiles the scene footprint (not just reusing drone positions)
  - FOV narrows at lower altitudes (zoom effect)
  - Pure render + retrain loop — no generative enhancement (deterministic)
"""

import argparse
import json
import math
import os
import shutil
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


def notify_slack(message, webhook_url, job_id, status="info"):
    """Non-blocking Slack notification."""
    if not webhook_url:
        return
    import urllib.request
    emoji = {"info": "\U0001f504", "success": "\u2705", "error": "\u274c"}.get(status, "\U0001f504")
    payload = json.dumps({"text": f"{emoji} *[{job_id[:8]}] walkable* \u2014 {message}"})
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


def upload_to_cdn(local_path, remote_key):
    """Upload a file to DO Spaces CDN via boto3. Returns the public URL."""
    endpoint = os.environ.get("SPACES_ENDPOINT", "")
    bucket = os.environ.get("SPACES_BUCKET", "")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not endpoint or not bucket or not access_key:
        return f"file://{os.path.abspath(local_path)}"
    try:
        import boto3
        session = boto3.session.Session()
        client = session.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.environ.get("AWS_DEFAULT_REGION", "nyc3"),
        )
        content_type = "image/jpeg" if local_path.endswith(".jpg") else "application/octet-stream"
        client.upload_file(
            local_path, bucket, remote_key,
            ExtraArgs={"ACL": "public-read", "ContentType": content_type},
        )
        return f"{endpoint}/{bucket}/{remote_key}"
    except Exception as e:
        print(f"  CDN upload failed: {e}")
        return f"file://{os.path.abspath(local_path)}"


def notify_slack_with_image(message, image_path, webhook_url, job_id, status="info"):
    """Send Slack message with an image preview uploaded to CDN."""
    if not webhook_url:
        return
    import urllib.request

    remote_key = f"jobs/{job_id}/descent/previews/{os.path.basename(image_path)}"
    image_url = upload_to_cdn(image_path, remote_key)

    if image_url.startswith("file://"):
        notify_slack(message, webhook_url, job_id, status)
        return

    emoji = {"info": "\U0001f504", "success": "\u2705", "error": "\u274c"}.get(status, "\U0001f504")
    payload = json.dumps({
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{emoji} *[{job_id[:8]}] descent* \u2014 {message}",
                },
            },
            {
                "type": "image",
                "image_url": image_url,
                "alt_text": message,
            },
        ]
    })
    try:
        req = urllib.request.Request(
            webhook_url, data=payload.encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        notify_slack(message, webhook_url, job_id, status)


def verify_mosaic_with_gemini(mosaic_path, level_name, webhook_url, job_id):
    """Send descent mosaic to Gemini Vision for quality check."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print(f"  Gemini verify skipped (no API key)")
        return

    try:
        import base64
        import urllib.request
        from PIL import Image as PILImage

        img = PILImage.open(mosaic_path).convert("RGB")
        img.thumbnail((512, 512))

        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = (
            "This is a stitched mosaic of rendered views from a Gaussian Splat at "
            f"{level_name}. Does it look like a coherent aerial view of terrain? "
            "Check for: black gaps, washed-out regions, splat artifacts, missing "
            "detail, blurry areas. Reply PASS or FAIL followed by a one-sentence reason."
        )

        payload = json.dumps({
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }]
        })

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_key}"
        )
        req = urllib.request.Request(
            url, data=payload.encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        passed = text.upper().startswith("PASS")
        status = "success" if passed else "error"
        print(f"  Gemini verify [{level_name}]: {text}")

        emoji = "\u2705" if passed else "\u26a0\ufe0f"
        notify_slack(
            f"{emoji} Gemini verify [{level_name}]: {text}",
            webhook_url, job_id, status,
        )
    except Exception as e:
        print(f"  Gemini verify failed ({e}), continuing")


def load_point_cloud(model_path):
    """Load the PLY point cloud and return positions as numpy array."""
    # Search for trained Gaussian splat checkpoints (prefer highest iteration)
    ckpt_dirs = sorted(Path(model_path).glob("point_cloud/iteration_*"))
    if ckpt_dirs:
        ply_path = str(ckpt_dirs[-1] / "point_cloud.ply")
        if os.path.exists(ply_path):
            ply_data = PlyData.read(ply_path)
            vertex = ply_data["vertex"]
            positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
            return positions, ply_path

    # Fallback: find any PLY
    for ply_file in sorted(Path(model_path).rglob("*.ply")):
        ply_data = PlyData.read(str(ply_file))
        if "vertex" in ply_data:
            vertex = ply_data["vertex"]
            positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
            return positions, str(ply_file)

    raise FileNotFoundError(f"No PLY file found in {model_path}")


def estimate_ground_level(positions):
    """Estimate ground Z from the lowest 5th percentile of points."""
    return float(np.percentile(positions[:, 2], 5))


def load_camera_poses(scene_path):
    """Load camera poses from COLMAP-style sparse reconstruction."""
    from scipy.spatial.transform import Rotation

    # Check sparse directories created by InstantSplat
    for sparse_dir in sorted(Path(scene_path).glob("sparse_*/0")):
        img_file = sparse_dir / "images.bin"
        if img_file.exists():
            return _load_poses_bin(img_file)
        img_file = sparse_dir / "images.txt"
        if img_file.exists():
            return _load_poses_txt(img_file)

    # Standard sparse/0 path
    for sparse_dir in [Path(scene_path) / "sparse" / "0", Path(scene_path) / "sparse"]:
        for fmt in ["images.bin", "images.txt"]:
            img_file = sparse_dir / fmt
            if img_file.exists():
                if fmt.endswith(".bin"):
                    return _load_poses_bin(img_file)
                else:
                    return _load_poses_txt(img_file)

    raise FileNotFoundError(f"No COLMAP sparse data found in {scene_path}")


def _load_poses_bin(images_path):
    """Load camera poses from COLMAP binary images.bin."""
    import struct
    from scipy.spatial.transform import Rotation

    poses = []
    with open(str(images_path), "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode())
            image_name = "".join(name_chars)

            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)

            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            center = -R.T @ t

            poses.append({
                "image_name": image_name,
                "center": center,
                "rotation": R,
                "quaternion": np.array([qw, qx, qy, qz]),
                "translation": t,
            })
    return poses


def _load_poses_txt(images_path):
    """Load camera poses from COLMAP text images.txt."""
    from scipy.spatial.transform import Rotation

    poses = []
    with open(str(images_path), "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            image_name = parts[9]

            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            center = -R.T @ t

            poses.append({
                "image_name": image_name,
                "center": center,
                "rotation": R,
                "quaternion": np.array([qw, qx, qy, qz]),
                "translation": t,
            })
            i += 2  # Skip 2D points line
        else:
            i += 1
    return poses


# ---------------------------------------------------------------------------
# Nadir camera grid generation
# ---------------------------------------------------------------------------

def generate_nadir_camera_grid(poses, ground_z, altitude_fraction, fov_deg=60.0,
                               max_images=64):
    """Generate a grid of nadir (straight-down) cameras at a given altitude.

    Args:
        poses: Original drone camera poses
        ground_z: Estimated ground Z coordinate
        altitude_fraction: Fraction of drone-to-ground height (1.0=drone, 0.05=near ground)
        fov_deg: Camera FOV in degrees (narrows at lower altitudes)
        max_images: Maximum number of images to generate at this level

    Returns:
        List of camera dicts with center, rotation, translation, fov
    """
    from scipy.spatial.transform import Rotation

    drone_centers = np.array([p["center"] for p in poses])
    avg_drone_z = np.mean(drone_centers[:, 2])

    # New altitude: interpolate between drone height and ground
    new_z = ground_z + altitude_fraction * (avg_drone_z - ground_z)

    # Scene footprint from drone positions
    xy_min = drone_centers[:, :2].min(axis=0)
    xy_max = drone_centers[:, :2].max(axis=0)
    xy_range = xy_max - xy_min

    # Inset by 10% to avoid edges where coverage is thin
    margin = 0.10
    xy_min_inset = xy_min + margin * xy_range
    xy_max_inset = xy_max - margin * xy_range

    # Ground footprint of one camera at this altitude
    height_above_ground = abs(new_z - ground_z)
    fov_rad = math.radians(fov_deg)
    footprint = 2.0 * height_above_ground * math.tan(fov_rad / 2.0)

    # Grid spacing: overlap ~30% between adjacent cameras
    spacing = footprint * 0.7 if footprint > 0 else xy_range.mean() / 4.0

    if spacing < 1e-6:
        spacing = xy_range.mean() / 4.0

    n_x = max(1, int(math.ceil((xy_max_inset[0] - xy_min_inset[0]) / spacing)))
    n_y = max(1, int(math.ceil((xy_max_inset[1] - xy_min_inset[1]) / spacing)))

    # Cap to max_images
    total = n_x * n_y
    if total > max_images:
        scale = math.sqrt(max_images / total)
        n_x = max(1, int(n_x * scale))
        n_y = max(1, int(n_y * scale))

    grid_x = np.linspace(xy_min_inset[0], xy_max_inset[0], n_x) if n_x > 1 else [np.mean([xy_min_inset[0], xy_max_inset[0]])]
    grid_y = np.linspace(xy_min_inset[1], xy_max_inset[1], n_y) if n_y > 1 else [np.mean([xy_min_inset[1], xy_max_inset[1]])]

    # Find the average drone "look-down" rotation from original poses
    # Most drone photos look roughly nadir — use the first pose's rotation as base
    # then force it to pure nadir (looking straight down)
    avg_rot = poses[0]["rotation"]  # w2c rotation

    # Pure nadir rotation: camera Z axis points down (toward -Z world if Z is up)
    # If drone Z > ground Z, camera looks in -Z direction
    z_sign = 1.0 if avg_drone_z > ground_z else -1.0

    # Build a nadir rotation matrix:
    # Camera convention: -Z is forward direction (OpenGL)
    # We want forward = [0, 0, -z_sign] (pointing down toward ground)
    # We'll use the average drone yaw to set the "up" direction in image
    avg_euler = Rotation.from_matrix(avg_rot).as_euler("zyx", degrees=True)
    avg_yaw = avg_euler[0]

    # Create nadir rotation: looking straight down, with yaw from drone
    # R_w2c that maps world-down to camera-forward
    if z_sign > 0:
        # Drone above ground: look in -Z direction
        # pitch = -90 degrees from horizontal = looking straight down
        nadir_rot = Rotation.from_euler("zyx", [avg_yaw, -90, 0], degrees=True).as_matrix()
    else:
        nadir_rot = Rotation.from_euler("zyx", [avg_yaw, 90, 0], degrees=True).as_matrix()

    cameras = []
    for gx in grid_x:
        for gy in grid_y:
            center = np.array([gx, gy, new_z])
            t_vec = -nadir_rot @ center

            cameras.append({
                "center": center,
                "rotation": nadir_rot,
                "translation": t_vec,
                "altitude_fraction": altitude_fraction,
                "fov": fov_rad,
            })

    print(f"  Altitude {altitude_fraction:.3f}: {len(cameras)} nadir cameras "
          f"(grid {n_x}x{n_y}, z={new_z:.4f}, FOV={fov_deg:.0f}deg, "
          f"footprint={footprint:.4f}, spacing={spacing:.4f})")
    return cameras


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_from_splat(model_path, cameras, output_dir, scene_path):
    """Render images from the current Gaussian splat at given camera poses."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        _render_native(model_path, cameras, output_dir, scene_path)
    except Exception as e:
        print(f"  Native rendering failed ({e}), using train.py fallback")
        _render_fallback(model_path, output_dir, scene_path)


def _render_native(model_path, cameras, output_dir, scene_path):
    """Render using gaussian-splatting rasterizer directly."""
    import torch
    from PIL import Image as PILImage

    sys.path.insert(0, "/mnt/splatwalk/InstantSplat")
    from gaussian_renderer import render
    from scene import GaussianModel
    from argparse import Namespace
    from scipy.spatial.transform import Rotation as SciRotation

    gaussians = GaussianModel(3)
    ckpt_dirs = sorted(Path(model_path).glob("point_cloud/iteration_*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {model_path}")
    latest_ckpt = ckpt_dirs[-1] / "point_cloud.ply"
    gaussians.load_ply(str(latest_ckpt))

    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    from scene.cameras import Camera

    for idx, cam in enumerate(cameras):
        R = cam["rotation"]
        T = cam["translation"]
        fov = cam.get("fov", 1.0)

        camera = Camera(
            colmap_id=idx,
            R=R,
            T=T,
            FoVx=fov,
            FoVy=fov,
            image=torch.zeros(3, 512, 512),
            gt_alpha_mask=None,
            image_name=f"zoom_{cam['altitude_fraction']:.3f}_{idx:03d}",
            uid=idx,
        )

        quat = SciRotation.from_matrix(R).as_quat()  # [x,y,z,w] scipy
        camera_pose = torch.tensor(
            [quat[3], quat[0], quat[1], quat[2],
             T[0], T[1], T[2]],
            dtype=torch.float32,
        ).cuda()
        camera.camera_pose = camera_pose

        rendering = render(camera, gaussians, pipe, bg_color, camera_pose=camera_pose)
        image = rendering["render"]

        img_np = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = PILImage.fromarray(img_np)
        out_path = os.path.join(output_dir, f"zoom_{cam['altitude_fraction']:.3f}_{idx:03d}.jpg")
        img_pil.save(out_path, "JPEG", quality=95)

    print(f"  Rendered {len(cameras)} nadir views")


def _render_fallback(model_path, output_dir, scene_path):
    """Fallback: use render.py."""
    cmd = [
        sys.executable,
        "/mnt/splatwalk/InstantSplat/render.py",
        "--source_path", scene_path,
        "--model_path", model_path,
        "--skip_train",
        "--skip_test",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        print("  render.py fallback also failed, continuing with existing images")


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

def retrain_model(scene_path, model_path, output_model_path, iterations, n_views):
    """Retrain the Gaussian splat with zoom descent images.

    Uses densification to grow new Gaussians for newly-revealed detail.
    """
    cmd = [
        sys.executable,
        "/mnt/splatwalk/InstantSplat/train.py",
        "--source_path", scene_path,
        "--model_path", output_model_path,
        "--iterations", str(iterations),
        "--n_views", str(n_views),
        "--pp_optimizer",
        "--optim_pose",
        # Densification enabled to grow detail at lower altitudes
        "--densify_from_iter", "200",
        "--opacity_reset_interval", "1000",
        "--densify_until_iter", str(iterations),
        "--densification_interval", "100",
    ]

    if output_model_path != model_path and not os.path.exists(output_model_path):
        shutil.copytree(model_path, output_model_path)

    # Clear old checkpoints — train.py starts from COLMAP, not checkpoints
    pc_dir = os.path.join(output_model_path, "point_cloud")
    if os.path.exists(pc_dir):
        shutil.rmtree(pc_dir)
        print(f"  Cleared old checkpoints from {pc_dir}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        ckpt_dirs = sorted(Path(output_model_path).glob("point_cloud/iteration_*"))
        if ckpt_dirs:
            print(f"  Retrain completed (non-zero exit), checkpoint at {ckpt_dirs[-1].name}")
        else:
            print(f"  Retrain stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"Retrain failed with code {result.returncode}")
    else:
        print(f"  Retrained for {iterations} iterations")


# ---------------------------------------------------------------------------
# Scene caption
# ---------------------------------------------------------------------------

def caption_scene(scene_images_dir):
    """Generate a scene description using Gemini."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return "grass, trees, pavement, residential area"

    try:
        import base64
        import urllib.request
        from PIL import Image as PILImage

        # Pick one aerial image
        images = sorted(Path(scene_images_dir).glob("*.jpg"))
        if not images:
            return "grass, trees, pavement, residential area"

        img = PILImage.open(images[0]).convert("RGB")
        img.thumbnail((512, 512))

        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = json.dumps({
            "contents": [{
                "parts": [
                    {"text": "Describe what terrain and features are visible in this aerial/drone photo in 10 words or less. Just list the terrain types, no sentences."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }]
        })

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        req = urllib.request.Request(url, data=payload.encode(),
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"  Scene caption: {text}")
        return text
    except Exception as e:
        print(f"  Captioning failed ({e}), using default")
        return "grass, trees, pavement, residential area"


# ---------------------------------------------------------------------------
# Mosaic stitching for quality check
# ---------------------------------------------------------------------------

def stitch_level_mosaic(render_dir, cameras, odm_orthophoto_path=None):
    """Stitch rendered tiles into a single mosaic from camera grid positions.

    Returns (mosaic_path, ssim_score). ssim_score is None if no ODM reference.
    """
    from PIL import Image as PILImage

    render_files = sorted(Path(render_dir).glob("*.jpg"))
    if not render_files or not cameras:
        return None, None

    # Get camera XY positions
    centers = np.array([c["center"][:2] for c in cameras[:len(render_files)]])
    xy_min = centers.min(axis=0)
    xy_max = centers.max(axis=0)
    xy_range = xy_max - xy_min

    # Determine grid dimensions from unique positions
    unique_x = np.unique(np.round(centers[:, 0], 6))
    unique_y = np.unique(np.round(centers[:, 1], 6))
    n_x = max(1, len(unique_x))
    n_y = max(1, len(unique_y))

    # Scale tiles so mosaic fits in 2048x2048 max
    tile_size = min(512, 2048 // max(n_x, n_y))
    mosaic_w = n_x * tile_size
    mosaic_h = n_y * tile_size

    mosaic = PILImage.new("RGB", (mosaic_w, mosaic_h), (0, 0, 0))

    for cam, render_file in zip(cameras[:len(render_files)], render_files):
        cx, cy = cam["center"][0], cam["center"][1]

        # Map to grid position
        if xy_range[0] > 1e-6:
            gx = int(round((cx - xy_min[0]) / xy_range[0] * (n_x - 1)))
        else:
            gx = 0
        if xy_range[1] > 1e-6:
            gy = int(round((cy - xy_min[1]) / xy_range[1] * (n_y - 1)))
        else:
            gy = 0

        gx = max(0, min(gx, n_x - 1))
        gy = max(0, min(gy, n_y - 1))

        tile = PILImage.open(render_file).resize(
            (tile_size, tile_size), PILImage.LANCZOS
        )
        mosaic.paste(tile, (gx * tile_size, gy * tile_size))

    # Save mosaic
    mosaic_path = os.path.join(render_dir, "..", "mosaic.jpg")
    mosaic_path = os.path.abspath(mosaic_path)
    mosaic.save(mosaic_path, "JPEG", quality=90)
    print(f"  Stitched mosaic: {mosaic_w}x{mosaic_h} ({n_x}x{n_y} grid)")

    # Compare against ODM orthophoto if available
    ssim_score = None
    if odm_orthophoto_path and os.path.exists(odm_orthophoto_path):
        try:
            import cv2
            from skimage.metrics import structural_similarity as ssim

            # Load ODM orthophoto and resize to match mosaic
            odm_img = cv2.imread(odm_orthophoto_path)
            if odm_img is not None:
                odm_resized = cv2.resize(odm_img, (mosaic_w, mosaic_h),
                                         interpolation=cv2.INTER_AREA)
                mosaic_cv = cv2.cvtColor(np.array(mosaic), cv2.COLOR_RGB2BGR)

                # Compute SSIM
                ssim_score = ssim(odm_resized, mosaic_cv, channel_axis=2)
                print(f"  SSIM vs ODM orthophoto: {ssim_score:.3f}")
        except ImportError:
            print("  skimage not available, skipping SSIM")
        except Exception as e:
            print(f"  SSIM computation failed: {e}")

    return mosaic_path, ssim_score


# ---------------------------------------------------------------------------
# FOV schedule
# ---------------------------------------------------------------------------

def get_fov_for_altitude(altitude_fraction):
    """Return camera FOV in degrees for a given altitude fraction.

    At drone altitude (1.0): wide FOV (75 deg) — like the original drone camera
    At low altitude (0.05): narrow FOV (35 deg) — simulates zoom, tighter crop
    """
    fov_high = 75.0   # at drone altitude
    fov_low = 35.0    # at lowest altitude
    # Linear interpolation
    fov = fov_low + altitude_fraction * (fov_high - fov_low)
    return fov


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Top-Down Progressive Zoom Descent")
    parser.add_argument("--model_path", required=True, help="Path to Stage 2 trained model")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for descent models")
    parser.add_argument(
        "--altitudes",
        default="1.0,0.5,0.25,0.12,0.05",
        help="Comma-separated altitude fractions (1=drone, 0=ground)",
    )
    parser.add_argument(
        "--drone_agl", type=float, default=60,
        help="Drone altitude above ground level in meters",
    )
    parser.add_argument(
        "--retrain_iterations", type=int, default=2000,
        help="Iterations per altitude level",
    )
    parser.add_argument(
        "--max_images_per_level", type=int, default=64,
        help="Max images to generate per altitude level",
    )
    parser.add_argument("--slack_webhook_url", default="", help="Slack webhook URL")
    parser.add_argument("--job_id", default="", help="Job ID for Slack notifications")
    parser.add_argument("--odm_orthophoto", default="",
                        help="Path to ODM orthophoto for SSIM comparison at each level")
    args = parser.parse_args()

    altitudes = [float(a) for a in args.altitudes.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Top-Down Progressive Zoom Descent ===")
    print(f"Altitude levels: {altitudes}")
    print(f"Drone AGL: {args.drone_agl}m")
    print(f"Retrain iterations: {args.retrain_iterations}")
    print(f"Max images/level: {args.max_images_per_level}")

    # Load geometry
    print(f"\nLoading point cloud from {args.model_path}...")
    positions, ply_path = load_point_cloud(args.model_path)
    print(f"  Loaded {len(positions)} points from {ply_path}")

    ground_z = estimate_ground_level(positions)
    print(f"  Ground level (5th percentile Z): {ground_z:.3f}")

    print(f"\nLoading camera poses from {args.scene_path}...")
    poses = load_camera_poses(args.scene_path)
    print(f"  Found {len(poses)} camera poses")

    scene_images_dir = os.path.join(args.scene_path, "images")

    current_model_path = args.model_path
    n_views = min(len(poses), 24)

    for level_idx, altitude in enumerate(altitudes):
        print(f"\n{'='*50}")
        real_alt = altitude * args.drone_agl
        print(f"Level {level_idx+1}/{len(altitudes)}: "
              f"altitude_fraction={altitude:.3f} (~{real_alt:.0f}m AGL)")
        print(f"{'='*50}")

        notify_slack(
            f"Stage 3 zoom: level {level_idx+1}/{len(altitudes)} "
            f"(~{real_alt:.0f}m AGL)",
            args.slack_webhook_url, args.job_id,
        )

        # Generate nadir camera grid
        fov_deg = get_fov_for_altitude(altitude)
        cameras = generate_nadir_camera_grid(
            poses, ground_z, altitude,
            fov_deg=fov_deg,
            max_images=args.max_images_per_level,
        )

        if not cameras:
            print("  No cameras generated, skipping level")
            continue

        # Render from current model
        level_dir = os.path.join(args.output_dir, f"level_{level_idx}_{altitude:.3f}")
        render_dir = os.path.join(level_dir, "renders")
        render_from_splat(current_model_path, cameras, render_dir, args.scene_path)

        # Stitch rendered tiles into mosaic + send to Slack
        mosaic_path, ssim_score = stitch_level_mosaic(
            render_dir, cameras,
            odm_orthophoto_path=args.odm_orthophoto if args.odm_orthophoto else None,
        )
        if mosaic_path:
            ssim_str = f" | SSIM={ssim_score:.3f}" if ssim_score is not None else ""
            notify_slack_with_image(
                f"Level {level_idx+1}/{len(altitudes)} renders (~{real_alt:.0f}m AGL, "
                f"{len(cameras)} views{ssim_str})",
                mosaic_path, args.slack_webhook_url, args.job_id,
            )
            verify_mosaic_with_gemini(
                mosaic_path,
                f"Level {level_idx+1} (~{real_alt:.0f}m AGL)",
                args.slack_webhook_url, args.job_id,
            )

        # Add renders to training set
        added = 0
        for img_file in Path(render_dir).glob("*.jpg"):
            shutil.copy2(str(img_file), os.path.join(scene_images_dir, img_file.name))
            added += 1
        print(f"  Added {added} render images to training set")

        # Retrain
        updated_n_views = len(list(Path(scene_images_dir).glob("*.jpg")))
        if updated_n_views > 24:
            updated_n_views = 24

        altitude_model = os.path.join(level_dir, "model")
        print(f"  Retraining for {args.retrain_iterations} iterations...")
        retrain_model(
            args.scene_path, current_model_path, altitude_model,
            args.retrain_iterations, updated_n_views,
        )

        current_model_path = altitude_model

        notify_slack(
            f"Stage 3 zoom: level {level_idx+1} complete "
            f"({len(cameras)} views at ~{real_alt:.0f}m)",
            args.slack_webhook_url, args.job_id,
        )

    # Copy final model
    final_dir = os.path.join(args.output_dir, "final")
    if current_model_path != final_dir:
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(current_model_path, final_dir)

    print(f"\n=== Zoom descent complete ===")
    print(f"Final model at: {final_dir}")
    print(f"Levels completed: {len(altitudes)}")

    notify_slack(
        f"Stage 3 zoom complete! {len(altitudes)} levels, final at ~{altitudes[-1]*args.drone_agl:.0f}m AGL",
        args.slack_webhook_url, args.job_id, "success",
    )


if __name__ == "__main__":
    main()
