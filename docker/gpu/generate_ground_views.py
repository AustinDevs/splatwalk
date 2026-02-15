#!/usr/bin/env python3
"""
Stage 3.5: ControlNet Ground View Generation

Generates photorealistic ground-level training images using FLUX.1-dev with
ControlNet-depth conditioning and IP-Adapter aerial texture matching.

Pipeline:
  1. Extract GPS from EXIF + geographic enrichment (species, hardiness zone)
  2. Generate scene description via Gemini API (falls back to BLIP)
  3. Generate ground-level camera poses (perimeter walk + interior grid)
  4. Render blurry RGB + depth maps from existing splat at those poses
  5. Extract aerial crops nearest each ground camera for IP-Adapter conditioning
  6. Generate photorealistic views with FLUX.1-dev + ControlNet-depth + IP-Adapter
  7. Add generated images to training set and retrain splat
"""

import argparse
import io
import json
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


# ---------------------------------------------------------------------------
# EXIF GPS extraction
# ---------------------------------------------------------------------------

def extract_gps_from_images(scene_images_dir):
    """Extract GPS coordinates from EXIF data of drone images.

    Returns (lat, lon, altitude_m) or (None, None, None) if no GPS found.
    """
    from PIL import ExifTags

    GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}

    def _dms_to_decimal(dms, ref):
        """Convert EXIF DMS (degrees, minutes, seconds) to decimal degrees."""
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal

    aerial_files = []
    for img_file in sorted(Path(scene_images_dir).glob("*")):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            name = img_file.stem.lower()
            if not (name.startswith("descent_") or name.startswith("enhanced_") or name.startswith("groundgen_")):
                aerial_files.append(img_file)

    for img_file in aerial_files:
        try:
            img = Image.open(img_file)
            exif_data = img._getexif()
            if not exif_data:
                continue

            gps_info = exif_data.get(ExifTags.Base.GPSInfo)
            if not gps_info:
                continue

            # Decode GPS tags
            gps = {}
            for tag_id, value in gps_info.items():
                tag_name = ExifTags.GPSTAGS.get(tag_id, tag_id)
                gps[tag_name] = value

            if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
                continue

            lat = _dms_to_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
            lon = _dms_to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "W"))

            # Extract altitude if available
            altitude = None
            if "GPSAltitude" in gps:
                altitude = float(gps["GPSAltitude"])
                if gps.get("GPSAltitudeRef", b'\x00') == b'\x01':
                    altitude = -altitude

            print(f"  EXIF GPS: {lat:.4f}, {lon:.4f}" + (f", alt={altitude:.1f}m" if altitude else ""))
            return lat, lon, altitude

        except Exception:
            continue

    print("  No EXIF GPS found in images")
    return None, None, None


# ---------------------------------------------------------------------------
# Geographic enrichment
# ---------------------------------------------------------------------------

def get_geographic_context(lat, lon, google_maps_key=None):
    """Build geographic context string from coordinates using external APIs.

    Returns a context string for Gemini prompt, or empty string on failure.
    """
    import urllib.request

    context_parts = []
    zip_code = None

    # Google Maps Geocoding
    if google_maps_key:
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={google_maps_key}"
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            if data.get("results"):
                address = data["results"][0].get("formatted_address", "")
                if address:
                    context_parts.append(f"Location: {address}")
                # Extract ZIP code
                for component in data["results"][0].get("address_components", []):
                    if "postal_code" in component.get("types", []):
                        zip_code = component["long_name"]
                        break
        except Exception as e:
            print(f"  Geocoding failed: {e}")

    # USDA Hardiness Zone (requires ZIP code)
    if zip_code:
        try:
            url = f"https://phzmapi.org/{zip_code}.json"
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            zone = data.get("zone", "")
            if zone:
                temp_range = data.get("temperature_range", "")
                zone_str = f"USDA Hardiness Zone: {zone}"
                if temp_range:
                    zone_str += f" ({temp_range})"
                context_parts.append(zone_str)
        except Exception as e:
            print(f"  Hardiness zone lookup failed: {e}")

    # Current weather from Open-Meteo
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read().decode())
        weather = data.get("current_weather", {})
        temp_c = weather.get("temperature")
        if temp_c is not None:
            temp_f = temp_c * 9 / 5 + 32
            context_parts.append(f"Current temperature: {temp_c:.0f}°C ({temp_f:.0f}°F)")
    except Exception as e:
        print(f"  Weather lookup failed: {e}")

    context = "\n".join(context_parts)
    if context:
        print(f"  Geographic context:\n    " + context.replace("\n", "\n    "))
    return context


def get_ground_elevation(lat, lon):
    """Get ground elevation in meters from Open-Meteo API."""
    import urllib.request

    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read().decode())
        elev = data.get("elevation")
        if elev is not None:
            if isinstance(elev, list):
                elev = elev[0]
            return float(elev)
    except Exception as e:
        print(f"  Ground elevation lookup failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Gemini API captioning (replaces BLIP)
# ---------------------------------------------------------------------------

def caption_aerial_scene(scene_images_dir, lat=None, lon=None, google_maps_key=None):
    """Generate a rich scene description using Gemini 2.5 Flash API.

    Falls back to BLIP if Gemini API is unavailable.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print("  No GEMINI_API_KEY, falling back to BLIP captioning")
        return _caption_with_blip(scene_images_dir)

    try:
        from google import genai
    except ImportError:
        print("  google-genai not installed, falling back to BLIP captioning")
        return _caption_with_blip(scene_images_dir)

    # Build geographic context
    geo_context = ""
    if lat is not None and lon is not None:
        geo_context = get_geographic_context(lat, lon, google_maps_key)

    # Select 3 representative aerial images
    aerial_files = []
    for img_file in sorted(Path(scene_images_dir).glob("*")):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            name = img_file.stem.lower()
            if not (name.startswith("descent_") or name.startswith("enhanced_") or name.startswith("groundgen_")):
                aerial_files.append(img_file)

    if not aerial_files:
        print("  No aerial images found for captioning")
        return _caption_with_blip(scene_images_dir)

    sample = aerial_files[:: max(1, len(aerial_files) // 3)][:3]

    # Prepare images for API (convert MPO to RGB JPEG, resize to 1024px max)
    image_parts = []
    for img_file in sample:
        try:
            img = Image.open(img_file).convert("RGB")
            # Resize to 1024px max dimension
            max_dim = max(img.size)
            if max_dim > 1024:
                scale = 1024 / max_dim
                img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            image_parts.append(buf.getvalue())
        except Exception as e:
            print(f"  Failed to prepare {img_file.name}: {e}")

    if not image_parts:
        return _caption_with_blip(scene_images_dir)

    # Build prompt
    geo_section = f"\nGeographic context:\n{geo_context}\n" if geo_context else ""
    prompt = f"""You are analyzing aerial drone photographs of a property to generate a detailed description for AI image generation of ground-level photorealistic views.
{geo_section}
Analyze these images and provide:
1. SCENE DESCRIPTION: Trees, grass, ground cover, structures, paths, terrain, water.
2. FLORA IDENTIFICATION: Specific species based on imagery + geographic location.
3. FLUX PROMPT: A single dense paragraph for AI image generation — include specific plant species, materials, lighting, atmosphere. Ground-level perspective."""

    try:
        client = genai.Client(api_key=gemini_key)

        # Build contents: text prompt + images
        contents = [prompt]
        for img_bytes in image_parts:
            contents.append(genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        text = response.text
        print(f"  Gemini response ({len(text)} chars)")

        # Extract FLUX PROMPT section
        flux_prompt = _extract_flux_prompt(text)
        if flux_prompt:
            print(f"  FLUX prompt: {flux_prompt[:120]}...")
            return flux_prompt
        else:
            # Use the full response as prompt if we can't extract the section
            print(f"  Using full Gemini response as prompt")
            return text[:500]

    except Exception as e:
        print(f"  Gemini API failed ({e}), falling back to BLIP captioning")
        return _caption_with_blip(scene_images_dir)


def _extract_flux_prompt(text):
    """Extract the FLUX PROMPT section from Gemini response."""
    # Look for "FLUX PROMPT:" or "3. FLUX PROMPT:" section
    markers = ["FLUX PROMPT:", "3. FLUX PROMPT", "**FLUX PROMPT**", "**3. FLUX PROMPT**"]
    for marker in markers:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            # Extract everything after the marker
            after = text[idx + len(marker):].strip()
            # Remove leading colon/asterisks
            after = after.lstrip(":* \n")
            # Take until next section header or end
            lines = []
            for line in after.split("\n"):
                stripped = line.strip()
                # Stop at next numbered section or empty separator
                if stripped and (stripped[0].isdigit() and "." in stripped[:3]):
                    break
                if stripped.startswith("---"):
                    break
                lines.append(line)
            result = " ".join(l.strip() for l in lines if l.strip())
            # Clean up any remaining markdown
            result = result.replace("**", "").replace("*", "").strip('"').strip()
            if len(result) > 50:
                return result
    return None


def _caption_with_blip(scene_images_dir):
    """Fallback: Auto-generate a scene prompt using BLIP."""
    from PIL import Image as PILImage

    try:
        from transformers import pipeline

        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0)

        aerial_files = []
        for img_file in sorted(Path(scene_images_dir).glob("*")):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                name = img_file.stem.lower()
                if not (name.startswith("descent_") or name.startswith("enhanced_") or name.startswith("groundgen_")):
                    aerial_files.append(img_file)

        sample = aerial_files[:: max(1, len(aerial_files) // 5)][:5]
        captions = []
        for img_file in sample:
            img = PILImage.open(img_file).convert("RGB")
            result = captioner(img, max_new_tokens=50)
            captions.append(result[0]["generated_text"])

        del captioner
        import torch
        torch.cuda.empty_cache()

        combined = ", ".join(captions)
        prompt = (
            f"photorealistic ground-level view of outdoor property, "
            f"based on aerial observation: {combined}, "
            f"natural lighting, high detail, 4k photography"
        )
        print(f"  BLIP-captioned prompt: {prompt[:100]}...")
        return prompt

    except Exception as e:
        print(f"  BLIP captioning failed ({e}), using generic prompt")
        return (
            "photorealistic ground-level view of outdoor property with grass, "
            "trees, buildings, natural lighting, high detail, 4k photography"
        )


# ---------------------------------------------------------------------------
# Point cloud and camera loading
# ---------------------------------------------------------------------------

def load_point_cloud(model_path):
    """Load PLY point cloud and return positions."""
    ply_candidates = [
        os.path.join(model_path, "point_cloud", "iteration_7000", "point_cloud.ply"),
        os.path.join(model_path, "point_cloud", "iteration_3000", "point_cloud.ply"),
        os.path.join(model_path, "point_cloud", "iteration_2000", "point_cloud.ply"),
    ]
    for candidate in ply_candidates:
        if os.path.exists(candidate):
            ply_data = PlyData.read(candidate)
            vertex = ply_data["vertex"]
            positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
            return positions, candidate

    for ply_file in sorted(Path(model_path).rglob("*.ply")):
        ply_data = PlyData.read(str(ply_file))
        if "vertex" in ply_data:
            vertex = ply_data["vertex"]
            positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
            return positions, str(ply_file)

    raise FileNotFoundError(f"No PLY file found in {model_path}")


def load_camera_poses(scene_path):
    """Load camera poses from COLMAP sparse reconstruction."""
    from scipy.spatial.transform import Rotation

    # Check sparse directories
    for sparse_dir in sorted(Path(scene_path).glob("sparse_*/0")):
        cam_file = sparse_dir / "cameras.bin"
        img_file = sparse_dir / "images.bin"
        if cam_file.exists() and img_file.exists():
            return _load_poses_bin(img_file)

    for sparse_dir in [Path(scene_path) / "sparse" / "0", Path(scene_path) / "sparse"]:
        cam_file = sparse_dir / "cameras.bin"
        img_file = sparse_dir / "images.bin"
        if cam_file.exists() and img_file.exists():
            return _load_poses_bin(img_file)

    # Try text format
    for sparse_dir in sorted(Path(scene_path).glob("sparse_*/0")) + [Path(scene_path) / "sparse" / "0"]:
        cam_file = sparse_dir / "cameras.txt"
        img_file = sparse_dir / "images.txt"
        if cam_file.exists() and img_file.exists():
            return _load_poses_txt(img_file)

    raise FileNotFoundError(f"No COLMAP sparse data found in {scene_path}")


def _load_poses_bin(images_path):
    """Load poses from COLMAP binary images file."""
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
    """Load poses from COLMAP text images file."""
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
            i += 2
        else:
            i += 1
    return poses


# ---------------------------------------------------------------------------
# Ground camera generation
# ---------------------------------------------------------------------------

def generate_ground_cameras(poses, positions, ground_z, num_perimeter=20, num_grid=12):
    """
    Generate ground-level camera poses with two strategies:
    1. Perimeter walk: cameras along convex hull looking inward
    2. Interior grid: regular grid with multiple look directions
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    drone_centers = np.array([p["center"] for p in poses])
    drone_xy = drone_centers[:, :2]
    avg_drone_z = np.mean(drone_centers[:, 2])

    # Eye height: scale from drone altitude ratio
    scene_height = avg_drone_z - ground_z
    eye_height = scene_height * 0.05  # ~5% of total scene height as eye level
    eye_height = max(eye_height, 0.005)  # minimum floor
    eye_z = ground_z + eye_height

    # Scene center for look-at targets
    scene_center_xy = np.mean(drone_xy, axis=0)

    ground_poses = []

    # --- Strategy 1: Perimeter walk ---
    try:
        hull = ConvexHull(drone_xy)
        hull_points = drone_xy[hull.vertices]
    except Exception:
        hull_points = drone_xy

    # Interpolate along hull perimeter
    perimeter_points = []
    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        n_interp = max(1, num_perimeter // len(hull_points))
        for t in np.linspace(0, 1, n_interp, endpoint=False):
            perimeter_points.append(p1 + t * (p2 - p1))

    for idx, pt in enumerate(perimeter_points[:num_perimeter]):
        look_dir = scene_center_xy - pt
        look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)
        yaw = np.arctan2(look_dir[1], look_dir[0])

        R = Rotation.from_euler("zyx", [yaw, -5, 0], degrees=True).as_matrix()
        center = np.array([pt[0], pt[1], eye_z])
        t_vec = -R @ center

        ground_poses.append({
            "center": center,
            "rotation": R,
            "translation": t_vec,
            "camera_type": "perimeter",
        })

    # --- Strategy 2: Interior grid ---
    xy_min = drone_xy.min(axis=0)
    xy_max = drone_xy.max(axis=0)
    xy_range = xy_max - xy_min

    n_grid_x = max(2, int(np.sqrt(num_grid * xy_range[0] / (xy_range[1] + 1e-8))))
    n_grid_y = max(2, num_grid // n_grid_x)
    grid_x = np.linspace(xy_min[0] + 0.1 * xy_range[0], xy_max[0] - 0.1 * xy_range[0], n_grid_x)
    grid_y = np.linspace(xy_min[1] + 0.1 * xy_range[1], xy_max[1] - 0.1 * xy_range[1], n_grid_y)

    cardinal_yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    grid_count = 0

    for gx in grid_x:
        for gy in grid_y:
            if grid_count >= num_grid:
                break
            for yaw in cardinal_yaws[:2]:
                R = Rotation.from_euler("zyx", [yaw, -5, 0], degrees=True).as_matrix()
                center = np.array([gx, gy, eye_z])
                t_vec = -R @ center

                ground_poses.append({
                    "center": center,
                    "rotation": R,
                    "translation": t_vec,
                    "camera_type": "grid",
                })
                grid_count += 1
                if grid_count >= num_grid:
                    break

    perimeter_actual = len(perimeter_points[:num_perimeter])
    print(f"  Generated {len(ground_poses)} ground cameras "
          f"({perimeter_actual} perimeter + {grid_count} grid)")
    return ground_poses


# ---------------------------------------------------------------------------
# Depth and RGB rendering
# ---------------------------------------------------------------------------

def render_depth_and_rgb(model_path, cameras, output_dir, scene_path):
    """
    Render blurry RGB and depth maps at each ground camera pose.
    RGB: rendered from current Gaussian splat
    Depth: estimated via Depth-Anything-V2 from the blurry render
    """
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print("  Rendering blurry RGB from splat model...")
    _render_rgb_from_splat(model_path, cameras, rgb_dir, scene_path)

    print("  Estimating depth maps with Depth-Anything-V2...")
    _estimate_depth_maps(rgb_dir, depth_dir)

    return rgb_dir, depth_dir


def _render_rgb_from_splat(model_path, cameras, output_dir, scene_path):
    """Render RGB images using gaussian-splatting rasterizer."""
    import torch
    from PIL import Image as PILImage

    sys.path.insert(0, "/mnt/splatwalk/InstantSplat")
    from gaussian_renderer import render
    from scene import GaussianModel
    from argparse import Namespace

    gaussians = GaussianModel(3)
    ckpt_dirs = sorted(Path(model_path).glob("point_cloud/iteration_*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {model_path}")
    latest_ckpt = ckpt_dirs[-1] / "point_cloud.ply"
    gaussians.load_ply(str(latest_ckpt))

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    from scene.cameras import Camera
    from scipy.spatial.transform import Rotation as SciRotation

    for idx, cam in enumerate(cameras):
        R = torch.tensor(cam["rotation"], dtype=torch.float32)
        T = torch.tensor(cam["translation"], dtype=torch.float32)

        camera = Camera(
            colmap_id=idx,
            R=R.numpy(),
            T=T.numpy(),
            FoVx=1.0,
            FoVy=1.0,
            image=torch.zeros(3, 512, 512),
            gt_alpha_mask=None,
            image_name=f"groundgen_{idx:03d}",
            uid=idx,
        )

        quat = SciRotation.from_matrix(cam["rotation"]).as_quat()  # [x,y,z,w] scipy
        camera_pose = torch.tensor(
            [quat[3], quat[0], quat[1], quat[2],
             cam["translation"][0], cam["translation"][1], cam["translation"][2]],
            dtype=torch.float32,
        ).cuda()
        camera.camera_pose = camera_pose

        rendering = render(camera, gaussians, pipe, bg_color, camera_pose=camera_pose)
        image = rendering["render"]

        img_np = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = PILImage.fromarray(img_np)
        img_pil.save(os.path.join(output_dir, f"groundgen_{idx:03d}.jpg"), "JPEG", quality=95)

    del gaussians
    torch.cuda.empty_cache()
    print(f"  Rendered {len(cameras)} ground-level RGB views")


def _estimate_depth_maps(rgb_dir, depth_dir):
    """Estimate depth maps from RGB images using Depth-Anything-V2."""
    import torch
    from PIL import Image as PILImage
    from transformers import pipeline

    depth_estimator = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=0,
    )

    for img_file in sorted(Path(rgb_dir).glob("*.jpg")):
        img = PILImage.open(img_file).convert("RGB")
        result = depth_estimator(img)
        depth_map = result["depth"]

        if not isinstance(depth_map, PILImage.Image):
            depth_arr = np.array(depth_map)
            depth_norm = ((depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8) * 255).astype(np.uint8)
            depth_map = PILImage.fromarray(depth_norm)

        depth_map = depth_map.resize((1024, 1024), PILImage.LANCZOS)
        depth_map.save(os.path.join(depth_dir, img_file.name), "JPEG", quality=95)

    del depth_estimator
    torch.cuda.empty_cache()
    print(f"  Generated {len(list(Path(depth_dir).glob('*.jpg')))} depth maps")


# ---------------------------------------------------------------------------
# IP-Adapter: aerial crop extraction
# ---------------------------------------------------------------------------

def extract_nearest_aerial_crop(camera, poses, scene_images_dir):
    """For a ground camera, find the nearest aerial image by XY distance
    and return a center-cropped PIL Image for IP-Adapter reference."""
    from PIL import Image as PILImage

    cam_xy = camera["center"][:2]

    # Find nearest aerial pose by XY distance
    min_dist = float("inf")
    nearest_name = None
    for pose in poses:
        dist = np.linalg.norm(pose["center"][:2] - cam_xy)
        if dist < min_dist:
            min_dist = dist
            nearest_name = pose["image_name"]

    if nearest_name is None:
        return None

    img_path = os.path.join(scene_images_dir, nearest_name)
    if not os.path.exists(img_path):
        # Try without extension
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(scene_images_dir, Path(nearest_name).stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        else:
            return None

    try:
        img = PILImage.open(img_path).convert("RGB")
        # Center crop (most relevant ground texture)
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
        img = img.resize((512, 512), PILImage.LANCZOS)
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FLUX generation with IP-Adapter
# ---------------------------------------------------------------------------

def generate_with_flux(depth_dir, rgb_dir, prompt, output_dir, poses, scene_images_dir,
                       ground_cameras, denoising_strength=0.85, controlnet_scale=0.4,
                       output_size=384):
    """
    Generate photorealistic ground views using FLUX.1-dev with ControlNet-depth
    and XLabs IP-Adapter for aerial texture matching.

    Uses FluxControlNetPipeline (txt2img + ControlNet) which supports IP-Adapter,
    unlike FluxControlNetImg2ImgPipeline which does not.
    """
    import torch
    from PIL import Image as PILImage
    from diffusers import FluxControlNetPipeline, FluxControlNetModel

    # Monkey-patch: PyTorch 2.4 doesn't support enable_gqa in scaled_dot_product_attention
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _patched_sdpa(*args, **kwargs):
        kwargs.pop('enable_gqa', None)
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

    # Load XLabs IP-Adapter (works with FluxControlNetPipeline, unlike InstantX)
    ip_adapter_loaded = False
    try:
        print("  Loading XLabs IP-Adapter v2 for aerial texture matching...")
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
        blurry_render = PILImage.open(rgb_file).convert("RGB").resize((1024, 1024), PILImage.LANCZOS)

        # Skip very dark renders (camera pointing into void)
        avg_brightness = np.array(blurry_render).mean()
        if avg_brightness < 20:
            print(f"  Skipping view {idx} (too dark, brightness={avg_brightness:.1f})")
            continue

        depth_map = PILImage.open(depth_file).convert("RGB").resize((1024, 1024), PILImage.LANCZOS)

        gen_kwargs = dict(
            prompt=prompt,
            control_image=depth_map,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=28,
            guidance_scale=4.0,
            height=1024,
            width=1024,
        )

        # Add IP-Adapter reference from nearest aerial image
        if ip_adapter_loaded and idx < len(ground_cameras):
            aerial_crop = extract_nearest_aerial_crop(
                ground_cameras[idx], poses, scene_images_dir
            )
            if aerial_crop is not None:
                gen_kwargs["ip_adapter_image"] = aerial_crop

        result = pipe(**gen_kwargs).images[0]

        # Save at reduced size (implicitly down-weighted vs real 512x512 aerials)
        result = result.resize((output_size, output_size), PILImage.LANCZOS)
        result.save(os.path.join(output_dir, f"groundgen_{generated_count:03d}.jpg"), "JPEG", quality=95)
        generated_count += 1

        if generated_count % 10 == 0:
            print(f"  Generated {generated_count} views (processed {idx + 1}/{len(depth_files)})")

    print(f"  Generated {generated_count} photorealistic ground views ({output_size}x{output_size})")

    del pipe, controlnet
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Scene integration and retraining
# ---------------------------------------------------------------------------

def add_images_to_scene(generated_dir, scene_path):
    """Copy generated images into the scene's training images directory."""
    scene_images = os.path.join(scene_path, "images")
    count = 0
    for img_file in sorted(Path(generated_dir).glob("groundgen_*.jpg")):
        shutil.copy2(str(img_file), os.path.join(scene_images, img_file.name))
        count += 1
    print(f"  Added {count} ground-level images to {scene_images}")
    return count


def write_colmap_cameras(cameras, scene_path):
    """
    Write ground-level camera poses to COLMAP text format.
    Converts existing binary files to text first, then appends ground cameras,
    then removes binary files so InstantSplat reads only text format.
    """
    import struct
    from scipy.spatial.transform import Rotation

    sparse_dir = None
    for sd in sorted(Path(scene_path).glob("sparse_*/0")):
        if (sd / "images.bin").exists() or (sd / "images.txt").exists():
            sparse_dir = sd
            break
    if sparse_dir is None:
        for sd in [Path(scene_path) / "sparse" / "0", Path(scene_path) / "sparse"]:
            if sd.exists():
                sparse_dir = sd
                break

    if sparse_dir is None:
        print("  WARNING: No sparse directory found, skipping COLMAP pose export")
        return

    images_bin = sparse_dir / "images.bin"
    cameras_bin = sparse_dir / "cameras.bin"
    images_txt = sparse_dir / "images.txt"
    cameras_txt = sparse_dir / "cameras.txt"

    existing_entries = []
    max_image_id = 0
    camera_id = 1

    if images_bin.exists():
        with open(str(images_bin), "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                tx, ty, tz = struct.unpack("<3d", f.read(24))
                cam_id = struct.unpack("<I", f.read(4))[0]
                name_chars = []
                while True:
                    c = f.read(1)
                    if c == b"\x00":
                        break
                    name_chars.append(c.decode())
                image_name = "".join(name_chars)
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.read(num_points2d * 24)

                existing_entries.append(
                    f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {image_name}\n\n"
                )
                max_image_id = max(max_image_id, image_id)
                camera_id = cam_id
        print(f"  Converted {len(existing_entries)} cameras from images.bin to text")
    elif images_txt.exists():
        # Read existing entries from text format (v4+ uses text-only COLMAP)
        with open(str(images_txt), "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 10:
                image_id = int(parts[0])
                max_image_id = max(max_image_id, image_id)
                camera_id = int(parts[8])
                existing_entries.append(line + "\n\n")
            i += 1
        print(f"  Read {len(existing_entries)} existing cameras from images.txt")

    if cameras_bin.exists():
        camera_entries = []
        with open(str(cameras_bin), "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                cam_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<i", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]
                num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
                params = struct.unpack(f"<{num_params}d", f.read(num_params * 8))
                model_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                              3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE"}.get(model_id, "PINHOLE")
                params_str = " ".join(f"{p}" for p in params)
                camera_entries.append(f"{cam_id} {model_name} {width} {height} {params_str}\n")

        with open(str(cameras_txt), "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(camera_entries)}\n")
            for entry in camera_entries:
                f.write(entry)
        print(f"  Converted {len(camera_entries)} camera models to cameras.txt")
        cameras_bin.unlink()

    # Only write COLMAP entries for ground images that actually exist on disk
    # (generate_with_flux skips dark renders, so not all cameras have images)
    scene_images_dir = Path(scene_path) / "images"
    ground_entries = []
    for idx, cam in enumerate(cameras):
        name = f"groundgen_{idx:03d}.jpg"
        if not (scene_images_dir / name).exists():
            continue
        image_id = max_image_id + idx + 1
        R_rot = Rotation.from_matrix(cam["rotation"])
        qw, qx, qy, qz = R_rot.as_quat()[[3, 0, 1, 2]]
        tx, ty, tz = cam["translation"]
        ground_entries.append(
            f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n\n"
        )

    total = len(existing_entries) + len(ground_entries)
    with open(str(images_txt), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {total}\n")
        for entry in existing_entries:
            f.write(entry)
        for entry in ground_entries:
            f.write(entry)

    if images_bin.exists():
        images_bin.unlink()
        print("  Removed images.bin (using text format only)")

    print(f"  Wrote {total} total camera poses ({len(existing_entries)} original + "
          f"{len(ground_entries)} ground) to {images_txt}")


def retrain_model(scene_path, model_path, output_model_path, iterations, n_views):
    """Retrain the Gaussian splat model with ground-level images added.

    Uses aggressive densification/pruning to suppress floaters from aerial Gaussians.
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
        # Aggressive floater suppression
        "--densify_from_iter", "200",
        "--opacity_reset_interval", "1000",
        "--densify_until_iter", str(iterations),
        "--densification_interval", "100",
    ]

    if output_model_path != model_path and not os.path.exists(output_model_path):
        shutil.copytree(model_path, output_model_path)

    # CRITICAL: Delete old point_cloud checkpoints so train.py starts fresh.
    # Train.py always starts from COLMAP (not checkpoints), so old ones
    # just cause confusion and mask retrain failures.
    pc_dir = os.path.join(output_model_path, "point_cloud")
    if os.path.exists(pc_dir):
        shutil.rmtree(pc_dir)
        print(f"  Cleared old checkpoints from {pc_dir}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        ckpt_dirs = sorted(Path(output_model_path).glob("point_cloud/iteration_*"))
        if ckpt_dirs:
            print(f"  Retrain completed with non-zero exit (save_pose error?), checkpoint at {ckpt_dirs[-1].name}")
        else:
            print(f"  Retrain stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"Retrain failed with code {result.returncode}")
    else:
        print(f"  Retrained for {iterations} iterations")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3.5: ControlNet Ground View Generation")
    parser.add_argument("--model_path", required=True, help="Path to Stage 3 descent model")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_perimeter_views", type=int, default=20)
    parser.add_argument("--num_grid_views", type=int, default=12)
    parser.add_argument("--denoising_strength", type=float, default=0.85)
    parser.add_argument("--controlnet_scale", type=float, default=0.4)
    parser.add_argument("--output_size", type=int, default=384)
    parser.add_argument("--retrain_iterations", type=int, default=3000)
    parser.add_argument("--coordinates", default="", help="lat,lon override (usually auto-detected from EXIF)")
    parser.add_argument("--slack_webhook_url", default="")
    parser.add_argument("--job_id", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load scene geometry
    print(f"Loading point cloud from {args.model_path}...")
    positions, ply_path = load_point_cloud(args.model_path)
    print(f"  Loaded {len(positions)} points from {ply_path}")

    ground_z = np.percentile(positions[:, 2], 5)
    drone_z = np.percentile(positions[:, 2], 95)
    print(f"  Ground Z: {ground_z:.4f}, Drone Z: {drone_z:.4f}")

    # 2. Load aerial camera poses
    print(f"Loading camera poses from {args.scene_path}...")
    poses = load_camera_poses(args.scene_path)
    print(f"  Found {len(poses)} camera poses")

    # 3. Extract GPS and geographic context
    scene_images_dir = os.path.join(args.scene_path, "images")
    lat, lon = None, None

    if args.coordinates:
        try:
            lat, lon = [float(x.strip()) for x in args.coordinates.split(",")]
            print(f"  Using provided coordinates: {lat}, {lon}")
        except Exception:
            print(f"  Invalid --coordinates format: {args.coordinates}")

    if lat is None:
        print("Extracting GPS from EXIF...")
        lat, lon, _alt = extract_gps_from_images(scene_images_dir)

    google_maps_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")

    # 4. Generate ground-level camera poses
    print("Generating ground-level cameras...")
    ground_cameras = generate_ground_cameras(
        poses, positions, ground_z,
        num_perimeter=args.num_perimeter_views,
        num_grid=args.num_grid_views,
    )

    # 5. Render blurry RGB + depth maps
    print("Rendering depth maps and blurry RGB...")
    notify_slack("Stage 3.5: Rendering ground-level depth/RGB...",
                 args.slack_webhook_url, args.job_id)
    rgb_dir, depth_dir = render_depth_and_rgb(
        args.model_path, ground_cameras, args.output_dir, args.scene_path
    )

    # 6. Caption the scene with Gemini (falls back to BLIP)
    print("Captioning aerial scene...")
    prompt = caption_aerial_scene(scene_images_dir, lat=lat, lon=lon, google_maps_key=google_maps_key)

    # 7. Generate photorealistic views with FLUX + IP-Adapter
    print("Generating photorealistic ground views with FLUX.1-dev + IP-Adapter...")
    notify_slack(f"Stage 3.5: Generating {len(ground_cameras)} ground views with FLUX...",
                 args.slack_webhook_url, args.job_id)
    generated_dir = os.path.join(args.output_dir, "generated")
    generate_with_flux(
        depth_dir, rgb_dir, prompt, generated_dir,
        poses=poses, scene_images_dir=scene_images_dir,
        ground_cameras=ground_cameras,
        denoising_strength=args.denoising_strength,
        controlnet_scale=args.controlnet_scale,
        output_size=args.output_size,
    )

    # 8. Add generated images to training set
    print("Adding generated images to training set...")
    num_added = add_images_to_scene(generated_dir, args.scene_path)

    # 9. Write COLMAP camera poses
    print("Writing COLMAP camera poses...")
    write_colmap_cameras(ground_cameras, args.scene_path)

    # 10. Retrain model
    n_views = min(len(list(Path(scene_images_dir).glob("*.jpg"))), 24)
    output_model = os.path.join(args.output_dir, "model")
    print(f"Retraining for {args.retrain_iterations} iterations with {num_added} new views...")
    notify_slack(f"Stage 3.5: Retraining with {num_added} ground views...",
                 args.slack_webhook_url, args.job_id)
    retrain_model(args.scene_path, args.model_path, output_model,
                  args.retrain_iterations, n_views)

    notify_slack(f"Stage 3.5 complete: {num_added} ground views generated and trained",
                 args.slack_webhook_url, args.job_id, "success")
    print(f"\nStage 3.5 complete. Model at: {output_model}")


if __name__ == "__main__":
    main()
