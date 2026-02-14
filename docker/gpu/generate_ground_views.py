#!/usr/bin/env python3
"""
Stage 3.5: ControlNet Ground View Generation

Generates photorealistic ground-level training images using FLUX.1-dev with
ControlNet-depth conditioning and IP-Adapter aerial texture matching.

Pipeline:
  1. Generate ground-level camera poses (perimeter walk + interior grid)
  2. Render blurry RGB + depth maps from existing splat at those poses
  3. Extract aerial crops nearest each ground camera for IP-Adapter conditioning
  4. Generate photorealistic views with FLUX.1-dev + ControlNet-depth + IP-Adapter
  5. Add generated images to training set and retrain splat
"""

import argparse
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


def generate_ground_cameras(poses, positions, ground_z):
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
        # Fallback: use all drone XY positions
        hull_points = drone_xy

    num_perimeter = min(40, len(hull_points) * 4)
    # Interpolate along hull perimeter
    perimeter_points = []
    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        n_interp = max(1, num_perimeter // len(hull_points))
        for t in np.linspace(0, 1, n_interp, endpoint=False):
            perimeter_points.append(p1 + t * (p2 - p1))

    for idx, pt in enumerate(perimeter_points[:40]):
        # Look toward scene center
        look_dir = scene_center_xy - pt
        look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)
        yaw = np.arctan2(look_dir[1], look_dir[0])

        # Horizontal look direction (slight downward tilt)
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

    # Create grid points
    n_grid_x = max(2, int(np.sqrt(32 * xy_range[0] / (xy_range[1] + 1e-8))))
    n_grid_y = max(2, 32 // n_grid_x)
    grid_x = np.linspace(xy_min[0] + 0.1 * xy_range[0], xy_max[0] - 0.1 * xy_range[0], n_grid_x)
    grid_y = np.linspace(xy_min[1] + 0.1 * xy_range[1], xy_max[1] - 0.1 * xy_range[1], n_grid_y)

    cardinal_yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # N, E, S, W
    grid_count = 0

    for gx in grid_x:
        for gy in grid_y:
            if grid_count >= 32:
                break
            # Pick 2 cardinal directions per grid point
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
                if grid_count >= 32:
                    break

    print(f"  Generated {len(ground_poses)} ground cameras "
          f"({len(perimeter_points[:40])} perimeter + {grid_count} grid)")
    return ground_poses


def render_depth_and_rgb(model_path, cameras, output_dir, scene_path):
    """
    Render blurry RGB and depth maps at each ground camera pose.
    RGB: rendered from current Gaussian splat
    Depth: estimated via MiDaS/DPT from the blurry render
    """
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Render RGB from Gaussian splat
    print("  Rendering blurry RGB from splat model...")
    _render_rgb_from_splat(model_path, cameras, rgb_dir, scene_path)

    # Estimate depth from rendered RGB using MiDaS
    print("  Estimating depth maps with MiDaS...")
    _estimate_depth_maps(rgb_dir, depth_dir)

    return rgb_dir, depth_dir


def _render_rgb_from_splat(model_path, cameras, output_dir, scene_path):
    """Render RGB images using gaussian-splatting rasterizer."""
    import torch
    from PIL import Image as PILImage

    sys.path.insert(0, "/opt/InstantSplat")
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

        # InstantSplat's render() requires camera_pose as a 7-element tensor
        # [qw, qx, qy, qz, tx, ty, tz] representing the w2c transform.
        # cam["rotation"] is already the w2c rotation matrix (from COLMAP quaternion).
        quat = SciRotation.from_matrix(cam["rotation"]).as_quat()  # [x,y,z,w] scipy
        camera_pose = torch.tensor(
            [quat[3], quat[0], quat[1], quat[2],  # wxyz
             cam["translation"][0], cam["translation"][1], cam["translation"][2]],
            dtype=torch.float32,
        ).cuda()
        camera.camera_pose = camera_pose

        rendering = render(camera, gaussians, pipe, bg_color, camera_pose=camera_pose)
        image = rendering["render"]

        img_np = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = PILImage.fromarray(img_np)
        img_pil.save(os.path.join(output_dir, f"groundgen_{idx:03d}.jpg"), "JPEG", quality=95)

    # Free GPU memory
    del gaussians
    torch.cuda.empty_cache()
    print(f"  Rendered {len(cameras)} ground-level RGB views")


def _estimate_depth_maps(rgb_dir, depth_dir):
    """Estimate depth maps from RGB images using MiDaS via transformers."""
    import torch
    from PIL import Image as PILImage
    from transformers import pipeline

    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large", device=0)

    for img_file in sorted(Path(rgb_dir).glob("*.jpg")):
        img = PILImage.open(img_file).convert("RGB")
        result = depth_estimator(img)
        depth_map = result["depth"]

        # Convert to PIL image (already is for pipeline output)
        if not isinstance(depth_map, PILImage.Image):
            depth_arr = np.array(depth_map)
            depth_norm = ((depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8) * 255).astype(np.uint8)
            depth_map = PILImage.fromarray(depth_norm)

        depth_map = depth_map.resize((1024, 1024), PILImage.LANCZOS)
        depth_map.save(os.path.join(depth_dir, img_file.name), "JPEG", quality=95)

    # Free depth model
    del depth_estimator
    torch.cuda.empty_cache()
    print(f"  Generated {len(list(Path(depth_dir).glob('*.jpg')))} depth maps")


def caption_aerial_scene(scene_images_dir):
    """Auto-generate a scene prompt from representative aerial images using BLIP."""
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

        # Sample up to 5 representative images
        sample = aerial_files[:: max(1, len(aerial_files) // 5)][:5]
        captions = []
        for img_file in sample:
            img = PILImage.open(img_file).convert("RGB")
            result = captioner(img, max_new_tokens=50)
            captions.append(result[0]["generated_text"])

        del captioner
        import torch
        torch.cuda.empty_cache()

        # Combine captions into a prompt
        combined = ", ".join(captions)
        prompt = (
            f"photorealistic ground-level view of outdoor property, "
            f"based on aerial observation: {combined}, "
            f"natural lighting, high detail, 4k photography"
        )
        print(f"  Auto-captioned prompt: {prompt[:100]}...")
        return prompt

    except Exception as e:
        print(f"  BLIP captioning failed ({e}), using generic prompt")
        return (
            "photorealistic ground-level view of outdoor property with grass, "
            "trees, buildings, natural lighting, high detail, 4k photography"
        )


def generate_with_flux(depth_dir, rgb_dir, prompt, output_dir,
                       denoising_strength=0.55, controlnet_scale=0.8):
    """
    Generate photorealistic ground views using FLUX.1-dev with ControlNet-depth.
    Uses img2img (blurry render as init) + depth ControlNet (geometry preservation).
    """
    import torch
    from PIL import Image as PILImage
    from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel

    # Monkey-patch: PyTorch 2.4 doesn't support enable_gqa in scaled_dot_product_attention
    # (added in PyTorch 2.5). Strip it so FLUX attention works on our snapshot.
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
    pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    depth_files = sorted(Path(depth_dir).glob("*.jpg"))
    rgb_files = sorted(Path(rgb_dir).glob("*.jpg"))

    for idx, (depth_file, rgb_file) in enumerate(zip(depth_files, rgb_files)):
        depth_map = PILImage.open(depth_file).convert("RGB").resize((1024, 1024), PILImage.LANCZOS)
        blurry_render = PILImage.open(rgb_file).convert("RGB").resize((1024, 1024), PILImage.LANCZOS)

        result = pipe(
            prompt=prompt,
            image=blurry_render,
            control_image=depth_map,
            strength=denoising_strength,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=28,
            guidance_scale=3.5,
            height=1024,
            width=1024,
        ).images[0]

        # Save at 512x512 to match training image size
        result = result.resize((512, 512), PILImage.LANCZOS)
        result.save(os.path.join(output_dir, f"groundgen_{idx:03d}.jpg"), "JPEG", quality=95)

        if (idx + 1) % 10 == 0:
            print(f"  Generated {idx + 1}/{len(depth_files)} views")

    print(f"  Generated {len(depth_files)} photorealistic ground views")

    # Free VRAM
    del pipe, controlnet
    torch.cuda.empty_cache()


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

    # Find the sparse directory
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

    # Step 1: Convert images.bin to images.txt if binary exists
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

    # Step 2: Convert cameras.bin to cameras.txt if binary exists
    if cameras_bin.exists():
        camera_entries = []
        with open(str(cameras_bin), "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                cam_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<i", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]
                # Number of params depends on camera model
                # PINHOLE=1: fx,fy,cx,cy (4 params), SIMPLE_PINHOLE=0: f,cx,cy (3 params)
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

    # Step 3: Write all images (existing + ground) to images.txt
    with open(str(images_txt), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(existing_entries) + len(cameras)}\n")
        for entry in existing_entries:
            f.write(entry)
        for idx, cam in enumerate(cameras):
            image_id = max_image_id + idx + 1
            R_rot = Rotation.from_matrix(cam["rotation"])
            qw, qx, qy, qz = R_rot.as_quat()[[3, 0, 1, 2]]
            tx, ty, tz = cam["translation"]
            name = f"groundgen_{idx:03d}.jpg"
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n")
            f.write("\n")  # Empty line for 2D points

    # Step 4: Remove binary files so InstantSplat reads text only
    if images_bin.exists():
        images_bin.unlink()
        print("  Removed images.bin (using text format only)")

    print(f"  Wrote {len(existing_entries) + len(cameras)} total camera poses to {images_txt}")


def retrain_model(scene_path, model_path, output_model_path, iterations, n_views):
    """Retrain the Gaussian splat model with ground-level images added."""
    cmd = [
        sys.executable,
        "/opt/InstantSplat/train.py",
        "--source_path", scene_path,
        "--model_path", output_model_path,
        "--iterations", str(iterations),
        "--n_views", str(n_views),
        "--pp_optimizer",
        "--optim_pose",
    ]

    if output_model_path != model_path and not os.path.exists(output_model_path):
        shutil.copytree(model_path, output_model_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Retrain stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"Retrain failed with code {result.returncode}")

    print(f"  Retrained for {iterations} iterations")


def main():
    parser = argparse.ArgumentParser(description="Stage 3.5: ControlNet Ground View Generation")
    parser.add_argument("--model_path", required=True, help="Path to Stage 3 descent model")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_perimeter_views", type=int, default=40)
    parser.add_argument("--num_grid_views", type=int, default=32)
    parser.add_argument("--denoising_strength", type=float, default=0.55)
    parser.add_argument("--controlnet_scale", type=float, default=0.8)
    parser.add_argument("--retrain_iterations", type=int, default=3000)
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

    # 3. Generate ground-level camera poses
    print("Generating ground-level cameras...")
    ground_cameras = generate_ground_cameras(poses, positions, ground_z)

    # 4. Render blurry RGB + depth maps
    print("Rendering depth maps and blurry RGB...")
    notify_slack("Stage 3.5: Rendering ground-level depth/RGB...",
                 args.slack_webhook_url, args.job_id)
    rgb_dir, depth_dir = render_depth_and_rgb(
        args.model_path, ground_cameras, args.output_dir, args.scene_path
    )

    # 5. Auto-caption the scene
    print("Captioning aerial scene...")
    scene_images_dir = os.path.join(args.scene_path, "images")
    prompt = caption_aerial_scene(scene_images_dir)

    # 6. Generate photorealistic views with FLUX
    print("Generating photorealistic ground views with FLUX.1-dev...")
    notify_slack(f"Stage 3.5: Generating {len(ground_cameras)} ground views with FLUX...",
                 args.slack_webhook_url, args.job_id)
    generated_dir = os.path.join(args.output_dir, "generated")
    generate_with_flux(
        depth_dir, rgb_dir, prompt, generated_dir,
        denoising_strength=args.denoising_strength,
        controlnet_scale=args.controlnet_scale,
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
