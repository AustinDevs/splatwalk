#!/usr/bin/env python3
"""
Stage 3: Iterative Virtual Descent

Loads the aerial Gaussian splat from Stage 2, estimates ground level,
then iteratively renders views at progressively lower altitudes and
retrains the model at each level to fill in detail.

Each altitude step:
  1. Generate camera poses interpolated between drone height and ground
  2. Render images from those poses using the current splat
  3. Add rendered images to the training set
  4. Retrain the model for N iterations (continuing from last checkpoint)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from pathlib import Path

try:
    import torch
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
    """Load the PLY point cloud and return positions as numpy array."""
    ply_candidates = [
        os.path.join(model_path, "point_cloud", "iteration_7000", "point_cloud.ply"),
        os.path.join(model_path, "point_cloud", "iteration_3000", "point_cloud.ply"),
    ]
    # Also search recursively
    for candidate in ply_candidates:
        if os.path.exists(candidate):
            ply_data = PlyData.read(candidate)
            vertex = ply_data["vertex"]
            positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
            return positions, candidate

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
    z_values = positions[:, 2]
    ground_z = np.percentile(z_values, 5)
    return ground_z


def load_camera_poses(scene_path):
    """Load camera poses from COLMAP-style sparse reconstruction."""
    cameras_path = None
    images_path = None

    # Check sparse directories created by InstantSplat
    for sparse_dir in sorted(Path(scene_path).glob("sparse_*/0")):
        cam_file = sparse_dir / "cameras.bin"
        img_file = sparse_dir / "images.bin"
        if cam_file.exists() and img_file.exists():
            cameras_path = cam_file
            images_path = img_file
            break

    # Also check standard sparse/0 path
    if cameras_path is None:
        for sparse_dir in [Path(scene_path) / "sparse" / "0", Path(scene_path) / "sparse"]:
            cam_file = sparse_dir / "cameras.bin"
            img_file = sparse_dir / "images.bin"
            if cam_file.exists() and img_file.exists():
                cameras_path = cam_file
                images_path = img_file
                break

    # Try text format
    if cameras_path is None:
        for sparse_dir in sorted(Path(scene_path).glob("sparse_*/0")) + [
            Path(scene_path) / "sparse" / "0"
        ]:
            cam_file = sparse_dir / "cameras.txt"
            img_file = sparse_dir / "images.txt"
            if cam_file.exists() and img_file.exists():
                return load_camera_poses_txt(cam_file, img_file)

    if cameras_path is None:
        raise FileNotFoundError(f"No COLMAP sparse data found in {scene_path}")

    return load_camera_poses_bin(cameras_path, images_path)


def load_camera_poses_bin(cameras_path, images_path):
    """Load camera poses from COLMAP binary format."""
    import struct

    poses = []

    with open(str(images_path), "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read image name (null-terminated)
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode())
            image_name = "".join(name_chars)

            # Skip 2D points
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)  # each point2D: x, y, point3D_id

            # Convert quaternion + translation to position
            # Camera center = -R^T * t
            from scipy.spatial.transform import Rotation

            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            center = -R.T @ t

            poses.append(
                {
                    "image_name": image_name,
                    "center": center,
                    "rotation": R,
                    "quaternion": np.array([qw, qx, qy, qz]),
                    "translation": t,
                }
            )

    return poses


def load_camera_poses_txt(cameras_path, images_path):
    """Load camera poses from COLMAP text format."""
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

            poses.append(
                {
                    "image_name": image_name,
                    "center": center,
                    "rotation": R,
                    "quaternion": np.array([qw, qx, qy, qz]),
                    "translation": t,
                }
            )
            i += 2  # Skip the 2D points line
        else:
            i += 1

    return poses


def generate_descent_cameras(poses, ground_z, altitude_fraction):
    """
    Generate descent camera poses at a given altitude fraction.

    altitude_fraction: 1.0 = drone altitude, 0.0 = ground level
    For each original pose, create a new pose that:
      - Keeps the same XY position
      - Interpolates Z between drone height and ground level
      - Gradually tilts camera from looking down to looking toward horizon
    """
    descent_poses = []
    drone_heights = [p["center"][2] for p in poses]
    avg_drone_z = np.mean(drone_heights)

    for pose in poses:
        center = pose["center"].copy()
        drone_z = center[2]

        # Interpolate altitude
        new_z = ground_z + altitude_fraction * (drone_z - ground_z)
        center[2] = new_z

        # Interpolate rotation: from looking down toward looking at horizon
        # At altitude_fraction=1.0: keep original (looking down)
        # At altitude_fraction=0.0: look toward horizon
        from scipy.spatial.transform import Rotation, Slerp

        original_rot = Rotation.from_matrix(pose["rotation"])

        # Create a "horizon-looking" rotation: keep yaw, tilt to horizontal
        # Extract the yaw from original rotation
        euler = original_rot.as_euler("zyx", degrees=True)
        # Gradually reduce pitch (tilt toward horizon)
        horizon_euler = euler.copy()
        horizon_euler[1] = 0  # Zero pitch = looking at horizon

        horizon_rot = Rotation.from_euler("zyx", horizon_euler, degrees=True)

        # Slerp between original and horizon
        t = 1.0 - altitude_fraction  # 0 at drone height, 1 at ground
        key_rots = Rotation.concatenate([original_rot, horizon_rot])
        slerp = Slerp([0, 1], key_rots)
        interpolated_rot = slerp(t)

        R = interpolated_rot.as_matrix()
        t_vec = -R @ center  # translation = -R * center

        descent_poses.append(
            {
                "center": center,
                "rotation": R,
                "translation": t_vec,
                "altitude_fraction": altitude_fraction,
            }
        )

    return descent_poses


def render_from_poses(model_path, descent_poses, output_dir, scene_path):
    """
    Render images from the current Gaussian splat model at the given poses.

    Uses the diff-gaussian-rasterization renderer directly.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        render_with_gsplat(model_path, descent_poses, output_dir, scene_path)
    except Exception as e:
        print(f"  Native rendering failed ({e}), falling back to train.py render")
        render_with_train_script(model_path, output_dir, scene_path)


def render_with_gsplat(model_path, descent_poses, output_dir, scene_path):
    """Render using gaussian-splatting rasterizer directly."""
    import torch
    from PIL import Image

    # Import gaussian splatting modules
    sys.path.insert(0, "/opt/InstantSplat")
    from gaussian_renderer import render
    from scene import GaussianModel
    from argparse import Namespace

    # Load the model
    gaussians = GaussianModel(3)  # 3 = SH degree

    # Find latest checkpoint
    ckpt_dirs = sorted(Path(model_path).glob("point_cloud/iteration_*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {model_path}")
    latest_ckpt = ckpt_dirs[-1] / "point_cloud.ply"
    gaussians.load_ply(str(latest_ckpt))

    # Set up a basic rendering pipeline
    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    for idx, pose in enumerate(descent_poses):
        # Create a minimal camera object
        from scene.cameras import Camera

        R = torch.tensor(pose["rotation"], dtype=torch.float32)
        T = torch.tensor(pose["translation"], dtype=torch.float32)

        camera = Camera(
            colmap_id=idx,
            R=R.numpy(),
            T=T.numpy(),
            FoVx=1.0,  # ~57 degrees
            FoVy=1.0,
            image=torch.zeros(3, 512, 512),
            gt_alpha_mask=None,
            image_name=f"descent_{pose['altitude_fraction']:.3f}_{idx:03d}",
            uid=idx,
        )

        rendering = render(camera, gaussians, pipe, bg_color)
        image = rendering["render"]

        # Save as JPEG
        img_np = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )
        img_pil = Image.fromarray(img_np)
        out_path = os.path.join(output_dir, f"descent_{pose['altitude_fraction']:.3f}_{idx:03d}.jpg")
        img_pil.save(out_path, "JPEG", quality=95)

    print(f"  Rendered {len(descent_poses)} descent views")


def render_with_train_script(model_path, output_dir, scene_path):
    """Fallback: use train.py in render-only mode to generate images."""
    # This produces renders from the trained model's camera poses
    cmd = [
        sys.executable,
        "/opt/InstantSplat/render.py",
        "--source_path", scene_path,
        "--model_path", model_path,
        "--skip_train",
        "--skip_test",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        print("  render.py fallback also failed, continuing with existing images")


def retrain_model(scene_path, model_path, output_model_path, iterations, n_views):
    """Retrain the Gaussian splat model with additional descent images."""
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

    # Copy the existing model as starting point if different path
    if output_model_path != model_path and not os.path.exists(output_model_path):
        shutil.copytree(model_path, output_model_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Retrain stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"Retrain failed with code {result.returncode}")

    print(f"  Retrained for {iterations} iterations")


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Iterative Virtual Descent")
    parser.add_argument("--model_path", required=True, help="Path to Stage 2 trained model")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for descent models")
    parser.add_argument(
        "--altitudes",
        default="0.75,0.5,0.25,0.1,0.025",
        help="Comma-separated altitude fractions (1=drone, 0=ground)",
    )
    parser.add_argument(
        "--retrain_iterations", type=int, default=2000, help="Iterations per altitude level"
    )
    parser.add_argument("--slack_webhook_url", default="", help="Slack webhook URL")
    parser.add_argument("--job_id", default="", help="Job ID for Slack notifications")
    args = parser.parse_args()

    altitudes = [float(a) for a in args.altitudes.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading point cloud from {args.model_path}...")
    positions, ply_path = load_point_cloud(args.model_path)
    print(f"  Loaded {len(positions)} points from {ply_path}")

    ground_z = estimate_ground_level(positions)
    drone_z = np.percentile(positions[:, 2], 95)
    print(f"  Ground level (5th percentile Z): {ground_z:.3f}")
    print(f"  Drone level (95th percentile Z): {drone_z:.3f}")

    print(f"\nLoading camera poses from {args.scene_path}...")
    poses = load_camera_poses(args.scene_path)
    print(f"  Found {len(poses)} camera poses")

    current_model_path = args.model_path
    n_views = min(len(poses), 24)

    for i, altitude in enumerate(altitudes):
        print(f"\n{'='*40}")
        print(f"Altitude level {i+1}/{len(altitudes)}: {altitude:.3f} ({altitude*100:.1f}% of drone height)")
        print(f"{'='*40}")

        # Generate descent camera poses
        descent_poses = generate_descent_cameras(poses, ground_z, altitude)
        print(f"  Generated {len(descent_poses)} descent camera poses")

        # Render from current model
        render_dir = os.path.join(args.output_dir, f"altitude_{altitude:.3f}", "renders")
        render_from_poses(current_model_path, descent_poses, render_dir, args.scene_path)

        # Copy rendered images into the scene's training images
        scene_images_dir = os.path.join(args.scene_path, "images")
        for img_file in Path(render_dir).glob("*.jpg"):
            shutil.copy2(str(img_file), os.path.join(scene_images_dir, img_file.name))

        updated_n_views = len(list(Path(scene_images_dir).glob("*.jpg")))
        if updated_n_views > 24:
            updated_n_views = 24

        # Retrain
        altitude_model = os.path.join(args.output_dir, f"altitude_{altitude:.3f}", "model")
        print(f"  Retraining for {args.retrain_iterations} iterations...")
        retrain_model(
            args.scene_path, current_model_path, altitude_model, args.retrain_iterations, updated_n_views
        )

        current_model_path = altitude_model

        notify_slack(
            f"Stage 3: Rendered {len(descent_poses)} views at {altitude*100:.0f}% altitude, retrained",
            args.slack_webhook_url,
            args.job_id,
        )

    # Copy final model to output
    final_dir = os.path.join(args.output_dir, "final")
    if current_model_path != final_dir:
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(current_model_path, final_dir)

    print(f"\nDescent complete. Final model at: {final_dir}")


if __name__ == "__main__":
    main()
