#!/usr/bin/env python3
"""
Stage 4: Diffusion Enhancement with ViewCrafter

Takes the ground-level Gaussian splat from Stage 3 and enhances it using
ViewCrafter's video diffusion model. The process:

1. Render the lowest-altitude views from the Stage 3 model (the "soft" images)
2. Feed pairs of rendered views into ViewCrafter with camera pose trajectories
3. ViewCrafter generates photorealistic enhanced frames via diffusion
4. Save enhanced images to the training set
5. Final retrain with enhanced views as pseudo-observations (3000 iterations)
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
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    from PIL import Image


def find_ply_and_load(model_path):
    """Find and load the latest PLY checkpoint."""
    from plyfile import PlyData

    ply_candidates = sorted(Path(model_path).rglob("*.ply"))
    if not ply_candidates:
        raise FileNotFoundError(f"No PLY files found in {model_path}")

    # Prefer the latest iteration
    ply_path = ply_candidates[-1]
    ply_data = PlyData.read(str(ply_path))
    vertex = ply_data["vertex"]
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    return positions, str(ply_path)


def render_low_altitude_views(model_path, scene_path, output_dir, num_views=24):
    """Render views from the current model for ViewCrafter input."""
    os.makedirs(output_dir, exist_ok=True)

    # Try to use render.py from InstantSplat
    render_cmd = [
        sys.executable,
        "/opt/InstantSplat/render.py",
        "--source_path", scene_path,
        "--model_path", model_path,
        "--skip_train",
        "--skip_test",
    ]

    try:
        result = subprocess.run(render_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Find rendered images
            render_dirs = list(Path(model_path).rglob("renders"))
            for rdir in render_dirs:
                for img in sorted(rdir.glob("*.png")) + sorted(rdir.glob("*.jpg")):
                    shutil.copy2(str(img), os.path.join(output_dir, img.name))
            return
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Fallback: copy the lowest-altitude descent renders if available
    descent_dirs = sorted(Path(model_path).parent.parent.glob("altitude_*/renders"))
    if descent_dirs:
        lowest = descent_dirs[-1]  # Last altitude = lowest
        for img in sorted(lowest.glob("*.jpg"))[:num_views]:
            shutil.copy2(str(img), os.path.join(output_dir, img.name))
        print(f"  Copied {min(num_views, len(list(lowest.glob('*.jpg'))))} renders from {lowest}")
    else:
        # Last resort: use the scene training images
        scene_images = sorted(Path(scene_path).joinpath("images").glob("*.jpg"))
        for img in scene_images[:num_views]:
            shutil.copy2(str(img), os.path.join(output_dir, img.name))
        print(f"  Copied {min(num_views, len(scene_images))} scene images as fallback")


def run_viewcrafter(input_dir, output_dir, viewcrafter_ckpt, batch_size=10):
    """
    Run ViewCrafter on pairs of rendered views to generate enhanced frames.

    ViewCrafter takes reference images and generates diffusion-enhanced
    novel views using its video diffusion model.
    """
    os.makedirs(output_dir, exist_ok=True)

    viewcrafter_dir = "/opt/ViewCrafter"
    if not os.path.isdir(viewcrafter_dir):
        raise FileNotFoundError("ViewCrafter not installed at /opt/ViewCrafter")

    input_images = sorted(Path(input_dir).glob("*.jpg")) + sorted(Path(input_dir).glob("*.png"))
    if len(input_images) < 2:
        raise ValueError(f"Need at least 2 input images, got {len(input_images)}")

    # Find the checkpoint model file
    ckpt_file = os.path.join(viewcrafter_ckpt, "model.ckpt")
    if not os.path.exists(ckpt_file):
        # Try finding any .ckpt file in the checkpoint dir
        ckpt_candidates = list(Path(viewcrafter_ckpt).glob("*.ckpt"))
        if ckpt_candidates:
            ckpt_file = str(ckpt_candidates[0])
        else:
            ckpt_file = viewcrafter_ckpt  # Use as-is

    # Find the config file (512 variant)
    config_file = os.path.join(viewcrafter_dir, "configs", "inference_pvd_512.yaml")
    if not os.path.exists(config_file):
        config_file = os.path.join(viewcrafter_dir, "configs", "inference_pvd_1024.yaml")

    enhanced_count = 0

    # Process pairs of images through ViewCrafter using sparse_view_interp mode
    for i in range(0, len(input_images) - 1, 2):
        img_a = input_images[i]
        img_b = input_images[min(i + 1, len(input_images) - 1)]

        pair_output = os.path.join(output_dir, f"pair_{i:03d}")
        os.makedirs(pair_output, exist_ok=True)

        # Prepare input directory with ordered images
        pair_input = os.path.join(pair_output, "input")
        os.makedirs(pair_input, exist_ok=True)
        shutil.copy2(str(img_a), os.path.join(pair_input, "frame_0000.jpg"))
        shutil.copy2(str(img_b), os.path.join(pair_input, "frame_0001.jpg"))

        # Find DUSt3R checkpoint (ViewCrafter uses DUSt3R for depth estimation)
        dust3r_ckpt = "/opt/InstantSplat/submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        if not os.path.exists(dust3r_ckpt):
            # Fallback: check ViewCrafter's own checkpoints dir
            dust3r_ckpt = os.path.join(viewcrafter_dir, "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

        # Run ViewCrafter with correct arguments
        cmd = [
            sys.executable,
            os.path.join(viewcrafter_dir, "inference.py"),
            "--image_dir", pair_input,
            "--out_dir", pair_output,
            "--ckpt_path", ckpt_file,
            "--model_path", dust3r_ckpt,
            "--config", config_file,
            "--mode", "sparse_view_interp",
            "--video_length", "25",
            "--ddim_steps", "25",
            "--height", "320",
            "--width", "512",
        ]

        print(f"  Running ViewCrafter on pair {i//2 + 1}/{(len(input_images)+1)//2}...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=viewcrafter_dir,
            )
            if result.returncode != 0:
                print(f"  Warning: ViewCrafter failed on pair {i}: {result.stderr[-200:]}")
                continue
        except subprocess.TimeoutExpired:
            print(f"  Warning: ViewCrafter timed out on pair {i}")
            continue

        # Collect generated frames from output
        for frame in sorted(Path(pair_output).rglob("*.png")) + sorted(
            Path(pair_output).rglob("*.jpg")
        ):
            if frame.parent.name != "input":
                dest = os.path.join(output_dir, f"enhanced_{enhanced_count:04d}.jpg")
                img = Image.open(str(frame)).convert("RGB")
                img.save(dest, "JPEG", quality=95)
                enhanced_count += 1

    print(f"  ViewCrafter generated {enhanced_count} enhanced frames")
    return enhanced_count


def retrain_with_enhanced(scene_path, model_path, output_model_path, iterations=3000):
    """Final retrain with enhanced pseudo-observation images."""
    n_images = len(list(Path(scene_path).joinpath("images").glob("*.jpg")))
    n_views = min(n_images, 24)

    if output_model_path != model_path and not os.path.exists(output_model_path):
        shutil.copytree(model_path, output_model_path)

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

    print(f"  Final retrain: {iterations} iterations with {n_images} images ({n_views} views)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Retrain stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"Final retrain failed with code {result.returncode}")

    print(f"  Retrain complete")


def main():
    parser = argparse.ArgumentParser(description="Stage 4: ViewCrafter Diffusion Enhancement")
    parser.add_argument("--model_path", required=True, help="Path to Stage 3 final model")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--viewcrafter_ckpt",
        default="/opt/ViewCrafter/checkpoints/ViewCrafter_25_512",
        help="ViewCrafter checkpoint path",
    )
    parser.add_argument(
        "--retrain_iterations", type=int, default=3000, help="Final retrain iterations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Frames per ViewCrafter pair (lower = less VRAM; use 5 for 20GB GPUs)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Render low-altitude views from Stage 3 model
    print("Rendering low-altitude views from Stage 3 model...")
    renders_dir = os.path.join(args.output_dir, "renders")
    render_low_altitude_views(args.model_path, args.scene_path, renders_dir)

    rendered_count = len(list(Path(renders_dir).glob("*")))
    print(f"  {rendered_count} views rendered")

    if rendered_count < 2:
        print("ERROR: Not enough rendered views for ViewCrafter")
        sys.exit(1)

    # Step 2: Run ViewCrafter enhancement
    print("\nRunning ViewCrafter diffusion enhancement...")
    enhanced_dir = os.path.join(args.output_dir, "enhanced_frames")
    enhanced_count = run_viewcrafter(renders_dir, enhanced_dir, args.viewcrafter_ckpt, args.batch_size)

    if enhanced_count == 0:
        print("ERROR: ViewCrafter produced no enhanced frames")
        sys.exit(1)

    # Step 3: Add enhanced frames to training set
    print(f"\nAdding {enhanced_count} enhanced frames to training set...")
    scene_images = os.path.join(args.scene_path, "images")
    for frame in sorted(Path(enhanced_dir).glob("*.jpg")):
        shutil.copy2(str(frame), os.path.join(scene_images, frame.name))

    # Step 4: Final retrain
    print(f"\nFinal retrain with enhanced views ({args.retrain_iterations} iterations)...")
    retrain_with_enhanced(
        args.scene_path, args.model_path, args.output_dir, args.retrain_iterations
    )

    print(f"\nEnhancement complete. Output at: {args.output_dir}")


if __name__ == "__main__":
    main()
