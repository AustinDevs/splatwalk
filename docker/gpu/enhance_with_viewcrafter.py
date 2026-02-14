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

# Monkey-patch for Pillow 10+ (ANTIALIAS removed, ViewCrafter still uses it)
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

# Ensure PyAV <14 is installed (av>=14 breaks torchvision 0.19's write_video
# with "TypeError: an integer is required" on frame.pict_type = "NONE").
print("Ensuring compatible PyAV version (av<14)...")
subprocess.call(
    [sys.executable, "-m", "pip", "install", "av<14"],
    timeout=120,
)


def extract_frames_from_video(video_path, output_dir, start_index=0):
    """Extract individual frames from a video file using PyAV."""
    import av

    count = 0
    try:
        container = av.open(str(video_path))
        for frame in container.decode(video=0):
            img = frame.to_image()  # PIL Image
            dest = os.path.join(output_dir, f"enhanced_{start_index + count:04d}.jpg")
            img.save(dest, "JPEG", quality=95)
            count += 1
        container.close()
    except Exception as e:
        print(f"  Warning: Could not extract frames from {video_path}: {e}")
    return count


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
    Run ViewCrafter in sparse_view_interp mode on pairs of rendered views.

    sparse_view_interp takes 2+ images of the same scene, uses DUSt3R to
    recover camera poses, builds a PyTorch3D point cloud, and generates
    diffusion-enhanced interpolated frames between the viewpoints.
    """
    os.makedirs(output_dir, exist_ok=True)

    viewcrafter_dir = "/opt/ViewCrafter"
    if not os.path.isdir(viewcrafter_dir):
        raise FileNotFoundError("ViewCrafter not installed at /opt/ViewCrafter")

    # Patch ANTIALIAS -> LANCZOS in ViewCrafter's bundled dust3r (Pillow 10+ removed ANTIALIAS)
    for pyfile in Path(viewcrafter_dir).rglob("*.py"):
        try:
            content = pyfile.read_text()
            if "Image.ANTIALIAS" in content:
                pyfile.write_text(content.replace("Image.ANTIALIAS", "Image.LANCZOS"))
                print(f"  Patched {pyfile} (ANTIALIAS -> LANCZOS)")
        except Exception:
            pass

    # Patch ViewCrafter's save_video to also save individual frames and catch video errors.
    # This ensures we get frames even if torchvision.io.write_video fails.
    pvd_utils = os.path.join(viewcrafter_dir, "utils", "pvd_utils.py")
    if os.path.exists(pvd_utils):
        try:
            content = Path(pvd_utils).read_text()
            if "# PATCHED_SAVE_FRAMES" not in content:
                # Add a patched save_video that saves frames alongside video
                patched_fn = '''
# PATCHED_SAVE_FRAMES
import os as _os
_original_save_video = save_video
def save_video(data, images_path, folder=None):
    """Patched save_video: saves individual frames and tries video."""
    import torch
    from PIL import Image as _Image
    # Save individual frames first (always works)
    frames_dir = images_path.rsplit(".", 1)[0] + "_frames"
    _os.makedirs(frames_dir, exist_ok=True)
    if isinstance(data, torch.Tensor):
        if data.dtype != torch.uint8:
            data_uint8 = (data.clamp(0, 1) * 255).byte() if data.max() <= 1.0 else data.byte()
        else:
            data_uint8 = data
        for i in range(data_uint8.shape[0]):
            frame = data_uint8[i].cpu().numpy()
            _Image.fromarray(frame).save(_os.path.join(frames_dir, f"{i:04d}.jpg"), "JPEG", quality=95)
        print(f"  Saved {data_uint8.shape[0]} frames to {frames_dir}")
    # Try video saving (may fail with av version issues)
    try:
        _original_save_video(data, images_path, folder)
    except Exception as e:
        print(f"  Video save failed (frames already saved): {e}")
'''
                content += patched_fn
                Path(pvd_utils).write_text(content)
                print(f"  Patched {pvd_utils} (added frame-saving save_video)")
        except Exception as e:
            print(f"  Warning: Could not patch pvd_utils.py: {e}")

    input_images = sorted(Path(input_dir).glob("*.jpg")) + sorted(Path(input_dir).glob("*.png"))
    if len(input_images) < 2:
        raise ValueError(f"Need at least 2 input images, got {len(input_images)}")

    # Find the SPARSE checkpoint (different from single-view model.ckpt)
    sparse_ckpt = os.path.join(viewcrafter_dir, "checkpoints", "model_sparse.ckpt")
    if not os.path.exists(sparse_ckpt):
        # Try downloading from HuggingFace at runtime
        print("  Sparse checkpoint not found, downloading from HuggingFace...")
        try:
            subprocess.check_call([
                sys.executable, "-c",
                "from huggingface_hub import hf_hub_download; "
                "import os, shutil; "
                f"path = hf_hub_download('Drexubery/ViewCrafter_25', 'model.ckpt'); "
                f"shutil.copy2(path, '{sparse_ckpt}')"
            ], timeout=600)
            print(f"  Downloaded sparse checkpoint: {sparse_ckpt}")
        except Exception as e:
            print(f"  WARNING: Could not download sparse checkpoint: {e}")
            # Fall back to the single-view checkpoint (may still work)
            ckpt_file = os.path.join(viewcrafter_ckpt, "model.ckpt")
            if os.path.exists(ckpt_file):
                sparse_ckpt = ckpt_file
                print(f"  Falling back to single-view checkpoint: {sparse_ckpt}")
            else:
                ckpt_candidates = list(Path(viewcrafter_ckpt).glob("*.ckpt"))
                if ckpt_candidates:
                    sparse_ckpt = str(ckpt_candidates[0])
                else:
                    raise FileNotFoundError("No ViewCrafter checkpoint found")

    # Find the config file
    config_file = os.path.join(viewcrafter_dir, "configs", "inference_pvd_512.yaml")
    if not os.path.exists(config_file):
        config_file = os.path.join(viewcrafter_dir, "configs", "inference_pvd_1024.yaml")

    # Find DUSt3R checkpoint (ViewCrafter uses DUSt3R for pose estimation + depth)
    dust3r_candidates = [
        "/opt/InstantSplat/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "/opt/InstantSplat/submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        os.path.join(viewcrafter_dir, "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
    ]
    dust3r_ckpt = next((p for p in dust3r_candidates if os.path.exists(p)), dust3r_candidates[0])

    enhanced_count = 0

    # Process consecutive pairs of images using sparse_view_interp mode.
    # Each pair produces ~25 interpolated frames via diffusion.
    num_pairs = min(len(input_images) - 1, batch_size)
    for i in range(num_pairs):
        # Create a temp directory with just this pair of images
        pair_dir = os.path.join(output_dir, f"pair_{i:03d}_input")
        view_output = os.path.join(output_dir, f"pair_{i:03d}_output")
        os.makedirs(pair_dir, exist_ok=True)
        os.makedirs(view_output, exist_ok=True)

        # Copy consecutive images as 000.jpg, 001.jpg (sorted order matters)
        img_a = Image.open(str(input_images[i])).convert("RGB")
        img_b = Image.open(str(input_images[i + 1])).convert("RGB")
        img_a.save(os.path.join(pair_dir, "000.jpg"), "JPEG", quality=95)
        img_b.save(os.path.join(pair_dir, "001.jpg"), "JPEG", quality=95)

        cmd = [
            sys.executable,
            os.path.join(viewcrafter_dir, "inference.py"),
            "--image_dir", pair_dir,
            "--out_dir", view_output,
            "--ckpt_path", sparse_ckpt,
            "--model_path", dust3r_ckpt,
            "--config", config_file,
            "--mode", "sparse_view_interp",
            "--bg_trd", "0.2",
            "--video_length", "25",
            "--ddim_steps", "25",
            "--height", "320",
            "--width", "512",
            "--seed", "123",
        ]

        print(f"  Running ViewCrafter sparse_view_interp on pair {i+1}/{num_pairs}...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min per pair (sparse mode is slower)
                cwd=viewcrafter_dir,
            )
            if result.returncode != 0:
                print(f"  Warning: ViewCrafter returned non-zero for pair {i}.")
                print(f"  stderr (last 2000 chars):\n{result.stderr[-2000:]}")
                # Don't continue — try to extract frames from partial output below
        except subprocess.TimeoutExpired:
            print(f"  Warning: ViewCrafter timed out on pair {i}")
            continue

        # Collect frames from output. Our patched save_video saves frames to
        # *_frames/ dirs. Also extract from MP4 videos as fallback.
        pre_count = enhanced_count

        # 1. Check for frame directories created by our patched save_video
        for frames_dir in sorted(Path(view_output).rglob("*_frames")):
            for frame in sorted(frames_dir.glob("*.jpg")):
                dest = os.path.join(output_dir, f"enhanced_{enhanced_count:04d}.jpg")
                shutil.copy2(str(frame), dest)
                enhanced_count += 1

        # 2. Extract from MP4 videos if no frames found yet
        if enhanced_count == pre_count:
            for video in sorted(Path(view_output).rglob("diffusion.mp4")):
                n = extract_frames_from_video(video, output_dir, start_index=enhanced_count)
                if n > 0:
                    print(f"  Extracted {n} frames from {video.name}")
                    enhanced_count += n

        if enhanced_count == pre_count:
            for video in sorted(Path(view_output).rglob("render.mp4")):
                n = extract_frames_from_video(video, output_dir, start_index=enhanced_count)
                if n > 0:
                    print(f"  Extracted {n} frames from {video.name} (fallback)")
                    enhanced_count += n

        # 3. Collect any other loose PNG/JPG frames
        for frame in sorted(Path(view_output).rglob("*.png")) + sorted(
            Path(view_output).rglob("*.jpg")
        ):
            if "input" not in str(frame.parent) and "_frames" not in str(frame.parent):
                dest = os.path.join(output_dir, f"enhanced_{enhanced_count:04d}.jpg")
                try:
                    img = Image.open(str(frame)).convert("RGB")
                    img.save(dest, "JPEG", quality=95)
                    enhanced_count += 1
                except Exception:
                    pass

        pair_frames = enhanced_count - pre_count
        if pair_frames > 0:
            print(f"  Collected {pair_frames} frames from pair {i}")

        # Clean up pair input dir to save disk space
        shutil.rmtree(pair_dir, ignore_errors=True)

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
        # Training may succeed but save_pose crashes after (non-contiguous UIDs).
        # Check if checkpoints were actually saved.
        ckpt_dirs = sorted(Path(output_model_path).glob("point_cloud/iteration_*"))
        if ckpt_dirs:
            print(f"  Retrain completed (post-training save_pose error ignored, checkpoint at {ckpt_dirs[-1].name})")
        else:
            print(f"  Retrain stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"Final retrain failed with code {result.returncode}")
    else:
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
        print("WARNING: ViewCrafter produced no enhanced frames, skipping enhancement")
        print("Copying Stage 3 model as output instead...")
        if args.model_path != args.output_dir:
            if os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir, ignore_errors=True)
            shutil.copytree(args.model_path, args.output_dir, dirs_exist_ok=True)
        sys.exit(1)  # Signal failure so run_pipeline.sh uses fallback path

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
