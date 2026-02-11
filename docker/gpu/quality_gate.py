#!/usr/bin/env python3
"""
Stage 5: Quality Gate — Confidence Scoring

Analyzes the final Gaussian splat model and produces per-Gaussian confidence
scores based on how much each Gaussian was informed by real vs synthetic
training views.

Confidence tiers:
  - >0.7: "captured" — primarily from real drone imagery
  - 0.4-0.7: "mixed" — blend of real and AI-generated observations
  - <0.4: "ai_generated" — primarily from synthetic/diffusion views

Output:
  - Copies the model to output_dir
  - Writes confidence.json with aggregate stats
  - Writes per_gaussian_confidence.npy with per-Gaussian scores
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData


def count_real_vs_synthetic(scene_images_dir):
    """Count real vs synthetic training images by filename convention."""
    real_count = 0
    synthetic_count = 0

    for img_file in Path(scene_images_dir).glob("*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        name = img_file.stem.lower()
        # Synthetic images from our pipeline have these prefixes
        if name.startswith("descent_") or name.startswith("enhanced_"):
            synthetic_count += 1
        else:
            real_count += 1

    return real_count, synthetic_count


def estimate_per_gaussian_confidence(ply_path, real_count, synthetic_count, scene_images_dir):
    """
    Estimate per-Gaussian confidence scores.

    Heuristic approach:
    1. Global ratio: what fraction of training data is real
    2. Spatial prior: Gaussians near the original drone altitude are more likely
       to be well-captured; those near ground are more likely synthetic
    3. Opacity as quality signal: low-opacity Gaussians are often artifacts
    """
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]
    n_gaussians = len(vertex["x"])

    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

    # 1. Global real/synthetic ratio
    total = real_count + synthetic_count
    if total == 0:
        global_real_ratio = 1.0
    else:
        global_real_ratio = real_count / total

    # 2. Spatial prior: Gaussians at higher Z are more likely from real images
    z_values = positions[:, 2]
    z_min, z_max = np.percentile(z_values, [5, 95])
    z_range = max(z_max - z_min, 1e-6)

    # Normalize Z to [0, 1] where 1 = drone altitude
    z_normalized = np.clip((z_values - z_min) / z_range, 0, 1)

    # Spatial confidence: higher altitude = more likely captured by real drone imagery
    spatial_confidence = 0.3 + 0.7 * z_normalized

    # 3. Opacity as quality signal
    if "opacity" in vertex.data.dtype.names:
        opacity_raw = vertex["opacity"]
        # Sigmoid to get actual opacity
        opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
    else:
        opacity = np.ones(n_gaussians)

    opacity_confidence = np.clip(opacity, 0, 1)

    # Combined confidence score
    confidence = (
        0.4 * global_real_ratio
        + 0.35 * spatial_confidence
        + 0.25 * opacity_confidence
    )

    # Clamp to [0, 1]
    confidence = np.clip(confidence, 0, 1)

    return confidence.astype(np.float32)


def classify_confidence(confidence):
    """Classify Gaussians into confidence tiers."""
    captured = np.sum(confidence > 0.7)
    mixed = np.sum((confidence >= 0.4) & (confidence <= 0.7))
    ai_generated = np.sum(confidence < 0.4)
    total = len(confidence)

    return {
        "total_gaussians": int(total),
        "captured": {"count": int(captured), "fraction": round(float(captured / max(total, 1)), 4)},
        "mixed": {"count": int(mixed), "fraction": round(float(mixed / max(total, 1)), 4)},
        "ai_generated": {
            "count": int(ai_generated),
            "fraction": round(float(ai_generated / max(total, 1)), 4),
        },
        "mean_confidence": round(float(np.mean(confidence)), 4),
        "median_confidence": round(float(np.median(confidence)), 4),
        "thresholds": {"captured": ">0.7", "mixed": "0.4-0.7", "ai_generated": "<0.4"},
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Quality Gate Confidence Scoring")
    parser.add_argument("--model_path", required=True, help="Path to enhanced model from Stage 4")
    parser.add_argument("--real_images", required=True, help="Path to original real images")
    parser.add_argument("--output_dir", required=True, help="Output directory for final model + metadata")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find the PLY file
    ply_path = None
    for candidate in sorted(Path(args.model_path).rglob("*.ply")):
        ply_path = str(candidate)
        break

    if ply_path is None:
        print(f"ERROR: No PLY file found in {args.model_path}")
        sys.exit(1)

    print(f"Analyzing model: {ply_path}")

    # Count training image composition
    # The scene images dir is the parent of real_images that contains
    # both real and synthetic images added during descent/enhancement
    scene_images_dir = args.real_images
    real_count, synthetic_count = count_real_vs_synthetic(scene_images_dir)
    print(f"Training data: {real_count} real, {synthetic_count} synthetic images")

    # Compute per-Gaussian confidence
    confidence = estimate_per_gaussian_confidence(
        ply_path, real_count, synthetic_count, scene_images_dir
    )
    print(f"Computed confidence for {len(confidence)} Gaussians")

    # Classify
    classification = classify_confidence(confidence)
    print(f"\nConfidence breakdown:")
    print(f"  Captured (>0.7):     {classification['captured']['count']} ({classification['captured']['fraction']*100:.1f}%)")
    print(f"  Mixed (0.4-0.7):     {classification['mixed']['count']} ({classification['mixed']['fraction']*100:.1f}%)")
    print(f"  AI-generated (<0.4): {classification['ai_generated']['count']} ({classification['ai_generated']['fraction']*100:.1f}%)")
    print(f"  Mean confidence:     {classification['mean_confidence']:.4f}")

    # Copy model to output
    print(f"\nCopying model to {args.output_dir}...")
    # Copy the point_cloud directory structure
    for src_file in Path(args.model_path).rglob("*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(args.model_path)
            dest = Path(args.output_dir) / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_file), str(dest))

    # Save confidence metadata
    confidence_data = {
        "real_images": real_count,
        "synthetic_images": synthetic_count,
        **classification,
    }

    confidence_path = os.path.join(args.output_dir, "confidence.json")
    with open(confidence_path, "w") as f:
        json.dump(confidence_data, f, indent=2)
    print(f"Wrote confidence metadata to {confidence_path}")

    # Save per-Gaussian confidence as numpy array
    npy_path = os.path.join(args.output_dir, "per_gaussian_confidence.npy")
    np.save(npy_path, confidence)
    print(f"Wrote per-Gaussian confidence ({len(confidence)} values) to {npy_path}")

    print("\nQuality gate complete.")


if __name__ == "__main__":
    main()
