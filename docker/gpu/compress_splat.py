#!/usr/bin/env python3
"""
Splat Compression Pipeline: Prune + .splat conversion

Replaces the simple PLY-to-.splat conversion with a proper compression pipeline:
  Stage A: Importance-based pruning (remove low-contribution Gaussians)
  Stage B: .splat binary conversion (32 bytes/Gaussian, ~3x smaller than PLY)

Expected compression: 200-500MB PLY -> ~60-70% after pruning -> ~50% via .splat = 30-100MB final

Usage:
    python compress_splat.py input.ply output.splat [--prune_ratio 0.3] [--confidence_npy path]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData, PlyElement


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def prune_gaussians(ply_path, output_ply_path, prune_ratio=0.3, confidence_path=None):
    """
    Remove low-contribution Gaussians based on importance scoring.

    Scoring: combine opacity + scale (proxy for pixel contribution).
    If confidence.npy available, also factor in per-Gaussian confidence.
    """
    print(f"Reading PLY: {ply_path}")
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]
    original_count = len(vertex.data)

    input_size_mb = Path(ply_path).stat().st_size / 1024 / 1024
    print(f"  {original_count} Gaussians, {input_size_mb:.1f} MB")

    # Verify this is a Gaussian splat PLY (not a raw point cloud)
    field_names = [p.name for p in vertex.properties]
    if "opacity" not in field_names:
        print(f"  ERROR: PLY has no 'opacity' field. Available fields: {field_names[:20]}")
        print(f"  This looks like a raw point cloud, not a trained Gaussian splat.")
        raise ValueError(f"PLY file is not a Gaussian splat (no opacity field): {ply_path}")

    # Compute importance score per Gaussian
    opacities = sigmoid(np.array(vertex["opacity"]))

    scales = np.exp(np.stack([
        np.array(vertex["scale_0"]),
        np.array(vertex["scale_1"]),
        np.array(vertex["scale_2"]),
    ], axis=-1))
    volume = np.prod(scales, axis=-1)  # proxy for pixel area
    importance = opacities * np.cbrt(volume)  # opacity x size

    # Boost with confidence scores if available
    if confidence_path and os.path.exists(confidence_path):
        confidence = np.load(confidence_path)
        if len(confidence) == original_count:
            importance *= (0.5 + 0.5 * confidence)
            print(f"  Applied confidence weighting from {confidence_path}")
        else:
            print(f"  WARNING: confidence array length mismatch ({len(confidence)} vs {original_count}), skipping")

    # Keep top (1 - prune_ratio) Gaussians
    threshold = np.percentile(importance, prune_ratio * 100)
    keep_mask = importance >= threshold

    kept_count = keep_mask.sum()
    pruned_count = original_count - kept_count

    # Write pruned PLY
    pruned_data = vertex.data[keep_mask]
    pruned_element = PlyElement.describe(pruned_data, "vertex")
    PlyData([pruned_element]).write(output_ply_path)

    output_size_mb = Path(output_ply_path).stat().st_size / 1024 / 1024
    print(f"  Pruned: {original_count} -> {kept_count} ({100 * kept_count / original_count:.0f}% kept, {pruned_count} removed)")
    print(f"  Size: {input_size_mb:.1f} MB -> {output_size_mb:.1f} MB ({100 * output_size_mb / input_size_mb:.0f}%)")

    return kept_count


def convert_to_splat(ply_path, splat_path):
    """Convert PLY to .splat binary format (antimatter15 format, 32 bytes/Gaussian).

    Format per Gaussian (32 bytes):
      float32 x, y, z         (12 bytes - position)
      float32 s0, s1, s2      (12 bytes - scale, exponentiated)
      uint8   r, g, b, a      ( 4 bytes - color + opacity)
      uint8   q0, q1, q2, q3  ( 4 bytes - quaternion, 128-biased)
    """
    print(f"Converting to .splat: {ply_path} -> {splat_path}")

    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]
    N = len(vertex.data)
    field_names = set(vertex.data.dtype.names)

    # Positions: float32 [N, 3]
    positions = np.stack([
        np.array(vertex["x"], dtype=np.float32),
        np.array(vertex["y"], dtype=np.float32),
        np.array(vertex["z"], dtype=np.float32),
    ], axis=-1)

    # Scales: exp(log_scale) -> float32 [N, 3]
    if "scale_0" in field_names:
        scales = np.exp(np.stack([
            np.array(vertex["scale_0"], dtype=np.float32),
            np.array(vertex["scale_1"], dtype=np.float32),
            np.array(vertex["scale_2"], dtype=np.float32),
        ], axis=-1))
    else:
        scales = np.full((N, 3), 0.01, dtype=np.float32)

    # Colors: SH DC -> RGB [0,255] uint8 [N, 3]
    SH_C0 = 0.28209479177387814
    if "f_dc_0" in field_names:
        colors = np.stack([
            np.array(vertex["f_dc_0"], dtype=np.float32),
            np.array(vertex["f_dc_1"], dtype=np.float32),
            np.array(vertex["f_dc_2"], dtype=np.float32),
        ], axis=-1)
        colors = (colors * SH_C0 + 0.5) * 255.0
    elif "red" in field_names:
        colors = np.stack([
            np.array(vertex["red"], dtype=np.float32),
            np.array(vertex["green"], dtype=np.float32),
            np.array(vertex["blue"], dtype=np.float32),
        ], axis=-1)
        if colors.max() <= 1.0:
            colors *= 255.0
    else:
        colors = np.full((N, 3), 128.0, dtype=np.float32)
    colors_u8 = colors.clip(0, 255).astype(np.uint8)

    # Opacity: sigmoid(logit) -> [0,255] uint8 [N]
    if "opacity" in field_names:
        alpha = (sigmoid(np.array(vertex["opacity"], dtype=np.float32)) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        alpha = np.full(N, 255, dtype=np.uint8)

    # RGBA: [N, 4]
    rgba = np.column_stack([colors_u8, alpha])

    # Rotation quaternion: normalize then map [-1,1] -> [0,255] uint8
    if "rot_0" in field_names:
        rots = np.stack([
            np.array(vertex["rot_0"], dtype=np.float32),
            np.array(vertex["rot_1"], dtype=np.float32),
            np.array(vertex["rot_2"], dtype=np.float32),
            np.array(vertex["rot_3"], dtype=np.float32),
        ], axis=-1)
        norms = np.linalg.norm(rots, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        rots = rots / norms
    else:
        rots = np.zeros((N, 4), dtype=np.float32)
        rots[:, 0] = 1.0
    rot_u8 = (rots * 128 + 128).clip(0, 255).astype(np.uint8)

    # Sort by importance (most opaque/large first) for progressive rendering
    sort_score = -np.exp(
        np.array(vertex.data["scale_0"] if "scale_0" in field_names else np.zeros(N), dtype=np.float32) +
        np.array(vertex.data["scale_1"] if "scale_1" in field_names else np.zeros(N), dtype=np.float32) +
        np.array(vertex.data["scale_2"] if "scale_2" in field_names else np.zeros(N), dtype=np.float32)
    ) / (1 + np.exp(-np.array(vertex.data["opacity"] if "opacity" in field_names else np.zeros(N), dtype=np.float32)))
    order = np.argsort(sort_score)

    positions = positions[order]
    scales = scales[order]
    rgba = rgba[order]
    rot_u8 = rot_u8[order]

    # Write interleaved binary: vectorized (fast for millions of Gaussians)
    buf = np.empty((N, 32), dtype=np.uint8)
    buf[:, 0:12] = positions.view(np.uint8).reshape(N, 12)
    buf[:, 12:24] = scales.view(np.uint8).reshape(N, 12)
    buf[:, 24:28] = rgba
    buf[:, 28:32] = rot_u8

    with open(splat_path, "wb") as f:
        f.write(buf.tobytes())

    splat_size_mb = Path(splat_path).stat().st_size / 1024 / 1024
    print(f"  {N} Gaussians -> {splat_size_mb:.1f} MB (.splat)")
    return splat_size_mb


def main():
    parser = argparse.ArgumentParser(description="Splat Compression: Prune + .splat conversion")
    parser.add_argument("input_ply", help="Input PLY file path")
    parser.add_argument("output_splat", help="Output .splat file path")
    parser.add_argument("--prune_ratio", type=float, default=0.3,
                        help="Fraction of Gaussians to prune (0.0-1.0, default 0.3)")
    parser.add_argument("--confidence_npy", default=None,
                        help="Path to per-Gaussian confidence.npy for weighted pruning")
    parser.add_argument("--skip_prune", action="store_true",
                        help="Skip pruning, only do .splat conversion")
    args = parser.parse_args()

    if not Path(args.input_ply).exists():
        print(f"Error: Input file not found: {args.input_ply}")
        sys.exit(1)

    input_size_mb = Path(args.input_ply).stat().st_size / 1024 / 1024
    print(f"Input PLY: {input_size_mb:.1f} MB")

    # Stage A: Prune
    if args.skip_prune:
        pruned_ply = args.input_ply
        print("Skipping pruning stage")
    else:
        pruned_ply = args.input_ply.replace(".ply", "_pruned.ply")
        if pruned_ply == args.input_ply:
            pruned_ply = args.input_ply + ".pruned.ply"

        print(f"\n=== Stage A: Importance-Based Pruning (ratio={args.prune_ratio}) ===")
        prune_gaussians(args.input_ply, pruned_ply,
                        prune_ratio=args.prune_ratio,
                        confidence_path=args.confidence_npy)

    # Stage B: .splat Conversion
    print(f"\n=== Stage B: .splat Conversion ===")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_splat)), exist_ok=True)
    splat_size = convert_to_splat(pruned_ply, args.output_splat)

    # Cleanup intermediate file
    if not args.skip_prune and pruned_ply != args.input_ply and os.path.exists(pruned_ply):
        os.remove(pruned_ply)
        print(f"  Cleaned up intermediate: {pruned_ply}")

    # Summary
    final_size_mb = Path(args.output_splat).stat().st_size / 1024 / 1024
    compression_ratio = input_size_mb / max(final_size_mb, 0.01)
    print(f"\n=== Compression Summary ===")
    print(f"  Input:  {input_size_mb:.1f} MB (PLY)")
    print(f"  Output: {final_size_mb:.1f} MB (.splat)")
    print(f"  Ratio:  {compression_ratio:.1f}x ({100 * final_size_mb / input_size_mb:.1f}%)")


if __name__ == "__main__":
    main()
