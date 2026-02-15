#!/usr/bin/env python3
"""
Splat Compression Pipeline: Prune + .splat conversion

Replaces the simple PLY-to-.splat conversion with a proper compression pipeline:
  Stage A: Importance-based pruning (remove low-contribution Gaussians)
  Stage B: .splat binary conversion (32 bytes/Gaussian, ~3x smaller than PLY)
           Includes floater removal + uniform scene scaling for web viewer compatibility

Uses UNIFORM scene scaling: positions AND scales are multiplied by the same factor.
This preserves the trained proportions between Gaussians, unlike separate scale multipliers
which distort the overlap relationships that training optimized.

Expected: 1GB PLY -> ~54MB .splat (1.8M Gaussians, 60fps in browser)

Usage:
    python compress_splat.py input.ply output.splat [--prune_ratio 0.20] [--scene_scale 50.0]
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

    # Vegetation-aware importance boost: green Gaussians get 1.3x multiplier
    SH_C0 = 0.28209479177387814
    if "f_dc_0" in [p.name for p in vertex.properties]:
        r = np.array(vertex["f_dc_0"]) * SH_C0 + 0.5
        g = np.array(vertex["f_dc_1"]) * SH_C0 + 0.5
        b = np.array(vertex["f_dc_2"]) * SH_C0 + 0.5
        greenness = np.clip(g - 0.5 * (r + b), 0, 1)
        importance *= (1.0 + 0.3 * greenness)
        green_count = (greenness > 0.05).sum()
        print(f"  Vegetation boost: {green_count} green Gaussians ({100*green_count/original_count:.1f}%)")

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


def convert_to_splat(ply_path, splat_path, scene_scale=50.0):
    """Convert PLY to .splat binary format (antimatter15 format, 32 bytes/Gaussian).

    Uses UNIFORM scene scaling: both positions and scales are multiplied by
    the same factor, preserving the trained proportions between Gaussians.

    Includes floater removal tuned for aerial drone reconstructions:
      - Low opacity (< 10%) Gaussians removed
      - Giant Gaussians removed (max scale > 97th percentile)
      - Elongated Gaussians removed (scale ratio > 25:1) — kills streak floaters
      - Position outliers removed (> 3σ from centroid)
      - White floaters removed (bright + low saturation + high opacity)

    Format per Gaussian (32 bytes):
      float32 x, y, z         (12 bytes - position, centered + scaled)
      float32 s0, s1, s2      (12 bytes - scale, exponentiated + scaled)
      uint8   r, g, b, a      ( 4 bytes - color + opacity)
      uint8   q0, q1, q2, q3  ( 4 bytes - quaternion, 128-biased)
    """
    print(f"Converting to .splat: {ply_path} -> {splat_path} (scene_scale={scene_scale})")

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

    # Scales (log space for filtering, then exponentiate)
    if "scale_0" in field_names:
        log_scales = np.stack([
            np.array(vertex["scale_0"], dtype=np.float32),
            np.array(vertex["scale_1"], dtype=np.float32),
            np.array(vertex["scale_2"], dtype=np.float32),
        ], axis=-1)
        scales = np.exp(log_scales)
    else:
        log_scales = np.full((N, 3), -4.6, dtype=np.float32)
        scales = np.full((N, 3), 0.01, dtype=np.float32)

    # Opacity (float for filtering)
    if "opacity" in field_names:
        opacity_logit = np.array(vertex["opacity"], dtype=np.float32)
        opacity = sigmoid(opacity_logit)
    else:
        opacity_logit = np.zeros(N, dtype=np.float32)
        opacity = np.ones(N, dtype=np.float32)

    # Colors: SH DC -> RGB [0,1] float
    SH_C0 = 0.28209479177387814
    if "f_dc_0" in field_names:
        colors_rgb = np.stack([
            np.array(vertex["f_dc_0"], dtype=np.float32),
            np.array(vertex["f_dc_1"], dtype=np.float32),
            np.array(vertex["f_dc_2"], dtype=np.float32),
        ], axis=-1) * SH_C0 + 0.5
    elif "red" in field_names:
        colors_rgb = np.stack([
            np.array(vertex["red"], dtype=np.float32),
            np.array(vertex["green"], dtype=np.float32),
            np.array(vertex["blue"], dtype=np.float32),
        ], axis=-1)
        if colors_rgb.max() > 1.0:
            colors_rgb /= 255.0
    else:
        colors_rgb = np.full((N, 3), 0.5, dtype=np.float32)

    # === Floater removal filters ===
    keep = np.ones(N, dtype=bool)

    # 1. Remove low opacity (< 10%)
    opacity_mask = opacity >= 0.10
    print(f"  Filter: opacity >= 0.10: keeping {opacity_mask.sum()} / {N} ({100*opacity_mask.mean():.1f}%)")
    keep &= opacity_mask

    # 2. Remove giant Gaussians (max scale > 97th percentile)
    max_scale = scales.max(axis=-1)
    scale_cap = np.percentile(max_scale[keep], 97.0)
    scale_mask = max_scale <= scale_cap
    print(f"  Filter: max_scale <= {scale_cap:.6f}: keeping {(keep & scale_mask).sum()} / {keep.sum()}")
    keep &= scale_mask

    # 3. Remove elongated Gaussians (scale ratio > 25:1)
    min_scale = scales.min(axis=-1)
    min_scale[min_scale == 0] = 1e-10
    elongation = max_scale / min_scale
    elong_mask = elongation <= 25.0
    print(f"  Filter: elongation <= 25: keeping {(keep & elong_mask).sum()} / {keep.sum()}")
    keep &= elong_mask

    # 4. Remove position outliers (> 3σ from centroid)
    centroid = positions[keep].mean(axis=0)
    dists = np.linalg.norm(positions - centroid, axis=-1)
    dist_std = dists[keep].std()
    pos_mask = dists <= 3.0 * dist_std
    print(f"  Filter: position < 3σ: keeping {(keep & pos_mask).sum()} / {keep.sum()}")
    keep &= pos_mask

    # 5. Remove white floaters (bright + low saturation + high opacity)
    brightness = (colors_rgb[:, 0] + colors_rgb[:, 1] + colors_rgb[:, 2]) / 3.0
    max_rgb = np.maximum(colors_rgb[:, 0], np.maximum(colors_rgb[:, 1], colors_rgb[:, 2]))
    min_rgb = np.minimum(colors_rgb[:, 0], np.minimum(colors_rgb[:, 1], colors_rgb[:, 2]))
    saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1e-6)
    white_mask = ~((brightness > 0.8) & (saturation < 0.15) & (opacity > 0.4))
    white_removed = keep.sum() - (keep & white_mask).sum()
    print(f"  Filter: white floaters: removed {white_removed}")
    keep &= white_mask

    kept = keep.sum()
    removed = N - kept
    print(f"  Total after filtering: {kept} ({100*kept/N:.1f}% of {N}, removed {removed} floaters)")

    # Apply filter
    positions = positions[keep]
    scales = scales[keep]
    opacity = opacity[keep].copy()
    colors_rgb = colors_rgb[keep]
    N = kept

    # Uniform scene scaling: positions AND scales by same factor
    # Centers scene at origin first, then scales uniformly
    positions = (positions - centroid) * scene_scale
    scales = scales * scene_scale

    print(f"  Uniform {scene_scale}x scene scale applied (centered at origin)")
    print(f"  Position range: x=[{positions[:,0].min():.0f},{positions[:,0].max():.0f}] "
          f"y=[{positions[:,1].min():.0f},{positions[:,1].max():.0f}] "
          f"z=[{positions[:,2].min():.0f},{positions[:,2].max():.0f}]")
    print(f"  Scale stats: median={np.median(scales):.5f}, p5={np.percentile(scales,5):.5f}, p95={np.percentile(scales,95):.5f}")

    # Reduce opacity of large Gaussians to prevent wash-out from aerial viewing
    vol = np.prod(scales, axis=-1)
    vol_p50 = np.percentile(vol, 50)
    vol_p95 = np.percentile(vol, 95)
    large_mask = vol > vol_p50
    if large_mask.any():
        vol_normalized = np.clip((vol[large_mask] - vol_p50) / (vol_p95 - vol_p50), 0, 1)
        opacity[large_mask] *= (1.0 - 0.7 * vol_normalized).astype(np.float32)
        print(f"  Opacity fade: reduced opacity for {large_mask.sum()} large Gaussians")

    # Colors to uint8
    colors_u8 = (colors_rgb * 255.0).clip(0, 255).astype(np.uint8)

    # Alpha: [0,255] uint8
    alpha = (opacity * 255.0).clip(0, 255).astype(np.uint8)
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
        rots = np.zeros((len(vertex.data), 4), dtype=np.float32)
        rots[:, 0] = 1.0
    rots = rots[keep]
    rot_u8 = (rots * 128 + 128).clip(0, 255).astype(np.uint8)

    # Sort by importance (most opaque/large first) for progressive rendering
    sort_score = -(np.prod(scales, axis=-1) * opacity)
    order = np.argsort(sort_score)

    positions = positions[order].astype(np.float32)
    scales = scales[order].astype(np.float32)
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
    parser.add_argument("--prune_ratio", type=float, default=0.20,
                        help="Fraction of Gaussians to prune (0.0-1.0, default 0.20)")
    parser.add_argument("--confidence_npy", default=None,
                        help="Path to per-Gaussian confidence.npy for weighted pruning")
    parser.add_argument("--scene_scale", type=float, default=50.0,
                        help="Uniform scene scale for web viewer (default 50.0, applied to positions AND scales)")
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
    splat_size = convert_to_splat(pruned_ply, args.output_splat, scene_scale=args.scene_scale)

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
