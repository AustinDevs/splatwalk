#!/usr/bin/env python3
"""
Splat Compression Pipeline: Prune + SPZ

Replaces the simple PLY-to-.splat conversion with a proper compression pipeline:
  Stage A: Importance-based pruning (remove low-contribution Gaussians)
  Stage B: SPZ conversion (quantized + compressed format)

Expected compression: 200-500MB PLY -> ~60-70% after pruning -> ~10-15% after SPZ = 10-30MB final

Usage:
    python compress_splat.py input.ply output.spz [--prune_ratio 0.3] [--confidence_npy path]
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


def convert_to_spz(ply_path, spz_path):
    """Convert PLY to SPZ using gsconverter (3dgsconverter package)."""
    print(f"Converting to SPZ: {ply_path} -> {spz_path}")

    cmd = ["gsconverter", "-i", ply_path, "-o", spz_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(f"  {result.stdout.strip()}")
        if result.stderr:
            print(f"  gsconverter stderr: {result.stderr.strip()[:500]}")
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
    except FileNotFoundError:
        # Fall back to python -m
        cmd = [sys.executable, "-m", "gsconverter", "-i", ply_path, "-o", spz_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(f"  {result.stdout.strip()}")
        if result.stderr:
            print(f"  gsconverter stderr: {result.stderr.strip()[:500]}")
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)

    spz_size_mb = Path(spz_path).stat().st_size / 1024 / 1024
    print(f"  SPZ output: {spz_size_mb:.1f} MB")
    return spz_size_mb


def main():
    parser = argparse.ArgumentParser(description="Splat Compression: Prune + SPZ")
    parser.add_argument("input_ply", help="Input PLY file path")
    parser.add_argument("output_spz", help="Output SPZ file path")
    parser.add_argument("--prune_ratio", type=float, default=0.3,
                        help="Fraction of Gaussians to prune (0.0-1.0, default 0.3)")
    parser.add_argument("--confidence_npy", default=None,
                        help="Path to per-Gaussian confidence.npy for weighted pruning")
    parser.add_argument("--skip_prune", action="store_true",
                        help="Skip pruning, only do SPZ conversion")
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

    # Stage B: SPZ Conversion
    print(f"\n=== Stage B: SPZ Conversion ===")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_spz)), exist_ok=True)
    spz_size = convert_to_spz(pruned_ply, args.output_spz)

    # Cleanup intermediate file
    if not args.skip_prune and pruned_ply != args.input_ply and os.path.exists(pruned_ply):
        os.remove(pruned_ply)
        print(f"  Cleaned up intermediate: {pruned_ply}")

    # Summary
    final_size_mb = Path(args.output_spz).stat().st_size / 1024 / 1024
    compression_ratio = input_size_mb / max(final_size_mb, 0.01)
    print(f"\n=== Compression Summary ===")
    print(f"  Input:  {input_size_mb:.1f} MB (PLY)")
    print(f"  Output: {final_size_mb:.1f} MB (SPZ)")
    print(f"  Ratio:  {compression_ratio:.1f}x ({100 * final_size_mb / input_size_mb:.1f}%)")


if __name__ == "__main__":
    main()
