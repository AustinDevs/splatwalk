#!/usr/bin/env python3
"""
Convert PLY Gaussian Splat files to .splat format.
.splat is the antimatter15 binary format (32 bytes per splat, no header)
supported by all major web-based Gaussian Splat viewers.

Format per splat (32 bytes total):
  float32 x, y, z           (12 bytes - position)
  float32 scale_0, 1, 2     (12 bytes - scale in exp space)
  uint8   r, g, b, a        ( 4 bytes - color + opacity)
  uint8   rot_0, 1, 2, 3    ( 4 bytes - quaternion, 128-biased)
"""

import sys
import struct
import numpy as np
from pathlib import Path

try:
    from plyfile import PlyData
except ImportError:
    print("Installing plyfile...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def read_ply(ply_path: str) -> dict:
    """Read a PLY file and extract Gaussian splat data."""
    ply_data = PlyData.read(ply_path)
    vertex = ply_data['vertex']

    # Extract positions
    positions = np.stack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ], axis=-1).astype(np.float32)

    # Extract colors (spherical harmonics DC component)
    if 'f_dc_0' in vertex.data.dtype.names:
        colors = np.stack([
            vertex['f_dc_0'],
            vertex['f_dc_1'],
            vertex['f_dc_2']
        ], axis=-1).astype(np.float32)
        # Convert SH to RGB: color = SH_DC * C0 + 0.5
        colors = (colors * 0.28209479177387814 + 0.5).clip(0, 1)
    elif 'red' in vertex.data.dtype.names:
        colors = np.stack([
            vertex['red'],
            vertex['green'],
            vertex['blue']
        ], axis=-1).astype(np.float32)
        if colors.max() > 1:
            colors = colors / 255.0
    else:
        colors = np.ones((len(positions), 3), dtype=np.float32) * 0.5

    # Extract opacity (stored as logit in PLY, convert via sigmoid)
    if 'opacity' in vertex.data.dtype.names:
        opacities = sigmoid(vertex['opacity']).astype(np.float32)
    else:
        opacities = np.ones(len(positions), dtype=np.float32)

    # Extract scales (stored as log-scale in PLY, convert via exp)
    if 'scale_0' in vertex.data.dtype.names:
        scales = np.stack([
            np.exp(vertex['scale_0']),
            np.exp(vertex['scale_1']),
            np.exp(vertex['scale_2'])
        ], axis=-1).astype(np.float32)
    else:
        scales = np.ones((len(positions), 3), dtype=np.float32) * 0.01

    # Extract rotations (quaternion wxyz in PLY)
    if 'rot_0' in vertex.data.dtype.names:
        rotations = np.stack([
            vertex['rot_0'],
            vertex['rot_1'],
            vertex['rot_2'],
            vertex['rot_3']
        ], axis=-1).astype(np.float32)
        # Normalize quaternions
        norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        rotations = rotations / norms
    else:
        rotations = np.zeros((len(positions), 4), dtype=np.float32)
        rotations[:, 0] = 1.0

    return {
        'positions': positions,
        'colors': colors,
        'opacities': opacities,
        'scales': scales,
        'rotations': rotations
    }


def write_splat(data: dict, output_path: str):
    """Write Gaussian splat data to .splat binary format (antimatter15 format).

    32 bytes per splat, no header:
      float32 x, y, z          (position)
      float32 s0, s1, s2       (scale, already exponentiated)
      uint8   r, g, b, a       (color + opacity)
      uint8   q0, q1, q2, q3   (quaternion, 128-biased: val_uint8 = val_float * 128 + 128)
    """
    num_splats = len(data['positions'])

    # Prepare arrays
    positions = data['positions'].astype(np.float32)       # [N, 3]
    scales = data['scales'].astype(np.float32)             # [N, 3]

    # Colors: float [0,1] -> uint8 [0,255]
    colors_rgb = (data['colors'] * 255).clip(0, 255).astype(np.uint8)  # [N, 3]

    # Opacity: float [0,1] -> uint8 [0,255]
    alphas = (data['opacities'] * 255).clip(0, 255).astype(np.uint8)   # [N]

    # RGBA combined
    rgba = np.column_stack([colors_rgb, alphas])  # [N, 4]

    # Quaternion: float [-1,1] -> uint8 [0,255] with 128 = 0.0
    rot_uint8 = ((data['rotations'] * 128) + 128).clip(0, 255).astype(np.uint8)  # [N, 4]

    # Write interleaved: 32 bytes per splat
    buf = bytearray(num_splats * 32)
    for i in range(num_splats):
        offset = i * 32
        struct.pack_into('<3f', buf, offset, *positions[i])
        struct.pack_into('<3f', buf, offset + 12, *scales[i])
        buf[offset + 24:offset + 28] = rgba[i].tobytes()
        buf[offset + 28:offset + 32] = rot_uint8[i].tobytes()

    with open(output_path, 'wb') as f:
        f.write(buf)

    print(f"Written {num_splats} splats to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


def convert_ply_to_splat(ply_path: str, output_path: str):
    """Convert a PLY file to .splat format."""
    print(f"Reading PLY file: {ply_path}")
    data = read_ply(ply_path)

    num = len(data['positions'])
    print(f"Found {num} Gaussian splats")

    bbox_min = data['positions'].min(axis=0)
    bbox_max = data['positions'].max(axis=0)
    print(f"Bounding box: {bbox_min} to {bbox_max}")

    # Sort by opacity (most opaque first) for better progressive rendering
    order = np.argsort(-data['opacities'])
    data = {k: v[order] for k, v in data.items()}

    print(f"Writing .splat file: {output_path}")
    write_splat(data, output_path)
    print("Conversion complete!")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: convert_to_ksplat.py <input.ply> <output.splat>")
        sys.exit(1)

    ply_path = sys.argv[1]
    output_path = sys.argv[2]

    if not Path(ply_path).exists():
        print(f"Error: Input file not found: {ply_path}")
        sys.exit(1)

    convert_ply_to_splat(ply_path, output_path)
