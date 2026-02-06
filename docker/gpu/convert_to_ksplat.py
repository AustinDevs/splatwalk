#!/usr/bin/env python3
"""
Convert PLY Gaussian Splat files to KSPLAT format.
KSPLAT is an optimized binary format for web-based Gaussian Splat rendering.
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
    return 1 / (1 + np.exp(-x))


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
        # Convert SH to RGB
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

    # Extract opacity
    if 'opacity' in vertex.data.dtype.names:
        opacities = sigmoid(vertex['opacity']).astype(np.float32)
    else:
        opacities = np.ones(len(positions), dtype=np.float32)

    # Extract scales
    if 'scale_0' in vertex.data.dtype.names:
        scales = np.stack([
            np.exp(vertex['scale_0']),
            np.exp(vertex['scale_1']),
            np.exp(vertex['scale_2'])
        ], axis=-1).astype(np.float32)
    else:
        scales = np.ones((len(positions), 3), dtype=np.float32) * 0.01

    # Extract rotations (quaternion)
    if 'rot_0' in vertex.data.dtype.names:
        rotations = np.stack([
            vertex['rot_0'],
            vertex['rot_1'],
            vertex['rot_2'],
            vertex['rot_3']
        ], axis=-1).astype(np.float32)
        # Normalize quaternions
        rotations = rotations / np.linalg.norm(rotations, axis=-1, keepdims=True)
    else:
        # Identity quaternion
        rotations = np.zeros((len(positions), 4), dtype=np.float32)
        rotations[:, 0] = 1.0

    return {
        'positions': positions,
        'colors': colors,
        'opacities': opacities,
        'scales': scales,
        'rotations': rotations
    }


def write_ksplat(data: dict, output_path: str):
    """Write Gaussian splat data to KSPLAT binary format."""
    num_splats = len(data['positions'])

    with open(output_path, 'wb') as f:
        # Header
        f.write(b'KSPT')  # Magic number
        f.write(struct.pack('<I', 1))  # Version
        f.write(struct.pack('<I', num_splats))  # Number of splats

        # Section flags (positions, colors, opacities, scales, rotations)
        f.write(struct.pack('<B', 0x1F))  # All sections present

        # Padding for alignment
        f.write(b'\x00' * 7)

        # Write positions (3 x float32 per splat)
        positions = data['positions'].astype(np.float32)
        f.write(positions.tobytes())

        # Write colors (3 x uint8 per splat)
        colors = (data['colors'] * 255).clip(0, 255).astype(np.uint8)
        f.write(colors.tobytes())

        # Write opacities (1 x uint8 per splat)
        opacities = (data['opacities'] * 255).clip(0, 255).astype(np.uint8)
        f.write(opacities.tobytes())

        # Write scales (3 x float32 per splat)
        scales = data['scales'].astype(np.float32)
        f.write(scales.tobytes())

        # Write rotations (4 x int8 per splat, normalized -127 to 127)
        rotations = (data['rotations'] * 127).clip(-127, 127).astype(np.int8)
        f.write(rotations.tobytes())

    print(f"Written {num_splats} splats to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


def convert_ply_to_ksplat(ply_path: str, ksplat_path: str):
    """Convert a PLY file to KSPLAT format."""
    print(f"Reading PLY file: {ply_path}")
    data = read_ply(ply_path)

    print(f"Found {len(data['positions'])} Gaussian splats")

    # Compute bounding box
    bbox_min = data['positions'].min(axis=0)
    bbox_max = data['positions'].max(axis=0)
    print(f"Bounding box: {bbox_min} to {bbox_max}")

    print(f"Writing KSPLAT file: {ksplat_path}")
    write_ksplat(data, ksplat_path)

    print("Conversion complete!")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: convert_to_ksplat.py <input.ply> <output.ksplat>")
        sys.exit(1)

    ply_path = sys.argv[1]
    ksplat_path = sys.argv[2]

    if not Path(ply_path).exists():
        print(f"Error: Input file not found: {ply_path}")
        sys.exit(1)

    convert_ply_to_ksplat(ply_path, ksplat_path)
