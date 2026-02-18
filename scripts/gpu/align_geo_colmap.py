#!/usr/bin/env python3
"""
Coordinate alignment between EXIF GPS (UTM meters) and COLMAP's arbitrary
coordinate system using Umeyama's method with RANSAC.

Correspondences: EXIF GPS -> UTM easting/northing vs COLMAP camera centers.

Usage:
    from align_geo_colmap import compute_geo_to_colmap_transform

    s, R, T, inliers = compute_geo_to_colmap_transform(
        exif_gps_list,   # [{lat, lon, alt, image_name}, ...]
        colmap_poses,    # [{center: np.array([x,y,z]), image_name: str}, ...]
    )
    # Transform: colmap_point = s * R @ utm_point + T
"""

import numpy as np


def project_gps_to_utm(lat, lon, alt=0.0):
    """Project lat/lon/alt to UTM easting/northing/alt using pyproj.

    Returns (easting, northing, alt) in meters.
    """
    from pyproj import Proj

    # Determine UTM zone
    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    proj = Proj(proj="utm", zone=zone, datum="WGS84", south=(hemisphere == "south"))
    easting, northing = proj(lon, lat)
    return np.array([easting, northing, alt])


def umeyama_alignment(src, tgt):
    """Compute similarity transform (s, R, T) from src to tgt using Umeyama's method.

    tgt = s * R @ src + T

    Args:
        src: (N, 3) source points
        tgt: (N, 3) target points

    Returns:
        s: scale factor
        R: (3, 3) rotation matrix
        T: (3,) translation vector
    """
    assert src.shape == tgt.shape and src.shape[0] >= 3

    n, d = src.shape
    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)

    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    sigma_src_sq = (src_c ** 2).sum() / n

    # Cross-covariance
    Sigma = tgt_c.T @ src_c / n

    U, D, Vt = np.linalg.svd(Sigma)
    V = Vt.T

    # Handle reflection
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[d - 1, d - 1] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma_src_sq
    T = mu_tgt - s * R @ mu_src

    return s, R, T


def ransac_similarity_transform(src, tgt, max_iters=200, inlier_thresh=10.0, min_inliers=4):
    """Robust Umeyama alignment with RANSAC outlier rejection.

    Args:
        src: (N, 3) source points (UTM)
        tgt: (N, 3) target points (COLMAP)
        max_iters: RANSAC iterations
        inlier_thresh: max residual in COLMAP units to count as inlier
        min_inliers: minimum inliers to accept a model

    Returns:
        s, R, T, inlier_mask
    """
    N = len(src)
    if N < 4:
        raise ValueError(f"Need >= 4 correspondences, got {N}")

    best_inliers = None
    best_count = 0

    rng = np.random.RandomState(42)

    for _ in range(max_iters):
        # Sample 4 random correspondences
        idx = rng.choice(N, size=4, replace=False)
        try:
            s, R, T = umeyama_alignment(src[idx], tgt[idx])
        except Exception:
            continue

        # Compute residuals for all points
        transformed = s * (R @ src.T).T + T
        residuals = np.linalg.norm(transformed - tgt, axis=1)
        inlier_mask = residuals < inlier_thresh
        count = inlier_mask.sum()

        if count > best_count:
            best_count = count
            best_inliers = inlier_mask

    if best_count < min_inliers:
        raise ValueError(
            f"RANSAC failed: only {best_count} inliers (need {min_inliers}). "
            f"GPS-COLMAP alignment too poor."
        )

    # Refit on all inliers
    s, R, T = umeyama_alignment(src[best_inliers], tgt[best_inliers])

    # Compute final residuals
    transformed = s * (R @ src.T).T + T
    residuals = np.linalg.norm(transformed - tgt, axis=1)
    final_inliers = residuals < inlier_thresh

    print(f"  RANSAC alignment: {final_inliers.sum()}/{N} inliers, "
          f"mean residual={residuals[final_inliers].mean():.3f}")

    return s, R, T, final_inliers


def compute_geo_to_colmap_transform(exif_gps_list, colmap_poses):
    """Compute the full transform from geographic (UTM) to COLMAP coordinates.

    Args:
        exif_gps_list: list of dicts with keys: lat, lon, alt, image_name
        colmap_poses: list of dicts with keys: center (np.array), image_name

    Returns:
        s, R, T, inlier_mask — where colmap_point = s * R @ utm_point + T
    """
    # Build correspondence by matching image names
    colmap_by_name = {}
    for p in colmap_poses:
        name = p["image_name"].lower()
        # Strip path prefixes
        if "/" in name:
            name = name.split("/")[-1]
        colmap_by_name[name] = p["center"]

    utm_points = []
    colmap_points = []

    for gps in exif_gps_list:
        name = gps["image_name"].lower()
        if "/" in name:
            name = name.split("/")[-1]
        # Try with and without extension changes (.JPG -> .jpg)
        candidates = [name, name.replace(".jpg", ".jpeg"), name.replace(".jpeg", ".jpg")]
        matched = None
        for c in candidates:
            if c in colmap_by_name:
                matched = c
                break
        if matched is None:
            continue

        utm = project_gps_to_utm(gps["lat"], gps["lon"], gps["alt"])
        utm_points.append(utm)
        colmap_points.append(colmap_by_name[matched])

    utm_arr = np.array(utm_points)
    colmap_arr = np.array(colmap_points)

    if len(utm_arr) < 4:
        raise ValueError(
            f"Only {len(utm_arr)} GPS-COLMAP correspondences found (need >= 4). "
            f"Check that EXIF image names match COLMAP image names."
        )

    # Check for collinearity
    centered = utm_arr[:, :2] - utm_arr[:, :2].mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if singular_values[-1] / singular_values[0] < 0.01:
        raise ValueError("GPS points are nearly collinear — cannot compute alignment")

    print(f"  Found {len(utm_arr)} GPS-COLMAP correspondences")
    s, R, T, inliers = ransac_similarity_transform(utm_arr, colmap_arr)

    # Validate residual
    transformed = s * (R @ utm_arr.T).T + T
    residuals = np.linalg.norm(transformed - colmap_arr, axis=1)
    mean_residual = residuals[inliers].mean()
    if mean_residual > 10.0:
        raise ValueError(
            f"Alignment residual too high: {mean_residual:.2f} COLMAP units. "
            f"GPS data may be unreliable."
        )

    return s, R, T, inliers


def extract_exif_gps(input_dir):
    """Extract GPS data from all images in a directory.

    Returns list of dicts: [{image_name, lat, lon, alt}, ...]
    """
    import json
    from pathlib import Path
    from PIL import Image, ExifTags

    results = []
    for img_path in sorted(Path(input_dir).glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            img = Image.open(img_path)
            exif = img._getexif()
            if not exif:
                continue
            gps_info = exif.get(ExifTags.Base.GPSInfo)
            if not gps_info:
                continue

            tags = {}
            for k, v in gps_info.items():
                tags[ExifTags.GPSTAGS.get(k, k)] = v

            if "GPSLatitude" not in tags:
                continue

            def dms_to_decimal(dms, ref):
                val = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
                return -val if ref in ("S", "W") else val

            lat = dms_to_decimal(tags["GPSLatitude"], tags.get("GPSLatitudeRef", "N"))
            lon = dms_to_decimal(tags["GPSLongitude"], tags.get("GPSLongitudeRef", "W"))
            alt = float(tags.get("GPSAltitude", 0))

            results.append({
                "image_name": img_path.name,
                "lat": lat,
                "lon": lon,
                "alt": alt,
            })
        except Exception:
            continue

    return results


def save_exif_gps_json(exif_gps_list, output_path):
    """Save EXIF GPS data to JSON for reuse."""
    import json
    serializable = []
    for entry in exif_gps_list:
        serializable.append({
            "image_name": entry["image_name"],
            "lat": float(entry["lat"]),
            "lon": float(entry["lon"]),
            "alt": float(entry["alt"]),
        })
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved {len(serializable)} GPS entries to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GPS-COLMAP alignment")
    parser.add_argument("--input_dir", required=True, help="Directory with EXIF images")
    parser.add_argument("--scene_path", required=True, help="Scene path with COLMAP sparse data")
    parser.add_argument("--output", default="exif_gps.json", help="Output GPS JSON")
    args = parser.parse_args()

    # Extract GPS
    gps_list = extract_exif_gps(args.input_dir)
    print(f"Extracted GPS from {len(gps_list)} images")
    save_exif_gps_json(gps_list, args.output)

    # Test alignment if scene path has COLMAP data
    import sys
    sys.path.insert(0, ".")
    try:
        from render_zoom_descent import load_camera_poses
        poses = load_camera_poses(args.scene_path)
        s, R, T, inliers = compute_geo_to_colmap_transform(gps_list, poses)
        print(f"Scale: {s:.6f}")
        print(f"Inliers: {inliers.sum()}/{len(inliers)}")
    except Exception as e:
        print(f"Alignment test failed: {e}")
