#!/usr/bin/env python3
"""
Generate 2D Orthomosaic Deep-Zoom Tile Pyramid

Takes an ODM orthophoto (GeoTIFF), tiles it as Level 0, then progressively
enhances tiles with FLUX to create multiple zoom levels.
Uploads tile pyramid to CDN in {z}/{x}/{y}.jpg format for Leaflet viewer.

Pipeline:
  1. Load ODM orthophoto as Level 0
  2. FLUX Img2Img refinement (Levels 1-2): upscale 2x, enhance tiles
  3. Bicubic upscale + sharpen (Levels 3-4)
  4. Upload tile pyramid to CDN + generate ortho_manifest.json

Usage:
    python generate_ortho_tiles.py \
        --odm_orthophoto /workspace/job/output/odm/odm_orthophoto/odm_orthophoto.tif \
        --scene_path /workspace/job/output/scene \
        --output_dir /workspace/job/output/ortho \
        --job_id aukerman
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Import helpers from render_zoom_descent.py (both live on the volume)
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from render_zoom_descent import (
    notify_slack,
    caption_scene,
)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_image(image, tile_size=512):
    """Split image into tile_size x tile_size tiles. Returns list of (x, y, tile)."""
    h, w = image.shape[:2]
    tiles = []
    nx = math.ceil(w / tile_size)
    ny = math.ceil(h / tile_size)
    for ty in range(ny):
        for tx in range(nx):
            x0 = tx * tile_size
            y0 = ty * tile_size
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            tile = image[y0:y1, x0:x1]
            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            tiles.append((tx, ty, tile))
    return tiles


def save_tiles(tiles, zoom_level, output_dir):
    """Save tiles in {z}/{x}/{y}.jpg format."""
    level_dir = os.path.join(output_dir, str(zoom_level))
    os.makedirs(level_dir, exist_ok=True)
    for tx, ty, tile in tiles:
        col_dir = os.path.join(level_dir, str(tx))
        os.makedirs(col_dir, exist_ok=True)
        tile_path = os.path.join(col_dir, f"{ty}.jpg")
        cv2.imwrite(tile_path, tile, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Level {zoom_level}: saved {len(tiles)} tiles")


# ---------------------------------------------------------------------------
# FLUX enhancement for tiles
# ---------------------------------------------------------------------------

def reconstruct_image_from_tiles(tiles, tile_size=512):
    """Reconstruct a full image from a list of (x, y, tile) tuples."""
    if not tiles:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    max_x = max(tx for tx, ty, _ in tiles)
    max_y = max(ty for tx, ty, _ in tiles)
    w = (max_x + 1) * tile_size
    h = (max_y + 1) * tile_size
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for tx, ty, tile in tiles:
        x0 = tx * tile_size
        y0 = ty * tile_size
        image[y0:y0 + tile_size, x0:x0 + tile_size] = tile
    return image


def enhance_tiles_with_flux(parent_tiles, zoom_level, output_dir, scene_images_dir,
                            prompt_base="", skip_flux=False, flux_strength=0.25,
                            tile_size=512):
    """Stitch-first FLUX Img2Img enhancement for a zoom level.

    1. Reconstruct full parent image from tiles
    2. Upscale 2x with Lanczos
    3. Slice into grid of tile_size patches (with overlap for seamless edges)
    4. Run FluxImg2ImgPipeline on each patch at low strength
    5. Stitch enhanced patches back and slice into final tiles

    Returns list of (x, y, tile) for the child level.
    """
    if skip_flux:
        return _upscale_tiles_bicubic(parent_tiles, zoom_level, output_dir)

    import torch
    from PIL import Image as PILImage

    # Reconstruct parent-level full image
    parent_image = reconstruct_image_from_tiles(parent_tiles, tile_size)
    ph, pw = parent_image.shape[:2]
    print(f"  Reconstructed parent image: {pw}x{ph}")

    # Upscale 2x with Lanczos
    new_w, new_h = pw * 2, ph * 2
    upscaled = cv2.resize(parent_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    print(f"  Upscaled to {new_w}x{new_h}")

    # Monkey-patch: PyTorch 2.4 doesn't support enable_gqa
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _patched_sdpa(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

    from diffusers import FluxImg2ImgPipeline

    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    prompt = f"aerial orthophoto, nadir view, {prompt_base}, sharp, photorealistic"

    # Process tiles with overlap for seamless edges
    overlap = 64
    nx = math.ceil(new_w / tile_size)
    ny = math.ceil(new_h / tile_size)

    # Output canvas for enhanced image
    enhanced = np.zeros((new_h, new_w, 3), dtype=np.float64)
    weight_map = np.zeros((new_h, new_w), dtype=np.float64)

    total_patches = nx * ny
    patch_idx = 0

    for ty in range(ny):
        for tx in range(nx):
            patch_idx += 1

            # Tile region in the upscaled image
            x0 = tx * tile_size
            y0 = ty * tile_size
            x1 = min(x0 + tile_size, new_w)
            y1 = min(y0 + tile_size, new_h)

            # Expand with overlap (clamped to image bounds)
            ex0 = max(x0 - overlap, 0)
            ey0 = max(y0 - overlap, 0)
            ex1 = min(x1 + overlap, new_w)
            ey1 = min(y1 + overlap, new_h)

            patch = upscaled[ey0:ey1, ex0:ex1]

            # Skip very dark patches (background)
            if patch.mean() < 15:
                # Just use the upscaled data directly
                enhanced[y0:y1, x0:x1] += upscaled[y0:y1, x0:x1].astype(np.float64)
                weight_map[y0:y1, x0:x1] += 1.0
                continue

            # Resize patch to 512x512 for FLUX input
            patch_pil = PILImage.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            patch_pil = patch_pil.resize((512, 512), PILImage.LANCZOS)

            result = pipe(
                prompt=prompt,
                image=patch_pil,
                strength=flux_strength,
                num_inference_steps=20,
                guidance_scale=3.5,
                height=512,
                width=512,
            ).images[0]

            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

            # Resize result back to the expanded patch size
            result_resized = cv2.resize(result_np, (ex1 - ex0, ey1 - ey0),
                                        interpolation=cv2.INTER_LANCZOS4)

            # Compute offset of the core tile within the expanded patch
            core_x0 = x0 - ex0
            core_y0 = y0 - ey0
            core_x1 = core_x0 + (x1 - x0)
            core_y1 = core_y0 + (y1 - y0)

            # Only use the core region (discard overlap margins)
            core = result_resized[core_y0:core_y1, core_x0:core_x1]
            enhanced[y0:y1, x0:x1] += core.astype(np.float64)
            weight_map[y0:y1, x0:x1] += 1.0

            if patch_idx % 10 == 0 or patch_idx == total_patches:
                print(f"  FLUX Img2Img: {patch_idx}/{total_patches} patches "
                      f"(strength={flux_strength})")

    # Normalize
    valid = weight_map > 0
    for c in range(3):
        enhanced[:, :, c][valid] /= weight_map[valid]
    enhanced = enhanced.clip(0, 255).astype(np.uint8)

    del pipe
    torch.cuda.empty_cache()

    # Slice final enhanced image into tiles
    child_tiles = tile_image(enhanced, tile_size)

    print(f"  FLUX enhanced level {zoom_level}: {len(child_tiles)} tiles "
          f"({new_w}x{new_h})")
    return child_tiles


def _upscale_tiles_bicubic(tiles, zoom_level, output_dir):
    """Bicubic upscale: each parent tile becomes 4 child tiles."""
    child_tiles = []
    for tx, ty, tile in tiles:
        upscaled = cv2.resize(tile, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        # Unsharp mask for sharpening
        blurred = cv2.GaussianBlur(upscaled, (0, 0), 3)
        upscaled = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

        for dy in range(2):
            for dx in range(2):
                child = upscaled[dy*512:(dy+1)*512, dx*512:(dx+1)*512]
                child_tiles.append((tx*2 + dx, ty*2 + dy, child))

    print(f"  Bicubic upscaled {len(tiles)} -> {len(child_tiles)} tiles")
    return child_tiles


# ---------------------------------------------------------------------------
# CDN upload
# ---------------------------------------------------------------------------

def upload_to_cdn(local_path, remote_key):
    """Upload a file to DO Spaces CDN. Returns the public URL."""
    endpoint = os.environ.get("SPACES_ENDPOINT", "")
    bucket = os.environ.get("SPACES_BUCKET", "")

    if not endpoint or not bucket:
        print(f"  SKIP upload (no Spaces credentials): {local_path}")
        return f"file://{os.path.abspath(local_path)}"

    content_types = {
        ".jpg": "image/jpeg", ".json": "application/json",
        ".png": "image/png", ".splat": "application/octet-stream",
    }
    ext = Path(local_path).suffix.lower()
    content_type = content_types.get(ext, "application/octet-stream")

    cmd = [
        "aws", "s3", "cp", local_path,
        f"s3://{bucket}/{remote_key}",
        "--endpoint-url", endpoint,
        "--acl", "public-read",
        "--content-type", content_type,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Upload failed: {result.stderr[:200]}")
        return f"file://{os.path.abspath(local_path)}"

    return f"{endpoint}/{bucket}/{remote_key}"


def upload_tile_pyramid(output_dir, job_id, total_levels):
    """Upload all tiles to CDN as jobs/{job_id}/ortho/{z}/{x}/{y}.jpg."""
    endpoint = os.environ.get("SPACES_ENDPOINT", "")
    bucket = os.environ.get("SPACES_BUCKET", "")
    if not endpoint or not bucket:
        print("  SKIP tile upload (no Spaces credentials)")
        return f"file://{os.path.abspath(output_dir)}/{{z}}/{{x}}/{{y}}.jpg"

    remote_prefix = f"jobs/{job_id}/ortho"

    # Use aws s3 sync for bulk upload
    cmd = [
        "aws", "s3", "sync", output_dir,
        f"s3://{bucket}/{remote_prefix}",
        "--endpoint-url", endpoint,
        "--acl", "public-read",
        "--content-type", "image/jpeg",
        "--exclude", "*.json",
        "--exclude", "*.py",
    ]
    print(f"  Uploading tile pyramid to CDN...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Sync failed: {result.stderr[:300]}")

    return f"{endpoint}/{bucket}/{remote_prefix}/{{z}}/{{x}}/{{y}}.jpg"


# ---------------------------------------------------------------------------
# Quality verification + Slack image previews
# ---------------------------------------------------------------------------

def verify_with_gemini(image_np, level_name, webhook_url, job_id):
    """Send image to Gemini Vision for quality check. Returns (passed, reason)."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print(f"  Gemini verify skipped (no API key)")
        return True, "skipped"

    try:
        import base64
        import io
        import urllib.request

        # Resize to 512x512 for Gemini
        small = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_AREA)
        _, jpg_buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(jpg_buf.tobytes()).decode()

        # Level-specific prompt
        if "Level 0" in level_name:
            prompt = (
                "Does this aerial orthomosaic look like a seamless stitched image? "
                "Check for: black gaps, visible seams, misaligned images, vertically "
                "flipped regions. Reply PASS or FAIL followed by a one-sentence reason."
            )
        else:
            prompt = (
                "This is a zoomed-in version of an aerial orthomosaic. Does it look "
                "like a believable zoom-in of the same scene? Check for: hallucinated "
                "objects, inconsistent textures, visible tile boundaries, incoherent "
                "features. Reply PASS or FAIL followed by a one-sentence reason."
            )

        payload = json.dumps({
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }]
        })

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_key}"
        )
        req = urllib.request.Request(
            url, data=payload.encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        passed = text.upper().startswith("PASS")
        status = "success" if passed else "error"
        print(f"  Gemini verify [{level_name}]: {text}")

        # Notify Slack with result
        emoji = "\u2705" if passed else "\u26a0\ufe0f"
        notify_slack(
            f"{emoji} Gemini verify [{level_name}]: {text}",
            webhook_url, job_id, status,
        )
        return passed, text

    except Exception as e:
        print(f"  Gemini verify failed ({e}), continuing")
        return True, f"error: {e}"


def notify_slack_with_image(message, image_path, webhook_url, job_id, status="info"):
    """Send Slack message with an image preview uploaded to CDN."""
    if not webhook_url:
        return

    import urllib.request

    # Upload preview image to CDN
    filename = os.path.basename(image_path)
    remote_key = f"jobs/{job_id}/ortho/previews/{filename}"
    image_url = upload_to_cdn(image_path, remote_key)

    emoji = {"info": "\U0001f504", "success": "\u2705", "error": "\u274c"}.get(status, "\U0001f504")

    # If CDN upload returned a file:// URL, fall back to text-only
    if image_url.startswith("file://"):
        notify_slack(message, webhook_url, job_id, status)
        return

    # Slack blocks payload with image
    payload = json.dumps({
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{emoji} *[{job_id[:8]}] ortho* \u2014 {message}",
                },
            },
            {
                "type": "image",
                "image_url": image_url,
                "alt_text": message,
            },
        ]
    })

    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        # Fall back to text-only
        notify_slack(message, webhook_url, job_id, status)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate 2D Orthomosaic Deep-Zoom Tiles")
    parser.add_argument("--odm_orthophoto", required=True,
                        help="Path to ODM orthophoto GeoTIFF")
    parser.add_argument("--scene_path", default="",
                        help="Scene dir with images/ (for FLUX captioning)")
    parser.add_argument("--output_dir", required=True, help="Output directory for tiles")
    parser.add_argument("--job_id", default="aukerman", help="Job ID for CDN paths")
    parser.add_argument("--max_enhance_level", type=int, default=2,
                        help="Max zoom level to FLUX-enhance (default 2)")
    parser.add_argument("--total_levels", type=int, default=5,
                        help="Total zoom levels 0-N (default 5)")
    parser.add_argument("--canvas_size", type=int, default=2048,
                        help="Base orthomosaic canvas size (default 2048)")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size (default 512)")
    parser.add_argument("--flux_strength", type=float, default=0.25,
                        help="FLUX Img2Img strength (default 0.25, lower=subtler)")
    parser.add_argument("--skip_flux", action="store_true",
                        help="Skip FLUX enhancement (bicubic only)")
    parser.add_argument("--slack_webhook_url", default="")
    args = parser.parse_args()

    if not args.slack_webhook_url:
        args.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Orthomosaic Deep-Zoom Tile Generator ===")
    print(f"ODM orthophoto: {args.odm_orthophoto}")
    print(f"Levels: 0-{args.total_levels - 1} (FLUX through level {args.max_enhance_level})")

    notify_slack("Ortho: starting tile generation...",
                 args.slack_webhook_url, args.job_id)

    # --- Step 1: Load ODM orthophoto as Level 0 ---
    print("\nStep 1: Loading ODM orthophoto...")
    notify_slack("Ortho: loading ODM orthophoto...",
                 args.slack_webhook_url, args.job_id)

    ortho_raw = cv2.imread(args.odm_orthophoto, cv2.IMREAD_COLOR)
    if ortho_raw is None:
        raise ValueError(f"Failed to load ODM orthophoto: {args.odm_orthophoto}")

    h, w = ortho_raw.shape[:2]
    print(f"  ODM orthophoto: {w}x{h}")

    # Resize maintaining aspect ratio, center-pad to square canvas
    scale = args.canvas_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    ortho_resized = cv2.resize(ortho_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ortho = np.zeros((args.canvas_size, args.canvas_size, 3), dtype=np.uint8)
    y_off = (args.canvas_size - new_h) // 2
    x_off = (args.canvas_size - new_w) // 2
    ortho[y_off:y_off + new_h, x_off:x_off + new_w] = ortho_resized

    print(f"  Placed {new_w}x{new_h} into {args.canvas_size}x{args.canvas_size} canvas "
          f"(offset {x_off},{y_off})")

    # Save full orthomosaic for reference
    ortho_path = os.path.join(args.output_dir, "orthomosaic.jpg")
    cv2.imwrite(ortho_path, ortho, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Save a downscaled preview for Slack
    preview_path = os.path.join(args.output_dir, "preview_level_0.jpg")
    preview = cv2.resize(ortho, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(preview_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Gemini quality check
    verify_with_gemini(ortho, "Level 0 orthomosaic",
                       args.slack_webhook_url, args.job_id)

    # Slack image update
    notify_slack_with_image(
        f"Level 0 ODM orthophoto ({args.canvas_size}x{args.canvas_size})",
        preview_path, args.slack_webhook_url, args.job_id)

    # Tile Level 0
    tiles_dir = os.path.join(args.output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    current_tiles = tile_image(ortho, args.tile_size)
    save_tiles(current_tiles, 0, tiles_dir)

    # Caption scene for FLUX prompts
    scene_images_dir = os.path.join(args.scene_path, "images") if args.scene_path else ""
    if scene_images_dir and os.path.isdir(scene_images_dir):
        prompt_base = caption_scene(scene_images_dir)
    else:
        prompt_base = "grass, trees, pavement, residential area"

    # --- Step 2: Progressive zoom levels ---
    image_dim = args.canvas_size
    for level in range(1, args.total_levels):
        image_dim *= 2
        print(f"\nStep 2.{level}: Zoom level {level} ({image_dim}x{image_dim})...")

        if level <= args.max_enhance_level and not args.skip_flux:
            notify_slack(f"Ortho: FLUX enhancing level {level} ({len(current_tiles)} tiles)...",
                         args.slack_webhook_url, args.job_id)
            child_tiles = enhance_tiles_with_flux(
                current_tiles, level, tiles_dir, scene_images_dir,
                prompt_base=prompt_base, skip_flux=False,
                flux_strength=args.flux_strength, tile_size=args.tile_size,
            )
        else:
            # Bicubic upscale + sharpen
            child_tiles = _upscale_tiles_bicubic(current_tiles, level, tiles_dir)

        save_tiles(child_tiles, level, tiles_dir)

        # Preview + quality check for this level
        level_image = reconstruct_image_from_tiles(child_tiles, args.tile_size)
        preview_path = os.path.join(args.output_dir, f"preview_level_{level}.jpg")
        preview = cv2.resize(level_image, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(preview_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 85])

        verify_with_gemini(level_image, f"Level {level} zoom",
                           args.slack_webhook_url, args.job_id)

        notify_slack_with_image(
            f"Level {level} complete ({image_dim}x{image_dim}, {len(child_tiles)} tiles)",
            preview_path, args.slack_webhook_url, args.job_id)

        current_tiles = child_tiles

    # --- Step 3: Upload ---
    print("\nStep 3: Uploading tile pyramid...")
    notify_slack("Ortho: uploading tiles to CDN...",
                 args.slack_webhook_url, args.job_id)

    tile_url_template = upload_tile_pyramid(tiles_dir, args.job_id, args.total_levels)

    # Count total tiles
    total_tiles = 0
    for level in range(args.total_levels):
        level_dir = os.path.join(tiles_dir, str(level))
        if os.path.exists(level_dir):
            for root, dirs, files in os.walk(level_dir):
                total_tiles += len([f for f in files if f.endswith(".jpg")])

    # Final image dimensions
    final_dim = args.canvas_size * (2 ** (args.total_levels - 1))

    # --- Step 4: Generate manifest ---
    # Check if splat manifest exists for cross-linking
    splat_manifest_url = None
    endpoint = os.environ.get("SPACES_ENDPOINT", "")
    bucket = os.environ.get("SPACES_BUCKET", "")
    if endpoint and bucket:
        splat_manifest_url = f"{endpoint}/{bucket}/demo/{args.job_id}/manifest.json"

    manifest = {
        "viewer_mode": "ortho",
        "tile_url_template": tile_url_template,
        "tile_size": args.tile_size,
        "min_zoom": 0,
        "max_zoom": args.total_levels - 1,
        "max_enhanced_zoom": min(args.max_enhance_level, args.total_levels - 1),
        "image_width": final_dim,
        "image_height": final_dim,
        "scene_caption": prompt_base,
        "total_tiles": total_tiles,
    }
    if splat_manifest_url:
        manifest["splat_manifest_url"] = splat_manifest_url

    manifest_path = os.path.join(args.output_dir, "ortho_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Upload manifest
    if endpoint and bucket:
        upload_to_cdn(manifest_path, f"jobs/{args.job_id}/ortho/ortho_manifest.json")
        # Also upload to demo path
        upload_to_cdn(manifest_path, f"demo/{args.job_id}/ortho_manifest.json")

    notify_slack(
        f"Ortho complete! {total_tiles} tiles, {args.total_levels} levels, "
        f"FLUX through level {args.max_enhance_level}",
        args.slack_webhook_url, args.job_id, "success",
    )

    print(f"\n=== Orthomosaic tile generation complete ===")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Levels: 0-{args.total_levels - 1}")
    print(f"  FLUX enhanced: 0-{args.max_enhance_level}")
    print(f"  Final resolution: {final_dim}x{final_dim}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
