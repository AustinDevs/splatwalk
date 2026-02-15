#!/usr/bin/env python3
"""
Gemini Render Scoring: Score splat renders at multiple altitudes.

Renders the Gaussian splat at several altitudes (aerial → ground level),
sends snapshots to Gemini for quality scoring, uploads results to Spaces,
and posts progress to Slack.

Returns a JSON summary with per-altitude scores.

Usage:
    python score_renders.py \
        --model_path /root/output/aukerman-v12/model \
        --scene_path /root/output/scene \
        --output_dir /root/output/scores \
        --gemini_api_key AIzaSy... \
        --slack_webhook_url https://hooks.slack.com/... \
        --spaces_key DO801... \
        --spaces_secret YUGh... \
        --spaces_bucket splatwalk \
        --spaces_endpoint https://nyc3.digitaloceanspaces.com \
        --viewer_base_url "https://splatwalk.com/view"
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from PIL import Image
    from plyfile import PlyData
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData


# ---------------------------------------------------------------------------
# Slack + Spaces helpers
# ---------------------------------------------------------------------------

def notify_slack(message, webhook_url, status="info", blocks=None):
    """Send a Slack notification."""
    if not webhook_url:
        return
    import urllib.request
    emoji = {"info": "\U0001f504", "success": "\u2705", "error": "\u274c",
             "score": "\U0001f3af"}.get(status, "\U0001f504")
    payload = {"text": f"{emoji} *[splatwalk scoring]* \u2014 {message}"}
    if blocks:
        payload["blocks"] = blocks
    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  Slack notification failed: {e}")


def upload_to_spaces(local_path, remote_key, spaces_key, spaces_secret,
                     spaces_bucket, spaces_endpoint, content_type="image/jpeg"):
    """Upload a file to DO Spaces and return public URL."""
    import subprocess
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = spaces_key
    env["AWS_SECRET_ACCESS_KEY"] = spaces_secret
    cmd = [
        "aws", "s3", "cp", local_path,
        f"s3://{spaces_bucket}/{remote_key}",
        "--endpoint-url", spaces_endpoint,
        "--acl", "public-read",
        "--content-type", content_type,
    ]
    subprocess.run(cmd, env=env, capture_output=True, check=True)
    return f"{spaces_endpoint}/{spaces_bucket}/{remote_key}"


# ---------------------------------------------------------------------------
# Render at multiple altitudes
# ---------------------------------------------------------------------------

ALTITUDE_CONFIGS = [
    {"name": "aerial_high",    "fraction": 0.90, "label": "Aerial (high orbit)"},
    {"name": "aerial_mid",     "fraction": 0.60, "label": "Aerial (mid altitude)"},
    {"name": "tree_canopy",    "fraction": 0.30, "label": "Tree canopy level"},
    {"name": "low_altitude",   "fraction": 0.12, "label": "Low altitude (rooftop)"},
    {"name": "eye_level",      "fraction": 0.04, "label": "Eye level (~5ft)"},
    {"name": "ground_level",   "fraction": 0.02, "label": "Ground level (~3ft)"},
]


def load_model_and_scene(model_path, scene_path):
    """Load the Gaussian splat model and camera poses."""
    # Find PLY
    ply_path = None
    for candidate in [
        Path(model_path) / "point_cloud" / "iteration_10000" / "point_cloud.ply",
        Path(model_path) / "point_cloud" / "iteration_7000" / "point_cloud.ply",
        Path(model_path) / "point_cloud" / "iteration_3000" / "point_cloud.ply",
        Path(model_path) / "point_cloud" / "iteration_2000" / "point_cloud.ply",
    ]:
        if candidate.exists():
            ply_path = candidate
            break

    if ply_path is None:
        for p in sorted(Path(model_path).rglob("point_cloud.ply")):
            ply_path = p
            break

    if ply_path is None:
        raise FileNotFoundError(f"No point_cloud.ply found in {model_path}")

    ply_data = PlyData.read(str(ply_path))
    vertex = ply_data["vertex"]
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

    ground_z = np.percentile(positions[:, 2], 5)
    drone_z = np.percentile(positions[:, 2], 95)

    # Load camera poses
    poses = _load_camera_poses(scene_path)

    return positions, ply_path, ground_z, drone_z, poses


def _load_camera_poses(scene_path):
    """Load camera poses from COLMAP sparse reconstruction."""
    import struct
    from scipy.spatial.transform import Rotation

    for sparse_dir in sorted(Path(scene_path).glob("sparse_*/0")):
        img_file = sparse_dir / "images.bin"
        if img_file.exists():
            return _load_poses_bin(img_file)

    for sparse_dir in [Path(scene_path) / "sparse" / "0", Path(scene_path) / "sparse"]:
        img_file = sparse_dir / "images.bin"
        if img_file.exists():
            return _load_poses_bin(img_file)

    # Try text format
    for sparse_dir in sorted(Path(scene_path).glob("sparse_*/0")) + [Path(scene_path) / "sparse" / "0"]:
        img_file = sparse_dir / "images.txt"
        if img_file.exists():
            return _load_poses_txt(img_file)

    raise FileNotFoundError(f"No COLMAP sparse data found in {scene_path}")


def _load_poses_bin(images_path):
    import struct
    from scipy.spatial.transform import Rotation
    poses = []
    with open(str(images_path), "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode())
            image_name = "".join(name_chars)
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            center = -R.T @ t
            poses.append({"image_name": image_name, "center": center,
                          "rotation": R, "translation": t})
    return poses


def _load_poses_txt(images_path):
    from scipy.spatial.transform import Rotation
    poses = []
    with open(str(images_path), "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            center = -R.T @ t
            poses.append({"image_name": parts[9], "center": center,
                          "rotation": R, "translation": t})
            i += 2
        else:
            i += 1
    return poses


def render_at_altitude(model_path, poses, ground_z, drone_z, altitude_fraction,
                       output_path, scene_path):
    """Render the splat at a given altitude fraction using the GS rasterizer."""
    import torch
    from PIL import Image as PILImage
    from scipy.spatial.transform import Rotation, Slerp

    sys.path.insert(0, "/mnt/splatwalk/InstantSplat")
    from gaussian_renderer import render
    from scene import GaussianModel
    from scene.cameras import Camera
    from argparse import Namespace

    gaussians = GaussianModel(3)
    ckpt_dirs = sorted(Path(model_path).glob("point_cloud/iteration_*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {model_path}")
    latest_ckpt = ckpt_dirs[-1] / "point_cloud.ply"
    gaussians.load_ply(str(latest_ckpt))

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Pick the pose with the median XY position (most central view)
    centers = np.array([p["center"] for p in poses])
    median_xy = np.median(centers[:, :2], axis=0)
    dists = np.linalg.norm(centers[:, :2] - median_xy, axis=1)
    best_idx = np.argmin(dists)
    pose = poses[best_idx]

    # Interpolate altitude
    center = pose["center"].copy()
    new_z = ground_z + altitude_fraction * (drone_z - ground_z)
    center[2] = new_z

    # Interpolate rotation: from looking down toward horizon
    original_rot = Rotation.from_matrix(pose["rotation"])
    euler = original_rot.as_euler("zyx", degrees=True)
    horizon_euler = euler.copy()
    horizon_euler[1] = 0  # Zero pitch = looking at horizon
    horizon_rot = Rotation.from_euler("zyx", horizon_euler, degrees=True)

    t = 1.0 - altitude_fraction  # 0 at drone, 1 at ground
    key_rots = Rotation.concatenate([original_rot, horizon_rot])
    slerp = Slerp([0, 1], key_rots)
    interpolated_rot = slerp(min(t, 0.95))  # Don't go fully horizontal

    R = interpolated_rot.as_matrix()
    t_vec = -R @ center

    camera = Camera(
        colmap_id=0,
        R=R,
        T=t_vec,
        FoVx=1.2,  # ~69 degrees, wider FoV for ground views
        FoVy=1.2,
        image=torch.zeros(3, 768, 768),
        gt_alpha_mask=None,
        image_name=f"score_{altitude_fraction:.3f}",
        uid=0,
    )

    from scipy.spatial.transform import Rotation as SciRotation
    quat = SciRotation.from_matrix(R).as_quat()
    camera_pose = torch.tensor(
        [quat[3], quat[0], quat[1], quat[2], t_vec[0], t_vec[1], t_vec[2]],
        dtype=torch.float32,
    ).cuda()
    camera.camera_pose = camera_pose

    rendering = render(camera, gaussians, pipe, bg_color, camera_pose=camera_pose)
    image = rendering["render"]

    img_np = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_pil = PILImage.fromarray(img_np)
    img_pil.save(output_path, "JPEG", quality=95)

    del gaussians
    torch.cuda.empty_cache()

    return img_pil


def render_all_altitudes(model_path, scene_path, output_dir):
    """Render at all configured altitudes and return image paths."""
    positions, ply_path, ground_z, drone_z, poses = load_model_and_scene(model_path, scene_path)
    print(f"Loaded model: {len(positions)} Gaussians, ground_z={ground_z:.3f}, drone_z={drone_z:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    rendered = []

    for config in ALTITUDE_CONFIGS:
        output_path = os.path.join(output_dir, f"{config['name']}.jpg")
        print(f"  Rendering {config['label']} (fraction={config['fraction']})...")
        try:
            render_at_altitude(
                model_path, poses, ground_z, drone_z,
                config["fraction"], output_path, scene_path
            )
            rendered.append({**config, "path": output_path})
            print(f"    Saved: {output_path}")
        except Exception as e:
            print(f"    FAILED: {e}")
            rendered.append({**config, "path": None, "error": str(e)})

    return rendered


# ---------------------------------------------------------------------------
# Gemini scoring
# ---------------------------------------------------------------------------

SCORING_PROMPT = """You are evaluating renders from a 3D Gaussian Splat reconstruction of a real property from aerial drone footage.

Score this render on a scale of 1-10 for photorealism and visual quality. Consider:
- Clarity and sharpness of structures (buildings, roads, vehicles)
- Vegetation quality (trees, grass, landscaping)
- Absence of visual artifacts (floaters, noise, blurring, ghosting)
- Overall photorealism (would this pass as a real photograph?)
- Geometric consistency (do structures look correct?)

The view is from: {altitude_label}

Respond in this EXACT JSON format and nothing else:
{{
  "score": <number 1-10>,
  "strengths": "<brief list of what looks good>",
  "weaknesses": "<brief list of main issues>",
  "suggestions": "<actionable improvements>"
}}"""


def score_with_gemini(image_path, altitude_label, gemini_key):
    """Score a single render using Gemini 2.5 Flash."""
    try:
        from google import genai
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
        from google import genai

    client = genai.Client(api_key=gemini_key)

    # Load and prepare image
    img = Image.open(image_path).convert("RGB")
    max_dim = max(img.size)
    if max_dim > 1024:
        scale = 1024 / max_dim
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    img_bytes = buf.getvalue()

    prompt = SCORING_PROMPT.format(altitude_label=altitude_label)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        ],
    )

    text = response.text.strip()
    # Extract JSON from response (handle markdown code blocks)
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        result = json.loads(text)
        return result
    except json.JSONDecodeError:
        # Try to extract score from text
        import re
        score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
        if score_match:
            return {"score": float(score_match.group(1)),
                    "strengths": "parse error", "weaknesses": text[:200],
                    "suggestions": ""}
        print(f"  Failed to parse Gemini response: {text[:300]}")
        return {"score": 0, "strengths": "", "weaknesses": "Failed to parse",
                "suggestions": ""}


def score_all_renders(rendered, gemini_key):
    """Score all rendered altitude images with Gemini."""
    scores = []
    for r in rendered:
        if r.get("path") is None:
            scores.append({**r, "score": 0, "gemini": {"score": 0, "error": r.get("error", "render failed")}})
            continue

        print(f"  Scoring {r['label']}...")
        try:
            result = score_with_gemini(r["path"], r["label"], gemini_key)
            score = float(result.get("score", 0))
            print(f"    Score: {score}/10 - {result.get('strengths', '')[:80]}")
            scores.append({**r, "score": score, "gemini": result})
        except Exception as e:
            print(f"    Scoring failed: {e}")
            scores.append({**r, "score": 0, "gemini": {"score": 0, "error": str(e)}})

    return scores


# ---------------------------------------------------------------------------
# Upload + Slack reporting
# ---------------------------------------------------------------------------

def upload_renders_and_report(scores, args):
    """Upload renders to Spaces and send a Slack report."""
    timestamp = int(time.time())
    image_urls = {}

    for s in scores:
        if s.get("path") and os.path.exists(s["path"]):
            try:
                remote_key = f"scenes/aukerman/scores/{timestamp}/{s['name']}.jpg"
                url = upload_to_spaces(
                    s["path"], remote_key,
                    args.spaces_key, args.spaces_secret,
                    args.spaces_bucket, args.spaces_endpoint,
                )
                image_urls[s["name"]] = url
                print(f"  Uploaded {s['name']}: {url}")
            except Exception as e:
                print(f"  Upload failed for {s['name']}: {e}")

    # Build Slack message
    avg_score = np.mean([s["score"] for s in scores if s["score"] > 0]) if scores else 0
    ground_scores = [s for s in scores if s["fraction"] <= 0.05]
    ground_avg = np.mean([s["score"] for s in ground_scores if s["score"] > 0]) if ground_scores else 0

    lines = [
        f"*Render Scoring Results* (avg: {avg_score:.1f}/10, ground: {ground_avg:.1f}/10)",
        "",
    ]
    for s in scores:
        emoji = "\u2705" if s["score"] >= 9 else "\u26a0\ufe0f" if s["score"] >= 7 else "\u274c"
        url = image_urls.get(s["name"], "")
        link = f"<{url}|view>" if url else ""
        gemini = s.get("gemini", {})
        lines.append(
            f"{emoji} *{s['label']}*: {s['score']}/10 {link}"
        )
        if gemini.get("weaknesses"):
            lines.append(f"    Issues: {gemini['weaknesses'][:120]}")

    if args.viewer_base_url:
        lines.append("")
        lines.append(f"*Viewer*: <{args.viewer_base_url}|Open 3D Viewer>")

    lines.append("")
    if ground_avg >= 9:
        lines.append("\u2705 *Ground-level target reached (9/10)!*")
    else:
        lines.append(f"\u274c Ground-level at {ground_avg:.1f}/10 — needs improvement")
        # Add specific suggestions
        for s in ground_scores:
            gemini = s.get("gemini", {})
            if gemini.get("suggestions"):
                lines.append(f"    \u27a1\ufe0f {gemini['suggestions'][:150]}")

    message = "\n".join(lines)
    notify_slack(message, args.slack_webhook_url, "score")

    # Also save full JSON report
    report = {
        "timestamp": timestamp,
        "average_score": round(avg_score, 1),
        "ground_average": round(ground_avg, 1),
        "scores": [{
            "name": s["name"],
            "label": s["label"],
            "fraction": s["fraction"],
            "score": s["score"],
            "image_url": image_urls.get(s["name"], ""),
            "gemini": s.get("gemini", {}),
        } for s in scores],
    }

    report_path = os.path.join(args.output_dir, "score_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Upload report
    try:
        remote_key = f"scenes/aukerman/scores/{timestamp}/report.json"
        upload_to_spaces(
            report_path, remote_key,
            args.spaces_key, args.spaces_secret,
            args.spaces_bucket, args.spaces_endpoint,
            content_type="application/json",
        )
    except Exception:
        pass

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score splat renders with Gemini at multiple altitudes")
    parser.add_argument("--model_path", required=True, help="Path to trained Gaussian splat model")
    parser.add_argument("--scene_path", required=True, help="Path to scene directory with COLMAP data")
    parser.add_argument("--output_dir", required=True, help="Output directory for renders and scores")
    parser.add_argument("--gemini_api_key", required=True, help="Gemini API key")
    parser.add_argument("--slack_webhook_url", default="", help="Slack webhook URL")
    parser.add_argument("--spaces_key", default="", help="DO Spaces access key")
    parser.add_argument("--spaces_secret", default="", help="DO Spaces secret key")
    parser.add_argument("--spaces_bucket", default="splatwalk", help="DO Spaces bucket")
    parser.add_argument("--spaces_endpoint", default="https://nyc3.digitaloceanspaces.com")
    parser.add_argument("--viewer_base_url", default="", help="URL to the web viewer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    notify_slack("Starting render scoring at 6 altitudes...", args.slack_webhook_url, "info")

    # 1. Render at all altitudes
    print("=== Rendering at multiple altitudes ===")
    rendered = render_all_altitudes(args.model_path, args.scene_path, args.output_dir)

    # 2. Score with Gemini
    print("\n=== Scoring with Gemini ===")
    scores = score_all_renders(rendered, args.gemini_api_key)

    # 3. Upload and report
    print("\n=== Uploading and reporting ===")
    report = upload_renders_and_report(scores, args)

    # 4. Print summary
    print(f"\n{'='*50}")
    print(f"SCORING SUMMARY")
    print(f"{'='*50}")
    for s in scores:
        print(f"  {s['label']:30s} {s['score']:4.1f}/10")
    print(f"  {'─'*40}")
    print(f"  {'Average':30s} {report['average_score']:4.1f}/10")
    print(f"  {'Ground average':30s} {report['ground_average']:4.1f}/10")
    print(f"{'='*50}")

    # Return the ground average score as exit code hint
    # 0 = success (score >= 9), 1 = needs improvement
    ground_avg = report["ground_average"]
    if ground_avg >= 9.0:
        print("\nGROUND-LEVEL TARGET REACHED (9/10)!")
        sys.exit(0)
    else:
        print(f"\nGround level at {ground_avg:.1f}/10 — needs improvement")
        sys.exit(1)


if __name__ == "__main__":
    main()
