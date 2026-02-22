#!/usr/bin/env python3
"""
Gemini Quality Scoring Loop for Ground GLB

Renders the ground GLB from multiple viewpoints, scores with Gemini 2.5 Flash,
sends snapshots to Slack, and iterates on build_game_scene.py parameters until
the scene scores 9/10 (or max iterations reached).

Usage:
    python gemini_score_scene.py \
        --glb_path /root/output/scene.glb \
        --orthophoto /root/output/odm/odm_orthophoto/odm_orthophoto.tif \
        --scene_transform /root/output/scene_transform.json \
        --build_args '--orthophoto ... --dsm ... --output ...' \
        --target_score 9 \
        --max_iterations 3 \
        --gemini_api_key $GEMINI_API_KEY \
        --slack_webhook_url $SLACK_WEBHOOK_URL \
        --job_id demo-123
"""

import argparse
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Headless GLB rendering with pyrender + EGL
# ---------------------------------------------------------------------------

def render_glb_views(glb_path, scene_bounds, output_dir, resolution=1024):
    """Render 5 views of the GLB: 1 top-down nadir + 4 ground-level cardinal directions.

    Uses pyrender with EGL backend for headless GPU rendering.
    Returns list of JPEG paths.
    """
    # Must set these BEFORE any OpenGL/pyglet imports
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # Prevent pyglet from creating a shadow window (fails without X11 display)
    try:
        import pyglet
        pyglet.options['shadow_window'] = False
    except Exception:
        pass

    import trimesh
    import pyrender

    os.makedirs(output_dir, exist_ok=True)

    # Load GLB
    scene_trimesh = trimesh.load(glb_path)
    scene = pyrender.Scene.from_trimesh_scene(scene_trimesh)

    # Scene center and dimensions
    center = np.array(scene_bounds.get("center", [0, 0, 0]), dtype=np.float32)
    size = np.array(scene_bounds.get("size", [100, 100, 20]), dtype=np.float32)
    ground_z = scene_bounds.get("ground_z", center[2] - size[2] / 2)
    max_extent = max(size[0], size[1])

    # Camera setup — znear/zfar must cover the scene (default zfar=100 is way too small)
    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 3.0, aspectRatio=1.0,
        znear=max_extent * 0.001,
        zfar=max_extent * 5.0,
    )
    renderer = pyrender.OffscreenRenderer(resolution, resolution)

    # Add ambient + directional light so the scene is visible
    scene.ambient_light = np.array([0.3, 0.3, 0.3, 1.0])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_pose = np.eye(4)
    light_pose[:3, 2] = [0, -0.5, -1]  # sun from above-front
    scene.add(light, pose=light_pose)

    print(f"  Scene: center={center.tolist()}, size={size.tolist()}, "
          f"ground_z={ground_z:.1f}, max_extent={max_extent:.0f}")

    views = []

    # --- View 0: Top-down nadir ---
    cam_pos = np.array([center[0], center[1], center[2] + max_extent * 0.8])
    look_at = np.array([center[0], center[1], ground_z])
    cam_node = _add_camera_looking_at(scene, camera, cam_pos, look_at, up=[0, 1, 0])

    color, _ = renderer.render(scene)
    path = os.path.join(output_dir, "view_nadir.jpg")
    _save_render(color, path)
    views.append({"path": path, "label": "nadir (top-down)"})
    scene.remove_node(cam_node)

    # --- Views 1-4: Ground-level cardinal directions ---
    eye_height = ground_z + 3.0  # ~3m above ground (person height)
    distance = max_extent * 0.3  # look from edge toward center
    cardinals = [
        ("north", [center[0], center[1] - distance, eye_height], [center[0], center[1], eye_height]),
        ("south", [center[0], center[1] + distance, eye_height], [center[0], center[1], eye_height]),
        ("east",  [center[0] - distance, center[1], eye_height], [center[0], center[1], eye_height]),
        ("west",  [center[0] + distance, center[1], eye_height], [center[0], center[1], eye_height]),
    ]

    for label, pos, target in cardinals:
        cam_pos = np.array(pos, dtype=np.float32)
        look_at = np.array(target, dtype=np.float32)
        cam_node = _add_camera_looking_at(scene, camera, cam_pos, look_at, up=[0, 0, 1])

        color, _ = renderer.render(scene)
        path = os.path.join(output_dir, f"view_{label}.jpg")
        _save_render(color, path)
        views.append({"path": path, "label": f"{label} (ground-level)"})
        scene.remove_node(cam_node)

    renderer.delete()
    print(f"  Rendered {len(views)} views to {output_dir}")
    return views


def _add_camera_looking_at(scene, camera, eye, target, up=None):
    """Add camera to pyrender scene looking from eye toward target."""
    import pyrender

    if up is None:
        up = [0, 0, 1]

    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    cam_up = np.cross(right, forward)

    # Build camera-to-world matrix (OpenGL convention: -Z forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = cam_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye

    return scene.add(camera, pose=pose)


def _save_render(color_array, path, quality=90):
    """Save render as JPEG."""
    from PIL import Image
    img = Image.fromarray(color_array).convert("RGB")
    img.save(path, "JPEG", quality=quality)


# ---------------------------------------------------------------------------
# Gemini scoring
# ---------------------------------------------------------------------------

def score_with_gemini(views, orthophoto_path, api_key):
    """Send rendered views + orthophoto to Gemini 2.5 Flash for quality scoring.

    Returns dict: {overall_score, per_view_scores, issues[], suggestions[]}
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    # Encode images
    contents = []

    # Add orthophoto as reference (resize to max 1024px for API)
    from PIL import Image
    ortho = Image.open(orthophoto_path).convert("RGB")
    ortho.thumbnail((1024, 1024))
    ortho_path_tmp = orthophoto_path + ".thumb.jpg"
    ortho.save(ortho_path_tmp, "JPEG", quality=85)
    with open(ortho_path_tmp, "rb") as f:
        ortho_b64 = base64.b64encode(f.read()).decode()
    os.remove(ortho_path_tmp)
    contents.append({"inline_data": {"mime_type": "image/jpeg", "data": ortho_b64}})
    contents.append("Above: Original aerial orthophoto (ground truth reference).\n")

    # Add each rendered view
    for view in views:
        with open(view["path"], "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        contents.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
        contents.append(f"Above: Rendered 3D scene — {view['label']}\n")

    # Scoring prompt
    contents.append(
        "You are evaluating a procedurally generated 3D scene (ground-level GLB) against "
        "the original aerial orthophoto. Rate the scene 1-10 for realism.\n\n"
        "Consider:\n"
        "- Do trees, roads, water, buildings match what's visible in the aerial photo?\n"
        "- Are tree heights and canopy sizes proportional?\n"
        "- Is the terrain shape realistic?\n"
        "- Are materials (grass, concrete, asphalt) correctly assigned?\n"
        "- Is vegetation density appropriate?\n\n"
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "overall_score": <1-10>,\n'
        '  "per_view_scores": {"nadir": <1-10>, "north": <1-10>, "south": <1-10>, "east": <1-10>, "west": <1-10>},\n'
        '  "issues": ["issue 1", "issue 2", ...],\n'
        '  "suggestions": ["suggestion 1", "suggestion 2", ...]\n'
        "}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        text = response.text.strip()
        # Extract JSON
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        print(f"  Gemini score: {result.get('overall_score', '?')}/10")
        if result.get("issues"):
            for issue in result["issues"][:3]:
                print(f"    Issue: {issue}")
        return result
    except Exception as e:
        print(f"  Gemini scoring failed: {e}")
        return {"overall_score": 5, "per_view_scores": {}, "issues": [str(e)], "suggestions": []}


# ---------------------------------------------------------------------------
# Slack notifications with images
# ---------------------------------------------------------------------------

def upload_image_to_cdn(local_path, remote_key):
    """Upload image to DO Spaces CDN. Returns URL."""
    endpoint = os.environ.get("SPACES_ENDPOINT", "")
    bucket = os.environ.get("SPACES_BUCKET", "")
    if not endpoint or not bucket:
        return f"file://{os.path.abspath(local_path)}"

    cmd = [
        "aws", "s3", "cp", local_path,
        f"s3://{bucket}/{remote_key}",
        "--endpoint-url", endpoint,
        "--acl", "public-read",
        "--content-type", "image/jpeg",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"file://{os.path.abspath(local_path)}"
    return f"{endpoint}/{bucket}/{remote_key}"


def send_to_slack(views, scores, iteration, webhook_url, job_id):
    """Upload snapshots to CDN, send Slack message with score + image links."""
    if not webhook_url:
        return

    import urllib.request

    overall = scores.get("overall_score", "?")
    issues = scores.get("issues", [])

    # Upload renders to CDN
    image_urls = []
    for view in views:
        remote_key = f"jobs/{job_id}/quality/iter{iteration}/{Path(view['path']).name}"
        url = upload_image_to_cdn(view["path"], remote_key)
        image_urls.append({"label": view["label"], "url": url})

    # Build Slack message
    emoji = ":white_check_mark:" if overall >= 9 else ":mag:"
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{emoji} *[{job_id[:8]}] Quality Loop — Iteration {iteration}*\n"
                        f"Score: *{overall}/10*"
            }
        }
    ]

    if issues:
        issue_text = "\n".join(f"- {i}" for i in issues[:5])
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Issues:*\n{issue_text}"}
        })

    # Add image links
    if image_urls:
        links = " | ".join(f"<{img['url']}|{img['label']}>" for img in image_urls)
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Renders:* {links}"}
        })

    payload = json.dumps({"blocks": blocks})
    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  Slack notification failed: {e}")


# ---------------------------------------------------------------------------
# Feedback → parameter mapping
# ---------------------------------------------------------------------------

def map_feedback_to_params(scores, current_params):
    """Map Gemini feedback to adjusted build_game_scene.py parameters.

    Returns dict of updated CLI params for next iteration.
    """
    params = dict(current_params)
    issues = [i.lower() for i in scores.get("issues", [])]
    suggestions = [s.lower() for s in scores.get("suggestions", [])]
    all_feedback = " ".join(issues + suggestions)

    # Tree height issues
    if any(kw in all_feedback for kw in ["tree", "tall", "short", "height", "canopy"]):
        if any(kw in all_feedback for kw in ["too tall", "too large", "oversized", "too big"]):
            params["tree_height_scale"] = params.get("tree_height_scale", 1.0) * 0.75
            params["canopy_diameter_scale"] = params.get("canopy_diameter_scale", 1.0) * 0.8
        elif any(kw in all_feedback for kw in ["too small", "too short", "stubby", "thin"]):
            params["tree_height_scale"] = params.get("tree_height_scale", 1.0) * 1.3
            params["canopy_diameter_scale"] = params.get("canopy_diameter_scale", 1.0) * 1.2
        elif any(kw in all_feedback for kw in ["missing tree", "no tree", "fewer tree"]):
            params["sam2_points_per_side"] = min(64, params.get("sam2_points_per_side", 32) + 8)

    # Terrain issues
    if any(kw in all_feedback for kw in ["terrain", "bumpy", "rough", "jagged", "smooth"]):
        if any(kw in all_feedback for kw in ["too smooth", "flat", "no elevation"]):
            params["dsm_smooth_sigma"] = max(0.5, params.get("dsm_smooth_sigma", 1.5) * 0.6)
        elif any(kw in all_feedback for kw in ["bumpy", "jagged", "rough", "noisy"]):
            params["dsm_smooth_sigma"] = min(5.0, params.get("dsm_smooth_sigma", 1.5) * 1.5)

    # Vegetation density
    if any(kw in all_feedback for kw in ["sparse", "empty", "bare", "more vegetation", "more grass"]):
        params["grass_density"] = min(500, params.get("grass_density", 200) + 100)
    elif any(kw in all_feedback for kw in ["cluttered", "too dense", "too much", "overgrown"]):
        params["grass_density"] = max(50, params.get("grass_density", 200) - 80)

    # Material issues → re-segment with finer grid
    if any(kw in all_feedback for kw in ["wrong material", "misclassified", "material"]):
        params["sam2_points_per_side"] = min(64, params.get("sam2_points_per_side", 32) + 8)

    return params


def params_to_cli_args(params):
    """Convert params dict to CLI argument string for build_game_scene.py."""
    args = []
    param_map = {
        "tree_height_scale": "--tree_height_scale",
        "canopy_diameter_scale": "--canopy_diameter_scale",
        "grass_density": "--grass_density",
        "dsm_smooth_sigma": "--dsm_smooth_sigma",
        "sam2_points_per_side": "--sam2_points_per_side",
    }
    for key, flag in param_map.items():
        if key in params:
            args.extend([flag, str(params[key])])
    return args


# ---------------------------------------------------------------------------
# Determine which steps need re-running based on changed params
# ---------------------------------------------------------------------------

def changed_steps(old_params, new_params):
    """Determine which build_game_scene steps to re-run based on param changes."""
    steps = set()

    if old_params.get("dsm_smooth_sigma") != new_params.get("dsm_smooth_sigma"):
        steps.update(["terrain", "assets", "export"])

    if old_params.get("sam2_points_per_side") != new_params.get("sam2_points_per_side"):
        steps.update(["segment", "materials", "assets", "export"])

    if (old_params.get("tree_height_scale") != new_params.get("tree_height_scale") or
            old_params.get("canopy_diameter_scale") != new_params.get("canopy_diameter_scale") or
            old_params.get("grass_density") != new_params.get("grass_density")):
        steps.update(["assets", "export"])

    return ",".join(sorted(steps)) if steps else "assets,export"


# ---------------------------------------------------------------------------
# Main quality loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemini Quality Scoring Loop")
    parser.add_argument("--glb_path", required=True, help="Path to scene.glb")
    parser.add_argument("--orthophoto", required=True, help="Path to ODM orthophoto")
    parser.add_argument("--scene_transform", default="", help="scene_transform.json path")
    parser.add_argument("--build_args", required=True,
                        help="Base CLI args for build_game_scene.py (quoted string)")
    parser.add_argument("--target_score", type=int, default=9, help="Target quality score (default 9)")
    parser.add_argument("--max_iterations", type=int, default=3, help="Max quality loop iterations")
    parser.add_argument("--gemini_api_key", default="", help="Gemini API key")
    parser.add_argument("--slack_webhook_url", default="", help="Slack webhook URL")
    parser.add_argument("--job_id", default="unknown", help="Job ID for CDN paths")
    args = parser.parse_args()

    if not args.gemini_api_key:
        args.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not args.slack_webhook_url:
        args.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")

    if not args.gemini_api_key:
        print("WARNING: No Gemini API key — skipping quality loop")
        return args.glb_path

    # Load scene bounds
    scene_bounds = {"center": [0, 0, 0], "size": [100, 100, 20], "ground_z": -5}
    if args.scene_transform and os.path.exists(args.scene_transform):
        with open(args.scene_transform) as f:
            data = json.load(f)
        scene_bounds = data.get("scene_bounds", scene_bounds)

    # Initial params (defaults)
    current_params = {
        "tree_height_scale": 1.0,
        "canopy_diameter_scale": 1.0,
        "grass_density": 200,
        "dsm_smooth_sigma": 1.5,
        "sam2_points_per_side": 32,
    }

    best_glb = args.glb_path
    best_score = 0

    print("=" * 60)
    print("Gemini Quality Scoring Loop")
    print(f"  Target: {args.target_score}/10, Max iterations: {args.max_iterations}")
    print("=" * 60)

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{args.max_iterations} ---")

        # 1. Render views
        render_dir = os.path.join(os.path.dirname(args.glb_path),
                                  f"quality_renders_iter{iteration}")
        views = render_glb_views(args.glb_path, scene_bounds, render_dir)

        # 2. Score with Gemini
        scores = score_with_gemini(views, args.orthophoto, args.gemini_api_key)
        overall = scores.get("overall_score", 0)

        # 3. Send to Slack
        send_to_slack(views, scores, iteration, args.slack_webhook_url, args.job_id)

        # Track best
        if overall > best_score:
            best_score = overall
            best_glb = args.glb_path
            # Keep a copy of the best GLB
            best_copy = args.glb_path.replace(".glb", f"_best_iter{iteration}.glb")
            shutil.copy2(args.glb_path, best_copy)
            best_glb = best_copy

        print(f"  Score: {overall}/10 (best: {best_score}/10, target: {args.target_score}/10)")

        # 4. Check if target met
        if overall >= args.target_score:
            print(f"  Target score {args.target_score} achieved!")
            best_glb = args.glb_path  # Current is best
            break

        # 5. If not last iteration, adjust params and rebuild
        if iteration < args.max_iterations:
            old_params = dict(current_params)
            current_params = map_feedback_to_params(scores, current_params)

            steps = changed_steps(old_params, current_params)
            print(f"  Adjusted params: {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in current_params.items()})}")
            print(f"  Re-running steps: {steps}")

            # Rebuild GLB with adjusted params
            build_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_game_scene.py"),
            ]
            build_cmd.extend(shlex.split(args.build_args))
            build_cmd.extend(["--step", steps])
            build_cmd.extend(params_to_cli_args(current_params))

            print(f"  Running: {' '.join(build_cmd)}")
            result = subprocess.run(build_cmd, capture_output=False)
            if result.returncode != 0:
                print(f"  WARNING: Rebuild failed (exit {result.returncode}), using best GLB")
                break
        else:
            print(f"  Max iterations reached. Best score: {best_score}/10")

    # Ensure best GLB is at the expected output path
    if best_glb != args.glb_path and os.path.exists(best_glb):
        shutil.copy2(best_glb, args.glb_path)
        print(f"  Restored best GLB (score {best_score}) to {args.glb_path}")

    print(f"\nQuality loop complete. Final score: {best_score}/10")
    return best_glb


if __name__ == "__main__":
    main()
