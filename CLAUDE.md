# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

SplatWalk — a SaaS that converts overhead drone or satellite images of undeveloped land into walkable ground-level 3D Gaussian Splat experiences. Users submit aerial imagery, the system spins up an ephemeral DigitalOcean H100 GPU droplet, runs a 5-stage AI pipeline (DUSt3R → InstantSplat training → descent rendering → FLUX ground view generation → .splat compression), uploads results to CDN, destroys the droplet, and serves an interactive WebGL/WebXR viewer.

## Commands

```bash
npm run dev          # Start dev server with --watch (port 3000)
npm start            # Production server
```

No build step, no test suite, no linter. Pure Node.js ESM (`"type": "module"`).

## Architecture

### Backend (Express + SQLite)

- `src/index.js` — Entry point, Express app with COEP/COOP headers for SharedArrayBuffer
- `src/routes/pages.js` — Server-rendered EJS pages (home, job status, viewer)
- `src/routes/jobs.js` — REST API + SSE streaming for job progress
- `src/services/jobProcessor.js` — Orchestrates the full pipeline: upload → provision GPU → process → complete
- `src/services/gpuOrchestrator.js` — DigitalOcean API: creates/destroys GPU droplets, SSH pipeline execution, cloud-init with Volume mount
- `src/db/index.js` — SQLite schema (jobs + rate_limits tables), sync via better-sqlite3

### Frontend (Vanilla JS + EJS)

- `src/public/js/viewer.js` — 3D Gaussian Splat viewer using `@mkkellogg/gaussian-splats-3d`. Handles .splat and .ply formats, scene bounds computation, VR support
- `src/public/js/home.js` — Image upload / job submission form
- `src/public/js/job-status.js` — SSE-based live progress tracking

### GPU Pipeline (Python, runs on remote GPU droplet)

All scripts in `docker/gpu/`, fetched from GitHub at runtime for hotfixes:

- `run_pipeline.sh` — Main orchestrator (5 stages), EXIF GPS extraction, drone AGL calculation
- `render_descent.py` — Stage 3: renders cameras at descending altitudes, retrains at each level
- `generate_ground_views.py` — Stage 3.5: FLUX ControlNet + IP-Adapter + Gemini captioning + Depth-Anything-V2
- `compress_splat.py` — Stage 5: importance-based pruning + .splat binary conversion with uniform scene scaling
- `quality_gate.py` — Confidence scoring

### Infrastructure

- **Everything-on-Volume**: Runtime (conda, repos, models ~36GB) lives on a persistent DO Volume at `/mnt/splatwalk/`. Droplets are stateless — attach Volume at boot, detach on self-destruct.
- `scripts/setup-volume.sh` — Installs full runtime onto Volume (idempotent)
- `.github/workflows/build-gpu-image.yml` — Docker image CI/CD to GHCR
- DO has a **1 GPU droplet limit** — always check/destroy existing before creating new

## Critical Technical Details

### Uniform Scene Scaling (.splat format)
Positions AND scales must be multiplied by the **same factor** (default 50x). Separate scale multipliers distort trained Gaussian overlap proportions, causing wash-out or speckle. Scene is centered at centroid before scaling.

### InstantSplat Training
- `n_views` is a **path component** (`sparse_{n_views}/0/`), NOT a camera count limit
- `Scene()` always starts from COLMAP, never from checkpoints
- Densification is disabled — initial point count determines final Gaussian count
- **Never retrain with FLUX-generated images** — even 500 iterations bloats Gaussian scales 4x

### PyTorch3D Compilation
Pre-built pip wheel has CPU-only `_C.so` (~1MB). GPU rasterization requires building from source with both `FORCE_CUDA=1` AND `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`. Properly compiled `_C.so` is ~24-27MB.

### Web Viewer
- `/api/splat-proxy?url=...` proxies CDN files with correct COEP headers for SharedArrayBuffer
- `computeSceneBounds()` only works for .splat files (reads binary format directly)
- GaussianSplats3D viewer options: `gpuAcceleratedSort`, `antialiased`, `kernel2DSize`, `sphericalHarmonicsDegree` are compile-time (set at construction, not runtime)

### COLMAP from cameras.json
cameras.json `rotation` field = R_c2w (camera-to-world). To get COLMAP format: `R_w2c = R_c2w.T`, `T_w2c = -R_w2c @ position`.

## Environment

Copy `.env.example` to `.env`. Key variables:
- `DO_API_TOKEN`, `DO_SPACES_KEY/SECRET`, `DO_VOLUME_ID` — DigitalOcean infra
- `DO_SSH_PRIVATE_KEY_PATH` — SSH key for GPU droplets (default `~/.ssh/splatwalk_gpu`)
- `GEMINI_API_KEY` — Scene captioning (Gemini 2.5 Flash)
- `GOOGLE_MAPS_API_KEY` — Geocoding for botanical identification
- `SLACK_WEBHOOK_URL` — Pipeline progress notifications
- `GPU_DROPLET_SIZE` — `gpu-h100x1-80gb` (default) or `gpu-4000adax1-20gb` (budget)
