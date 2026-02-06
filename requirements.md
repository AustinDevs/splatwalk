# SplatWalk - Zillow Listing to Walkable WebVR

## Overview

A web application that takes a Zillow listing URL, scrapes the listing photos, processes them through three 3D Gaussian splatting pipelines on a cloud GPU, and presents walkable WebVR experiences the user can explore on their phone or headset.

---

## Architecture

```
User Input (Zillow URL)
    ↓
Playwright Scraper (extract listing images + address + metadata)
    ↓
DigitalOcean GPU Droplet (spun up on demand)
    ↓ (pulls pre-built Docker image from GitHub Container Registry)
    ├── Pipeline A: ViewCrafter only
    ├── Pipeline B: InstantSplat only
    └── Pipeline C: ViewCrafter → InstantSplat (combined)
    ↓
Export .ply files → convert to .ksplat
    ↓
Upload to DigitalOcean Spaces (CDN-enabled)
    ↓
Destroy GPU Droplet
    ↓
UI shows listing with three walkable WebVR viewers
```

---

## Stack

- **Frontend:** React (Next.js or Vite) with TypeScript
- **Backend:** Node.js API routes or standalone Express server
- **Scraping:** Playwright (headless Chromium)
- **Cloud GPU:** DigitalOcean GPU Droplets via `doctl` CLI or DO API
- **3D Processing:** Pre-built Docker image on GitHub Container Registry (ghcr.io)
- **3D Viewer:** GaussianSplats3D (`@mkkellogg/gaussian-splats-3d`) with WebXR
- **Storage:** DigitalOcean Spaces (S3-compatible object storage with CDN)
- **Database:** SQLite (or Postgres if you prefer) for job tracking

---

## Docker Image (Pre-built, hosted on ghcr.io)

The GPU processing Docker image should be built and pushed separately. It contains:

- CUDA 12.1 + PyTorch 2.1
- ViewCrafter (with `model_sparse.ckpt` baked in)
- DUSt3R (with weights baked in)
- InstantSplat (with MASt3R weights baked in)
- `s3cmd` or `aws` CLI for uploading results to DO Spaces
- A single entrypoint script: `/workspace/run_pipeline.sh`

### Entrypoint Script Contract

The Docker container expects:

```bash
# Environment variables passed at runtime:
#   INPUT_DIR       - path to mounted directory containing listing images
#   OUTPUT_DIR      - path to mounted directory for results
#   PIPELINE_MODE   - "viewcrafter" | "instantsplat" | "combined"
#   SPACES_BUCKET   - DO Spaces bucket name
#   SPACES_REGION   - DO Spaces region (e.g., nyc3)
#   SPACES_KEY      - DO Spaces access key
#   SPACES_SECRET   - DO Spaces secret key
#   JOB_ID          - unique job identifier for organizing uploads

/workspace/run_pipeline.sh
```

The entrypoint script should:

1. Run the pipeline(s) based on `PIPELINE_MODE`
2. Convert output `.ply` files to `.ksplat` format (using GaussianSplats3D's converter or a Python script)
3. Upload `.ksplat` files to `s3://${SPACES_BUCKET}/splats/${JOB_ID}/` with public-read ACL
4. Write a `manifest.json` to the same path with metadata about the outputs
5. Exit cleanly

### Manifest Output Format

```json
{
  "job_id": "abc123",
  "pipeline": "viewcrafter",
  "splat_url": "https://bucket.nyc3.cdn.digitaloceanspaces.com/splats/abc123/viewcrafter.ksplat",
  "splat_size_mb": 45.2,
  "num_input_images": 8,
  "num_generated_views": 20,
  "processing_time_seconds": 842
}
```

---

## App Components

### 1. Zillow Scraper Module

**File:** `lib/scraper.ts`

Use Playwright to:

- Navigate to the provided Zillow listing URL
- Handle Zillow's anti-bot measures (wait for page load, handle any modals/captchas)
- Extract the full-resolution listing photos (click through the photo gallery to get all images)
- Extract listing metadata:
  - Street address
  - City, state, zip
  - Price
  - Beds/baths/sqft
  - Zillow listing URL (canonical)
  - Thumbnail image URL (for the listing card)
- Download all photos to a local temp directory
- Return a structured object with metadata + array of local image file paths

**Important notes:**
- Zillow aggressively blocks scrapers. Playwright should use a realistic user agent, viewport size, and add random delays between actions.
- Photos are often lazy-loaded in a gallery/carousel — the scraper needs to click through each one to load full-res versions.
- Look for `<img>` tags or `<picture>` sources with the highest resolution available. Zillow often serves photos via their CDN with dimension parameters in the URL — try to get the largest version.
- If Zillow blocks the request, surface a clear error to the user rather than failing silently.

### 2. GPU Job Orchestrator

**File:** `lib/gpu-orchestrator.ts`

Manages the lifecycle of a DigitalOcean GPU Droplet for processing. Uses the DigitalOcean API directly (via `fetch` or the `do-wrapper` npm package) rather than shelling out to `doctl`.

#### Configuration (via environment variables)

```
DO_API_TOKEN        - DigitalOcean API token
DO_SSH_KEY_ID       - SSH key fingerprint or ID registered with DO
DO_GPU_SIZE         - GPU Droplet size slug (e.g., "gpu-h100x1-80gb")
DO_GPU_IMAGE        - Image slug or snapshot ID (AI/ML-ready NVIDIA image)
DO_GPU_REGION       - Region slug (e.g., "nyc1")
DO_SPACES_BUCKET    - Spaces bucket name
DO_SPACES_REGION    - Spaces region
DO_SPACES_KEY       - Spaces access key
DO_SPACES_SECRET    - Spaces secret key
GHCR_IMAGE          - Full Docker image path (e.g., "ghcr.io/youruser/splatwalk-gpu:latest")
```

#### Orchestration Flow

1. **Create Droplet** via DO API (`POST /v2/droplets`)
   - Use the AI/ML-ready NVIDIA image so CUDA drivers are pre-installed
   - Pass a `user_data` cloud-init script that:
     - Installs Docker and NVIDIA container toolkit (if not in image)
     - Pulls the pre-built Docker image from ghcr.io
     - Signals readiness (e.g., touches a file or calls a webhook)
   - Tag the droplet with `splatwalk` for easy cleanup

2. **Wait for Ready** — poll the DO API until the droplet is active, then poll via SSH until the Docker image is pulled and ready

3. **Upload Images** — SCP the scraped listing photos to the droplet (or upload to Spaces first and have the droplet pull from there)

4. **Run Three Pipelines** — SSH into the droplet and execute the Docker container three times:
   ```bash
   # Run all three in sequence on the same droplet
   docker run --gpus all \
     -v /root/images:/workspace/input \
     -v /root/output_vc:/workspace/output \
     -e PIPELINE_MODE=viewcrafter \
     -e SPACES_BUCKET=$BUCKET \
     -e SPACES_REGION=$REGION \
     -e SPACES_KEY=$KEY \
     -e SPACES_SECRET=$SECRET \
     -e JOB_ID=$JOB_ID \
     $GHCR_IMAGE

   docker run --gpus all \
     -v /root/images:/workspace/input \
     -v /root/output_is:/workspace/output \
     -e PIPELINE_MODE=instantsplat \
     # ... same env vars ...
     $GHCR_IMAGE

   docker run --gpus all \
     -v /root/images:/workspace/input \
     -v /root/output_combined:/workspace/output \
     -e PIPELINE_MODE=combined \
     # ... same env vars ...
     $GHCR_IMAGE
   ```

5. **Verify Uploads** — check that all three `.ksplat` files and manifests exist in Spaces

6. **Destroy Droplet** — `DELETE /v2/droplets/{id}`. Do this in a `finally` block so it always runs even if the pipeline fails.

#### Safety Measures

- **Timeout watchdog:** If the droplet has been running for more than 60 minutes, destroy it regardless. Use a setTimeout or a separate cron check.
- **Error handling:** If any pipeline step fails, still destroy the droplet. Log the error and mark the job as failed.
- **Cost tracking:** Log droplet create/destroy times so you can audit GPU costs.
- **Concurrent job limit:** Only allow 1-2 GPU droplets at a time to avoid surprise bills. Queue additional jobs.

### 3. Job Queue & Status Tracking

**File:** `lib/job-manager.ts`

Track processing jobs in a database (SQLite is fine for a simple version).

#### Job Schema

```sql
CREATE TABLE jobs (
  id TEXT PRIMARY KEY,           -- UUID
  zillow_url TEXT NOT NULL,
  address TEXT,
  city TEXT,
  state TEXT,
  zip TEXT,
  price TEXT,
  beds INTEGER,
  baths REAL,
  sqft INTEGER,
  thumbnail_url TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
    -- pending | scraping | uploading | processing | complete | failed
  error_message TEXT,
  droplet_id INTEGER,
  num_images INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,

  -- Splat URLs (populated when processing completes)
  viewcrafter_splat_url TEXT,
  instantsplat_splat_url TEXT,
  combined_splat_url TEXT,

  -- Processing metadata
  viewcrafter_time_seconds INTEGER,
  instantsplat_time_seconds INTEGER,
  combined_time_seconds INTEGER,
  total_cost_estimate REAL
);
```

#### Status Updates

The backend should update job status at each stage. The frontend polls or uses SSE/WebSocket to show progress.

### 4. API Routes

**Backend endpoints:**

```
POST   /api/jobs              - Submit a new Zillow URL for processing
GET    /api/jobs               - List all jobs (paginated)
GET    /api/jobs/:id           - Get job details + splat URLs
DELETE /api/jobs/:id           - Delete a job and its assets from Spaces
GET    /api/jobs/:id/status    - Lightweight status poll endpoint
```

#### POST /api/jobs Request

```json
{
  "zillow_url": "https://www.zillow.com/homedetails/123-Main-St/12345678_zpid/"
}
```

#### POST /api/jobs Response

```json
{
  "job_id": "abc-123-def",
  "status": "pending",
  "message": "Job queued. Processing will begin shortly."
}
```

#### GET /api/jobs/:id Response (when complete)

```json
{
  "id": "abc-123-def",
  "zillow_url": "https://www.zillow.com/homedetails/...",
  "address": "123 Main St",
  "city": "Los Angeles",
  "state": "CA",
  "zip": "90042",
  "price": "$850,000",
  "beds": 3,
  "baths": 2,
  "sqft": 1450,
  "thumbnail_url": "https://photos.zillowstatic.com/...",
  "status": "complete",
  "num_images": 8,
  "splats": {
    "viewcrafter": {
      "url": "https://bucket.nyc3.cdn.digitaloceanspaces.com/splats/abc-123-def/viewcrafter.ksplat",
      "viewer_url": "/view/abc-123-def/viewcrafter",
      "processing_time_seconds": 842,
      "size_mb": 45.2
    },
    "instantsplat": {
      "url": "https://bucket.nyc3.cdn.digitaloceanspaces.com/splats/abc-123-def/instantsplat.ksplat",
      "viewer_url": "/view/abc-123-def/instantsplat",
      "processing_time_seconds": 180,
      "size_mb": 38.7
    },
    "combined": {
      "url": "https://bucket.nyc3.cdn.digitaloceanspaces.com/splats/abc-123-def/combined.ksplat",
      "viewer_url": "/view/abc-123-def/combined",
      "processing_time_seconds": 1020,
      "size_mb": 52.1
    }
  },
  "total_cost_estimate": 1.85,
  "created_at": "2026-02-05T10:30:00Z",
  "completed_at": "2026-02-05T10:55:00Z"
}
```

### 5. Frontend Pages

#### Page: Home / Job Submission (`/`)

- Clean, minimal form with a single input field for a Zillow URL
- "Generate 3D Walkthrough" submit button
- Below the form: a list/grid of previously processed listings showing:
  - Thumbnail image
  - Address
  - Price / beds / baths
  - Processing status (with animated indicator if in progress)
  - Link to view results when complete

#### Page: Job Status (`/jobs/:id`)

- Listing details at the top (address, price, photo, link to Zillow)
- Processing progress indicator showing current stage:
  - ✅ Scraping photos (X images found)
  - ✅ GPU instance created
  - 🔄 Processing ViewCrafter... (elapsed time)
  - ⏳ Processing InstantSplat...
  - ⏳ Processing Combined...
  - ⏳ Uploading results
- When complete: three cards, one for each pipeline result, each with:
  - Pipeline name and description
  - Processing time and file size
  - "View in 3D" button → links to the viewer page
  - "Enter VR" button → links to the viewer page with WebXR auto-enabled

#### Page: 3D Viewer (`/view/:jobId/:pipeline`)

- Full-screen GaussianSplats3D viewer
- Loads the `.ksplat` file from DO Spaces CDN
- Controls:
  - **Desktop:** WASD/arrow keys to move, mouse to look
  - **Mobile:** Touch joystick (nipplejs) for movement, gyro for look
  - **VR headset:** WebXR mode with room-scale or teleport locomotion
- "Enter VR" button (only shown if WebXR is supported)
- Minimal overlay UI:
  - Back button
  - Address text
  - Link to Zillow listing
  - Toggle between the three pipeline results without leaving the viewer
- Loading progress bar while the splat file downloads

#### GaussianSplats3D Viewer Implementation

```typescript
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const viewer = new GaussianSplats3D.Viewer({
  cameraUp: [0, -1, -0.6],
  initialCameraPosition: [-1, -4, 6],
  initialCameraLookAt: [0, 4, 0],
  webXRMode: GaussianSplats3D.WebXRMode.VR,  // Enable WebVR
  // For mobile: use orbit controls with touch
});

viewer.addSplatScene(splatUrl, {
  splatAlphaRemovalThreshold: 5,
  showLoadingUI: true,
  progressiveLoad: true,
})
.then(() => {
  viewer.start();
});
```

For mobile touch controls, add nipplejs as a virtual joystick:

```typescript
import nipplejs from 'nipplejs';

const joystick = nipplejs.create({
  zone: document.getElementById('joystick-zone'),
  mode: 'static',
  position: { left: '60px', bottom: '60px' },
  size: 100,
});

joystick.on('move', (evt, data) => {
  // Translate joystick input to camera movement
  const speed = data.force * 0.05;
  const angle = data.angle.radian;
  // Move the camera based on angle and speed
});
```

---

## Environment Variables

Create a `.env.example` with all required configuration:

```bash
# DigitalOcean API
DO_API_TOKEN=your_do_api_token
DO_SSH_KEY_ID=your_ssh_key_id

# GPU Droplet Config
DO_GPU_SIZE=gpu-h100x1-80gb
DO_GPU_IMAGE=gpu-h100x1-80gb-gen0-ubuntu-2204-ai-ml-20240730
DO_GPU_REGION=nyc1

# DigitalOcean Spaces
DO_SPACES_BUCKET=splatwalk
DO_SPACES_REGION=nyc3
DO_SPACES_KEY=your_spaces_key
DO_SPACES_SECRET=your_spaces_secret
DO_SPACES_CDN_ENDPOINT=https://splatwalk.nyc3.cdn.digitaloceanspaces.com

# GitHub Container Registry
GHCR_IMAGE=ghcr.io/youruser/splatwalk-gpu:latest
GHCR_TOKEN=your_github_token

# App Settings
MAX_CONCURRENT_JOBS=2
GPU_TIMEOUT_MINUTES=60
DATABASE_URL=file:./splatwalk.db
```

---

## File Structure

```
splatwalk/
├── README.md
├── package.json
├── .env.example
├── docker/
│   ├── gpu/
│   │   ├── Dockerfile              # GPU processing image
│   │   ├── run_pipeline.sh         # Entrypoint script
│   │   ├── convert_to_ksplat.py    # PLY → KSPLAT converter
│   │   └── upload_to_spaces.sh     # Upload results to DO Spaces
│   └── docker-compose.yml          # For local dev (app only, not GPU)
├── src/
│   ├── app/                        # Next.js app directory (or pages/)
│   │   ├── page.tsx                # Home - URL input + job list
│   │   ├── jobs/
│   │   │   └── [id]/
│   │   │       └── page.tsx        # Job status page
│   │   └── view/
│   │       └── [jobId]/
│   │           └── [pipeline]/
│   │               └── page.tsx    # 3D viewer page
│   ├── components/
│   │   ├── JobSubmitForm.tsx
│   │   ├── JobList.tsx
│   │   ├── JobStatusCard.tsx
│   │   ├── ProcessingProgress.tsx
│   │   ├── SplatViewer.tsx         # GaussianSplats3D wrapper
│   │   ├── VRButton.tsx
│   │   └── TouchControls.tsx       # nipplejs joystick
│   ├── lib/
│   │   ├── scraper.ts              # Playwright Zillow scraper
│   │   ├── gpu-orchestrator.ts     # DO GPU Droplet lifecycle
│   │   ├── job-manager.ts          # Job queue + DB operations
│   │   ├── spaces.ts               # DO Spaces upload/URL helpers
│   │   └── do-api.ts               # DigitalOcean API client wrapper
│   └── api/                        # API routes
│       └── jobs/
│           ├── route.ts            # POST + GET /api/jobs
│           └── [id]/
│               ├── route.ts        # GET + DELETE /api/jobs/:id
│               └── status/
│                   └── route.ts    # GET /api/jobs/:id/status
├── public/
│   └── joystick-zone.css
└── scripts/
    ├── build-gpu-image.sh          # Build + push Docker image to ghcr.io
    ├── create-spaces-bucket.sh     # One-time Spaces setup
    └── cleanup-stale-droplets.sh   # Safety script to kill orphaned GPU droplets
```

---

## Safety & Cost Controls

1. **Always destroy droplets in a finally block.** No matter what fails, the droplet must be destroyed.
2. **60-minute hard timeout.** If a droplet has been running longer than this, kill it.
3. **Max concurrent jobs: 2.** Queue additional requests. Don't spin up unlimited GPU instances.
4. **Stale droplet cleanup cron.** Run `cleanup-stale-droplets.sh` every 15 minutes to find and destroy any droplets tagged `splatwalk` that are older than 60 minutes.
5. **Cost estimation.** Track GPU time per job and display estimated cost in the UI.
6. **Rate limit the submission endpoint.** Prevent abuse — max 5 jobs per hour per IP.

---

## Deployment

The web app itself (not the GPU processing) should be deployed cheaply:

- **Option A:** DigitalOcean App Platform (easy, stays in DO ecosystem)
- **Option B:** Vercel or Cloudflare Pages (free tier works, but Playwright may need a separate server)
- **Option C:** A basic $6/mo DO Droplet running the Node.js app

Playwright scraping needs a real server environment (not serverless/edge), so if using Vercel/Cloudflare for the frontend, run the scraper + orchestrator as a separate service on a small DO Droplet or use a Playwright cloud service like Browserbase.

---

## Notes for Claude Code

- Start by getting the Playwright scraper working and tested against a real Zillow URL before building anything else. Zillow's anti-scraping is aggressive and this is the most likely thing to break.
- The GPU Docker image should be built and tested manually first. Don't try to automate the Docker build as part of the app — it's a separate artifact.
- For local development, mock the GPU orchestrator to return fake splat URLs pointing to sample `.ksplat` files so you can build and test the viewer without spinning up real GPU instances.
- The GaussianSplats3D viewer is the most user-facing piece — make sure it works well on mobile with touch controls before optimizing anything else.
- Use Server-Sent Events (SSE) for real-time job status updates rather than polling. Simpler than WebSockets and works fine for one-directional updates.
