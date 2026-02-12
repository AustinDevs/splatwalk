import { NodeSSH } from 'node-ssh';
import { readFile } from 'fs/promises';
import { extname } from 'path';
import { S3Client, PutObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3';

const DO_API_TOKEN = process.env.DO_API_TOKEN || '';
const DO_SSH_KEY_ID = process.env.DO_SSH_KEY_ID || '';
const DO_SSH_PRIVATE_KEY_PATH = process.env.DO_SSH_PRIVATE_KEY_PATH || '~/.ssh/splatwalk_gpu';

const GPU_DROPLET_SIZE = process.env.GPU_DROPLET_SIZE || 'g-2vcpu-8gb';
const GPU_DROPLET_REGION = process.env.GPU_DROPLET_REGION || 'nyc1';
const GPU_DROPLET_IMAGE = process.env.GPU_DROPLET_IMAGE || 'docker-20-04';
const GPU_DOCKER_IMAGE = process.env.GPU_DOCKER_IMAGE || 'ghcr.io/your-org/splatwalk-gpu:latest';

const GPU_TIMEOUT_MS = parseInt(process.env.GPU_TIMEOUT_MS || '3600000', 10);

// GitHub Container Registry credentials (for private repos)
const GHCR_USERNAME = process.env.GHCR_USERNAME || '';
const GHCR_TOKEN = process.env.GHCR_TOKEN || '';

const DO_SPACES_KEY = process.env.DO_SPACES_KEY || '';
const DO_SPACES_SECRET = process.env.DO_SPACES_SECRET || '';
const DO_SPACES_BUCKET = process.env.DO_SPACES_BUCKET || 'splatwalk';
const DO_SPACES_REGION = process.env.DO_SPACES_REGION || 'nyc3';
const DO_SPACES_ENDPOINT = process.env.DO_SPACES_ENDPOINT || 'https://nyc3.digitaloceanspaces.com';

const SLACK_WEBHOOK_URL = process.env.SLACK_WEBHOOK_URL || '';

// Pipeline tuning (lower values for smaller GPUs like RTX 4000 Ada 20GB)
const MAX_N_VIEWS = process.env.MAX_N_VIEWS || '';
const VIEWCRAFTER_BATCH_SIZE = process.env.VIEWCRAFTER_BATCH_SIZE || '';
const TRAIN_ITERATIONS = process.env.TRAIN_ITERATIONS || '';

const DO_API_BASE = 'https://api.digitalocean.com/v2';

class GPUOrchestrator {
  constructor() {
    this.s3Client = new S3Client({
      endpoint: DO_SPACES_ENDPOINT,
      region: DO_SPACES_REGION,
      credentials: {
        accessKeyId: DO_SPACES_KEY,
        secretAccessKey: DO_SPACES_SECRET,
      },
      forcePathStyle: false,
    });
    this.activeDroplets = new Map();
  }

  async doRequest(method, path, body = null) {
    const response = await fetch(`${DO_API_BASE}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${DO_API_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`DigitalOcean API error: ${response.status} ${error}`);
    }

    if (response.status === 204) return null;
    return response.json();
  }

  static VIDEO_EXTENSIONS = ['.mov', '.mp4'];
  static VIDEO_CONTENT_TYPES = {
    '.mov': 'video/quicktime',
    '.mp4': 'video/mp4',
  };

  isVideoFile(filePath) {
    return GPUOrchestrator.VIDEO_EXTENSIONS.includes(extname(filePath).toLowerCase());
  }

  async uploadImages(jobId, filePaths) {
    const uploadedUrls = [];

    for (let i = 0; i < filePaths.length; i++) {
      const filePath = filePaths[i];
      const ext = extname(filePath).toLowerCase();
      const isVideo = this.isVideoFile(filePath);

      const key = isVideo
        ? `jobs/${jobId}/input/video${ext}`
        : `jobs/${jobId}/input/image-${i.toString().padStart(3, '0')}.jpg`;
      const contentType = isVideo
        ? GPUOrchestrator.VIDEO_CONTENT_TYPES[ext] || 'video/mp4'
        : 'image/jpeg';

      const fileContent = await readFile(filePath);

      await this.s3Client.send(
        new PutObjectCommand({
          Bucket: DO_SPACES_BUCKET,
          Key: key,
          Body: fileContent,
          ContentType: contentType,
          ACL: 'public-read',
        })
      );

      uploadedUrls.push(`${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/${key}`);
    }

    return uploadedUrls;
  }

  getDatasetUrl(datasetName) {
    return `${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/datasets/${datasetName}.zip`;
  }

  async createDroplet(jobId, cloudInitOverride) {
    const cloudInit = cloudInitOverride || this.generateCloudInit(jobId);

    const response = await this.doRequest('POST', '/droplets', {
      name: `splatwalk-gpu-${jobId.slice(0, 8)}`,
      region: GPU_DROPLET_REGION,
      size: GPU_DROPLET_SIZE,
      image: GPU_DROPLET_IMAGE,
      ssh_keys: [DO_SSH_KEY_ID],
      user_data: cloudInit,
      monitoring: true,
    });

    const dropletId = response.droplet.id.toString();

    const timeout = setTimeout(async () => {
      console.warn(`GPU timeout reached for droplet ${dropletId}, destroying...`);
      await this.destroyDroplet(dropletId).catch(console.error);
    }, GPU_TIMEOUT_MS);

    this.activeDroplets.set(dropletId, timeout);

    return {
      id: dropletId,
      ip: '',
      status: 'new',
    };
  }

  async launchJob(jobId, pipeline, { datasetName, imageUrls } = {}) {
    const datasetUrl = datasetName ? this.getDatasetUrl(datasetName) : null;
    const cloudInit = this.generateAutonomousCloudInit(jobId, pipeline, { datasetUrl, imageUrls });
    const droplet = await this.createDroplet(jobId, cloudInit);

    // Clear the local timeout — autonomous droplet self-destructs
    const timeout = this.activeDroplets.get(droplet.id);
    if (timeout) {
      clearTimeout(timeout);
      this.activeDroplets.delete(droplet.id);
    }

    return {
      jobId,
      dropletId: droplet.id,
      manifestUrl: `${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/jobs/${jobId}/output/${pipeline}/manifest.json`,
    };
  }

  async waitForDroplet(dropletId) {
    const maxAttempts = 60;
    const pollInterval = 5000;

    for (let i = 0; i < maxAttempts; i++) {
      const response = await this.doRequest('GET', `/droplets/${dropletId}`);
      const droplet = response.droplet;

      if (droplet.status === 'active') {
        const publicNetwork = droplet.networks.v4.find((n) => n.type === 'public');
        if (publicNetwork) {
          const ip = publicNetwork.ip_address;
          const sshReady = await this.waitForSSH(ip);
          if (sshReady) {
            return { id: dropletId, ip, status: 'active' };
          }
        }
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new Error('Droplet failed to become ready');
  }

  async waitForSSH(ip) {
    const ssh = new NodeSSH();
    const maxAttempts = 12;
    const privateKeyPath = DO_SSH_PRIVATE_KEY_PATH.replace('~', process.env.HOME || '');

    for (let i = 0; i < maxAttempts; i++) {
      try {
        await ssh.connect({
          host: ip,
          username: 'root',
          privateKeyPath,
          readyTimeout: 10000,
        });
        await ssh.dispose();
        return true;
      } catch {
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
    }

    return false;
  }

  async runPipeline(jobId, dropletId, pipeline, imageUrls) {
    const droplet = await this.getDropletInfo(dropletId);
    const ssh = new NodeSSH();
    const privateKeyPath = DO_SSH_PRIVATE_KEY_PATH.replace('~', process.env.HOME || '');

    try {
      // Retry SSH connection (cloud-init may restart services)
      const maxRetries = 6;
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          await ssh.connect({
            host: droplet.ip,
            username: 'root',
            privateKeyPath,
            readyTimeout: 30000,
          });
          break;
        } catch (err) {
          if (attempt === maxRetries) throw err;
          console.log(`[${pipeline}] SSH connection attempt ${attempt} failed, retrying in 10s...`);
          await new Promise((resolve) => setTimeout(resolve, 10000));
        }
      }

      // Wait for cloud-init to complete (ensures docker/nvidia are installed)
      console.log(`[${pipeline}] Waiting for cloud-init to complete...`);
      const cloudInitResult = await ssh.execCommand(
        'cloud-init status --wait || true',
        { execOptions: { timeout: 300000 } }
      );
      console.log(`[${pipeline}] Cloud-init status: ${cloudInitResult.stdout.trim() || 'done'}`);

      await ssh.execCommand(`mkdir -p /workspace/${jobId}/input`);

      // Debug: Check GPU and docker status
      console.log(`[${pipeline}] Checking GPU availability...`);
      const nvidiaSmi = await ssh.execCommand('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 || echo "nvidia-smi not available"');
      console.log(`[${pipeline}] GPU: ${nvidiaSmi.stdout.trim()}`);

      const dockerInfo = await ssh.execCommand('docker info 2>&1 | grep -E "(Runtimes|Default Runtime)" || echo "No runtime info"');
      console.log(`[${pipeline}] Docker runtimes: ${dockerInfo.stdout.trim()}`);

      // Ensure docker is logged in to GHCR
      if (GHCR_USERNAME && GHCR_TOKEN) {
        console.log(`[${pipeline}] Authenticating with GitHub Container Registry...`);
        const loginResult = await ssh.execCommand(
          `echo "${GHCR_TOKEN}" | docker login ghcr.io -u ${GHCR_USERNAME} --password-stdin`
        );
        if (loginResult.code !== 0) {
          console.error(`[${pipeline}] Docker login failed: ${loginResult.stderr}`);
        }
      }

      for (let i = 0; i < imageUrls.length; i++) {
        const url = imageUrls[i];
        // Derive filename from the URL key (preserves video.mov / image-NNN.jpg)
        const urlPath = new URL(url).pathname;
        const filename = urlPath.split('/').pop();
        await ssh.execCommand(
          `curl -s -o /workspace/${jobId}/input/${filename} "${url}"`,
          { cwd: '/workspace' }
        );
      }

      const dockerCmd = [
        'docker run --rm --gpus all',
        `-v /workspace/${jobId}:/data`,
        `-e INPUT_DIR=/data/input`,
        `-e OUTPUT_DIR=/data/output`,
        `-e PIPELINE_MODE=${pipeline}`,
        `-e JOB_ID=${jobId}`,
        `-e SPACES_KEY=${DO_SPACES_KEY}`,
        `-e SPACES_SECRET=${DO_SPACES_SECRET}`,
        `-e SPACES_BUCKET=${DO_SPACES_BUCKET}`,
        `-e SPACES_REGION=${DO_SPACES_REGION}`,
        `-e SPACES_ENDPOINT=${DO_SPACES_ENDPOINT}`,
        SLACK_WEBHOOK_URL ? `-e SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}` : '',
        MAX_N_VIEWS ? `-e MAX_N_VIEWS=${MAX_N_VIEWS}` : '',
        VIEWCRAFTER_BATCH_SIZE ? `-e VIEWCRAFTER_BATCH_SIZE=${VIEWCRAFTER_BATCH_SIZE}` : '',
        TRAIN_ITERATIONS ? `-e TRAIN_ITERATIONS=${TRAIN_ITERATIONS}` : '',
        GHCR_TOKEN ? `-e GITHUB_TOKEN=${GHCR_TOKEN}` : '',
        GPU_DOCKER_IMAGE,
      ].join(' ');

      console.log(`Running pipeline ${pipeline} on droplet ${dropletId}`);
      const result = await ssh.execCommand(dockerCmd, {
        cwd: '/workspace',
        onStdout: (chunk) => console.log(`[${pipeline}] ${chunk.toString()}`),
        onStderr: (chunk) => console.error(`[${pipeline}] ${chunk.toString()}`),
      });

      if (result.code !== 0) {
        throw new Error(`Pipeline ${pipeline} failed: ${result.stderr}`);
      }

      const manifestUrl = `${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/jobs/${jobId}/output/${pipeline}/manifest.json`;
      const manifestResponse = await fetch(manifestUrl);

      if (!manifestResponse.ok) {
        throw new Error(`Pipeline ${pipeline} did not produce manifest`);
      }

      const manifest = await manifestResponse.json();

      return {
        splatUrl: manifest.splatUrl,
        thumbnailUrl: manifest.thumbnailUrl,
      };
    } finally {
      await ssh.dispose();
    }
  }

  async getDropletInfo(dropletId) {
    const response = await this.doRequest('GET', `/droplets/${dropletId}`);
    const droplet = response.droplet;
    const publicNetwork = droplet.networks.v4.find((n) => n.type === 'public');

    return {
      id: dropletId,
      ip: publicNetwork?.ip_address || '',
      status: droplet.status,
    };
  }

  async destroyDroplet(dropletId) {
    const timeout = this.activeDroplets.get(dropletId);
    if (timeout) {
      clearTimeout(timeout);
      this.activeDroplets.delete(dropletId);
    }

    console.log(`Destroying droplet ${dropletId}`);
    await this.doRequest('DELETE', `/droplets/${dropletId}`);
  }

  generateCloudInit(jobId) {
    // Add docker login for private repos if credentials provided
    const dockerLogin = GHCR_USERNAME && GHCR_TOKEN
      ? `  - echo "${GHCR_TOKEN}" | docker login ghcr.io -u ${GHCR_USERNAME} --password-stdin`
      : '';

    // gpu-h100x1-base image has NVIDIA drivers and docker pre-installed
    return `#cloud-config
runcmd:
  - nvidia-smi
${dockerLogin}
  - docker pull ${GPU_DOCKER_IMAGE}
  - echo "Ready for job ${jobId}"
`;
  }

  generateAutonomousCloudInit(jobId, pipeline, { datasetUrl, imageUrls }) {
    // Build the input download block: prefer dataset zip, fall back to individual URLs
    let downloadBlock;
    if (datasetUrl) {
      downloadBlock = `
# --- Download dataset zip from Spaces ---
mkdir -p /workspace/${jobId}/input
echo "Downloading dataset: ${datasetUrl}"
curl -fSL -o /workspace/${jobId}/dataset.zip "${datasetUrl}"
apt-get install -y unzip || true
unzip -q -o /workspace/${jobId}/dataset.zip -d /workspace/${jobId}/input
rm /workspace/${jobId}/dataset.zip
echo "Extracted $(ls -1 /workspace/${jobId}/input | wc -l) files"
`;
    } else if (imageUrls?.length) {
      const urlList = imageUrls.join(' ');
      downloadBlock = `
# --- Download input files individually from Spaces ---
mkdir -p /workspace/${jobId}/input
for url in ${urlList}; do
  filename=$(basename "$url")
  echo "Downloading $filename..."
  curl -s -o "/workspace/${jobId}/input/$filename" "$url"
done
echo "Downloaded ${imageUrls.length} files"
`;
    } else {
      throw new Error('launchJob requires either datasetName or imageUrls');
    }

    return `#!/bin/bash
exec > /var/log/splatwalk-job.log 2>&1

echo "=== SplatWalk Autonomous Job Runner ==="
echo "Job: ${jobId}"
echo "Pipeline: ${pipeline}"

# --- Slack helper (same as in run_pipeline.sh) ---
notify_slack() {
  local message="\$1"
  local status="\${2:-info}"
  if [ -z "${SLACK_WEBHOOK_URL}" ]; then return; fi
  local emoji=":arrows_counterclockwise:"
  [ "\$status" = "success" ] && emoji=":white_check_mark:"
  [ "\$status" = "error" ] && emoji=":x:"
  local payload="{\\"text\\":\\"\${emoji} *[${jobId.slice(0, 8)}] ${pipeline}* -- \${message}\\"}"
  curl -s -X POST -H 'Content-type: application/json' --data "\$payload" "${SLACK_WEBHOOK_URL}" > /dev/null 2>&1 &
}

# --- Upload log to Spaces ---
upload_log() {
  local log_key="jobs/${jobId}/logs/cloud-init.log"
  echo "Uploading log to s3://${DO_SPACES_BUCKET}/\$log_key ..."
  local date_str=\$(date -u +"%a, %d %b %Y %H:%M:%S GMT")
  local content_type="text/plain"
  local resource="/${DO_SPACES_BUCKET}/\$log_key"
  local string_to_sign="PUT\\n\\n\${content_type}\\n\${date_str}\\n\${resource}"
  local signature=\$(echo -en "\$string_to_sign" | openssl dgst -sha1 -hmac "${DO_SPACES_SECRET}" -binary | base64)
  curl -s -X PUT \\
    -H "Date: \$date_str" \\
    -H "Content-Type: \$content_type" \\
    -H "Authorization: AWS ${DO_SPACES_KEY}:\$signature" \\
    --data-binary @/var/log/splatwalk-job.log \\
    "${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/\$log_key" || true
  notify_slack "Log uploaded: ${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/\$log_key"
}

# --- Always self-destruct, no matter what ---
self_destruct() {
  echo "Self-destructing droplet..."
  upload_log
  notify_slack "Droplet self-destructing." "info"
  sleep 2  # let Slack message send
  DROPLET_ID=\$(curl -s http://169.254.169.254/metadata/v1/id)
  curl -s -X DELETE \\
    -H "Authorization: Bearer ${DO_API_TOKEN}" \\
    "https://api.digitalocean.com/v2/droplets/\$DROPLET_ID"
}
trap self_destruct EXIT

# --- Safety timeout: self-destruct after 5 hours no matter what ---
(
  sleep 18000
  echo "Safety timeout reached..."
  notify_slack "Safety timeout (5h) reached. Force killing." "error"
  kill -9 \$\$ 2>/dev/null  # kill the main script, triggers EXIT trap
) &

notify_slack "Droplet booted, waiting for GPU driver..."

# --- Wait for GPU/docker readiness ---
for i in \$(seq 1 30); do
  nvidia-smi && break
  echo "Waiting for GPU driver... (attempt \$i)"
  sleep 10
done

GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
notify_slack "GPU ready: \$GPU_NAME. Configuring Docker GPU runtime..."

# --- Ensure nvidia-container-toolkit is configured ---
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
sleep 5

notify_slack "Docker GPU configured. Pulling image..."

# --- Docker login + pull (with retry for rate limits) ---
${GHCR_USERNAME && GHCR_TOKEN ? `echo "${GHCR_TOKEN}" | docker login ghcr.io -u ${GHCR_USERNAME} --password-stdin` : 'echo "No GHCR credentials, skipping login"'}
for attempt in 1 2 3; do
  docker pull ${GPU_DOCKER_IMAGE} && break
  echo "Docker pull failed (attempt \$attempt/3), waiting 30s..."
  notify_slack "Docker pull failed (attempt \$attempt/3), retrying..." "error"
  sleep 30
done

notify_slack "Docker image pulled. Downloading input data..."

${downloadBlock}

notify_slack "Input data ready. Starting pipeline..."

# --- Run the pipeline (stream output to log) ---
echo "=== DOCKER RUN START ==="
docker run --rm --gpus all \\
  -v /workspace/${jobId}:/data \\
  -e INPUT_DIR=/data/input \\
  -e OUTPUT_DIR=/data/output \\
  -e PIPELINE_MODE=${pipeline} \\
  -e JOB_ID=${jobId} \\
  -e SPACES_KEY=${DO_SPACES_KEY} \\
  -e SPACES_SECRET=${DO_SPACES_SECRET} \\
  -e SPACES_BUCKET=${DO_SPACES_BUCKET} \\
  -e SPACES_REGION=${DO_SPACES_REGION} \\
  -e SPACES_ENDPOINT=${DO_SPACES_ENDPOINT} \\
  -e SLACK_WEBHOOK_URL='${SLACK_WEBHOOK_URL}' \\
  -e MAX_N_VIEWS=${MAX_N_VIEWS || '24'} \\
  -e VIEWCRAFTER_BATCH_SIZE=${VIEWCRAFTER_BATCH_SIZE || '10'} \\
  -e TRAIN_ITERATIONS=${TRAIN_ITERATIONS || '7000'} \\
  -e GITHUB_TOKEN=${GHCR_TOKEN || ''} \\
  --entrypoint bash \\
  ${GPU_DOCKER_IMAGE} -c '
    echo "Installing runtime deps..."
    pip install --no-cache-dir icecream open3d trimesh "pyglet<2" evo matplotlib tensorboard imageio gdown roma opencv-python transformers huggingface_hub omegaconf pytorch-lightning open-clip-torch kornia decord imageio-ffmpeg scikit-image moviepy 2>&1 | tail -3

    echo "Building PyTorch3D from source with CUDA support (required for ViewCrafter)..."
    # Check if PyTorch3D already has GPU support
    if python -c "from pytorch3d.renderer.points.rasterize_points import rasterize_points; print(\"PyTorch3D GPU OK\")" 2>/dev/null; then
      echo "  PyTorch3D already has GPU support"
    else
      echo "  Removing pre-built PyTorch3D (no GPU support)..."
      pip uninstall -y pytorch3d 2>/dev/null || true
      echo "  Compiling PyTorch3D from source (this takes ~15-20 min)..."
      FORCE_CUDA=1 pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" 2>&1 | tail -10
      # Verify it works
      if python -c "from pytorch3d.renderer.points.rasterize_points import rasterize_points; print(\"PyTorch3D GPU OK\")" 2>/dev/null; then
        echo "  PyTorch3D compiled successfully with GPU support!"
      else
        echo "  WARNING: PyTorch3D GPU compilation failed, ViewCrafter Stage 4 will be skipped"
      fi
    fi

    echo "Fetching latest pipeline scripts from GitHub..."
    CURL_ARGS=(-fsSL -H "Accept: application/vnd.github.raw")
    [ -n "\$GITHUB_TOKEN" ] && CURL_ARGS+=(-H "Authorization: token \$GITHUB_TOKEN")
    BASE_URL="https://api.github.com/repos/AustinDevs/splatwalk/contents/docker/gpu"
    for script in render_descent.py enhance_with_viewcrafter.py quality_gate.py convert_to_ksplat.py; do
      curl "\${CURL_ARGS[@]}" "\$BASE_URL/\$script" -o "/opt/\$script" && echo "  Updated \$script"
    done

    echo "Patching ViewCrafter for Pillow 10+ (ANTIALIAS -> LANCZOS)..."
    grep -rl "Image.ANTIALIAS" /opt/ViewCrafter/ 2>/dev/null | while read f; do
      sed -i "s/Image.ANTIALIAS/Image.LANCZOS/g" "\$f"
      echo "  Patched \$f"
    done

    exec /opt/entrypoint.sh
  ' 2>&1
DOCKER_EXIT=\$?
echo "=== DOCKER RUN END (exit \$DOCKER_EXIT) ==="

if [ "\$DOCKER_EXIT" -eq 0 ]; then
  notify_slack "Pipeline finished successfully!" "success"
else
  notify_slack "Pipeline failed (exit \$DOCKER_EXIT). Check log in Spaces." "error"
fi
# EXIT trap fires here → self_destruct → droplet deleted
`;
  }
}

export const gpuOrchestrator = new GPUOrchestrator();
