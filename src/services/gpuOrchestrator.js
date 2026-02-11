import { NodeSSH } from 'node-ssh';
import { readFile } from 'fs/promises';
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

  async uploadImages(jobId, imagePaths) {
    const uploadedUrls = [];

    for (let i = 0; i < imagePaths.length; i++) {
      const imagePath = imagePaths[i];
      const key = `jobs/${jobId}/input/image-${i.toString().padStart(3, '0')}.jpg`;

      const fileContent = await readFile(imagePath);

      await this.s3Client.send(
        new PutObjectCommand({
          Bucket: DO_SPACES_BUCKET,
          Key: key,
          Body: fileContent,
          ContentType: 'image/jpeg',
          ACL: 'public-read',  // GPU droplet needs to download without auth
        })
      );

      uploadedUrls.push(`${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/${key}`);
    }

    return uploadedUrls;
  }

  async createDroplet(jobId) {
    const cloudInit = this.generateCloudInit(jobId);

    const response = await this.doRequest('POST', '/droplets', {
      name: `splatwalk-gpu-${jobId.slice(0, 8)}`,
      region: GPU_DROPLET_REGION,
      size: GPU_DROPLET_SIZE,
      image: GPU_DROPLET_IMAGE,
      ssh_keys: [DO_SSH_KEY_ID],
      user_data: cloudInit,
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
        const filename = `image-${i.toString().padStart(3, '0')}.jpg`;
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
}

export const gpuOrchestrator = new GPUOrchestrator();
