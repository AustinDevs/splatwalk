import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const canvas = document.getElementById('viewer-canvas');
const pipelineSwitcher = document.getElementById('pipeline-switcher');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const vrBtn = document.getElementById('vr-btn');
const header = document.getElementById('viewer-header');
const help = document.getElementById('viewer-help');
const joystickZone = document.getElementById('joystick-zone');

let viewer = null;
let camera = null;
let movement = { x: 0, y: 0 };
let isMobile = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
let joystick = null;

// Compute scene bounds by sampling the .splat binary
async function computeSceneBounds(splatUrl) {
  try {
    // Use same-origin proxy to bypass COEP cross-origin restrictions
    const proxyUrl = `/api/splat-proxy?url=${encodeURIComponent(splatUrl)}`;

    const headResp = await fetch(proxyUrl, { method: 'HEAD' });
    const totalBytes = parseInt(headResp.headers.get('content-length') || '0');
    if (!totalBytes) return null;

    const totalSplats = Math.floor(totalBytes / 32);
    const samplesToTake = Math.min(totalSplats, 2000);
    const bytesToFetch = samplesToTake * 32;

    // Sample from start and multiple locations
    const offsets = [0];
    if (totalSplats > 2000) {
      offsets.push(Math.floor(totalSplats * 0.25) * 32);
      offsets.push(Math.floor(totalSplats * 0.5) * 32);
      offsets.push(Math.floor(totalSplats * 0.75) * 32);
    }

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (const offset of offsets) {
      const end = Math.min(offset + bytesToFetch - 1, totalBytes - 1);
      const resp = await fetch(proxyUrl, { headers: { Range: `bytes=${offset}-${end}` } });
      const buf = await resp.arrayBuffer();
      const view = new DataView(buf);

      for (let i = 0; i < Math.floor(buf.byteLength / 32); i++) {
        const off = i * 32;
        const x = view.getFloat32(off, true);
        const y = view.getFloat32(off + 4, true);
        const z = view.getFloat32(off + 8, true);

        if (isFinite(x) && isFinite(y) && isFinite(z)) {
          minX = Math.min(minX, x); maxX = Math.max(maxX, x);
          minY = Math.min(minY, y); maxY = Math.max(maxY, y);
          minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
        }
      }
    }

    return {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
      center: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
      size: [maxX - minX, maxY - minY, maxZ - minZ],
    };
  } catch (e) {
    console.warn('Could not compute scene bounds:', e);
    return null;
  }
}

// Compute a good camera position from scene bounds
function computeCameraFromBounds(bounds) {
  const { center, size } = bounds;
  const maxDim = Math.max(size[0], size[1], size[2]);

  // Find the thinnest axis (for aerial scenes this is the height axis)
  const minAxisIdx = size[0] <= size[1] && size[0] <= size[2] ? 0
                   : size[1] <= size[2] ? 1 : 2;

  // Camera up should align with the thinnest axis
  const cameraUp = [0, 0, 0];
  cameraUp[minAxisIdx] = 1;

  // Place camera along the thin axis, offset at a 30° angle from above
  const distance = maxDim * 1.5;
  const camPos = [...center];
  camPos[minAxisIdx] += distance * 0.5; // Above the scene along thin axis

  // Also offset along the longest ground axis for a 3/4 view
  const groundAxes = [0, 1, 2].filter(i => i !== minAxisIdx);
  camPos[groundAxes[0]] -= distance * 0.3;

  return {
    position: camPos,
    lookAt: center,
    up: cameraUp,
  };
}

// Initialize viewer
async function initViewer(splatUrl) {
  if (viewer) {
    viewer.dispose();
  }

  try {
    // Pre-compute scene bounds for smart camera placement
    const bounds = await computeSceneBounds(splatUrl);
    let camConfig = {
      position: [0, 1.6, 5],
      lookAt: [0, 0, 0],
      up: [0, 1, 0],
    };
    if (bounds) {
      camConfig = computeCameraFromBounds(bounds);
    }

    viewer = new GaussianSplats3D.Viewer({
      cameraUp: camConfig.up,
      initialCameraPosition: camConfig.position,
      initialCameraLookAt: camConfig.lookAt,
      selfDrivenMode: true,
      useBuiltInControls: !isMobile,
      rootElement: canvas,
    });

    camera = viewer.camera;
    window.viewer = viewer;

    await viewer.addSplatScene(splatUrl, {
      progressiveLoad: true,
    });

    viewer.start();

    // Check VR support
    if ('xr' in navigator) {
      navigator.xr.isSessionSupported('immersive-vr').then(function(supported) {
        if (supported) {
          vrBtn.style.display = 'block';
        }
      });
    }

    // Start animation loop for mobile
    if (isMobile) {
      initJoystick();
      animateMovement();
    } else {
      initKeyboardControls();
    }
  } catch (err) {
    console.error('Failed to load splat:', err);
    canvas.innerHTML = '<div style="color: white; text-align: center; padding: 2rem;">Failed to load 3D scene</div>';
  }
}

function initKeyboardControls() {
  const speed = 0.1;

  document.addEventListener('keydown', function(e) {
    if (!camera) return;

    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);

    const right = new THREE.Vector3();
    right.crossVectors(camera.up, direction).normalize();

    switch (e.key.toLowerCase()) {
      case 'w':
        camera.position.addScaledVector(direction, speed);
        break;
      case 's':
        camera.position.addScaledVector(direction, -speed);
        break;
      case 'a':
        camera.position.addScaledVector(right, speed);
        break;
      case 'd':
        camera.position.addScaledVector(right, -speed);
        break;
    }
  });
}

function initJoystick() {
  joystick = nipplejs.create({
    zone: joystickZone,
    mode: 'static',
    position: { left: '80px', bottom: '80px' },
    color: 'rgba(255, 255, 255, 0.5)',
    size: 100,
    restOpacity: 0.5,
  });

  joystick.on('move', function(evt, data) {
    if (data.vector) {
      movement.x = data.vector.x;
      movement.y = data.vector.y;
    }
  });

  joystick.on('end', function() {
    movement.x = 0;
    movement.y = 0;
  });
}

function animateMovement() {
  const speed = 0.05;

  function animate() {
    if (camera && (movement.x !== 0 || movement.y !== 0)) {
      const direction = new THREE.Vector3();
      camera.getWorldDirection(direction);

      camera.position.addScaledVector(direction, -movement.y * speed);

      const right = new THREE.Vector3();
      right.crossVectors(camera.up, direction).normalize();
      camera.position.addScaledVector(right, movement.x * speed);
    }

    requestAnimationFrame(animate);
  }

  animate();
}

// Pipeline switching
pipelineSwitcher.addEventListener('click', function(e) {
  const btn = e.target.closest('.pipeline-btn');
  if (!btn) return;

  document.querySelectorAll('.pipeline-btn').forEach(function(b) {
    b.classList.remove('active');
  });
  btn.classList.add('active');

  const url = btn.dataset.url;
  if (url) {
    initViewer(url);
  }
});

// Fullscreen
fullscreenBtn.addEventListener('click', function() {
  const container = document.getElementById('viewer-container');

  if (!document.fullscreenElement) {
    container.requestFullscreen().then(function() {
      fullscreenBtn.textContent = 'Exit Fullscreen';
    });
  } else {
    document.exitFullscreen().then(function() {
      fullscreenBtn.textContent = 'Fullscreen';
    });
  }
});

document.addEventListener('fullscreenchange', function() {
  const isFullscreen = !!document.fullscreenElement;
  header.style.display = isFullscreen ? 'none' : 'flex';
  help.style.display = isFullscreen ? 'none' : 'block';
});

// VR
vrBtn.addEventListener('click', async function() {
  if (!viewer) return;

  try {
    const session = await navigator.xr.requestSession('immersive-vr', {
      optionalFeatures: ['local-floor', 'bounded-floor'],
    });

    viewer.renderer.xr.setSession(session);
  } catch (err) {
    console.error('Failed to enter VR:', err);
  }
});

// Load initial splat
if (window.INITIAL_SPLAT_URL) {
  initViewer(window.INITIAL_SPLAT_URL);
}
