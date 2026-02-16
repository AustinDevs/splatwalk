import * as THREE from 'three';

// Lazy-load GaussianSplats3D
let GaussianSplats3D = null;

// ---------------------------------------------------------------------------
// DOM elements
// ---------------------------------------------------------------------------

const splatContainer = document.getElementById('splat-container');
const loadingOverlay = document.getElementById('loading-overlay');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const joystickZone = document.getElementById('joystick-zone');
const infoPanel = document.getElementById('info-panel');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let manifest = null;
let splatViewer = null;
let splatCamera = null;
let moveSpeed = 0.25;
let altitudeSpeed = 0.15;

// Key state
const keys = {};

// Mobile joystick
let joystick = null;
let joystickMovement = { x: 0, y: 0 };

const isMobile = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

// ---------------------------------------------------------------------------
// Load manifest
// ---------------------------------------------------------------------------

async function loadManifest() {
  const url = window.MANIFEST_URL;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('Manifest not found (' + resp.status + '). Assets may not be generated yet.');
  manifest = await resp.json();

  // Support both old (positions-based) and new (topdown) manifest formats
  if (!manifest.splat_url) {
    throw new Error('Manifest missing splat_url');
  }
}

// ---------------------------------------------------------------------------
// Scene bounds computation (from .splat binary)
// ---------------------------------------------------------------------------

async function computeSceneBounds(splatUrl) {
  if (!splatUrl.toLowerCase().endsWith('.splat')) return null;
  try {
    const headResp = await fetch(splatUrl, { method: 'HEAD' });
    const totalBytes = parseInt(headResp.headers.get('content-length') || '0');
    if (!totalBytes) return null;

    const totalSplats = Math.floor(totalBytes / 32);
    const samplesToTake = Math.min(totalSplats, 2000);
    const bytesToFetch = samplesToTake * 32;

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
      const resp = await fetch(splatUrl, { headers: { Range: 'bytes=' + offset + '-' + end } });
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

// ---------------------------------------------------------------------------
// Initialize splat viewer
// ---------------------------------------------------------------------------

async function initSplat() {
  // Dynamically import GaussianSplats3D
  if (!GaussianSplats3D) {
    console.log('[splat] loading GaussianSplats3D library...');
    GaussianSplats3D = await import('@mkkellogg/gaussian-splats-3d');
    console.log('[splat] GaussianSplats3D loaded');
  }

  const splatUrl = manifest.splat_url;

  // Determine initial camera position from scene bounds
  // Scene uses Z-up (from COLMAP). For top-down: camera above max Z, looking down.
  let camPos, camLookAt, camUp;

  let bounds = manifest.scene_bounds || null;
  if (!bounds) {
    console.log('[splat] computing bounds from .splat file...');
    const computed = await Promise.race([
      computeSceneBounds(splatUrl),
      new Promise(function (resolve) { setTimeout(function () { resolve(null); }, 5000); }),
    ]);
    if (computed) {
      bounds = {
        center: computed.center,
        size: computed.size,
        min: computed.min,
        max: computed.max,
        ground_z: computed.min[2],
      };
    }
  }

  if (bounds) {
    // Position camera above the scene, looking straight down
    const sceneHeight = bounds.size ? bounds.size[2] : 10;
    const maxZ = bounds.max ? bounds.max[2] : sceneHeight;
    const cx = bounds.center ? bounds.center[0] : 0;
    const cy = bounds.center ? bounds.center[1] : 0;
    // Start well above the top of the scene
    camPos = [cx, cy, maxZ + sceneHeight * 0.5];
    camLookAt = [cx, cy, bounds.ground_z || 0];
    // For looking down -Z, up must be perpendicular to Z axis
    camUp = [0, 1, 0];
  } else {
    camPos = [0, 0, 50];
    camLookAt = [0, 0, 0];
    camUp = [0, 1, 0];
  }

  console.log('[splat] camera position:', camPos, 'lookAt:', camLookAt);

  splatViewer = new GaussianSplats3D.Viewer({
    cameraUp: camUp,
    initialCameraPosition: camPos,
    initialCameraLookAt: camLookAt,
    selfDrivenMode: true,
    useBuiltInControls: !isMobile,
    rootElement: splatContainer,
    sphericalHarmonicsDegree: 0,
    gpuAcceleratedSort: true,
    antialiased: true,
    kernel2DSize: 0.3,
    logLevel: 1,
    sceneFadeInRateMultiplier: 5.0,
  });

  splatCamera = splatViewer.camera;

  // Stream the splat scene
  splatViewer.addSplatScene(splatUrl, { progressiveLoad: true })
    .then(function () { console.log('[splat] scene fully loaded'); })
    .catch(function (err) { console.warn('[splat] scene load error:', err); });
  splatViewer.start();

  // Set movement speed based on scene size
  const sceneBounds = manifest.scene_bounds;
  const sceneSize = sceneBounds
    ? Math.max(sceneBounds.size[0], sceneBounds.size[1], sceneBounds.size[2])
    : 10;
  moveSpeed = sceneSize * 0.03;
  altitudeSpeed = sceneSize * 0.02;

  // Set up controls
  if (isMobile) {
    initJoystick();
  }
  initKeyboardControls();
  startMovementLoop();

  console.log('[splat] viewer initialized, moveSpeed:', moveSpeed);
}

// ---------------------------------------------------------------------------
// Keyboard controls: WASD for XY movement, Q/E for altitude
// ---------------------------------------------------------------------------

function initKeyboardControls() {
  document.addEventListener('keydown', function (e) {
    keys[e.key.toLowerCase()] = true;
  });
  document.addEventListener('keyup', function (e) {
    keys[e.key.toLowerCase()] = false;
  });
}

// ---------------------------------------------------------------------------
// Mobile joystick
// ---------------------------------------------------------------------------

function initJoystick() {
  if (joystick) return;
  joystick = nipplejs.create({
    zone: joystickZone,
    mode: 'static',
    position: { left: '80px', bottom: '80px' },
    color: 'rgba(255, 255, 255, 0.5)',
    size: 100,
    restOpacity: 0.5,
  });
  joystick.on('move', function (evt, data) {
    if (data.vector) {
      joystickMovement.x = data.vector.x;
      joystickMovement.y = data.vector.y;
    }
  });
  joystick.on('end', function () {
    joystickMovement.x = 0;
    joystickMovement.y = 0;
  });
}

// ---------------------------------------------------------------------------
// Movement loop
// ---------------------------------------------------------------------------

function startMovementLoop() {
  function animate() {
    requestAnimationFrame(animate);
    if (!splatCamera) return;

    const direction = new THREE.Vector3();
    splatCamera.getWorldDirection(direction);

    // Project direction onto XY plane for horizontal movement
    const forward = new THREE.Vector3(direction.x, direction.y, 0).normalize();
    const right = new THREE.Vector3();
    right.crossVectors(new THREE.Vector3(0, 0, 1), forward).normalize();

    // Keyboard input
    if (keys['w']) splatCamera.position.addScaledVector(forward, moveSpeed);
    if (keys['s']) splatCamera.position.addScaledVector(forward, -moveSpeed);
    if (keys['a']) splatCamera.position.addScaledVector(right, moveSpeed);
    if (keys['d']) splatCamera.position.addScaledVector(right, -moveSpeed);
    if (keys['q']) splatCamera.position.z += altitudeSpeed;
    if (keys['e']) splatCamera.position.z -= altitudeSpeed;

    // Mobile joystick input
    if (joystickMovement.x !== 0 || joystickMovement.y !== 0) {
      splatCamera.position.addScaledVector(forward, -joystickMovement.y * moveSpeed * 0.1);
      splatCamera.position.addScaledVector(right, joystickMovement.x * moveSpeed * 0.1);
    }
  }
  animate();
}

// ---------------------------------------------------------------------------
// Loading overlay
// ---------------------------------------------------------------------------

function showLoading(text) {
  var textEl = loadingOverlay.querySelector('.loading-text');
  if (textEl) textEl.textContent = text || 'Loading...';
  loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
  loadingOverlay.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Fullscreen
// ---------------------------------------------------------------------------

fullscreenBtn.addEventListener('click', function () {
  var container = document.getElementById('demo-container');
  if (!document.fullscreenElement) {
    container.requestFullscreen().then(function () {
      fullscreenBtn.textContent = 'Exit Fullscreen';
    });
  } else {
    document.exitFullscreen().then(function () {
      fullscreenBtn.textContent = 'Fullscreen';
    });
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  try {
    showLoading('Loading manifest...');
    await loadManifest();

    // SharedArrayBuffer requires crossOriginIsolated
    if (!self.crossOriginIsolated) {
      console.warn('crossOriginIsolated is false — SharedArrayBuffer unavailable');
      showLoading('Requires HTTPS with COEP/COOP headers for SharedArrayBuffer');
      return;
    }

    showLoading('Initializing splat viewer...');
    await initSplat();

    // Hide info panel after 8 seconds
    setTimeout(function () {
      if (infoPanel) infoPanel.style.opacity = '0';
      setTimeout(function () {
        if (infoPanel) infoPanel.style.display = 'none';
      }, 300);
    }, 8000);

    hideLoading();
  } catch (err) {
    console.error('Init error:', err);
    var textEl = loadingOverlay.querySelector('.loading-text');
    if (textEl) textEl.textContent = 'Failed to load: ' + err.message;
  }
}

init();
