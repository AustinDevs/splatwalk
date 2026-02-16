import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Lazy-load GaussianSplats3D only when splat mode is activated
let GaussianSplats3D = null;

// ---------------------------------------------------------------------------
// DOM elements
// ---------------------------------------------------------------------------

const splatContainer = document.getElementById('splat-container');
const threeContainer = document.getElementById('three-container');
const modeSwitcher = document.getElementById('mode-switcher');
const positionSelector = document.getElementById('position-selector');
const infoPanel = document.getElementById('info-panel');
const loadingOverlay = document.getElementById('loading-overlay');
const fullscreenBtn = document.getElementById('fullscreen-btn');
const joystickZone = document.getElementById('joystick-zone');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let manifest = null;
let currentMode = 'skybox';
let currentPositionIdx = 0;

// Splat mode state
let splatViewer = null;
let splatCamera = null;
let splatBounds = null;
let splatMoveSpeed = 0.25;
let splatMovement = { x: 0, y: 0 };
let joystick = null;

// Three.js shared state (skybox + mesh)
let threeRenderer = null;
let threeScene = null;
let threeCamera = null;
let threeControls = null;
let meshSphere = null;

// Texture caches
const panoramaTextures = new Map();
const depthTextures = new Map();

const isMobile = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

const INFO_TEXT = {
  splat: '<strong>Gaussian Splat</strong><br>Native 3D Gaussians rendered in real-time. Free camera movement. Blurry at ground level (trained from aerial only).',
  skybox: '<strong>Panoramic Skybox</strong><br>FLUX-generated 360\u00b0 panorama. Photorealistic but rotation only \u2014 no parallax or depth.',
  mesh: '<strong>Depth Mesh</strong><br>Panorama projected onto depth sphere. Subtle parallax on head movement. Limited zoom range.',
};

// ---------------------------------------------------------------------------
// URL helper
// ---------------------------------------------------------------------------

function resolveUrl(url) {
  if (!url || url === '') return '';
  return url;
}

// ---------------------------------------------------------------------------
// Load manifest
// ---------------------------------------------------------------------------

async function loadManifest() {
  const url = window.MANIFEST_URL;
  const fetchUrl = resolveUrl(url);
  const resp = await fetch(fetchUrl);
  if (!resp.ok) throw new Error('Manifest not found (' + resp.status + '). Assets may not be generated yet.');
  manifest = await resp.json();

  // Build position dots
  manifest.positions.forEach(function (pos, idx) {
    const dot = document.createElement('button');
    dot.className = 'pos-dot' + (idx === 0 ? ' active' : '');
    dot.textContent = idx + 1;
    dot.dataset.idx = idx;
    positionSelector.appendChild(dot);
  });

  positionSelector.addEventListener('click', function (e) {
    const dot = e.target.closest('.pos-dot');
    if (!dot) return;
    const idx = parseInt(dot.dataset.idx, 10);
    switchPosition(idx);
  });
}

// ---------------------------------------------------------------------------
// Scene bounds computation (from viewer.js)
// ---------------------------------------------------------------------------

async function computeSceneBounds(splatUrl) {
  if (!splatUrl.toLowerCase().endsWith('.splat')) return null;
  try {
    const fetchBase = resolveUrl(splatUrl);
    const headResp = await fetch(fetchBase, { method: 'HEAD' });
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
      const resp = await fetch(fetchBase, { headers: { Range: 'bytes=' + offset + '-' + end } });
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
// Splat mode
// ---------------------------------------------------------------------------

async function initSplat() {
  if (splatViewer) return; // already initialized

  // Dynamically import GaussianSplats3D only when needed
  if (!GaussianSplats3D) {
    console.log('[demo] loading GaussianSplats3D library...');
    GaussianSplats3D = await import('@mkkellogg/gaussian-splats-3d');
    console.log('[demo] GaussianSplats3D loaded');
  }

  const splatUrl = manifest.splat_url;
  console.log('[demo] initSplat: computing bounds for', splatUrl);

  // Compute bounds with a timeout — skip if it takes too long
  try {
    splatBounds = await Promise.race([
      computeSceneBounds(splatUrl),
      new Promise(function (resolve) { setTimeout(function () { resolve(null); }, 5000); }),
    ]);
  } catch (e) {
    console.warn('[demo] bounds failed:', e);
    splatBounds = null;
  }
  console.log('[demo] bounds:', splatBounds);

  // Determine initial camera from first position
  const pos = manifest.positions[currentPositionIdx].world_xyz;
  console.log('[demo] creating viewer at position', pos);

  splatViewer = new GaussianSplats3D.Viewer({
    cameraUp: [0, 1, 0],
    initialCameraPosition: pos,
    initialCameraLookAt: [pos[0] + 1, pos[1], pos[2]],
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
  console.log('[demo] viewer created, loading splat scene...');

  // Don't await — let splats stream in while we show the UI
  splatViewer.addSplatScene(resolveUrl(splatUrl), { progressiveLoad: true })
    .then(function () { console.log('[demo] splat scene fully loaded'); })
    .catch(function (err) { console.warn('[demo] splat scene load error:', err); });
  splatViewer.start();
  console.log('[demo] viewer started');

  const sceneSize = splatBounds ? Math.max(splatBounds.size[0], splatBounds.size[1], splatBounds.size[2]) : 5;
  splatMoveSpeed = sceneSize * 0.05;

  if (isMobile) {
    initJoystick(splatMoveSpeed);
    animateSplatMovement(splatMoveSpeed);
  } else {
    initKeyboardControls(splatMoveSpeed);
  }
  console.log('[demo] initSplat complete');
}

function initKeyboardControls(speed) {
  document.addEventListener('keydown', function (e) {
    if (!splatCamera || currentMode !== 'splat') return;
    const direction = new THREE.Vector3();
    splatCamera.getWorldDirection(direction);
    const right = new THREE.Vector3();
    right.crossVectors(splatCamera.up, direction).normalize();

    switch (e.key.toLowerCase()) {
      case 'w': splatCamera.position.addScaledVector(direction, speed); break;
      case 's': splatCamera.position.addScaledVector(direction, -speed); break;
      case 'a': splatCamera.position.addScaledVector(right, speed); break;
      case 'd': splatCamera.position.addScaledVector(right, -speed); break;
    }
  });
}

function initJoystick(speed) {
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
    if (data.vector) { splatMovement.x = data.vector.x; splatMovement.y = data.vector.y; }
  });
  joystick.on('end', function () { splatMovement.x = 0; splatMovement.y = 0; });
}

function animateSplatMovement(speed) {
  speed = (speed || 0.25) * 0.1;
  function animate() {
    if (splatCamera && currentMode === 'splat' && (splatMovement.x !== 0 || splatMovement.y !== 0)) {
      const direction = new THREE.Vector3();
      splatCamera.getWorldDirection(direction);
      splatCamera.position.addScaledVector(direction, -splatMovement.y * speed);
      const right = new THREE.Vector3();
      right.crossVectors(splatCamera.up, direction).normalize();
      splatCamera.position.addScaledVector(right, splatMovement.x * speed);
    }
    requestAnimationFrame(animate);
  }
  animate();
}

// ---------------------------------------------------------------------------
// Three.js initialization (shared by skybox + mesh)
// ---------------------------------------------------------------------------

function initThree() {
  if (threeRenderer) return;

  threeRenderer = new THREE.WebGLRenderer({ antialias: true });
  threeRenderer.setPixelRatio(window.devicePixelRatio);
  threeRenderer.setSize(window.innerWidth, window.innerHeight);
  threeContainer.appendChild(threeRenderer.domElement);

  threeScene = new THREE.Scene();

  threeCamera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  threeCamera.position.set(0, 0, 0.01);

  threeControls = new OrbitControls(threeCamera, threeRenderer.domElement);
  threeControls.enableDamping = true;
  threeControls.dampingFactor = 0.1;
  threeControls.rotateSpeed = -0.5; // inverted for inside-sphere
  threeControls.target.set(0, 0, 0);

  // Ambient light for mesh mode
  threeScene.add(new THREE.AmbientLight(0xffffff, 1.0));

  // Create depth mesh sphere (hidden by default)
  const geo = new THREE.SphereGeometry(50, 128, 64);
  const mat = new THREE.MeshStandardMaterial({
    side: THREE.BackSide,
    displacementScale: -15,
  });
  meshSphere = new THREE.Mesh(geo, mat);
  meshSphere.visible = false;
  threeScene.add(meshSphere);

  function animateThree() {
    requestAnimationFrame(animateThree);
    if (currentMode !== 'splat') {
      threeControls.update();
      threeRenderer.render(threeScene, threeCamera);
    }
  }
  animateThree();

  window.addEventListener('resize', function () {
    if (threeRenderer) {
      threeCamera.aspect = window.innerWidth / window.innerHeight;
      threeCamera.updateProjectionMatrix();
      threeRenderer.setSize(window.innerWidth, window.innerHeight);
    }
  });
}

// ---------------------------------------------------------------------------
// Texture loading (with proxy + cache)
// ---------------------------------------------------------------------------

async function loadTexture(url) {
  const resolved = resolveUrl(url);
  if (!resolved) return null;
  return new Promise(function (resolve, reject) {
    new THREE.TextureLoader().load(resolved, resolve, undefined, reject);
  });
}

async function getPanoramaTexture(posIdx) {
  if (panoramaTextures.has(posIdx)) return panoramaTextures.get(posIdx);
  const url = manifest.positions[posIdx].panorama_url;
  const tex = await loadTexture(url);
  if (tex) {
    tex.mapping = THREE.EquirectangularReflectionMapping;
    tex.colorSpace = THREE.SRGBColorSpace;
  }
  panoramaTextures.set(posIdx, tex);
  return tex;
}

async function getDepthTexture(posIdx) {
  if (depthTextures.has(posIdx)) return depthTextures.get(posIdx);
  const url = manifest.positions[posIdx].depth_panorama_url;
  const tex = await loadTexture(url);
  depthTextures.set(posIdx, tex);
  return tex;
}

// ---------------------------------------------------------------------------
// Mode: Skybox
// ---------------------------------------------------------------------------

async function activateSkybox(posIdx) {
  initThree();
  meshSphere.visible = false;
  threeControls.enableZoom = false;
  threeControls.enablePan = false;

  const tex = await getPanoramaTexture(posIdx);
  threeScene.background = tex || new THREE.Color(0x1e293b);
}

// ---------------------------------------------------------------------------
// Mode: Mesh (depth displacement sphere)
// ---------------------------------------------------------------------------

async function activateMesh(posIdx) {
  initThree();
  threeScene.background = new THREE.Color(0x000000);
  meshSphere.visible = true;
  threeControls.enableZoom = true;
  threeControls.enablePan = false;
  threeControls.minDistance = 0.1;
  threeControls.maxDistance = 5;

  const [colorTex, depthTex] = await Promise.all([
    getPanoramaTexture(posIdx),
    getDepthTexture(posIdx),
  ]);

  meshSphere.material.map = colorTex || null;
  meshSphere.material.displacementMap = depthTex || null;
  meshSphere.material.displacementScale = depthTex ? -15 : 0;
  meshSphere.material.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Get/set camera direction for sync between modes
// ---------------------------------------------------------------------------

function getCurrentDirection() {
  if (currentMode === 'splat' && splatCamera) {
    const dir = new THREE.Vector3();
    splatCamera.getWorldDirection(dir);
    return dir;
  }
  if (threeCamera) {
    const dir = new THREE.Vector3();
    threeCamera.getWorldDirection(dir);
    return dir;
  }
  return new THREE.Vector3(0, 0, -1);
}

function applyDirectionToSplat(dir) {
  if (!splatCamera) return;
  const pos = splatCamera.position.clone();
  splatCamera.lookAt(pos.x + dir.x, pos.y + dir.y, pos.z + dir.z);
}

function applyDirectionToThree(dir) {
  if (!threeCamera) return;
  threeCamera.position.set(0, 0, 0.01);
  threeCamera.lookAt(dir.x, dir.y, dir.z);
  if (threeControls) threeControls.target.set(0, 0, 0);
}

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------

async function switchMode(newMode) {
  if (newMode === currentMode) return;

  showLoading('Switching to ' + newMode + '...');
  const dir = getCurrentDirection();

  currentMode = newMode;

  // Update button styles
  document.querySelectorAll('.mode-btn').forEach(function (btn) {
    btn.classList.toggle('active', btn.dataset.mode === newMode);
  });

  // Toggle containers
  splatContainer.style.display = newMode === 'splat' ? 'block' : 'none';
  threeContainer.style.display = newMode === 'splat' ? 'none' : 'block';
  joystickZone.style.display = newMode === 'splat' ? 'block' : 'none';

  try {
    if (newMode === 'splat') {
      await initSplat();
      applyDirectionToSplat(dir);
    } else if (newMode === 'skybox') {
      await activateSkybox(currentPositionIdx);
      applyDirectionToThree(dir);
    } else if (newMode === 'mesh') {
      await activateMesh(currentPositionIdx);
      applyDirectionToThree(dir);
    }
  } catch (err) {
    console.error('Mode switch error:', err);
  }

  infoPanel.innerHTML = INFO_TEXT[newMode];
  hideLoading();
}

// ---------------------------------------------------------------------------
// Position switching
// ---------------------------------------------------------------------------

async function switchPosition(newIdx) {
  if (newIdx === currentPositionIdx && currentMode === 'splat') return;
  if (newIdx < 0 || newIdx >= manifest.positions.length) return;

  currentPositionIdx = newIdx;

  // Update dot styles
  document.querySelectorAll('.pos-dot').forEach(function (dot) {
    dot.classList.toggle('active', parseInt(dot.dataset.idx, 10) === newIdx);
  });

  const pos = manifest.positions[newIdx].world_xyz;

  if (currentMode === 'splat') {
    if (splatCamera) {
      splatCamera.position.set(pos[0], pos[1], pos[2]);
    }
  } else if (currentMode === 'skybox') {
    showLoading('Loading position ' + (newIdx + 1) + '...');
    await activateSkybox(newIdx);
    hideLoading();
  } else if (currentMode === 'mesh') {
    showLoading('Loading position ' + (newIdx + 1) + '...');
    await activateMesh(newIdx);
    hideLoading();
  }
}

// ---------------------------------------------------------------------------
// Loading overlay
// ---------------------------------------------------------------------------

function showLoading(text) {
  const textEl = loadingOverlay.querySelector('.loading-text');
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
  const container = document.getElementById('demo-container');
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
// Event listeners
// ---------------------------------------------------------------------------

modeSwitcher.addEventListener('click', function (e) {
  const btn = e.target.closest('.mode-btn');
  if (!btn) return;
  switchMode(btn.dataset.mode);
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  try {
    await loadManifest();

    // SharedArrayBuffer requires crossOriginIsolated (HTTPS or localhost).
    // When unavailable, GaussianSplats3D can't work — disable splat button.
    if (!self.crossOriginIsolated) {
      console.warn('crossOriginIsolated is false — splat mode unavailable');
      const splatBtn = document.querySelector('[data-mode="splat"]');
      if (splatBtn) { splatBtn.disabled = true; splatBtn.title = 'Requires HTTPS or localhost'; splatBtn.style.opacity = '0.4'; }
    }

    // Default to skybox — always shows visual content immediately.
    // Splat mode initializes lazily when user clicks the button.
    await activateSkybox(currentPositionIdx);

    // Show skybox container, hide splat container
    splatContainer.style.display = 'none';
    threeContainer.style.display = 'block';
    joystickZone.style.display = 'none';

    // Set active button
    document.querySelectorAll('.mode-btn').forEach(function (btn) {
      btn.classList.toggle('active', btn.dataset.mode === 'skybox');
    });
    infoPanel.innerHTML = INFO_TEXT['skybox'];

    hideLoading();
  } catch (err) {
    console.error('Init error:', err);
    const textEl = loadingOverlay.querySelector('.loading-text');
    if (textEl) textEl.textContent = 'Failed to load demo: ' + err.message;
  }
}

init();
