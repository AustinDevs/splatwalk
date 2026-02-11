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

// Initialize viewer
async function initViewer(splatUrl) {
  if (viewer) {
    viewer.dispose();
  }

  try {
    viewer = new GaussianSplats3D.Viewer({
      cameraUp: [0, 1, 0],
      initialCameraPosition: [0, 1.6, 5],
      initialCameraLookAt: [0, 1.6, 0],
      selfDrivenMode: true,
      useBuiltInControls: !isMobile,
      rootElement: canvas,
    });

    camera = viewer.camera;

    await viewer.addSplatScene(splatUrl, {
      progressiveLoad: true,
    });

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
