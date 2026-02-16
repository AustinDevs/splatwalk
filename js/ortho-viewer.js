// ---------------------------------------------------------------------------
// Ortho Viewer — Leaflet-based 2D deep-zoom tile viewer
// ---------------------------------------------------------------------------

(function () {
  'use strict';

  var loadingOverlay = document.getElementById('loading-overlay');
  var fullscreenBtn = document.getElementById('fullscreen-btn');
  var splatBtn = document.getElementById('splat-btn');

  var manifest = null;
  var map = null;

  // -------------------------------------------------------------------------
  // Load manifest
  // -------------------------------------------------------------------------

  function loadManifest(callback) {
    var url = window.ORTHO_MANIFEST_URL;
    var textEl = loadingOverlay.querySelector('.loading-text');
    if (textEl) textEl.textContent = 'Loading manifest...';

    fetch(url)
      .then(function (resp) {
        if (!resp.ok) throw new Error('Manifest not found (' + resp.status + ')');
        return resp.json();
      })
      .then(function (data) {
        manifest = data;
        callback(null, data);
      })
      .catch(function (err) {
        callback(err);
      });
  }

  // -------------------------------------------------------------------------
  // Initialize map
  // -------------------------------------------------------------------------

  function initMap() {
    var imgWidth = manifest.image_width || 32768;
    var imgHeight = manifest.image_height || 32768;
    var tileSize = manifest.tile_size || 512;
    var minZoom = manifest.min_zoom || 0;
    var maxZoom = manifest.max_zoom || 4;
    var maxEnhanced = manifest.max_enhanced_zoom || 2;

    // Create map first so we can use unproject()
    map = L.map('map', {
      crs: L.CRS.Simple,
      minZoom: minZoom,
      maxZoom: maxZoom,
      zoomSnap: 1,
      zoomDelta: 1,
      attributionControl: false,
    });

    // Convert pixel coordinates at max zoom to CRS.Simple latlng.
    // unproject([x, y], zoom) maps pixel coords to map coords.
    var southWest = map.unproject([0, imgHeight], maxZoom);
    var northEast = map.unproject([imgWidth, 0], maxZoom);
    var bounds = L.latLngBounds(southWest, northEast);

    map.setMaxBounds(bounds.pad(0.25));
    map.options.maxBoundsViscosity = 0.8;
    map.fitBounds(bounds);

    // Tile layer — use tms:true since our tiles are stored with Y=0 at top
    var tileUrl = manifest.tile_url_template;
    L.tileLayer(tileUrl, {
      tileSize: tileSize,
      minZoom: minZoom,
      maxZoom: maxZoom,
      bounds: bounds,
      noWrap: true,
      tms: true,
      crossOrigin: '',
      errorTileUrl: '',
    }).addTo(map);

    // Zoom info control
    var ZoomInfo = L.Control.extend({
      options: { position: 'bottomleft' },
      onAdd: function () {
        var div = L.DomUtil.create('div', 'zoom-info');
        div.id = 'zoom-info';
        this._update(div, map.getZoom());
        return div;
      },
      _update: function (el, zoom) {
        var level = Math.round(zoom);
        var badge = level <= maxEnhanced ? 'enhanced' : 'upscaled';
        var label = level <= maxEnhanced ? 'AI enhanced' : 'upscaled';
        el.innerHTML = 'Zoom ' + level + '/' + maxZoom +
          ' <span class="zoom-badge ' + badge + '">' + label + '</span>';
      },
    });

    var zoomInfo = new ZoomInfo();
    zoomInfo.addTo(map);

    map.on('zoomend', function () {
      var el = document.getElementById('zoom-info');
      if (el) zoomInfo._update(el, map.getZoom());
    });

    // "3D Splat" button — link to splat viewer if available
    if (manifest.splat_manifest_url) {
      splatBtn.style.display = '';
      splatBtn.addEventListener('click', function () {
        window.location.href = 'index.html?manifest=' +
          encodeURIComponent(manifest.splat_manifest_url);
      });
    }
  }

  // -------------------------------------------------------------------------
  // Fullscreen
  // -------------------------------------------------------------------------

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

  // -------------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------------

  loadManifest(function (err) {
    if (err) {
      var textEl = loadingOverlay.querySelector('.loading-text');
      if (textEl) textEl.textContent = 'Failed to load: ' + err.message;
      console.error('Manifest error:', err);
      return;
    }

    initMap();
    loadingOverlay.classList.add('hidden');
  });
})();
