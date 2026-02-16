/*
 * Service worker that injects COEP/COOP headers so crossOriginIsolated = true.
 * Required for SharedArrayBuffer (used by GaussianSplats3D web worker sort).
 * GitHub Pages doesn't support custom response headers, so we intercept here.
 */

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (e) => {
  if (e.request.cache === 'only-if-cached' && e.request.mode !== 'same-origin') return;

  e.respondWith(
    fetch(e.request).then((response) => {
      // Opaque responses (cross-origin no-cors) have status 0 which is
      // invalid for the Response constructor — return them unchanged.
      if (response.status === 0) return response;

      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    })
  );
});
