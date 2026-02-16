import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { initDb } from './db/index.js';
import { jobsRouter } from './routes/jobs.js';
import { pagesRouter } from './routes/pages.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// View engine
app.set('view engine', 'ejs');
app.set('views', join(__dirname, 'views'));

// Middleware
app.use(cors());
app.use(express.json());

// Required for SharedArrayBuffer (used by gaussian-splats-3d web workers)
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
  next();
});

app.use(express.static(join(__dirname, 'public')));

// Routes
app.use('/api/jobs', jobsRouter);
app.use('/', pagesRouter);

// Proxy for splat files (bypasses COEP cross-origin restrictions)
app.get('/api/splat-proxy', async (req, res) => {
  const url = req.query.url;
  if (!url || !url.startsWith('https://')) {
    return res.status(400).json({ error: 'Missing or invalid url parameter' });
  }
  try {
    const headers = {};
    if (req.headers.range) headers.Range = req.headers.range;
    const upstream = await fetch(url, { method: req.method === 'HEAD' ? 'HEAD' : 'GET', headers });
    res.status(upstream.status);
    if (upstream.headers.get('content-length')) res.setHeader('Content-Length', upstream.headers.get('content-length'));
    if (upstream.headers.get('content-range')) res.setHeader('Content-Range', upstream.headers.get('content-range'));
    if (upstream.headers.get('content-type')) res.setHeader('Content-Type', upstream.headers.get('content-type'));
    if (req.method === 'HEAD') return res.end();
    const reader = upstream.body.getReader();
    const pump = async () => {
      const { done, value } = await reader.read();
      if (done) { res.end(); return; }
      res.write(value);
      await pump();
    };
    await pump();
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

app.head('/api/splat-proxy', async (req, res) => {
  const url = req.query.url;
  if (!url || !url.startsWith('https://')) return res.status(400).end();
  try {
    const upstream = await fetch(url, { method: 'HEAD' });
    if (upstream.headers.get('content-length')) res.setHeader('Content-Length', upstream.headers.get('content-length'));
    res.status(upstream.status).end();
  } catch (e) {
    res.status(502).end();
  }
});

// Test manifest for /demo when CDN assets aren't available yet
app.get('/api/demo-manifest', (req, res) => {
  res.json({
    splat_url: '/test-scene.splat',
    positions: [
      { id: 'pos_0', world_xyz: [0.0, 1.76, 1.16], panorama_url: '/test-panorama.jpg', depth_panorama_url: '/test-depth.jpg' },
      { id: 'pos_1', world_xyz: [-0.8, 1.76, 1.16], panorama_url: '/test-panorama.jpg', depth_panorama_url: '/test-depth.jpg' },
      { id: 'pos_2', world_xyz: [0.8, 1.76, 1.16], panorama_url: '/test-panorama.jpg', depth_panorama_url: '/test-depth.jpg' },
    ],
  });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

async function start() {
  await initDb();

  app.listen(PORT, () => {
    console.log(`SplatWalk running on http://localhost:${PORT}`);
  });
}

start().catch(console.error);
