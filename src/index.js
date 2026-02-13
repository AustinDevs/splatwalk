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
