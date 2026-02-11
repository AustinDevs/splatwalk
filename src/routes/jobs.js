import { Router } from 'express';
import betterSse from 'better-sse';
const { createSession } = betterSse;
import { jobManager } from '../services/jobManager.js';
import { processJob } from '../services/jobProcessor.js';

export const jobsRouter = Router();

// Rate limiting middleware
const rateLimitMiddleware = async (req, res, next) => {
  const clientIp = req.ip || req.socket.remoteAddress || 'unknown';
  const allowed = jobManager.checkRateLimit(clientIp);

  if (!allowed) {
    res.status(429).json({
      success: false,
      error: 'Rate limit exceeded. Maximum 5 jobs per hour.',
    });
    return;
  }

  next();
};

// Validate Realtor.com URL
function isValidRealtorUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.hostname.includes('realtor.com');
  } catch {
    return false;
  }
}

// POST /api/jobs - Create new job
jobsRouter.post('/', rateLimitMiddleware, async (req, res) => {
  try {
    const { listingUrl } = req.body;

    if (!listingUrl) {
      res.status(400).json({
        success: false,
        error: 'listingUrl is required',
      });
      return;
    }

    if (!isValidRealtorUrl(listingUrl)) {
      res.status(400).json({
        success: false,
        error: 'Invalid Realtor.com URL',
      });
      return;
    }

    const clientIp = req.ip || req.socket.remoteAddress || 'unknown';
    const job = jobManager.createJob(listingUrl, clientIp);

    // Start processing in background
    processJob(job.id).catch((err) => {
      console.error(`Job ${job.id} failed:`, err);
    });

    res.status(201).json({
      success: true,
      data: job,
    });
  } catch (error) {
    console.error('Error creating job:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create job',
    });
  }
});

// GET /api/jobs - List jobs
jobsRouter.get('/', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const pageSize = Math.min(parseInt(req.query.pageSize) || 20, 100);
    const status = req.query.status || null;

    const { jobs, total } = jobManager.listJobs(page, pageSize, status);

    res.json({
      success: true,
      data: jobs,
      pagination: {
        page,
        pageSize,
        total,
        totalPages: Math.ceil(total / pageSize),
      },
    });
  } catch (error) {
    console.error('Error listing jobs:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to list jobs',
    });
  }
});

// GET /api/jobs/:id - Get job by ID
jobsRouter.get('/:id', async (req, res) => {
  try {
    const job = jobManager.getJob(req.params.id);

    if (!job) {
      res.status(404).json({
        success: false,
        error: 'Job not found',
      });
      return;
    }

    res.json({
      success: true,
      data: job,
    });
  } catch (error) {
    console.error('Error getting job:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get job',
    });
  }
});

// DELETE /api/jobs/:id - Delete job
jobsRouter.delete('/:id', async (req, res) => {
  try {
    const job = jobManager.getJob(req.params.id);

    if (!job) {
      res.status(404).json({
        success: false,
        error: 'Job not found',
      });
      return;
    }

    jobManager.deleteJob(req.params.id);

    res.json({
      success: true,
      data: { deleted: true },
    });
  } catch (error) {
    console.error('Error deleting job:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete job',
    });
  }
});

// GET /api/jobs/:id/events - SSE for real-time updates
jobsRouter.get('/:id/events', async (req, res) => {
  try {
    const job = jobManager.getJob(req.params.id);

    if (!job) {
      res.status(404).json({
        success: false,
        error: 'Job not found',
      });
      return;
    }

    const session = await createSession(req, res);

    // Send initial state
    session.push({
      type: 'status_update',
      jobId: job.id,
      status: job.status,
      progress: job.progress,
      timestamp: new Date().toISOString(),
    });

    // Subscribe to updates
    const unsubscribe = jobManager.subscribe(job.id, (event) => {
      session.push(event);
    });

    req.on('close', () => {
      unsubscribe();
    });
  } catch (error) {
    console.error('Error setting up SSE:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to establish SSE connection',
    });
  }
});
