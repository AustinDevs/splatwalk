import { v4 as uuidv4 } from 'uuid';
import { getDb } from '../db/index.js';

const MAX_CONCURRENT_JOBS = parseInt(process.env.MAX_CONCURRENT_JOBS || '2', 10);
const RATE_LIMIT_JOBS_PER_HOUR = parseInt(process.env.RATE_LIMIT_JOBS_PER_HOUR || '5', 10);
const RATE_LIMIT_WINDOW_MS = 60 * 60 * 1000;

class JobManager {
  constructor() {
    this.queue = [];
    this.activeJobs = new Set();
    this.eventEmitter = new Map();
  }

  rowToJob(row) {
    return {
      id: row.id,
      zillowUrl: row.zillow_url,
      status: row.status,
      progress: row.progress,
      metadata: row.metadata ? JSON.parse(row.metadata) : null,
      results: row.results ? JSON.parse(row.results) : [],
      error: row.error || null,
      dropletId: row.droplet_id || null,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      completedAt: row.completed_at || null,
    };
  }

  checkRateLimit(clientIp) {
    const db = getDb();
    const now = new Date();
    const windowStart = new Date(now.getTime() - RATE_LIMIT_WINDOW_MS);

    const record = db.prepare('SELECT * FROM rate_limits WHERE ip = ?').get(clientIp);

    if (!record) {
      db.prepare('INSERT INTO rate_limits (ip, count, window_start) VALUES (?, 1, ?)').run(
        clientIp,
        now.toISOString()
      );
      return true;
    }

    const recordWindowStart = new Date(record.window_start);

    if (recordWindowStart < windowStart) {
      db.prepare('UPDATE rate_limits SET count = 1, window_start = ? WHERE ip = ?').run(
        now.toISOString(),
        clientIp
      );
      return true;
    }

    if (record.count >= RATE_LIMIT_JOBS_PER_HOUR) {
      return false;
    }

    db.prepare('UPDATE rate_limits SET count = count + 1 WHERE ip = ?').run(clientIp);
    return true;
  }

  createJob(zillowUrl, clientIp) {
    const db = getDb();
    const now = new Date().toISOString();
    const id = uuidv4();

    db.prepare(`
      INSERT INTO jobs (id, zillow_url, status, progress, results, client_ip, created_at, updated_at)
      VALUES (?, ?, 'pending', 0, '[]', ?, ?, ?)
    `).run(id, zillowUrl, clientIp, now, now);

    const job = this.getJob(id);
    this.processQueue();
    return job;
  }

  getJob(id) {
    const db = getDb();
    const row = db.prepare('SELECT * FROM jobs WHERE id = ?').get(id);
    if (!row) return null;
    return this.rowToJob(row);
  }

  listJobs(page = 1, pageSize = 20, statusFilter = null) {
    const db = getDb();
    const offset = (page - 1) * pageSize;

    let query = 'SELECT * FROM jobs';
    let countQuery = 'SELECT COUNT(*) as count FROM jobs';
    const params = [];

    if (statusFilter) {
      query += ' WHERE status = ?';
      countQuery += ' WHERE status = ?';
      params.push(statusFilter);
    }

    query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?';

    const rows = db.prepare(query).all(...params, pageSize, offset);
    const countResult = db.prepare(countQuery).get(...params);

    return {
      jobs: rows.map((row) => this.rowToJob(row)),
      total: countResult.count,
    };
  }

  updateJobStatus(id, status, progress = null, additionalData = {}) {
    const db = getDb();
    const now = new Date().toISOString();

    const updates = ['status = ?', 'updated_at = ?'];
    const params = [status, now];

    if (progress !== null) {
      updates.push('progress = ?');
      params.push(progress);
    }

    if (additionalData.metadata) {
      updates.push('metadata = ?');
      params.push(JSON.stringify(additionalData.metadata));
    }

    if (additionalData.error) {
      updates.push('error = ?');
      params.push(additionalData.error);
    }

    if (additionalData.dropletId) {
      updates.push('droplet_id = ?');
      params.push(additionalData.dropletId);
    }

    if (status === 'completed' || status === 'failed') {
      updates.push('completed_at = ?');
      params.push(now);
      this.activeJobs.delete(id);
      this.processQueue();
    }

    params.push(id);
    db.prepare(`UPDATE jobs SET ${updates.join(', ')} WHERE id = ?`).run(...params);

    const job = this.getJob(id);
    if (job) {
      this.emitEvent(id, {
        type: 'status_update',
        jobId: id,
        status: job.status,
        progress: job.progress,
        timestamp: now,
      });
    }

    return job;
  }

  addPipelineResult(id, result) {
    const db = getDb();
    const job = this.getJob(id);
    if (!job) return null;

    const results = [...job.results, result];
    const now = new Date().toISOString();

    db.prepare('UPDATE jobs SET results = ?, updated_at = ? WHERE id = ?').run(
      JSON.stringify(results),
      now,
      id
    );

    this.emitEvent(id, {
      type: 'progress',
      jobId: id,
      result,
      timestamp: now,
    });

    return this.getJob(id);
  }

  deleteJob(id) {
    const db = getDb();
    db.prepare('DELETE FROM jobs WHERE id = ?').run(id);
    this.activeJobs.delete(id);
    return true;
  }

  canAcceptJob() {
    return this.activeJobs.size < MAX_CONCURRENT_JOBS;
  }

  waitForSlot(jobId) {
    if (this.canAcceptJob()) {
      this.activeJobs.add(jobId);
      return Promise.resolve();
    }

    return new Promise((resolve) => {
      this.queue.push({ id: jobId, resolve });
    });
  }

  processQueue() {
    while (this.queue.length > 0 && this.canAcceptJob()) {
      const next = this.queue.shift();
      if (next) {
        this.activeJobs.add(next.id);
        next.resolve();
      }
    }
  }

  subscribe(jobId, callback) {
    if (!this.eventEmitter.has(jobId)) {
      this.eventEmitter.set(jobId, new Set());
    }
    this.eventEmitter.get(jobId).add(callback);

    return () => {
      this.eventEmitter.get(jobId)?.delete(callback);
    };
  }

  emitEvent(jobId, event) {
    const listeners = this.eventEmitter.get(jobId);
    if (listeners) {
      listeners.forEach((cb) => cb(event));
    }
  }
}

export const jobManager = new JobManager();
