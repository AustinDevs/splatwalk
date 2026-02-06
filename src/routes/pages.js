import { Router } from 'express';
import { jobManager } from '../services/jobManager.js';

export const pagesRouter = Router();

const STATUS_LABELS = {
  pending: 'Queued',
  scraping: 'Scraping Zillow',
  uploading: 'Uploading Images',
  provisioning: 'Starting GPU',
  processing_viewcrafter: 'Running ViewCrafter',
  processing_instantsplat: 'Running InstantSplat',
  processing_combined: 'Running Combined Pipeline',
  converting: 'Converting to KSPLAT',
  completed: 'Complete',
  failed: 'Failed',
};

// Home page
pagesRouter.get('/', (req, res) => {
  const { jobs } = jobManager.listJobs(1, 12);
  res.render('home', { jobs, statusLabels: STATUS_LABELS });
});

// Job status page
pagesRouter.get('/jobs/:id', (req, res) => {
  const job = jobManager.getJob(req.params.id);

  if (!job) {
    return res.status(404).render('error', {
      title: 'Job Not Found',
      message: 'The requested job could not be found.',
    });
  }

  res.render('job-status', { job, statusLabels: STATUS_LABELS });
});

// Viewer page
pagesRouter.get('/view/:id', (req, res) => {
  const job = jobManager.getJob(req.params.id);

  if (!job) {
    return res.status(404).render('error', {
      title: 'Job Not Found',
      message: 'The requested job could not be found.',
    });
  }

  if (job.status !== 'completed') {
    return res.redirect(`/jobs/${job.id}`);
  }

  const successfulResults = job.results.filter((r) => r.status === 'success');

  if (successfulResults.length === 0) {
    return res.render('error', {
      title: 'No Results',
      message: 'All pipelines failed for this listing.',
    });
  }

  res.render('viewer', { job, results: successfulResults });
});
