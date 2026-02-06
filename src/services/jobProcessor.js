import { jobManager } from './jobManager.js';
import { scrapeZillowListing } from './scraper.js';
import { gpuOrchestrator } from './gpuOrchestrator.js';

const PIPELINES = ['viewcrafter', 'instantsplat', 'combined'];

export async function processJob(jobId) {
  console.log(`Starting job ${jobId}`);

  try {
    await jobManager.waitForSlot(jobId);

    // Phase 1: Scraping
    await jobManager.updateJobStatus(jobId, 'scraping', 5);

    const job = jobManager.getJob(jobId);
    if (!job) throw new Error('Job not found');

    const scrapeResult = await scrapeZillowListing(job.zillowUrl);

    await jobManager.updateJobStatus(jobId, 'scraping', 15, {
      metadata: scrapeResult.metadata,
    });

    // Phase 2: Upload images
    await jobManager.updateJobStatus(jobId, 'uploading', 20);

    const imageUrls = await gpuOrchestrator.uploadImages(jobId, scrapeResult.images);

    await jobManager.updateJobStatus(jobId, 'uploading', 25);

    // Phase 3: Provision GPU droplet
    await jobManager.updateJobStatus(jobId, 'provisioning', 30);

    const droplet = await gpuOrchestrator.createDroplet(jobId);

    await jobManager.updateJobStatus(jobId, 'provisioning', 40, {
      dropletId: droplet.id,
    });

    try {
      await gpuOrchestrator.waitForDroplet(droplet.id);
      await jobManager.updateJobStatus(jobId, 'provisioning', 45);

      // Phase 4: Run pipelines
      let progressBase = 45;
      const progressPerPipeline = 15;

      for (const pipeline of PIPELINES) {
        const statusKey = `processing_${pipeline}`;
        await jobManager.updateJobStatus(jobId, statusKey, progressBase);

        const startTime = Date.now();

        try {
          const result = await gpuOrchestrator.runPipeline(
            jobId,
            droplet.id,
            pipeline,
            imageUrls
          );

          jobManager.addPipelineResult(jobId, {
            pipeline,
            splatUrl: result.splatUrl,
            thumbnailUrl: result.thumbnailUrl,
            processingTimeMs: Date.now() - startTime,
            status: 'success',
          });
        } catch (error) {
          console.error(`Pipeline ${pipeline} failed:`, error);
          jobManager.addPipelineResult(jobId, {
            pipeline,
            splatUrl: '',
            processingTimeMs: Date.now() - startTime,
            status: 'failed',
            error: error.message,
          });
        }

        progressBase += progressPerPipeline;
        await jobManager.updateJobStatus(jobId, statusKey, progressBase);
      }

      // Phase 5: Complete
      await jobManager.updateJobStatus(jobId, 'completed', 100);
      console.log(`Job ${jobId} completed successfully`);
    } finally {
      await gpuOrchestrator.destroyDroplet(droplet.id).catch((err) => {
        console.error(`Failed to destroy droplet ${droplet.id}:`, err);
      });
    }
  } catch (error) {
    console.error(`Job ${jobId} failed:`, error);
    await jobManager.updateJobStatus(jobId, 'failed', 0, {
      error: error.message,
    });
    throw error;
  }
}
