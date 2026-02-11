#!/usr/bin/env node
/**
 * Run a GPU splat pipeline.
 *
 * Usage:
 *   node scripts/test-gpu-pipeline.js <pipeline> <dataset-or-path> [--wait]
 *
 * Fire-and-forget (walkable pipeline):
 *   node scripts/test-gpu-pipeline.js walkable aukerman
 *   node scripts/test-gpu-pipeline.js walkable aukerman --wait
 *
 * Interactive SSH (short pipelines):
 *   node scripts/test-gpu-pipeline.js instantsplat ./tmp/test-images
 *
 * Datasets are uploaded once with upload-dataset.js and referenced by name.
 * Local paths are uploaded per-job (individual files to Spaces).
 */

import 'dotenv/config';
import { gpuOrchestrator } from '../src/services/gpuOrchestrator.js';
import { readdir, stat } from 'fs/promises';
import { join } from 'path';
import { randomUUID } from 'crypto';

const PIPELINE = process.argv[2] || 'instantsplat';
const INPUT = process.argv[3] || './tmp/test-images';
const WAIT_MODE = process.argv.includes('--wait');

// Pipelines that use fire-and-forget (long-running)
const FIRE_AND_FORGET_PIPELINES = ['walkable'];

async function isLocalDir(path) {
  try {
    return (await stat(path)).isDirectory();
  } catch {
    return false;
  }
}

async function pollForManifest(manifestUrl, intervalMs = 60000, maxAttempts = 300) {
  console.log(`\nPolling for manifest every ${intervalMs / 1000}s...`);
  console.log(`URL: ${manifestUrl}\n`);

  for (let i = 1; i <= maxAttempts; i++) {
    try {
      const response = await fetch(manifestUrl);
      if (response.ok) {
        const manifest = await response.json();
        return manifest;
      }
    } catch {
      // Not ready yet
    }

    const elapsed = (i * intervalMs) / 60000;
    process.stdout.write(`\r  Waiting... ${elapsed.toFixed(0)}m elapsed (attempt ${i}/${maxAttempts})`);
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error('Manifest did not appear within timeout');
}

async function main() {
  const jobId = randomUUID();
  let dropletId = null;
  const useFireAndForget = FIRE_AND_FORGET_PIPELINES.includes(PIPELINE);
  const isLocal = await isLocalDir(INPUT);
  const datasetName = !isLocal ? INPUT : null;

  console.log('='.repeat(50));
  console.log('SplatWalk GPU Pipeline');
  console.log('='.repeat(50));
  console.log(`Job ID:   ${jobId}`);
  console.log(`Pipeline: ${PIPELINE}`);
  console.log(`Input:    ${datasetName ? `dataset "${datasetName}"` : INPUT}`);
  console.log(`Mode:     ${useFireAndForget ? 'fire-and-forget' : 'interactive (SSH)'}`);
  console.log('');

  try {
    if (useFireAndForget) {
      // --- Fire-and-forget flow ---
      if (datasetName) {
        // Dataset already in Spaces — just reference by name
        const datasetUrl = gpuOrchestrator.getDatasetUrl(datasetName);
        console.log(`Step 1: Verifying dataset exists at ${datasetUrl}...`);
        const check = await fetch(datasetUrl, { method: 'HEAD' });
        if (!check.ok) {
          throw new Error(
            `Dataset "${datasetName}" not found in Spaces. Upload it first:\n` +
            `  node scripts/upload-dataset.js ${datasetName} /path/to/images`
          );
        }
        console.log(`Dataset found (${(parseInt(check.headers.get('content-length') || '0') / 1024 / 1024).toFixed(1)} MB)`);

        console.log('\nStep 2: Launching autonomous GPU job...');
        const { manifestUrl, dropletId: dId } = await gpuOrchestrator.launchJob(jobId, PIPELINE, { datasetName });

        printLaunched(dId, manifestUrl);
      } else {
        // Local dir — zip + upload as ephemeral dataset, then launch
        console.log('Step 1: Uploading images to Spaces...');
        const files = await readdir(INPUT);
        const imageFiles = files
          .filter(f => f.match(/\.(jpg|jpeg|png|webp|mov|mp4)$/i))
          .sort()
          .map(f => join(INPUT, f));

        if (imageFiles.length < 2) {
          throw new Error('Need at least 2 images');
        }
        console.log(`Found ${imageFiles.length} files`);

        const startUpload = Date.now();
        const imageUrls = await gpuOrchestrator.uploadImages(jobId, imageFiles);
        console.log(`Uploaded ${imageUrls.length} file(s) in ${(Date.now() - startUpload) / 1000}s`);

        console.log('\nStep 2: Launching autonomous GPU job...');
        const { manifestUrl, dropletId: dId } = await gpuOrchestrator.launchJob(jobId, PIPELINE, { imageUrls });

        printLaunched(dId, manifestUrl);
      }

      if (WAIT_MODE) {
        const manifestUrl = `${process.env.DO_SPACES_ENDPOINT || 'https://nyc3.digitaloceanspaces.com'}/${process.env.DO_SPACES_BUCKET || 'splatwalk'}/jobs/${jobId}/output/${PIPELINE}/manifest.json`;
        console.log('\n--wait flag set, polling for completion...');
        const manifest = await pollForManifest(manifestUrl);

        console.log('\n\n' + '='.repeat(50));
        if (manifest.status === 'success') {
          console.log('SUCCESS!');
          console.log('='.repeat(50));
          console.log(`Splat URL: ${manifest.splatUrl}`);
          if (manifest.thumbnailUrl) console.log(`Thumbnail: ${manifest.thumbnailUrl}`);
          if (manifest.confidence) console.log(`Confidence: ${JSON.stringify(manifest.confidence)}`);
        } else {
          console.log('PIPELINE FAILED');
          console.log('='.repeat(50));
          console.log(`Error: ${manifest.error}`);
          process.exitCode = 1;
        }
      }
    } else {
      // --- Interactive SSH flow (existing behavior for short pipelines) ---
      console.log('Step 1: Loading images...');
      const files = await readdir(INPUT);
      const videoFiles = files
        .filter(f => f.match(/\.(mov|mp4)$/i))
        .sort()
        .map(f => join(INPUT, f));
      const imageFiles = files
        .filter(f => f.match(/\.(jpg|jpeg|png|webp)$/i))
        .sort()
        .map(f => join(INPUT, f));

      const inputFiles = videoFiles.length > 0 ? videoFiles : imageFiles;
      const isVideo = videoFiles.length > 0;

      if (isVideo) {
        console.log(`Found ${videoFiles.length} video file(s)`);
      } else {
        console.log(`Found ${imageFiles.length} images`);
      }

      if (!isVideo && inputFiles.length < 2) {
        throw new Error('Need at least 2 images or 1 video file');
      }

      console.log(`\nStep 2: Uploading ${isVideo ? 'video' : 'images'} to DO Spaces...`);
      const startUpload = Date.now();
      const imageUrls = await gpuOrchestrator.uploadImages(jobId, inputFiles);
      console.log(`Uploaded ${imageUrls.length} file(s) in ${(Date.now() - startUpload) / 1000}s`);

      console.log('\nStep 3: Creating GPU droplet...');
      const startDroplet = Date.now();
      const droplet = await gpuOrchestrator.createDroplet(jobId);
      dropletId = droplet.id;
      console.log(`Created droplet ${dropletId}`);

      console.log('\nStep 4: Waiting for droplet to be ready (this may take 2-5 minutes)...');
      const readyDroplet = await gpuOrchestrator.waitForDroplet(dropletId);
      console.log(`Droplet ready at ${readyDroplet.ip} in ${(Date.now() - startDroplet) / 1000}s`);

      console.log(`\nStep 5: Running ${PIPELINE} pipeline...`);
      const startPipeline = Date.now();
      const result = await gpuOrchestrator.runPipeline(jobId, dropletId, PIPELINE, imageUrls);
      console.log(`Pipeline completed in ${(Date.now() - startPipeline) / 1000}s`);

      console.log('\n' + '='.repeat(50));
      console.log('SUCCESS!');
      console.log('='.repeat(50));
      console.log(`Splat URL: ${result.splatUrl}`);
      if (result.thumbnailUrl) console.log(`Thumbnail: ${result.thumbnailUrl}`);
    }
  } catch (error) {
    console.error('\nERROR:', error.message);
    process.exitCode = 1;
  } finally {
    if (dropletId) {
      console.log('\nCleaning up droplet...');
      try {
        await gpuOrchestrator.destroyDroplet(dropletId);
        console.log('Droplet destroyed');
      } catch (e) {
        console.error('Failed to destroy droplet:', e.message);
      }
    }
  }
}

function printLaunched(dropletId, manifestUrl) {
  console.log('\n' + '='.repeat(50));
  console.log('JOB LAUNCHED (fire-and-forget)');
  console.log('='.repeat(50));
  console.log(`Droplet ID: ${dropletId}`);
  console.log(`Monitor results at: ${manifestUrl}`);
  console.log(`Expected completion: 2-4 hours`);
  console.log(`Droplet will self-destruct when done.`);
}

main();
