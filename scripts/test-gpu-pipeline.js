#!/usr/bin/env node
/**
 * Test script to run GPU splat pipeline on test images
 * Usage: node scripts/test-gpu-pipeline.js
 */

import 'dotenv/config';
import { gpuOrchestrator } from '../src/services/gpuOrchestrator.js';
import { readdir } from 'fs/promises';
import { join } from 'path';
import { randomUUID } from 'crypto';

const TEST_IMAGES_DIR = './tmp/test-images';
const PIPELINE = process.argv[2] || 'instantsplat';

async function main() {
  const jobId = randomUUID();
  let dropletId = null;

  console.log('='.repeat(50));
  console.log('SplatWalk GPU Pipeline Test');
  console.log('='.repeat(50));
  console.log(`Job ID: ${jobId}`);
  console.log(`Pipeline: ${PIPELINE}`);
  console.log('');

  try {
    // 1. Find test images
    console.log('Step 1: Loading test images...');
    const files = await readdir(TEST_IMAGES_DIR);
    const videoFiles = files
      .filter(f => f.match(/\.(mov|mp4)$/i))
      .sort()
      .map(f => join(TEST_IMAGES_DIR, f));
    const imageFiles = files
      .filter(f => f.match(/\.(jpg|jpeg|png|webp)$/i))
      .sort()
      .map(f => join(TEST_IMAGES_DIR, f));

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

    // 2. Upload files to Spaces
    console.log(`\nStep 2: Uploading ${isVideo ? 'video' : 'images'} to DO Spaces...`);
    const startUpload = Date.now();
    const imageUrls = await gpuOrchestrator.uploadImages(jobId, inputFiles);
    console.log(`Uploaded ${imageUrls.length} file(s) in ${(Date.now() - startUpload) / 1000}s`);

    // 3. Create GPU droplet
    console.log('\nStep 3: Creating GPU droplet...');
    const startDroplet = Date.now();
    const droplet = await gpuOrchestrator.createDroplet(jobId);
    dropletId = droplet.id;
    console.log(`Created droplet ${dropletId}`);

    // 4. Wait for droplet to be ready
    console.log('\nStep 4: Waiting for droplet to be ready (this may take 2-5 minutes)...');
    const readyDroplet = await gpuOrchestrator.waitForDroplet(dropletId);
    console.log(`Droplet ready at ${readyDroplet.ip} in ${(Date.now() - startDroplet) / 1000}s`);

    // 5. Run pipeline
    console.log(`\nStep 5: Running ${PIPELINE} pipeline...`);
    const startPipeline = Date.now();
    const result = await gpuOrchestrator.runPipeline(jobId, dropletId, PIPELINE, imageUrls);
    console.log(`Pipeline completed in ${(Date.now() - startPipeline) / 1000}s`);

    // 6. Output results
    console.log('\n' + '='.repeat(50));
    console.log('SUCCESS!');
    console.log('='.repeat(50));
    console.log(`Splat URL: ${result.splatUrl}`);
    if (result.thumbnailUrl) {
      console.log(`Thumbnail: ${result.thumbnailUrl}`);
    }

  } catch (error) {
    console.error('\nERROR:', error.message);
    process.exitCode = 1;
  } finally {
    // Always destroy droplet
    if (dropletId) {
      console.log('\nStep 6: Cleaning up droplet...');
      try {
        await gpuOrchestrator.destroyDroplet(dropletId);
        console.log('Droplet destroyed');
      } catch (e) {
        console.error('Failed to destroy droplet:', e.message);
      }
    }
  }
}

main();
