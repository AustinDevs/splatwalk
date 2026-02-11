#!/usr/bin/env node
/**
 * Upload a local image directory as a named dataset zip to DO Spaces.
 *
 * Usage:
 *   node scripts/upload-dataset.js <name> <path-to-images>
 *
 * Examples:
 *   node scripts/upload-dataset.js aukerman ./odm_data_aukerman-master/images
 *   node scripts/upload-dataset.js parking-lot ./tmp/parking-lot-photos
 *
 * This uploads to: s3://splatwalk/datasets/<name>.zip
 * Rerun with the same name to overwrite with a new set.
 */

import 'dotenv/config';
import { execSync } from 'child_process';
import { readFile, stat, readdir, mkdtemp, rm } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';

const DO_SPACES_KEY = process.env.DO_SPACES_KEY || '';
const DO_SPACES_SECRET = process.env.DO_SPACES_SECRET || '';
const DO_SPACES_BUCKET = process.env.DO_SPACES_BUCKET || 'splatwalk';
const DO_SPACES_REGION = process.env.DO_SPACES_REGION || 'nyc3';
const DO_SPACES_ENDPOINT = process.env.DO_SPACES_ENDPOINT || 'https://nyc3.digitaloceanspaces.com';

const name = process.argv[2];
const inputDir = process.argv[3];

if (!name || !inputDir) {
  console.error('Usage: node scripts/upload-dataset.js <name> <path-to-images>');
  console.error('  e.g. node scripts/upload-dataset.js aukerman ./odm_data_aukerman-master/images');
  process.exit(1);
}

async function main() {
  // Validate input dir
  const dirStat = await stat(inputDir).catch(() => null);
  if (!dirStat?.isDirectory()) {
    console.error(`Error: ${inputDir} is not a directory`);
    process.exit(1);
  }

  const files = await readdir(inputDir);
  const imageFiles = files.filter(f => f.match(/\.(jpg|jpeg|png|webp|mov|mp4)$/i));
  if (imageFiles.length === 0) {
    console.error(`Error: No image/video files found in ${inputDir}`);
    process.exit(1);
  }

  console.log(`Dataset: ${name}`);
  console.log(`Source:  ${inputDir}`);
  console.log(`Files:   ${imageFiles.length} images/videos`);
  console.log('');

  // Create zip in a temp directory
  const tmpDir = await mkdtemp(join(tmpdir(), 'dataset-'));
  const zipPath = join(tmpDir, `${name}.zip`);

  try {
    console.log('Zipping...');
    // zip just the image files (flat, no directory structure)
    execSync(`zip -j "${zipPath}" ${imageFiles.map(f => `"${join(inputDir, f)}"`).join(' ')}`, {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const zipStat = await stat(zipPath);
    const sizeMB = (zipStat.size / 1024 / 1024).toFixed(1);
    console.log(`Zip created: ${sizeMB} MB`);

    // Upload to Spaces
    console.log(`Uploading to datasets/${name}.zip ...`);
    const startUpload = Date.now();

    const s3Client = new S3Client({
      endpoint: DO_SPACES_ENDPOINT,
      region: DO_SPACES_REGION,
      credentials: {
        accessKeyId: DO_SPACES_KEY,
        secretAccessKey: DO_SPACES_SECRET,
      },
      forcePathStyle: false,
    });

    const zipContent = await readFile(zipPath);
    await s3Client.send(
      new PutObjectCommand({
        Bucket: DO_SPACES_BUCKET,
        Key: `datasets/${name}.zip`,
        Body: zipContent,
        ContentType: 'application/zip',
        ACL: 'public-read',
      })
    );

    const elapsed = ((Date.now() - startUpload) / 1000).toFixed(1);
    const url = `${DO_SPACES_ENDPOINT}/${DO_SPACES_BUCKET}/datasets/${name}.zip`;

    console.log(`\nUploaded in ${elapsed}s`);
    console.log(`URL: ${url}`);
    console.log(`\nTo run a job with this dataset:`);
    console.log(`  node scripts/test-gpu-pipeline.js walkable ${name}`);
  } finally {
    await rm(tmpDir, { recursive: true, force: true });
  }
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
