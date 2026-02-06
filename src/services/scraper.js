import { chromium } from 'playwright';
import { writeFile, mkdir, rm } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';

const SCRAPE_TIMEOUT = parseInt(process.env.SCRAPE_TIMEOUT_MS || '60000', 10);

const USER_AGENTS = [
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
];

function getRandomUserAgent() {
  return USER_AGENTS[Math.floor(Math.random() * USER_AGENTS.length)];
}

function randomDelay(min = 500, max = 2000) {
  const delay = Math.floor(Math.random() * (max - min + 1) + min);
  return new Promise((resolve) => setTimeout(resolve, delay));
}

async function downloadImage(url, filepath) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download image: ${response.statusText}`);
  }
  const buffer = Buffer.from(await response.arrayBuffer());
  await writeFile(filepath, buffer);
}

export async function scrapeZillowListing(url) {
  let browser = null;

  try {
    browser = await chromium.launch({
      headless: true,
      args: [
        '--disable-blink-features=AutomationControlled',
        '--disable-dev-shm-usage',
        '--no-sandbox',
      ],
    });

    const context = await browser.newContext({
      userAgent: getRandomUserAgent(),
      viewport: { width: 1920, height: 1080 },
      locale: 'en-US',
      timezoneId: 'America/New_York',
    });

    await context.addInitScript(() => {
      Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
      });
    });

    const page = await context.newPage();
    page.setDefaultTimeout(SCRAPE_TIMEOUT);

    await randomDelay(1000, 2000);

    console.log(`Navigating to ${url}`);
    await page.goto(url, { waitUntil: 'domcontentloaded' });

    await randomDelay();

    const blocked = await page.$('text=Please verify you are a human');
    if (blocked) {
      throw new Error('Zillow has blocked the request. CAPTCHA detected.');
    }

    const metadata = await extractMetadata(page);
    console.log('Extracted metadata:', metadata);

    const imageUrls = await extractImageUrls(page);
    console.log(`Found ${imageUrls.length} images`);

    if (imageUrls.length === 0) {
      throw new Error('No images found on the listing');
    }

    metadata.imageCount = imageUrls.length;

    const tempDir = join(process.cwd(), 'tmp', `scrape-${Date.now()}`);
    if (!existsSync(tempDir)) {
      await mkdir(tempDir, { recursive: true });
    }

    const imagePaths = [];
    for (let i = 0; i < imageUrls.length; i++) {
      const imageUrl = imageUrls[i];
      const extension = getImageExtension(imageUrl);
      const filepath = join(tempDir, `image-${i.toString().padStart(3, '0')}.${extension}`);

      try {
        console.log(`Downloading image ${i + 1}/${imageUrls.length}`);
        await downloadImage(imageUrl, filepath);
        imagePaths.push(filepath);
        await randomDelay(200, 500);
      } catch (error) {
        console.warn(`Failed to download image ${i}:`, error);
      }
    }

    if (imagePaths.length === 0) {
      throw new Error('Failed to download any images');
    }

    return {
      metadata,
      images: imagePaths,
    };
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

async function extractMetadata(page) {
  const metadata = {
    address: '',
    imageCount: 0,
    scrapedAt: new Date().toISOString(),
  };

  try {
    const addressEl = await page.$('[data-testid="bdp-address"]');
    if (addressEl) {
      metadata.address = (await addressEl.textContent()) || '';
    } else {
      const h1 = await page.$('h1');
      if (h1) {
        metadata.address = (await h1.textContent()) || '';
      }
    }

    const priceEl = await page.$('[data-testid="price"]');
    if (priceEl) {
      metadata.price = (await priceEl.textContent()) || undefined;
    }

    const factsEl = await page.$$('[data-testid="bed-bath-item"]');
    for (const factEl of factsEl) {
      const text = (await factEl.textContent()) || '';
      const lowerText = text.toLowerCase();

      if (lowerText.includes('bd') || lowerText.includes('bed')) {
        const match = text.match(/(\d+)/);
        if (match) metadata.beds = parseInt(match[1], 10);
      } else if (lowerText.includes('ba') || lowerText.includes('bath')) {
        const match = text.match(/(\d+)/);
        if (match) metadata.baths = parseInt(match[1], 10);
      } else if (lowerText.includes('sqft') || lowerText.includes('sq')) {
        const match = text.replace(/,/g, '').match(/(\d+)/);
        if (match) metadata.sqft = parseInt(match[1], 10);
      }
    }
  } catch (error) {
    console.warn('Error extracting metadata:', error);
  }

  return metadata;
}

async function extractImageUrls(page) {
  const imageUrls = new Set();

  try {
    const galleryButton = await page.$('[data-testid="gallery-main"]');
    if (galleryButton) {
      await galleryButton.click();
      await randomDelay(1000, 2000);
    }

    let attempts = 0;
    const maxAttempts = 50;
    let lastCount = 0;
    let staleCount = 0;

    while (attempts < maxAttempts && staleCount < 3) {
      const images = await page.$$('img[src*="zillowstatic"], img[src*="photos.zillowstatic"]');

      for (const img of images) {
        const src = await img.getAttribute('src');
        if (src && isValidImageUrl(src)) {
          const highResSrc = getHighResUrl(src);
          imageUrls.add(highResSrc);
        }
      }

      const bgElements = await page.$$('[style*="background-image"]');
      for (const el of bgElements) {
        const style = await el.getAttribute('style');
        if (style) {
          const match = style.match(/url\(['"]?([^'")\s]+)['"]?\)/);
          if (match && isValidImageUrl(match[1])) {
            const highResSrc = getHighResUrl(match[1]);
            imageUrls.add(highResSrc);
          }
        }
      }

      const nextButton = await page.$('[aria-label="Next photo"]');
      if (nextButton) {
        await nextButton.click();
        await randomDelay(300, 600);
      } else {
        break;
      }

      if (imageUrls.size === lastCount) {
        staleCount++;
      } else {
        staleCount = 0;
        lastCount = imageUrls.size;
      }

      attempts++;
    }

    if (imageUrls.size === 0) {
      const allImages = await page.$$('img');
      for (const img of allImages) {
        const src = await img.getAttribute('src');
        if (src && isValidImageUrl(src)) {
          const highResSrc = getHighResUrl(src);
          imageUrls.add(highResSrc);
        }
      }
    }
  } catch (error) {
    console.warn('Error extracting image URLs:', error);
  }

  return Array.from(imageUrls);
}

function isValidImageUrl(url) {
  if (!url) return false;
  if (url.startsWith('data:')) return false;
  if (url.includes('placeholder')) return false;
  if (url.includes('logo')) return false;
  if (url.includes('icon')) return false;

  return (
    url.includes('zillowstatic') ||
    url.includes('photos.zillow') ||
    url.includes('/p_e/') ||
    url.includes('/p_f/')
  );
}

function getHighResUrl(url) {
  return url
    .replace(/\/p_c\//, '/p_f/')
    .replace(/\/p_d\//, '/p_f/')
    .replace(/\/p_e\//, '/p_f/')
    .replace(/_uncropped_scaled_within_\d+_\d+/, '')
    .replace(/\?.*$/, '');
}

function getImageExtension(url) {
  const match = url.match(/\.(jpe?g|png|webp)/i);
  return match ? match[1].toLowerCase() : 'jpg';
}

export async function cleanupScrapeDir(dirPath) {
  try {
    if (existsSync(dirPath)) {
      await rm(dirPath, { recursive: true, force: true });
    }
  } catch (error) {
    console.warn(`Failed to cleanup scrape directory ${dirPath}:`, error);
  }
}
