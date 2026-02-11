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

export async function scrapeRealtorListing(url, retryCount = 0) {
  const MAX_RETRIES = 3;
  let browser = null;

  try {
    const headless = process.env.SCRAPER_HEADLESS !== 'false';
    browser = await chromium.launch({
      headless,
      channel: 'chrome',
      args: [
        '--disable-blink-features=AutomationControlled',
        '--disable-dev-shm-usage',
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-infobars',
        '--window-size=1920,1080',
        '--start-maximized',
      ],
    });

    const context = await browser.newContext({
      userAgent: getRandomUserAgent(),
      viewport: { width: 1920, height: 1080 },
      locale: 'en-US',
      timezoneId: 'America/New_York',
      javaScriptEnabled: true,
      ignoreHTTPSErrors: true,
      extraHTTPHeaders: {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
      },
    });

    await context.addInitScript(() => {
      // Hide webdriver
      Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
      });

      // Override plugins
      Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5],
      });

      // Override languages
      Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en'],
      });

      // Override permissions
      const originalQuery = window.navigator.permissions.query;
      window.navigator.permissions.query = (parameters) =>
        parameters.name === 'notifications'
          ? Promise.resolve({ state: Notification.permission })
          : originalQuery(parameters);

      // Override chrome
      window.chrome = { runtime: {} };
    });

    const page = await context.newPage();
    page.setDefaultTimeout(SCRAPE_TIMEOUT);

    // Random delay before navigation
    await randomDelay(2000, 4000);

    console.log(`Navigating to ${url}`);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });

    // Wait a bit for the page to fully load
    await randomDelay(2000, 3000);

    // Check if blocked by Realtor.com
    const blocked = await page.$('text=Your request could not be processed');
    if (blocked) {
      if (browser) await browser.close();
      if (retryCount < MAX_RETRIES) {
        console.log(`Blocked, retrying (${retryCount + 1}/${MAX_RETRIES})...`);
        await randomDelay(3000, 5000);
        return scrapeRealtorListing(url, retryCount + 1);
      }
      throw new Error('Realtor.com has blocked the request. Please try again later.');
    }

    const metadata = await extractMetadata(page);
    console.log('Extracted metadata:', metadata);

    const imageUrls = await extractImageUrls(page, metadata.address);
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
    // Extract address from h1 or breadcrumb
    const h1 = await page.$('h1');
    if (h1) {
      metadata.address = ((await h1.textContent()) || '').trim();
    }

    // Extract price
    const priceEl = await page.$('[data-testid="list-price"], [class*="price"]');
    if (priceEl) {
      const priceText = (await priceEl.textContent()) || '';
      const priceMatch = priceText.match(/\$[\d,]+/);
      if (priceMatch) {
        metadata.price = priceMatch[0];
      }
    }

    // Extract beds, baths, sqft from the listing details
    const detailsList = await page.$$('li');
    for (const item of detailsList) {
      const text = ((await item.textContent()) || '').toLowerCase();

      if (text.includes('bed')) {
        const match = text.match(/(\d+)\s*bed/);
        if (match) metadata.beds = parseInt(match[1], 10);
      } else if (text.includes('bath')) {
        const match = text.match(/([\d.]+)\s*bath/);
        if (match) metadata.baths = parseFloat(match[1]);
      } else if (text.includes('sqft') || text.includes('square')) {
        const match = text.replace(/,/g, '').match(/(\d+)\s*(?:sqft|square)/);
        if (match) metadata.sqft = parseInt(match[1], 10);
      }
    }

    // Try to get property type
    const propertyTypeEl = await page.$('[data-testid="property-type"], [class*="property-type"]');
    if (propertyTypeEl) {
      metadata.propertyType = ((await propertyTypeEl.textContent()) || '').trim();
    }
  } catch (error) {
    console.warn('Error extracting metadata:', error);
  }

  return metadata;
}

async function extractImageUrls(page, address) {
  const imageUrls = new Set();

  try {
    // Click on the main photo to open the gallery
    const galleryButton = await page.$('button:has-text("View all"), [aria-label*="photo"], [data-testid*="gallery"]');
    if (galleryButton) {
      await galleryButton.click();
      await randomDelay(1000, 2000);
    }

    // Get the property hash from the first listing image
    let propertyHash = null;
    const firstImage = await page.$(`img[alt*="featured at"]`);
    if (firstImage) {
      const src = await firstImage.getAttribute('src');
      if (src) {
        const hashMatch = src.match(/rdcpix\.com\/([a-f0-9]+l)-m/);
        if (hashMatch) {
          propertyHash = hashMatch[1];
        }
      }
    }

    // Scroll through the gallery to load all lazy-loaded images
    for (let i = 0; i < 20; i++) {
      await page.evaluate(() => {
        const containers = document.querySelectorAll('[role="dialog"] > div:last-child, [data-testid*="gallery"], [class*="gallery"]');
        containers.forEach(el => {
          if (el.scrollHeight > el.clientHeight) {
            el.scrollTop += 400;
          }
        });
        const dialog = document.querySelector('[role="dialog"]');
        if (dialog) {
          const scrollable = dialog.querySelector('div:nth-child(2)');
          if (scrollable && scrollable.scrollHeight > scrollable.clientHeight) {
            scrollable.scrollTop += 400;
          }
        }
      });
      await randomDelay(150, 300);
    }

    // Extract all images from rdcpix.com
    const images = await page.$$('img');
    for (const img of images) {
      const src = await img.getAttribute('src');
      if (src && isValidImageUrl(src, propertyHash)) {
        const highResSrc = getHighResUrl(src);
        imageUrls.add(highResSrc);
      }
    }

    // If we didn't find many images, try getting them from the carousel
    if (imageUrls.size < 5) {
      let attempts = 0;
      const maxAttempts = 30;

      while (attempts < maxAttempts) {
        const images = await page.$$('img');
        for (const img of images) {
          const src = await img.getAttribute('src');
          if (src && isValidImageUrl(src, propertyHash)) {
            const highResSrc = getHighResUrl(src);
            imageUrls.add(highResSrc);
          }
        }

        const nextButton = await page.$('button[aria-label*="Next"], button:has-text("Next")');
        if (nextButton) {
          await nextButton.click();
          await randomDelay(300, 600);
        } else {
          break;
        }

        attempts++;
      }
    }
  } catch (error) {
    console.warn('Error extracting image URLs:', error);
  }

  return Array.from(imageUrls);
}

function isValidImageUrl(url, propertyHash) {
  if (!url) return false;
  if (url.startsWith('data:')) return false;
  if (url.includes('placeholder')) return false;
  if (url.includes('logo')) return false;
  if (url.includes('icon')) return false;
  if (url.includes('agent')) return false;
  if (url.includes('avatar')) return false;

  // Must be from rdcpix.com (Realtor.com's image CDN)
  if (!url.includes('rdcpix')) return false;

  // Must have the listing image pattern
  if (!url.includes('-m') || !url.includes('rd-w')) return false;

  // If we have a property hash, only accept images from this listing
  if (propertyHash && !url.includes(propertyHash)) return false;

  return true;
}

function getHighResUrl(url) {
  // Convert to high resolution (1280x960)
  return url.replace(/-w\d+_h\d+/, '-w1280_h960');
}

function getImageExtension(url) {
  if (url.includes('.webp')) return 'webp';
  if (url.includes('.png')) return 'png';
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
