import assert from 'node:assert/strict';
import { before, beforeEach, afterEach, after, test } from 'node:test';
import { existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { createServer as createPortReservation } from 'node:net';
import { chromium } from 'playwright';
import { createServer } from 'vite';

let server, browser, context, page, url;
const screenshotDirectory = process.env.BROWSER_EVIDENCE_DIR;
before(async () => {
  const port = await new Promise((resolvePort, reject) => {
    const reservation = createPortReservation();
    reservation.once('error', reject);
    reservation.listen(0, '127.0.0.1', () => {
      const availablePort = reservation.address().port;
      reservation.close(() => resolvePort(availablePort));
    });
  });
  server = await createServer({
    server: { host: '127.0.0.1', port, strictPort: true },
    define: {
      'import.meta.env.VITE_API_URL': JSON.stringify('http://synthetic.invalid'),
      'import.meta.env.VITE_WS_URL': JSON.stringify('ws://synthetic.invalid/stt'),
    },
  });
  await server.listen();
  url = `http://127.0.0.1:${server.httpServer.address().port}`;
  const macChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  const executablePath = process.env.BROWSER_EXECUTABLE || (existsSync(macChrome) ? macChrome : undefined);
  browser = await chromium.launch({ executablePath, headless: true });
  console.log(`Headless browser: ${browser.version()}; isolated localhost server: ${url}`);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
beforeEach(async () => {
  context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  await context.route('http://synthetic.invalid/**', route => route.fulfill({ json: {
    active_connections: 0, status: 'healthy', whisper_model: 'synthetic', whisper_device: 'cpu',
    model_sharing: null, gpu_vram_used_gb: null, gpu_vram_total_gb: null,
    gpu_vram_usage_percent: null, gpu_name: null,
  } }));
  await context.addInitScript(() => {
    const mock = window.__mock = { sockets: [], tracks: [], contexts: [], processors: [], failAudio: false };
    const nativeSocket = window.WebSocket;
    class Socket {
      static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
      constructor(url) { this.url = url; this.readyState = 0; this.sent = []; mock.sockets.push(this); }
      open() { this.readyState = 1; this.onopen?.({}); }
      send(message) { if (this.readyState !== 1) throw new Error('Socket is not open'); this.sent.push(message); }
      close() { if (this.readyState === 3) return; this.readyState = 3; this.onclose?.({ code: 1000 }); }
      fail() { this.readyState = 3; this.onerror?.({ message: 'Synthetic connection failure' }); this.onclose?.({ code: 1006 }); }
      message(data) { this.onmessage?.({ data: JSON.stringify(data) }); }
    }
    window.WebSocket = class extends Socket {
      constructor(url, protocols) {
        if (!String(url).startsWith('ws://synthetic.invalid/')) return new nativeSocket(url, protocols);
        super(url);
      }
    };
    const media = async () => {
      const track = { kind: 'audio', stopped: false, stop() { this.stopped = true; } };
      mock.tracks.push(track);
      return { getTracks: () => [track], getAudioTracks: () => [track], getVideoTracks: () => [], removeTrack() {} };
    };
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
      getDisplayMedia: media, getUserMedia: media,
    } });
    const node = () => ({ channelCount: 1, connect() {}, disconnected: false,
      disconnect() { this.disconnected = true; } });
    window.AudioContext = class {
      constructor() {
        if (mock.failAudio) throw new Error('Synthetic audio initialization failure');
        this.sampleRate = 48000; this.destination = {}; this.closed = false; mock.contexts.push(this);
      }
      createMediaStreamSource() { return node(); }
      createScriptProcessor() { const processor = node(); mock.processors.push(processor); return processor; }
      close() { this.closed = true; return Promise.resolve(); }
    };
    mock.emitAudio = () => {
      const samples = Float32Array.from({ length: 48000 }, (_, i) => 0.25 * Math.sin(i * 2 * Math.PI * 440 / 48000));
      mock.processors.at(-1).onaudioprocess({ inputBuffer: {
        numberOfChannels: 1, getChannelData: () => samples,
      } });
    };
    mock.transcript = () => mock.sockets.at(-1).message({
      text: 'Synthetic transcript', is_final: true, start: 0, end: 1, timestamp: 1700000000000,
    });
  });
  page = await context.newPage();
  page.on('dialog', dialog => dialog.dismiss());
  await page.goto(url);
  await page.getByRole('heading', { name: 'Whisper Real-time Transcription' }).waitFor();
});
afterEach(async () => context?.close());
const start = async () => {
  await page.getByRole('button', { name: 'Start Caption', exact: true }).click();
  await page.getByRole('button', { name: 'Stop Caption', exact: true }).waitFor();
};
const open = async () => { await page.evaluate(() => window.__mock.sockets.at(-1).open()); };
const assertReleased = async () => {
  await page.getByRole('button', { name: 'Start Caption', exact: true }).waitFor({ timeout: 2000 });
  assert.deepEqual(await page.evaluate(() => ({
    tracks: window.__mock.tracks.every(x => x.stopped),
    contexts: window.__mock.contexts.every(x => x.closed),
    processors: window.__mock.processors.every(x => x.disconnected),
    sockets: window.__mock.sockets.every(x => x.readyState === 3),
  })), { tracks: true, contexts: true, processors: true, sockets: true });
};
const screenshot = async name => {
  if (screenshotDirectory) {
    mkdirSync(screenshotDirectory, { recursive: true });
    await page.screenshot({ path: resolve(screenshotDirectory, name), fullPage: true });
  }
};

test('renders both themes and the parameter panel in real Chrome', async () => {
  await screenshot('whisper-dark.png');
  await page.getByTitle('Switch to Light Mode').click();
  assert.equal(await page.evaluate(() => localStorage.getItem('darkMode')), 'false');
  await page.getByRole('button', { name: '⚙️ 詳細パラメータ', exact: true }).click();
  assert.ok(await page.getByText('Beam Size', { exact: true }).isVisible());
  await screenshot('whisper-light-parameters.png');
});
test('synthetic PCM and transcript survive explicit stop', async () => {
  await start(); await open();
  await page.evaluate(() => { window.__mock.emitAudio(); window.__mock.transcript(); });
  await page.waitForFunction(() => window.__mock.sockets[0].sent.some(x => x instanceof ArrayBuffer), undefined, { timeout: 3000 });
  assert.equal(await page.evaluate(() => window.__mock.sockets[0].sent.find(x => x instanceof ArrayBuffer).byteLength), 32000);
  await page.getByText('Synthetic transcript', { exact: true }).waitFor();
  assert.ok(await page.getByText('Synthetic transcript', { exact: true }).isVisible());
  await screenshot('whisper-synthetic-caption.png');
  await page.getByRole('button', { name: 'Stop Caption', exact: true }).click();
  await assertReleased();
  await page.getByText('Synthetic transcript', { exact: true }).waitFor();
  assert.ok(await page.getByText('Synthetic transcript', { exact: true }).isVisible());
});
test('connection failure releases synthetic media and permits retry', async () => {
  await start();
  await page.evaluate(() => window.__mock.sockets.at(-1).fail());
  await assertReleased();
});
test('remote close releases media while preserving captions', async () => {
  await start(); await open();
  await page.evaluate(() => { window.__mock.transcript(); window.__mock.sockets.at(-1).close(); });
  await assertReleased();
  await page.getByText('Synthetic transcript', { exact: true }).waitFor();
  assert.ok(await page.getByText('Synthetic transcript', { exact: true }).isVisible());
});
test('retry creates a fresh connection and can stop normally', async () => {
  await start();
  await page.evaluate(() => window.__mock.sockets.at(-1).fail());
  await assertReleased();
  await start(); await open();
  await page.evaluate(() => window.__mock.transcript());
  assert.equal(await page.evaluate(() => window.__mock.sockets.length), 2);
  await page.getByText('Synthetic transcript', { exact: true }).waitFor();
  assert.ok(await page.getByText('Synthetic transcript', { exact: true }).isVisible());
  await page.getByRole('button', { name: 'Stop Caption', exact: true }).click();
  await assertReleased();
});
test('audio initialization failure closes acquired media and socket', async () => {
  await page.evaluate(() => { window.__mock.failAudio = true; });
  await page.getByRole('button', { name: 'Start Caption', exact: true }).click();
  await page.waitForFunction(() => window.__mock.sockets.length === 1);
  await assertReleased();
});

test('late close from the failed socket cannot stop a new connection', async () => {
  await start();
  await page.evaluate(() => {
    window.__mock.staleClose = window.__mock.sockets.at(-1).onclose;
    window.__mock.sockets.at(-1).fail();
  });
  await assertReleased();
  await start(); await open();
  await page.evaluate(() => window.__mock.staleClose({ code: 1006 }));
  assert.ok(await page.getByRole('button', { name: 'Stop Caption', exact: true }).isVisible());
  assert.equal(await page.evaluate(() => window.__mock.sockets.at(-1).readyState), 1);
  await page.getByRole('button', { name: 'Stop Caption', exact: true }).click();
  await assertReleased();
});
test('stopping while connecting ignores a queued open callback', async () => {
  await start();
  await page.evaluate(() => { window.__mock.staleOpen = window.__mock.sockets.at(-1).onopen; });
  await page.getByRole('button', { name: 'Stop Caption', exact: true }).click();
  await assertReleased();
  await page.evaluate(() => window.__mock.staleOpen({}));
  assert.ok(await page.getByRole('button', { name: 'Start Caption', exact: true }).isVisible());
  assert.equal(await page.evaluate(() => window.__mock.sockets[0].sent.length), 0);
});
