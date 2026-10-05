import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { readFile, writeFile, unlink } from 'node:fs/promises';
import { JSDOM } from 'jsdom';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { transformWithEsbuild } from 'vite';

// Transform the real component using the same Vite TypeScript transform.
// All service/media operations are mocks, with no browser or audio access.
const generated = new URL(`./.App-${process.pid}.mjs`, import.meta.url);
const { code } = await transformWithEsbuild(
  await readFile(new URL('../src/App.tsx', import.meta.url), 'utf8'), 'App.tsx', {
    loader: 'tsx', jsx: 'automatic',
    define: { 'import.meta.env': JSON.stringify({ VITE_API_URL: 'http://synthetic.invalid' }) },
  });
await writeFile(generated, code);
after(() => unlink(generated));
const { default: App } = await import(generated.href);
let dom, root, alerts, captureOptions;
const saved = new Map();
const replaceGlobal = (name, value) => {
  saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
};
beforeEach(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'http://synthetic.invalid' });
  alerts = [];
  captureOptions = undefined;
  for (const [name, value] of Object.entries({
    window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
    localStorage: dom.window.localStorage, IS_REACT_ACT_ENVIRONMENT: true,
    alert: (message) => alerts.push(message),
    fetch: async (url) => {
      assert.equal(url, 'http://synthetic.invalid/stats');
      return { json: async () => ({ active_connections: 0, status: 'healthy',
        whisper_model: 'synthetic', whisper_device: 'cpu', model_sharing: null }) };
    },
  })) replaceGlobal(name, value);
  Object.defineProperty(navigator, 'mediaDevices', { value: {
    getDisplayMedia: async (options) => {
      captureOptions = options;
      const error = new Error('Synthetic permission denial');
      error.name = 'NotAllowedError';
      throw error;
    },
  } });
  root = createRoot(document.getElementById('root'));
  await act(async () => root.render(createElement(App)));
});
afterEach(async () => {
  await act(async () => root.unmount());
  dom.window.close();
  for (const [name, descriptor] of saved) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
  saved.clear();
});
const button = (text) => Array.from(document.querySelectorAll('button')).find(
  (element) => element.textContent.trim() === text);

test('renders the stopped transcription controls with mocked stats', () => {
  assert.equal(document.querySelector('h1').textContent, 'Whisper Real-time Transcription');
  assert.ok(button('Start Caption'));
  assert.ok(document.body.textContent.includes('Stopped'));
  assert.equal(document.querySelectorAll('select').length, 3);
});
test('theme toggle persists the existing preference', async () => {
  assert.equal(localStorage.getItem('darkMode'), 'true');
  await act(async () => document.querySelector('[title="Switch to Light Mode"]').click());
  assert.equal(localStorage.getItem('darkMode'), 'false');
  assert.ok(document.querySelector('[title="Switch to Dark Mode"]'));
});
test('advanced parameters open and close without losing defaults', async () => {
  await act(async () => button('⚙️ 詳細パラメータ').click());
  assert.ok(document.body.textContent.includes('Beam Size'));
  assert.ok(document.querySelector('input[value="5"]'));
  await act(async () => button('⚙️ 詳細パラメータ').click());
  assert.ok(!document.body.textContent.includes('Beam Size'));
});
test('capture hints are preserved and permission rejection leaves recording stopped', async () => {
  await act(async () => button('Start Caption').click());
  assert.deepEqual(captureOptions.video, { cursor: 'never' });
  assert.equal(captureOptions.audio.suppressLocalAudioPlayback, false);
  assert.equal(captureOptions.audio.sampleRate, 48000);
  assert.equal(captureOptions.systemAudio, 'include');
  assert.equal(captureOptions.preferCurrentTab, false);
  assert.equal(alerts.length, 1);
  assert.match(alerts[0], /Permission denied/);
  assert.ok(button('Start Caption'));
});
