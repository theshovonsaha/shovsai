#!/usr/bin/env node
/**
 * Cross-platform optional Docker service starter.
 * Tries to start SearXNG via Docker Compose; skips gracefully if Docker is not available.
 */
const { execSync } = require('child_process');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');

try {
  execSync('docker compose up -d searxng', { cwd: ROOT, stdio: 'pipe' });
  console.log('✓  SearXNG started via Docker.');
} catch (err) {
  const msg = err.stderr ? err.stderr.toString().trim() : err.message;
  const reason = msg.includes('not found') || msg.includes('not recognized')
    ? 'Docker is not installed'
    : msg.includes('Cannot connect') || msg.includes('daemon')
      ? 'Docker daemon is not running'
      : 'Docker Compose command failed';
  console.log(
    `[info] ${reason} — skipping SearXNG.\n` +
    '       Web search will fall back to DuckDuckGo.'
  );
}
