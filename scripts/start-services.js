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
} catch {
  console.log(
    '[info] SearXNG/Docker not available — skipping.\n' +
    '       Web search will fall back to DuckDuckGo.'
  );
}
