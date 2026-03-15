#!/usr/bin/env node
/**
 * Cross-platform backend starter.
 * Detects Windows/Mac/Linux and launches uvicorn from the correct venv path.
 */
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const ROOT = path.resolve(__dirname, '..');
const isWin = process.platform === 'win32';

// Locate venv Python
const venvPython = isWin
  ? path.join(ROOT, 'venv', 'Scripts', 'python.exe')
  : path.join(ROOT, 'venv', 'bin', 'python');

if (!fs.existsSync(venvPython)) {
  console.error(
    '\n❌  Virtual environment not found.\n' +
    '    Run "npm run setup" first to create the venv and install dependencies.\n'
  );
  process.exit(1);
}

const args = [
  '-m', 'uvicorn',
  'api.main:app',
  '--reload',
  '--host', '0.0.0.0',
  '--port', '8000',
];

const child = spawn(venvPython, args, {
  cwd: ROOT,
  stdio: 'inherit',
  env: { ...process.env, PYTHONUNBUFFERED: '1' },
});

child.on('error', (err) => {
  console.error(`Failed to start backend: ${err.message}`);
  process.exit(1);
});

child.on('exit', (code) => {
  process.exit(code ?? 1);
});
