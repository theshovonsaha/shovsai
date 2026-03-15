#!/usr/bin/env node
/**
 * Cross-platform project setup.
 * Creates a Python venv, installs Python requirements, and installs frontend npm packages.
 *
 * Usage:  node scripts/setup.js          (full setup)
 *         node scripts/setup.js --skip-frontend  (backend only)
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const isWin = process.platform === 'win32';
const skipFrontend = process.argv.includes('--skip-frontend');

// ── Helpers ──────────────────────────────────────────────────────────────────
function run(cmd, opts = {}) {
  console.log(`\n▶  ${cmd}`);
  execSync(cmd, { stdio: 'inherit', cwd: ROOT, ...opts });
}

function findPython() {
  // Try common Python commands in order of preference
  const candidates = isWin
    ? ['python', 'python3', 'py -3']
    : ['python3', 'python'];

  for (const cmd of candidates) {
    try {
      const version = execSync(`${cmd} --version`, { encoding: 'utf8', cwd: ROOT }).trim();
      // Ensure it's Python 3
      if (version.includes('Python 3')) {
        console.log(`✓  Found ${version} via "${cmd}"`);
        return cmd;
      }
    } catch {
      // not found, try next
    }
  }
  return null;
}

// ── 1. Detect Python ────────────────────────────────────────────────────────
console.log('🔍  Detecting Python installation...');
const pythonCmd = findPython();
if (!pythonCmd) {
  console.error(
    '\n❌  Python 3 not found.\n' +
    '    Please install Python 3.10+ from https://www.python.org/downloads/\n' +
    '    Make sure to check "Add Python to PATH" during installation.\n'
  );
  process.exit(1);
}

// ── 2. Create venv (if missing) ─────────────────────────────────────────────
const venvDir = path.join(ROOT, 'venv');
const venvPython = isWin
  ? path.join(venvDir, 'Scripts', 'python.exe')
  : path.join(venvDir, 'bin', 'python');
const venvPip = isWin
  ? path.join(venvDir, 'Scripts', 'pip.exe')
  : path.join(venvDir, 'bin', 'pip');

if (!fs.existsSync(venvPython)) {
  console.log('\n📦  Creating virtual environment...');
  run(`${pythonCmd} -m venv venv`);
} else {
  console.log('✓  Virtual environment already exists.');
}

// ── 3. Install Python dependencies ──────────────────────────────────────────
console.log('\n📦  Installing Python dependencies...');
// Use the venv pip directly (cross-platform path)
run(`"${venvPip}" install -r requirements.txt`);

// ── 4. Install root npm dependencies ────────────────────────────────────────
console.log('\n📦  Installing root npm dependencies...');
run('npm install');

// ── 5. Install frontend dependencies ────────────────────────────────────────
if (!skipFrontend) {
  console.log('\n📦  Installing frontend dependencies...');
  run('npm install', { cwd: path.join(ROOT, 'frontend') });
}

// ── 6. Create required directories ──────────────────────────────────────────
console.log('\n📁  Ensuring required directories exist...');
const requiredDirs = ['logs', 'chroma_db', 'agent_sandbox'];
for (const dir of requiredDirs) {
  const dirPath = path.join(ROOT, dir);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`    Created ${dir}/`);
  } else {
    console.log(`    ✓  ${dir}/ exists`);
  }
}

// ── 7. Create .env from example if missing ──────────────────────────────────
const envPath = path.join(ROOT, '.env');
const envExamplePath = path.join(ROOT, '.env.example');
if (!fs.existsSync(envPath) && fs.existsSync(envExamplePath)) {
  fs.copyFileSync(envExamplePath, envPath);
  console.log('\n✓  Created .env from .env.example (edit it with your API keys)');
}

console.log('\n✅  Setup complete!');
console.log('    Run "npm run dev" to start the development server.\n');
