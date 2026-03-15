# 🚀 Agent Platform Setup Guide

Complete setup instructions for the Agent Platform on fresh machines.

## Quick Start By Platform

### macOS & Linux
```bash
chmod +x setup-linux-mac.sh
./setup-linux-mac.sh
```

### Windows (PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup-windows.ps1
```

---

## 📋 Prerequisites

| Component | Version | Purpose | Link |
|-----------|---------|---------|------|
| **Python** | 3.10+ | Backend runtime | [python.org](https://www.python.org/downloads/) |
| **Node.js** | 18+ | Frontend build & dev server | [nodejs.org](https://nodejs.org/) |
| **Docker** | Latest | SearXNG search, code sandboxing (optional) | [docker.com](https://www.docker.com/products/docker-desktop) |
| **Git** | Latest | Version control (optional) | [git-scm.com](https://git-scm.com/) |
| **Ollama** | Latest | Local LLM inference (optional) | [ollama.ai](https://ollama.ai/) |

### Verify Prerequisites

**Python:**
```bash
python3 --version  # should be 3.10+
pip3 --version
```

**Node.js:**
```bash
node --version     # should be v18+
npm --version      # should be 9+
```

---

## 🛠️ Manual Setup (if scripts don't work)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/agent-platform.git
cd agent-platform
```

### 2. Python Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Node Dependencies

**All Platforms:**
```bash
npm install
cd frontend && npm install && cd ..
```

### 4. Environment Configuration

**Create `.env` file in project root:**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Essential Configuration:**
```env
# Choose at least ONE LLM provider:

# Option 1: Local (Recommended for development)
OLLAMA_BASE_URL=http://localhost:11434

# Option 2: Groq Cloud
GROQ_API_KEY=gsk_...

# Option 3: OpenAI
OPENAI_API_KEY=sk_...

# Option 4: Google Gemini
GEMINI_API_KEY=AIza...

# Search (optional)
SEARXNG_BASE_URL=http://localhost:8888
```

### 5. Create Data Directories
```bash
mkdir -p chroma_db logs data
```

**Note:** SQLite database files (`agents.db`, `sessions.db`, `memory_graph.db`) are created automatically by the backend on first run. Do not create these as directories.

---

## 🚀 Running the Application

### Quick Start (All Services)
```bash
npm run dev
```

### Individual Services

**Terminal 1 - Backend:**
```bash
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
npm run dev:backend
```

**Terminal 2 - Frontend:**
```bash
npm run dev:frontend
```

**Terminal 3 - Docker Services (optional):**
```bash
npm run dev:services
```

### Service URLs
- Frontend: [`http://localhost:5173`](http://localhost:5173)
- Backend API: [`http://localhost:8000`](http://localhost:8000)
- API Docs: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- SearXNG (if running): [`http://localhost:8888`](http://localhost:8888)

---

## 🧠 Optional Services Setup

### Ollama (Local LLMs)

**Install Ollama:**
- macOS: `brew install ollama`
- Windows: Download from [ollama.ai](https://ollama.ai/)
- Linux: `curl https://ollama.ai/install.sh | sh`

**Pull a Model:**
```bash
ollama pull llama3.2
# or
ollama pull mistral
ollama pull neural-chat
```

**Run Ollama:**
```bash
ollama serve
```

The API will be available at `http://localhost:11434`

### Docker & SearXNG

**Install Docker Desktop:**
- macOS: `brew install docker`
- Windows: [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- Linux: `sudo apt install docker.io docker-compose`

**Start Services:**
```bash
npm run dev:services
# or manually:
docker-compose up -d
```

**Check Status:**
```bash
docker-compose ps
```

**View Logs:**
```bash
docker-compose logs -f searxng
```

**Stop Services:**
```bash
docker-compose down
```

---

## 🔑 LLM Provider Setup

### 1. Groq (Cloud, Fast & Free)
1. Sign up: [`console.groq.com`](https://console.groq.com)
2. Create API key
3. Set in `.env`: `GROQ_API_KEY=gsk_...`

### 2. OpenAI (Cloud)
1. Sign up: [`platform.openai.com`](https://platform.openai.com)
2. Create API key
3. Set in `.env`: `OPENAI_API_KEY=sk_...`

### 3. Google Gemini (Cloud)
1. Get API key: [`ai.google.dev`](https://ai.google.dev)
2. Set in `.env`: `GEMINI_API_KEY=AIza...`

### 4. Local Ollama (Free, Private)
1. Install Ollama
2. Pull model: `ollama pull llama3.2`
3. Set in `.env`: `OLLAMA_BASE_URL=http://localhost:11434`

---

## 🐛 Troubleshooting

### Port Already in Use

**Find and kill process on port 8000:**
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
```

### Virtual Environment Issues

**Reset venv:**
```bash
rm -rf venv  # or rmdir venv /s on Windows
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Module Not Found Errors

**Ensure venv is activated:**
```bash
# macOS/Linux
which python3  # should show path in venv/

# Windows
Get-Command python  # should show path in venv\
```

**Reinstall dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Docker Connection Issues

**Verify Docker is running:**
```bash
docker ps
docker-compose --version
```

**Restart Docker:**
```bash
# macOS/Linux
sudo systemctl restart docker

# Windows: Restart Docker Desktop from system tray
```

### Frontend Can't Connect to Backend

**Check backend is running:**
```bash
curl http://localhost:8000/health
# or
curl http://127.0.0.1:8000/health
```

**Verify proxy config in `frontend/vite.config.ts`:**
```typescript
proxy: {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, '')
  }
}
```

---

## 📝 Environment Variables Reference

```env
# === LLM PROVIDERS (pick at least one) ===
GROQ_API_KEY=gsk_...
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=sk_...
GEMINI_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
DEEPGRAM_API_KEY=...

# === SEARCH & DATA ===
SEARXNG_BASE_URL=http://localhost:8888
TAVILY_API_KEY=...
BRAVE_SEARCH_KEY=...
EXA_API_KEY=...
GOOGLE_PLACES_API_KEY=...

# === SERVER CONFIGURATION ===
PORT=8000
HOST=0.0.0.0
DEBUG=True

# === STORAGE PATHS ===
DB_PATH=agents.db
CHROMA_PATH=./chroma_db
TRACE_DIR=./logs

# === AGENT SETTINGS ===
MAX_TOOL_TURNS=5
DEFAULT_MODEL=llama3.2
EMBED_MODEL=nomic-embed-text
```

---

## 🎯 Platform-Specific Notes

### macOS
- Use `python3` and `pip3` explicitly
- Homebrew recommended for dependencies: `brew install python@3.11 node@18`
- M1/M2 Macs: Some packages may need native builds—use `arch -arm64 python3 -m`

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip nodejs npm git
```

### Linux (Fedora/RHEL)
```bash
sudo dnf install -y python3.11 python3-pip nodejs npm git
```

### Windows
- Use PowerShell 5.0+ (or PowerShell Core 7+)
- May need to set execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- WSL2 recommended for better performance with Docker
- Use `python` and `pip` (not `python3`/`pip3`)

---

## ✅ Verification Checklist

After setup, verify everything works:

```bash
# 1. Python environment
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1
python --version

# 2. Dependencies installed
pip list | grep -E "fastapi|uvicorn|chromadb"

# 3. Node packages installed
npm list | head -20

# 4. Backend starts
npm run dev:backend
# Should see: "Uvicorn running on http://0.0.0.0:8000"

# 5. Frontend starts (in new terminal)
npm run dev:frontend
# Should see: "➜  Local: http://localhost:5173/"

# 6. API is responding
curl http://localhost:8000/health
# Should return: {"status": "ok"}

# 7. Frontend loads
# Open http://localhost:5173 in browser
```

---

## 🚀 Next Steps After Setup

1. **Configure LLM Provider**
   - Add API keys to `.env`
   - Test with `/health` endpoint

2. **Explore Features**
   - Read [FEATURES_AND_ROADMAP.md](FEATURES_AND_ROADMAP.md)
   - Check [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

3. **Run Tests**
   ```bash
   python -m pytest tests/
   ```

4. **Start Development**
   - Use `npm run dev` for live reload
   - Check API docs at [`http://localhost:8000/docs`](http://localhost:8000/docs)

---

## 📚 Documentation

- **[README.md](README.md)** - Project overview & philosophy
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Architecture & design patterns
- **[FEATURES_AND_ROADMAP.md](FEATURES_AND_ROADMAP.md)** - Features & planned work
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

---

## 🆘 Getting Help

1. Check [Troubleshooting](#-troubleshooting) section above
2. Review existing GitHub issues
3. Check project documentation
4. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for support guidelines

---

## 📅 Setup Time Estimates

| Task | Time |
|------|------|
| Prerequisites check | 2-5 min |
| Python setup | 3-5 min |
| Dependencies install | 5-10 min |
| Node setup | 3-5 min |
| Configuration | 2-3 min |
| Docker setup (optional) | 5-10 min |
| **Total** | **20-40 min** |

---

**Last Updated:** March 2026
**Tested On:** macOS 13.x, Ubuntu 22.04, Windows 11, Python 3.10-3.12, Node 18-20
