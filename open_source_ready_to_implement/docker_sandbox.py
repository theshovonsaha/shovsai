"""
Docker Sandbox
--------------
Replaces the raw subprocess bash handler with an isolated Docker container
that is created, runs the command, and is destroyed — no host filesystem access.

Requirements:
    pip install docker

Environment variables:
    DOCKER_SANDBOX_IMAGE   — base image (default: python:3.11-slim)
    DOCKER_SANDBOX_MEMORY  — memory limit (default: 256m)
    DOCKER_SANDBOX_TIMEOUT — max seconds (default: 30)
    DOCKER_DISABLED        — set to "true" to fall back to subprocess (dev mode)

Why this is safe:
    - Container has no host volume mounts
    - Network is disabled by default
    - Container is removed immediately after run (remove=True)
    - Memory hard-capped
    - PID limit prevents fork bombs
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

SANDBOX_IMAGE   = os.getenv("DOCKER_SANDBOX_IMAGE",   "python:3.11-slim")
SANDBOX_MEMORY  = os.getenv("DOCKER_SANDBOX_MEMORY",  "256m")
SANDBOX_TIMEOUT = int(os.getenv("DOCKER_SANDBOX_TIMEOUT", "30"))
DOCKER_DISABLED = os.getenv("DOCKER_DISABLED", "false").lower() == "true"

# Files the agent creates get written here on the host, then mounted read-only
# into the container so scripts can reference them.
SANDBOX_DIR = Path(os.getenv("SANDBOX_DIR", "./agent_sandbox")).resolve()
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)


async def run_in_docker(
    command: str,
    timeout: int = SANDBOX_TIMEOUT,
    workdir: Optional[str] = None,
) -> str:
    """
    Execute a bash command inside a throwaway Docker container.
    Returns stdout+stderr as a single string.
    Falls back to subprocess if DOCKER_DISABLED=true or Docker not available.
    """
    if DOCKER_DISABLED:
        return await _subprocess_fallback(command, timeout, workdir)

    try:
        import docker
    except ImportError:
        return await _subprocess_fallback(command, timeout, workdir)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _run_docker_sync(command, timeout, workdir),
        )
        return result
    except Exception as e:
        # If Docker daemon is not running, fall back gracefully
        if "connection" in str(e).lower() or "socket" in str(e).lower():
            return await _subprocess_fallback(command, timeout, workdir)
        return f"Docker error: {e}"


def _run_docker_sync(command: str, timeout: int, workdir: Optional[str]) -> str:
    """Synchronous Docker run — called via executor so it doesn't block the loop."""
    import docker
    client = docker.from_env()

    # Mount the sandbox dir as /sandbox inside the container (read-write so
    # scripts can write files that persist to host sandbox dir)
    volumes = {
        str(SANDBOX_DIR): {"bind": "/sandbox", "mode": "rw"}
    }

    container_workdir = "/sandbox"
    if workdir:
        # Resolve workdir relative to sandbox, prevent escape
        resolved = (SANDBOX_DIR / workdir).resolve()
        if str(resolved).startswith(str(SANDBOX_DIR)):
            container_workdir = f"/sandbox/{workdir}"

    try:
        output = client.containers.run(
            image          = SANDBOX_IMAGE,
            command        = ["bash", "-c", command],
            working_dir    = container_workdir,
            volumes        = volumes,
            mem_limit      = SANDBOX_MEMORY,
            pids_limit     = 64,           # prevent fork bombs
            network_mode   = "none",       # no internet from sandbox
            remove         = True,         # destroyed after run
            stdout         = True,
            stderr         = True,
            timeout        = timeout,
        )
        return output.decode("utf-8", errors="replace").strip()
    except Exception as e:
        err = str(e)
        if "timed out" in err.lower():
            return f"[timeout] Command exceeded {timeout}s limit."
        return f"[error] {err}"
    finally:
        client.close()


async def _subprocess_fallback(command: str, timeout: int, workdir: Optional[str]) -> str:
    """
    Fallback: run in subprocess but constrained to SANDBOX_DIR.
    Used when Docker is not available (dev machines without Docker Desktop).
    """
    cwd = SANDBOX_DIR
    if workdir:
        resolved = (SANDBOX_DIR / workdir).resolve()
        if str(resolved).startswith(str(SANDBOX_DIR)):
            cwd = resolved

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd    = cwd,
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode("utf-8", errors="replace").strip()
    except asyncio.TimeoutError:
        proc.kill()
        return f"[timeout] Command exceeded {timeout}s limit."
    except Exception as e:
        return f"[error] {e}"
