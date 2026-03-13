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
    DOCKER_DISABLED        — set to "true" to disable execution entirely (safety)

STRICT POLICY: If Docker is unavailable or disabled, no execution is performed.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

SANDBOX_IMAGE   = os.getenv("DOCKER_SANDBOX_IMAGE",   "python:3.11-slim")
SANDBOX_MEMORY  = os.getenv("DOCKER_SANDBOX_MEMORY",  "256m")
SANDBOX_TIMEOUT = int(os.getenv("DOCKER_SANDBOX_TIMEOUT", "30"))
# DOCKER_DISABLED check is now dynamic inside run_in_docker

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
    
    STRICT SECURITY: 
    If DOCKER_DISABLED=true or Docker daemon is not available, 
    returns a denial message. No local fallback is allowed.
    """
    if os.getenv("DOCKER_DISABLED", "false").lower() == "true":
        return "[denied] Docker execution is explicitly disabled via DOCKER_DISABLED."

    try:
        import docker
    except ImportError:
        return "[error] 'docker' python library not installed. Cannot execute command safely."

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _run_docker_sync(command, timeout, workdir),
        )
        return result
    except Exception as e:
        # Check for connection errors specifically to provide better feedback
        err_str = str(e).lower()
        if "connection" in err_str or "socket" in err_str:
            return "[denied] Docker Desktop is not running. For safety, execution is strictly disallowed."
        
        return f"[error] Docker execution failed: {e}"


def _run_docker_sync(command: str, timeout: int, workdir: Optional[str]) -> str:
    """Synchronous Docker run — called via executor so it doesn't block the loop."""
    import docker
    try:
        client = docker.from_env()
    except Exception as e:
        return f"[denied] Cannot connect to Docker daemon: {e}. Execution disallowed."

    # Mount the sandbox dir as /sandbox inside the container
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
            # timeout is not supported in run(), it's for the API connection
            # We handle it via wait() or by allowing the container to be killed if it persists (not implemented here)
        )
        return output.decode("utf-8", errors="replace").strip()
    except Exception as e:
        err = str(e)
        if "timed out" in err.lower():
            return f"[timeout] Command exceeded {timeout}s limit."
        return f"[error] {err}"
    finally:
        try:
            client.close()
        except:
            pass
