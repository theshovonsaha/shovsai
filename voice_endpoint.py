"""
Voice WebSocket Endpoint
-------------------------
WebSocket-based voice I/O for the agent platform.

Protocol:
  Client → Server: binary audio chunks (PCM 16kHz 16-bit mono or WebM/Opus)
  Server → Client: JSON events interleaved with binary TTS audio

  JSON Messages (server → client):
    {"type": "stt_result", "text": "user's transcribed speech"}
    {"type": "agent_token", "content": "streaming text token"}
    {"type": "agent_done", "full_response": "complete response text"}
    {"type": "tts_start"}
    {"type": "tts_end"}
    {"type": "error", "message": "..."}

  JSON Messages (client → server):
    {"type": "config", "session_id": "...", "agent_id": "...", "model": "..."}
    {"type": "stt_end"}  — signals end of speech input

Usage:
  ws://localhost:8000/ws/voice
  First message must be a config JSON.
  Then stream audio chunks, ending with {"type": "stt_end"}.
  Server responds with agent text + TTS audio.
"""

import json
import asyncio
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from logger import log


def setup_voice_routes(app: FastAPI, agent_manager) -> None:
    """Register voice WebSocket endpoint."""

    @app.websocket("/ws/voice")
    async def voice_endpoint(ws: WebSocket):
        await ws.accept()
        log("system", "system", "Voice WebSocket connected")

        session_id: Optional[str] = None
        agent_id: str = "default"
        model: Optional[str] = None

        try:
            # Phase 1: Receive config
            config_raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
            config = json.loads(config_raw)
            session_id = config.get("session_id")
            agent_id = config.get("agent_id", "default")
            model = config.get("model")

            await ws.send_json({"type": "config_ack", "status": "ok"})
            log("system", session_id or "voice", f"Voice config: agent={agent_id} model={model}")

            # Phase 2: Audio loop
            while True:
                audio_chunks: list[bytes] = []
                
                # Collect audio until stt_end signal
                while True:
                    msg = await ws.receive()
                    
                    if "bytes" in msg and msg["bytes"]:
                        # Binary audio data
                        audio_chunks.append(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        try:
                            cmd = json.loads(msg["text"])
                            if cmd.get("type") == "stt_end":
                                break
                            if cmd.get("type") == "disconnect":
                                await ws.close()
                                return
                        except json.JSONDecodeError:
                            pass

                if not audio_chunks:
                    await ws.send_json({"type": "error", "message": "No audio received"})
                    continue

                # STT: Convert audio → text
                audio_data = b"".join(audio_chunks)
                log("system", session_id or "voice", f"Received {len(audio_data)} bytes of audio")

                try:
                    user_text = await _transcribe_audio(audio_data)
                except Exception as e:
                    await ws.send_json({
                        "type": "error",
                        "message": f"STT failed: {e}. Install faster-whisper: pip install faster-whisper"
                    })
                    continue

                if not user_text or not user_text.strip():
                    await ws.send_json({"type": "stt_result", "text": ""})
                    continue

                await ws.send_json({"type": "stt_result", "text": user_text})
                log("system", session_id or "voice", f"STT result: {user_text[:80]}")

                # Agent: Process through AgentCore
                agent_instance = agent_manager.get_agent_instance(agent_id)
                full_response = ""

                async for event in agent_instance.chat_stream(
                    user_message=user_text,
                    session_id=session_id,
                    agent_id=agent_id,
                    model=model,
                ):
                    if event["type"] == "session":
                        session_id = event["session_id"]
                    elif event["type"] == "token":
                        full_response += event["content"]
                        await ws.send_json({"type": "agent_token", "content": event["content"]})
                    elif event["type"] == "error":
                        await ws.send_json({"type": "error", "message": event["message"]})

                await ws.send_json({"type": "agent_done", "full_response": full_response})

                # TTS: Convert response → audio
                if full_response.strip():
                    try:
                        await ws.send_json({"type": "tts_start"})
                        async for audio_chunk in _synthesize_speech(full_response):
                            await ws.send_bytes(audio_chunk)
                        await ws.send_json({"type": "tts_end"})
                    except Exception as e:
                        await ws.send_json({
                            "type": "error",
                            "message": f"TTS failed: {e}. Install a TTS engine (see voice docs)"
                        })

        except WebSocketDisconnect:
            log("system", session_id or "voice", "Voice WebSocket disconnected")
        except asyncio.TimeoutError:
            await ws.send_json({"type": "error", "message": "Config timeout — send config JSON first"})
            await ws.close()
        except Exception as e:
            log("system", session_id or "voice", f"Voice error: {e}", level="error")
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass


# ── STT Engine ────────────────────────────────────────────────────────────────

_whisper_model = None

async def _transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text using faster-whisper.

    Supports: PCM 16kHz 16-bit mono, WAV, WebM, MP3, FLAC, OGG
    Install: pip install faster-whisper

    Falls back to a stub message if faster-whisper is not installed,
    so the rest of the system can still be tested.
    """
    global _whisper_model

    try:
        from faster_whisper import WhisperModel
        import numpy as np
        import io
        import tempfile
        import os

        # Lazy-load model (downloads on first use, ~1GB for 'base')
        if _whisper_model is None:
            model_size = os.getenv("WHISPER_MODEL", "base")
            device = os.getenv("WHISPER_DEVICE", "cpu")
            compute_type = "int8" if device == "cpu" else "float16"
            _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            log("system", "system", f"Whisper model loaded: {model_size} on {device}")

        # Write to temp file (faster-whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            segments, info = _whisper_model.transcribe(
                temp_path,
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return text.strip()
        finally:
            os.unlink(temp_path)

    except ImportError:
        # Stub for development: return a message indicating STT isn't installed
        log("system", "system", "faster-whisper not installed — using stub STT", level="warn")
        return "[STT unavailable — install faster-whisper]"


# ── TTS Engine ────────────────────────────────────────────────────────────────

async def _synthesize_speech(text: str):
    """
    Convert text to speech audio chunks.

    Yields bytes of PCM audio (16kHz, 16-bit, mono).

    Supports multiple backends, auto-detected:
      1. Kokoro TTS (pip install kokoro) — fastest open source
      2. Piper TTS (pip install piper-tts) — lightweight
      3. Edge TTS (pip install edge-tts) — free Microsoft voices

    Falls back to a "TTS not available" error if none installed.
    """
    # Try Kokoro first
    try:
        import kokoro
        voice = kokoro.Voice("af_bella")  # Default voice
        audio = voice.speak(text)
        # Yield in 4096-byte chunks
        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]
        return
    except ImportError:
        pass

    # Try edge-tts (free, no install weight)
    try:
        import edge_tts
        import io

        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
        return
    except ImportError:
        pass

    # No TTS available — raise clear error
    raise RuntimeError(
        "No TTS engine installed. Install one of:\n"
        "  pip install edge-tts      (easiest, free, uses Microsoft voices)\n"
        "  pip install kokoro         (fast local)\n"
        "  pip install piper-tts      (lightweight local)"
    )
