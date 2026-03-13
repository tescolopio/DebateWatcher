"""
src/interface/overlay_server.py – FastAPI + WebSocket server for HTML overlays.

OBS Browser Sources connect to this server via WebSocket.  When the
Orchestrator detects a fallacy or contradiction, it calls ``broadcast()``
which pushes a JSON payload to all connected overlay clients, triggering
in-browser animations.

Endpoints
---------
* ``GET /``           – health-check / status page
* ``GET /overlay``    – serve the default overlay HTML page
* ``WS  /ws``         – WebSocket endpoint for overlay clients
* ``POST /alert``     – REST endpoint for manual alert injection
* ``GET /transcript`` – latest transcript text (SSE stream)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"


class OverlayServer:
    """
    Lightweight WebSocket broadcast server for browser-based OBS overlays.

    Example::

        server = OverlayServer()
        asyncio.create_task(server.serve())          # start in background
        await server.broadcast({"fallacy_type": "Straw Man", "confidence": 0.9})
    """

    def __init__(self) -> None:
        self._host = settings.overlay.host
        self._port = settings.overlay.port
        self._connections: set = set()
        self._app = None
        self._latest_transcript: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def serve(self) -> None:
        """Start the FastAPI/uvicorn server.  Runs until cancelled."""
        try:
            import uvicorn  # type: ignore

            app = self._build_app()
            config = uvicorn.Config(
                app,
                host=self._host,
                port=self._port,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            logger.info(
                "OverlayServer starting on %s:%d", self._host, self._port
            )
            await server.serve()
        except ImportError:
            logger.error(
                "uvicorn/fastapi not installed.  "
                "pip install fastapi uvicorn[standard]"
            )
        except Exception as exc:
            logger.error("OverlayServer failed to start: %s", exc)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """Send *payload* as JSON to all connected WebSocket clients."""
        if not self._connections:
            return
        message = json.dumps(payload)
        dead = set()
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self._connections -= dead
        logger.debug(
            "Broadcast to %d clients: %s",
            len(self._connections),
            payload.get("fallacy_type", payload.get("alert_type")),
        )

    def update_transcript(self, text: str) -> None:
        """Store the latest transcript text for SSE clients."""
        self._latest_transcript = text

    # ------------------------------------------------------------------
    # App construction
    # ------------------------------------------------------------------

    def _build_app(self):
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
            from fastapi.staticfiles import StaticFiles
        except ImportError:
            logger.error("fastapi not installed.  pip install fastapi")
            raise

        app = FastAPI(title="DebateWatcher Overlay Server")

        # Serve static assets directory if it exists
        if _ASSETS_DIR.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(_ASSETS_DIR)),
                name="assets",
            )

        @app.get("/")
        async def health():
            return JSONResponse(
                {
                    "status": "ok",
                    "service": "DebateWatcher Overlay Server",
                    "clients_connected": len(self._connections),
                    "timestamp": time.time(),
                }
            )

        @app.get("/overlay", response_class=HTMLResponse)
        async def overlay():
            overlay_path = _ASSETS_DIR / "overlay.html"
            if overlay_path.exists():
                return HTMLResponse(content=overlay_path.read_text(encoding="utf-8"))
            return HTMLResponse(content=_DEFAULT_OVERLAY_HTML)

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self._connections.add(ws)
            logger.info(
                "Overlay client connected (%d total)", len(self._connections)
            )
            try:
                while True:
                    # Keep the connection alive; clients are receive-only
                    await asyncio.sleep(30)
                    await ws.send_text(json.dumps({"type": "ping"}))
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                self._connections.discard(ws)
                logger.info(
                    "Overlay client disconnected (%d remaining)",
                    len(self._connections),
                )

        @app.post("/alert")
        async def inject_alert(payload: dict):
            """Manual alert injection endpoint for the research dashboard."""
            await self.broadcast(payload)
            return JSONResponse({"status": "broadcast", "recipients": len(self._connections)})

        @app.get("/transcript")
        async def transcript():
            return JSONResponse({"text": self._latest_transcript})

        self._app = app
        return app


# ── Default overlay HTML ─────────────────────────────────────────────────────

_DEFAULT_OVERLAY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DebateWatcher Overlay</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: transparent;
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden;
  }
  #alert-container {
    position: fixed;
    bottom: 80px;
    left: 40px;
    right: 40px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    pointer-events: none;
  }
  .alert-card {
    background: rgba(15, 15, 30, 0.92);
    border-left: 5px solid #e74c3c;
    border-radius: 8px;
    padding: 14px 18px;
    color: #fff;
    font-size: 18px;
    line-height: 1.4;
    animation: slideIn 0.35s ease-out, fadeOut 0.5s ease-in 4.5s forwards;
    max-width: 700px;
  }
  .alert-card.contradiction {
    border-left-color: #f39c12;
  }
  .alert-type {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.75;
    margin-bottom: 4px;
  }
  .alert-fallacy { color: #e74c3c; font-weight: 700; font-size: 20px; }
  .alert-explanation { opacity: 0.9; }
  .confidence-bar {
    height: 3px;
    background: rgba(255,255,255,0.15);
    border-radius: 2px;
    margin-top: 8px;
    overflow: hidden;
  }
  .confidence-fill {
    height: 100%;
    background: #e74c3c;
    transition: width 0.3s ease;
  }
  @keyframes slideIn {
    from { transform: translateX(-30px); opacity: 0; }
    to   { transform: translateX(0);     opacity: 1; }
  }
  @keyframes fadeOut {
    from { opacity: 1; }
    to   { opacity: 0; transform: translateY(10px); }
  }
</style>
</head>
<body>
<div id="alert-container"></div>
<script>
  const WS_PROTO = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const WS_URL = `${WS_PROTO}//${location.host}/ws`;
  let ws;

  function connect() {
    ws = new WebSocket(WS_URL);
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'ping') return;
      showAlert(data);
    };
    ws.onclose = () => setTimeout(connect, 2000);
  }

  function showAlert(data) {
    if (!data.fallacy_type && !data.explanation) return;
    const container = document.getElementById('alert-container');
    const card = document.createElement('div');
    const isFallacy = data.alert_type === 'fallacy' || data.fallacy_type;
    card.className = 'alert-card' + (isFallacy ? '' : ' contradiction');

    const pct = Math.round((data.confidence || 0) * 100);
    card.innerHTML = `
      <div class="alert-type">${isFallacy ? '⚠ Logical Fallacy Detected' : '⚡ Contradiction Detected'}</div>
      ${data.fallacy_type ? `<div class="alert-fallacy">${data.fallacy_type}</div>` : ''}
      <div class="alert-explanation">${data.explanation || ''}</div>
      <div class="confidence-bar">
        <div class="confidence-fill" style="width:${pct}%; background:${isFallacy ? '#e74c3c' : '#f39c12'}"></div>
      </div>
    `;
    container.appendChild(card);
    setTimeout(() => card.remove(), 5500);
  }

  connect();

  // Also handle URL hash injections from OBS SetInputSettings
  function checkHash() {
    const hash = decodeURIComponent(location.hash);
    const match = hash.match(/^#alert=(.+)/);
    if (match) {
      try { showAlert(JSON.parse(match[1])); } catch(e) {}
      history.replaceState(null, '', location.pathname);
    }
  }
  window.addEventListener('hashchange', checkHash);
  checkHash();
</script>
</body>
</html>
"""
