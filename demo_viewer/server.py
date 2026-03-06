"""
demo_viewer/server.py
=====================
Web visualization server for the LLM Agents for AMI demo.

Layout (served at /demo):
  Left panel  → Home Assistant iframe at /home/overview
                (full reverse proxy at root – strips X-Frame-Options,
                 so HA auth flow works natively inside the iframe)
  Right panel → Real-time log viewer (SSE stream from demo subprocess)

Usage:
    python demo_viewer/server.py
    open http://localhost:8765/demo

Auth note:
    HA is proxied at the server root (/).  The first time you visit /demo,
    if HA needs authentication it will show its login form inside the left
    iframe.  Log in there once – the token is stored in localStorage at
    http://localhost:8765 and subsequent visits work without re-login.

No modifications to any existing project files.
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_SCRIPT  = PROJECT_ROOT / "tests" / "demo_with_home_assistant.py"
HA_URL       = "http://localhost:8123"
HA_WS_URL    = "ws://localhost:8123/api/websocket"
YGG_URL      = "http://localhost:8080/"
SERVER_PORT  = 8765

log = logging.getLogger("demo_viewer")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class _State:
    process:          Optional[asyncio.subprocess.Process] = None
    running_sequence: Optional[str]                        = None
    log_history:      list                                 = []
    log_subscribers:  list                                 = []   # one asyncio.Queue per SSE client

state = _State()

# ---------------------------------------------------------------------------
# Log broadcast
# ---------------------------------------------------------------------------

async def _broadcast(line: str) -> None:
    state.log_history.append(line)
    if len(state.log_history) > 10_000:
        state.log_history = state.log_history[-10_000:]
    for q in list(state.log_subscribers):
        try:
            q.put_nowait(line)
        except asyncio.QueueFull:
            pass

async def _capture(proc: asyncio.subprocess.Process) -> None:
    """Read subprocess stdout line-by-line and broadcast to all SSE clients."""
    try:
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            await _broadcast(line)
    except Exception as exc:
        log.warning("capture error: %s", exc)
    await proc.wait()
    state.process          = None
    state.running_sequence = None
    await _broadcast("[DEMO_VIEWER] Process finished.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if state.process and state.process.returncode is None:
        state.process.terminate()

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Demo API endpoints
# MUST be defined BEFORE the catch-all HA proxy routes.
# ---------------------------------------------------------------------------

@app.get("/demo", response_class=HTMLResponse)
@app.get("/demo/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(_HTML)


@app.post("/api/demo/run")
async def demo_run(request: Request):
    body  = await request.json()
    seq   = str(body.get("sequence", "3"))
    clear = bool(body.get("clear_signifiers", True))

    if state.process and state.process.returncode is None:
        return Response(
            json.dumps({"error": "Demo already running"}),
            status_code=409,
            media_type="application/json",
        )

    state.log_history.clear()

    cmd = [
        sys.executable,
        "-u",              # force unbuffered stdout/stderr so print() and logging stay in order
        str(DEMO_SCRIPT),
        "--sequence", seq,
        "--signifier-matcher", "v2",
    ]
    if clear:
        cmd.append("--clear-signifiers")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    state.process          = proc
    state.running_sequence = seq
    asyncio.create_task(_capture(proc))

    return {"ok": True, "pid": proc.pid, "sequence": seq}


@app.delete("/api/demo/stop")
async def demo_stop():
    if state.process and state.process.returncode is None:
        state.process.terminate()
        try:
            await asyncio.wait_for(state.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            state.process.kill()
    state.process          = None
    state.running_sequence = None
    await _broadcast("[DEMO_VIEWER] Stopped by user.")
    return {"ok": True}


@app.get("/api/demo/status")
async def demo_status():
    ha_ok = ygg_ok = False
    async with httpx.AsyncClient(timeout=2.0) as client:
        for flag, url in [("ha", HA_URL), ("ygg", YGG_URL)]:
            try:
                r = await client.get(url)
                if flag == "ha":
                    ha_ok  = r.status_code < 500
                else:
                    ygg_ok = r.status_code < 500
            except Exception:
                pass
    return {
        "ha_ok":            ha_ok,
        "ygg_ok":           ygg_ok,
        "process_running":  state.process is not None and state.process.returncode is None,
        "running_sequence": state.running_sequence,
    }


@app.get("/api/demo/logs")
async def demo_logs(request: Request):
    """SSE: replay log history then stream live lines."""
    q: asyncio.Queue = asyncio.Queue(maxsize=2_000)
    state.log_subscribers.append(q)

    async def gen():
        for line in list(state.log_history):
            yield f"data: {json.dumps({'text': line})}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps({'text': line})}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            try:
                state.log_subscribers.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )

# ---------------------------------------------------------------------------
# HA HTTP reverse proxy
# ---------------------------------------------------------------------------

_STRIP_REQ = {"host", "content-length", "transfer-encoding"}
_STRIP_RES = {
    "x-frame-options",
    "content-security-policy",
    "transfer-encoding",
    "connection",
    # httpx decompresses gzip/br automatically – original length is no longer valid
    "content-length",
    "content-encoding",
}

_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]


async def _proxy_http(path: str, request: Request) -> Response:
    query = request.url.query
    url   = f"{HA_URL}/{path}" + (f"?{query}" if query else "")
    hdrs  = {k: v for k, v in request.headers.items()
             if k.lower() not in _STRIP_REQ}

    log.info("HTTP proxy → %s %s", request.method, url)

    # Force gzip/deflate only.  httpx does NOT support brotli (br) decompression
    # unless the optional brotlicffi package is installed.  If HA responds with
    # brotli (its preferred encoding) and httpx can't decode it, it silently
    # returns the raw compressed bytes.  We then strip Content-Encoding: br, so
    # the browser receives binary garbage it tries to parse as JS → SyntaxError.
    hdrs["accept-encoding"] = "gzip, deflate"

    # follow_redirects=False: we let the BROWSER follow redirects so that
    # the URL bar stays accurate (HA JS reads window.location to know its path).
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        try:
            resp = await client.request(
                method  = request.method,
                url     = url,
                headers = hdrs,
                content = await request.body(),
            )
        except httpx.ConnectError:
            log.warning("HTTP proxy: cannot reach HA at %s", url)
            return Response(
                b"Home Assistant not reachable at " + HA_URL.encode(),
                status_code=502,
            )

    content_type = resp.headers.get("content-type", "")
    log.info("HTTP proxy ← %d  %s (%d bytes, %s)", resp.status_code, url,
             len(resp.content), content_type.split(";")[0])

    out_hdrs = {k: v for k, v in resp.headers.items()
                if k.lower() not in _STRIP_RES}

    # Rewrite Location header so redirects point back to OUR server, not HA's.
    # e.g.  Location: http://localhost:8123/home/overview
    #    →  Location: http://localhost:8765/home/overview
    if "location" in out_hdrs:
        loc = out_hdrs["location"]
        if loc.startswith(HA_URL):
            out_hdrs["location"] = loc.replace(HA_URL,
                                                f"http://localhost:{SERVER_PORT}", 1)

    # Rewrite any hardcoded HA origin inside text responses (HTML, JS, JSON).
    # HA sometimes embeds absolute localhost:8123 URLs; these would be fetched
    # by the browser from the wrong origin, failing CORS and silently breaking.
    content = resp.content
    _HA_BYTES  = HA_URL.encode()
    _OWN_BYTES = f"http://localhost:{SERVER_PORT}".encode()
    if _HA_BYTES in content and any(t in content_type for t in ("text/", "javascript", "json")):
        content = content.replace(_HA_BYTES, _OWN_BYTES)
        log.info("HTTP proxy: rewrote HA origin in response body")

    return Response(
        content     = content,
        status_code = resp.status_code,
        headers     = out_hdrs,
        media_type  = content_type or None,
    )


@app.get("/service_worker.js")
@app.get("/service_worker_es5.js")
async def kill_switch_sw():
    """Serve a one-shot service worker that immediately unregisters itself.

    If HA's service worker was registered from http://localhost:8765 in a previous
    session, it would intercept ALL JS requests before they reach the network,
    silently serving stale or missing files from its cache.  This kill-switch SW
    overwrites the registration; on the next page load HA will be clean.
    """
    js = (
        "// Kill-switch service worker: unregister any previous SW at this origin.\n"
        "self.addEventListener('install', () => self.skipWaiting());\n"
        "self.addEventListener('activate', () => {\n"
        "  self.registration.unregister();\n"
        "  clients.claim();\n"
        "});\n"
    )
    return Response(js, media_type="application/javascript")


@app.get("/api/debug/ha-html")
async def debug_ha_html():
    """Return HA's raw HTML (plus response headers) as plain text, for debugging."""
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            resp = await client.get(f"{HA_URL}/")
        except Exception as exc:
            return Response(f"Error connecting to HA: {exc}", media_type="text/plain",
                            status_code=502)
    info = (
        f"=== HA Response at {HA_URL}/ ===\n"
        f"Status: {resp.status_code}\n"
        f"Headers:\n" + "\n".join(f"  {k}: {v}" for k, v in resp.headers.items()) +
        f"\n\nBody ({len(resp.text)} chars):\n{resp.text}"
    )
    return Response(info, media_type="text/plain; charset=utf-8")


# HA is proxied at the server ROOT.
# This ensures the auth flow works: HA stores tokens in localStorage at
# http://localhost:8765, and the iframe (same origin) can use them.

@app.api_route("/", methods=_METHODS)
async def ha_root(request: Request):
    return await _proxy_http("", request)


# Catch-all: proxy everything to HA except our own /demo and /api/demo/* routes.
# The specific routes above are matched first by FastAPI (registration order).
@app.api_route("/{path:path}", methods=_METHODS)
async def ha_catchall(path: str, request: Request):
    if path == "demo" or path.startswith("demo/"):
        return Response(status_code=404)
    if path.startswith("api/demo/") or path == "api/demo":
        return Response(status_code=404)
    if path.startswith("api/debug/") or path == "api/debug":
        return Response(status_code=404)
    return await _proxy_http(path, request)

# ---------------------------------------------------------------------------
# HA WebSocket proxy
# WebSocket routes are a separate ASGI scope – no conflict with HTTP catch-all.
# ---------------------------------------------------------------------------

@app.websocket("/api/websocket")
async def ha_ws_proxy(websocket: WebSocket):
    log.info("WS proxy: browser client connecting → attempting HA at %s", HA_WS_URL)

    # Connect to HA FIRST so we can reject browser cleanly if HA is unavailable.
    try:
        ha = await websockets.connect(HA_WS_URL)
        log.info("WS proxy: connected to HA successfully")
    except Exception as exc:
        log.warning("WS proxy: cannot connect to HA WS: %s – rejecting browser WS", exc)
        # Must accept before we can close (Starlette requires it).
        await websocket.accept()
        await websocket.close(code=1011, reason="HA WebSocket unavailable")
        return

    await websocket.accept()
    log.info("WS proxy: bidirectional bridge active (browser ↔ HA)")

    async def to_ha():
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    await ha.send(msg["text"])
                elif msg.get("bytes") is not None:
                    await ha.send(msg["bytes"])
        except Exception as exc:
            log.debug("WS to_ha ended: %s", exc)

    async def from_ha():
        try:
            async for msg in ha:
                if isinstance(msg, bytes):
                    await websocket.send_bytes(msg)
                else:
                    await websocket.send_text(str(msg))
        except Exception as exc:
            log.debug("WS from_ha ended: %s", exc)

    try:
        done, pending = await asyncio.wait(
            [asyncio.create_task(to_ha()), asyncio.create_task(from_ha())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        log.info("WS proxy: connection closed")
        try:
            await ha.close()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# HTML/JS UI  (served at /demo – NOT an f-string, port injected via .replace)
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Agents for AMI – Demo Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:'Segoe UI',system-ui,sans-serif;
  background:#0f172a;color:#e2e8f0;
  display:flex;flex-direction:column;height:100vh;overflow:hidden;
}

/* ── Header ─────────────────────────────────────────────────────────── */
header{
  background:#1e293b;border-bottom:1px solid #334155;
  padding:.55rem 1rem;display:flex;align-items:center;
  gap:.6rem;flex-shrink:0;flex-wrap:wrap;
}
header h1{
  font-size:.92rem;font-weight:700;color:#f8fafc;
  margin-right:auto;white-space:nowrap;
}
header h1 span{color:#60a5fa}

select,button{
  background:#334155;color:#e2e8f0;border:1px solid #475569;
  border-radius:6px;padding:.32rem .65rem;font-size:.8rem;
  cursor:pointer;transition:background .15s;
}
select:hover,button:hover{background:#475569}
button:disabled{opacity:.4;cursor:not-allowed}
button.run{background:#14532d;border-color:#16a34a}
button.run:hover:not(:disabled){background:#166534}
button.stop{background:#7f1d1d;border-color:#b91c1c}
button.stop:hover:not(:disabled){background:#991b1b}

.badge{
  display:inline-flex;align-items:center;gap:.3rem;
  padding:.18rem .5rem;border-radius:999px;font-size:.7rem;font-weight:600;
}
.badge::before{content:'';width:6px;height:6px;border-radius:50%;background:currentColor}
.badge.ok{background:#14532d;color:#86efac}
.badge.err{background:#7f1d1d;color:#fca5a5}
.badge.chk{background:#1e3a5f;color:#93c5fd}

.fgrp{
  display:flex;gap:2px;background:#0f172a;
  border:1px solid #334155;border-radius:6px;padding:2px;
}
.fgrp button{
  border:none;padding:.25rem .5rem;font-size:.73rem;
  border-radius:4px;background:transparent;
}
.fgrp button.active{background:#3b82f6;color:#fff}

.seq4w{
  font-size:.7rem;color:#fbbf24;
  background:#451a03;border:1px solid #92400e;
  padding:.18rem .45rem;border-radius:4px;display:none;
}

label.ck{
  display:flex;align-items:center;gap:.3rem;
  font-size:.8rem;color:#94a3b8;cursor:pointer;user-select:none;
}

/* ── Panels ──────────────────────────────────────────────────────────── */
.panels{display:flex;flex:1;overflow:hidden}
.panel{
  flex:1;display:flex;flex-direction:column;
  overflow:hidden;border-right:1px solid #334155;
}
.panel:last-child{border-right:none}
.phdr{
  background:#1e293b;padding:.42rem .9rem;
  font-size:.73rem;font-weight:700;color:#94a3b8;
  text-transform:uppercase;letter-spacing:.06em;
  border-bottom:1px solid #334155;flex-shrink:0;
  display:flex;align-items:center;justify-content:space-between;
}

#ha-frame{flex:1;border:none;background:#1e293b}

/* ── Log panel ───────────────────────────────────────────────────────── */
.logwrap{
  flex:1;overflow-y:auto;padding:.35rem .45rem;
  font-family:'Cascadia Code','JetBrains Mono','Fira Code',monospace;
  font-size:.69rem;line-height:1.55;background:#070d1a;
}
.ll{padding:1px 3px;border-radius:2px;white-space:pre-wrap;word-break:break-all}
.ll:hover{background:#1e293b40}
.ll.hidden{display:none}

.ts{color:#4b5563}
.lg{color:#818cf8}
.lv-info{color:#60a5fa}
.lv-warning{color:#fbbf24}
.lv-error{color:#f87171}
.tdemo{
  color:#4ade80;font-weight:700;
  background:rgba(74,222,128,.12);padding:0 2px;border-radius:2px;
}
.l-demo{color:#a7f3d0}
.l-warn{color:#fbbf24}
.l-err{color:#f87171}
.l-sys{color:#60a5fa;font-style:italic}
.l-dim{color:#e2e8f0}

/* ── Footer ──────────────────────────────────────────────────────────── */
footer{
  background:#1e293b;border-top:1px solid #334155;
  padding:.28rem 1rem;font-size:.68rem;color:#64748b;
  display:flex;gap:1rem;align-items:center;flex-shrink:0;
}
#st-txt{margin-left:auto;color:#94a3b8}
</style>
</head>
<body>

<header>
  <h1>LLM Agents for <span>AMI</span> &ndash; Demo Viewer</h1>

  <div id="b-ha"  class="badge chk">HA</div>
  <div id="b-ygg" class="badge chk">Yggdrasil</div>

  <select id="seq-sel" onchange="onSeqChange()">
    <option value="basic">Sequence 1 &ndash; Basic (quick debug)</option>
    <option value="2">Sequence 2 &ndash; Startup only</option>
    <option value="3">Sequence 3 &ndash; Demo (full interaction)</option>
    <option value="4">Sequence 4 &ndash; Reuse Demo</option>
  </select>

  <div class="seq4w" id="seq4w">&#9888; Seq 4 needs signifiers from Seq 3</div>

  <label class="ck">
    <input type="checkbox" id="clr-chk" checked>
    Clear signifiers
  </label>

  <button class="run"  id="btn-run"  onclick="runDemo()">&#9654; Run</button>
  <button class="stop" id="btn-stop" onclick="stopDemo()" disabled>&#9632; Stop</button>
</header>

<div class="panels">

  <!-- Left: Home Assistant (proxied at root – auth works inside iframe) -->
  <div class="panel" style="flex:1.3">
    <div class="phdr">&#127968; Home Assistant &mdash; lab308</div>
    <iframe id="ha-frame" src="/home/overview" allow="*"></iframe>
  </div>

  <!-- Right: Logs -->
  <div class="panel" style="flex:1">
    <div class="phdr">
      <span>&#128203; Interaction Logs</span>
      <div class="fgrp">
        <button id="f-all"  class="active" onclick="setFilter('all')">All</button>
        <button id="f-demo" onclick="setFilter('demo')">[DEMO] only</button>
      </div>
    </div>
    <div class="logwrap" id="logwrap">
      <div class="ll l-sys">Waiting for demo to start&hellip;</div>
      <div id="logend"></div>
    </div>
  </div>

</div>

<footer>
  <span>&#128279; http://localhost:DEMO_PORT_PLACEHOLDER/demo</span>
  <span id="st-lines">0 lines</span>
  <span id="st-stream">&#9675; Not connected</span>
  <span id="st-txt">Ready</span>
</footer>

<script>
let filterMode = 'all';
let lineCount  = 0;
let autoScroll = true;
let evtSrc     = null;

const wrap   = document.getElementById('logwrap');
const logEnd = document.getElementById('logend');

// ── Sequence selector ──────────────────────────────────────────────────
function onSeqChange() {
  const v    = document.getElementById('seq-sel').value;
  const warn = document.getElementById('seq4w');
  const chk  = document.getElementById('clr-chk');
  if (v === '4') {
    warn.style.display = 'block';
    chk.checked = false;
  } else {
    warn.style.display = 'none';
    if (v === '3') chk.checked = true;
  }
}

// ── Filter ─────────────────────────────────────────────────────────────
function setFilter(mode) {
  filterMode = mode;
  document.getElementById('f-all' ).classList.toggle('active', mode === 'all');
  document.getElementById('f-demo').classList.toggle('active', mode === 'demo');
  wrap.querySelectorAll('.ll').forEach(applyFilter);
}

function applyFilter(el) {
  if (el.classList.contains('l-sys')) return;
  el.classList.toggle('hidden', filterMode === 'demo' && el.dataset.demo !== '1');
}

// ── Log rendering ──────────────────────────────────────────────────────
const PAT = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[([^\]]+)\] (\w+): (.*)$/s;
// Strip ANSI escape sequences (e.g. \x1b[1;36m … \x1b[0m) from subprocess output.
const ANSI_RE = /\x1b\[[0-9;]*[A-Za-z]/g;

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function addLine(rawText) {
  const text = rawText.replace(ANSI_RE, '');
  const isSys = text.startsWith('[DEMO_VIEWER]');
  const div   = document.createElement('div');
  div.className = 'll';

  if (isSys) {
    div.classList.add('l-sys');
    div.textContent = text;
  } else {
    const m = PAT.exec(text);
    if (m) {
      const [, ts, logger, level, msg] = m;
      const isDemo = msg.includes('[DEMO]');
      const lv     = level.toLowerCase();

      if      (isDemo)           div.classList.add('l-demo');
      else if (lv === 'warning') div.classList.add('l-warn');
      else if (lv === 'error')   div.classList.add('l-err');

      const msgHtml = esc(msg).replace(/\[DEMO\]/g,
        '<span class="tdemo">[DEMO]</span>');

      div.innerHTML =
        '<span class="ts">'           + esc(ts)     + '</span> ' +
        '[<span class="lg">'          + esc(logger) + '</span>] ' +
        '<span class="lv-' + lv + '">' + esc(level) + '</span>: ' +
        msgHtml;

      div.dataset.demo = isDemo ? '1' : '0';
    } else {
      div.classList.add('l-dim');
      div.textContent  = text;
      div.dataset.demo = '0';
    }
    applyFilter(div);
  }

  wrap.insertBefore(div, logEnd);
  lineCount++;
  document.getElementById('st-lines').textContent = lineCount + ' lines';
  if (autoScroll) logEnd.scrollIntoView({ behavior: 'instant' });
}

// ── SSE stream ─────────────────────────────────────────────────────────
function connectStream() {
  if (evtSrc) evtSrc.close();
  evtSrc = new EventSource('/api/demo/logs');
  document.getElementById('st-stream').textContent = '\u25cb Connecting\u2026';
  evtSrc.onopen    = () => { document.getElementById('st-stream').textContent = '\u25cf Connected'; };
  evtSrc.onmessage = (e) => { addLine(JSON.parse(e.data).text); };
  evtSrc.onerror   = () => { document.getElementById('st-stream').textContent = '\u25cc Reconnecting\u2026'; };
}

wrap.addEventListener('scroll', () => {
  const { scrollTop, scrollHeight, clientHeight } = wrap;
  autoScroll = scrollHeight - scrollTop - clientHeight < 40;
});

// ── Demo control ───────────────────────────────────────────────────────
async function runDemo() {
  const seq = document.getElementById('seq-sel').value;
  const clr = document.getElementById('clr-chk').checked;

  wrap.querySelectorAll('.ll').forEach(el => el.remove());
  lineCount = 0;
  document.getElementById('st-lines').textContent = '0 lines';
  autoScroll = true;

  const res  = await fetch('/api/demo/run', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ sequence: seq, clear_signifiers: clr }),
  });
  const data = await res.json();

  if (res.ok) {
    document.getElementById('btn-run' ).disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('st-txt'  ).textContent = 'Running sequence ' + seq;
  } else {
    addLine('[DEMO_VIEWER] ERROR: ' + (data.error || JSON.stringify(data)));
  }
}

async function stopDemo() {
  await fetch('/api/demo/stop', { method: 'DELETE' });
  document.getElementById('btn-run' ).disabled = false;
  document.getElementById('btn-stop').disabled = true;
  document.getElementById('st-txt'  ).textContent = 'Stopped';
}

// ── Health polling ─────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const res  = await fetch('/api/demo/status');
    const data = await res.json();
    setBadge('b-ha',  data.ha_ok,  'HA');
    setBadge('b-ygg', data.ygg_ok, 'Yggdrasil');
    const running = data.process_running;
    document.getElementById('btn-run' ).disabled = running;
    document.getElementById('btn-stop').disabled = !running;
    if (!running && document.getElementById('st-txt').textContent.startsWith('Running'))
      document.getElementById('st-txt').textContent = 'Ready';
  } catch (_) {}
}

function setBadge(id, ok, label) {
  const el = document.getElementById(id);
  el.className   = 'badge ' + (ok ? 'ok' : 'err');
  el.textContent = label;
}

connectStream();
pollStatus();
setInterval(pollStatus, 5000);
</script>
</body>
</html>
""".replace("DEMO_PORT_PLACEHOLDER", str(SERVER_PORT))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    log.info("Demo Viewer → http://localhost:%d/demo", SERVER_PORT)
    log.info("HA proxied at → http://localhost:%d/  (same origin = auth works)", SERVER_PORT)
    # log_level="info" enables uvicorn access logs (one line per request) –
    # useful to see ALL requests hitting the server, even if routing bypasses our handlers.
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
