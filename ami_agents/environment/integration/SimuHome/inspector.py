#!/usr/bin/env python3
"""
Visual inspector for a loaded SimuHome home.

A read-only single page: a fixed header with the simulated clock and each room's
environmental variables, and a scrollable body of one box per device grouped by
room, listing its properties and — visually distinguished — its writable
attributes.

Everything comes from one call to `GET /api/home/state`, which already returns
the clock, the per-room aggregates and every device with all its attributes.

Writable-vs-read-only is decided by `classify.py`, the same module the TD
generator uses, so what shows here as settable is exactly what gets an
actuatable affordance in the Thing Description.

    python -m ami_agents.environment.integration.SimuHome.inspector \\
        --sim http://127.0.0.1:8099/api --port 8098

Then open http://127.0.0.1:8098/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

try:  # package-relative when imported, plain when run as a script
    from .classify import classify_device
    from .sim_client import SimuHomeClient, SimuHomeError
except ImportError:  # pragma: no cover
    from classify import classify_device  # type: ignore
    from sim_client import SimuHomeClient, SimuHomeError  # type: ignore

app = FastAPI(title="SimuHome Inspector")

_client: SimuHomeClient | None = None


def get_client() -> SimuHomeClient:
    global _client
    if _client is None:
        _client = SimuHomeClient(os.getenv("SIMULATOR_API_BASE_URL"))
    return _client


# Matter carries temperature and humidity in centi-units; show human values.
CENTI_STATES = {"temperature", "humidity"}
STATE_UNITS = {
    "temperature": "°C",
    "humidity": "%",
    "illuminance": "lx",
    "pm10": "µg/m³",
    "air_quality": "µg/m³",
}


def _room_state_display(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for name, raw in sorted((state or {}).items()):
        value = raw
        if isinstance(raw, (int, float)) and name in CENTI_STATES:
            value = round(raw / 100.0, 2)
        elif isinstance(raw, float):
            value = round(raw, 2)
        out.append({"name": name, "value": value, "unit": STATE_UNITS.get(name, "")})
    return out


@app.get("/api/snapshot")
def snapshot() -> JSONResponse:
    """The whole view model, ready to render: one upstream call, classified."""
    try:
        state = get_client().home_state()
    except SimuHomeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    rooms = []
    for room_id, room in sorted((state.get("rooms") or {}).items()):
        devices = []
        for device in room.get("devices", []):
            classified = classify_device(device.get("attributes") or {})
            props, controls, meta = [], [], []
            for path, record in sorted(classified.items()):
                if record["role"] == "drop":
                    continue
                entry = {
                    "path": path,
                    "cluster": record["cluster"],
                    "attribute": record["attribute"],
                    "value": record["value"],
                    "type": record["type"],
                    "mechanism": record["mechanism"],
                }
                if record["role"] == "thing_metadata":
                    meta.append(entry)
                elif record["affordance"] == "actuatable":
                    controls.append(entry)
                else:
                    props.append(entry)
            devices.append({
                "device_id": device.get("device_id"),
                "device_type": device.get("device_type"),
                "metadata": meta,
                "properties": props,
                "controls": controls,
            })
        rooms.append({
            "room_id": room_id,
            "state": _room_state_display(room.get("state") or {}),
            "devices": sorted(devices, key=lambda d: d["device_id"] or ""),
        })

    return JSONResponse({
        "current_time": state.get("current_time"),
        "current_tick": state.get("current_tick"),
        "tick_interval": state.get("tick_interval"),
        "base_time": state.get("base_time"),
        "rooms": rooms,
        "totals": {
            "rooms": len(rooms),
            "devices": sum(len(r["devices"]) for r in rooms),
            "controls": sum(len(d["controls"]) for r in rooms for d in r["devices"]),
        },
    })


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_PAGE)


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>SimuHome Inspector</title>
<style>
  :root {
    --bg:#0f1216; --panel:#171b21; --line:#252b34; --ink:#e6e9ee; --dim:#98a2b3;
    --act:#7ee787; --actbg:#132218; --obs:#79c0ff; --meta:#6e7681;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { position:sticky; top:0; z-index:10; background:var(--panel);
           border-bottom:1px solid var(--line); padding:10px 14px; }
  .clock { display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
  .clock b { font-size:17px; letter-spacing:.5px; }
  .clock .dim { color:var(--dim); }
  .envstrip { display:flex; gap:10px; margin-top:8px; overflow-x:auto; padding-bottom:2px; }
  .envroom { border:1px solid var(--line); border-radius:6px; padding:6px 9px;
             min-width:210px; background:#12161b; }
  .envroom h4 { margin:0 0 4px; font-size:11px; color:var(--dim);
                text-transform:uppercase; letter-spacing:.6px; }
  .envvals { display:flex; gap:9px; flex-wrap:wrap; }
  .envvals span { white-space:nowrap; }
  .envvals i { color:var(--dim); font-style:normal; }
  main { padding:14px; }
  section { margin-bottom:22px; }
  section > h2 { margin:0 0 9px; font-size:13px; letter-spacing:.6px;
                 text-transform:uppercase; color:var(--obs); }
  .grid { display:grid; gap:10px;
          grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); }
  .dev { border:1px solid var(--line); border-radius:7px; background:var(--panel);
         padding:9px 11px; }
  .dev h3 { margin:0; font-size:13px; }
  .dev .dt { color:var(--dim); font-size:11px; margin-bottom:7px; }
  .grp { margin-top:7px; }
  .grp h5 { margin:0 0 3px; font-size:10px; letter-spacing:.7px;
            text-transform:uppercase; color:var(--dim); }
  .row { display:flex; justify-content:space-between; gap:10px; padding:1px 0; }
  .row .k { color:var(--dim); overflow:hidden; text-overflow:ellipsis;
            white-space:nowrap; }
  .row .v { font-weight:600; text-align:right; word-break:break-all; }
  .ctl { background:var(--actbg); border-left:2px solid var(--act);
         padding:3px 6px; margin:2px 0; border-radius:0 4px 4px 0; }
  .ctl .v { color:var(--act); }
  .ctl .m { color:var(--meta); font-size:10px; }
  .err { background:#3b1618; border:1px solid #7d2b2f; padding:10px;
         border-radius:6px; margin:14px; }
  .legend { color:var(--dim); font-size:11px; margin-top:6px; }
  .legend b { color:var(--act); }
</style></head><body>
<header>
  <div class="clock">
    <b id="time">—</b>
    <span class="dim">tick <span id="tick">—</span> @ <span id="ti">—</span>s</span>
    <span class="dim" id="totals"></span>
  </div>
  <div class="envstrip" id="env"></div>
  <div class="legend">green = writable / command-driven <b>controls</b>; grey = read-only properties</div>
</header>
<main id="main">loading…</main>
<script>
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function fmt(v){ if(v===null||v===undefined) return '—';
  if(typeof v==='object') return esc(JSON.stringify(v)).slice(0,60);
  if(typeof v==='number') return Number.isInteger(v)?v:v.toFixed(2); return esc(v); }
function rows(list, cls){ return list.map(p =>
  `<div class="row ${cls}"><span class="k" title="${esc(p.path)}">${esc(p.cluster||'')}.${esc(p.attribute)}</span>`+
  `<span class="v">${fmt(p.value)}${cls==='ctl'?` <span class="m">${esc(p.mechanism)}</span>`:''}</span></div>`).join('');
}
async function tick(){
  let d;
  try { const r = await fetch('/api/snapshot'); d = await r.json(); }
  catch(e){ document.getElementById('main').innerHTML =
    `<div class="err">cannot reach the inspector backend: ${esc(e)}</div>`; return; }
  if(d.error){ document.getElementById('main').innerHTML =
    `<div class="err">simulator unreachable: ${esc(d.error)}</div>`; return; }
  document.getElementById('time').textContent = d.current_time || '—';
  document.getElementById('tick').textContent = d.current_tick;
  document.getElementById('ti').textContent = d.tick_interval;
  document.getElementById('totals').textContent =
    `${d.totals.rooms} rooms · ${d.totals.devices} devices · ${d.totals.controls} controls`;
  document.getElementById('env').innerHTML = d.rooms.map(r =>
    `<div class="envroom"><h4>${esc(r.room_id)}</h4><div class="envvals">`+
    r.state.map(s=>`<span><i>${esc(s.name)}</i> ${fmt(s.value)}${esc(s.unit)}</span>`).join('')+
    `</div></div>`).join('');
  document.getElementById('main').innerHTML = d.rooms.map(r =>
    `<section><h2>${esc(r.room_id)} · ${r.devices.length} devices</h2><div class="grid">`+
    r.devices.map(dev => {
      const name = (dev.metadata.find(m=>m.attribute==='ProductName')||{}).value || dev.device_type;
      return `<div class="dev"><h3>${esc(dev.device_id)}</h3>`+
        `<div class="dt">${esc(name)}</div>`+
        (dev.controls.length?`<div class="grp"><h5>controls (${dev.controls.length})</h5>${rows(dev.controls,'ctl')}</div>`:'')+
        (dev.properties.length?`<div class="grp"><h5>properties (${dev.properties.length})</h5>${rows(dev.properties,'')}</div>`:'')+
        `</div>`;
    }).join('')+`</div></section>`).join('');
}
tick(); setInterval(tick, 1000);
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual inspector for a loaded SimuHome home.")
    parser.add_argument("--sim", default=os.getenv("SIMULATOR_API_BASE_URL", "http://127.0.0.1:8000/api"),
                        help="SimuHome API base URL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()

    os.environ["SIMULATOR_API_BASE_URL"] = args.sim
    global _client
    _client = SimuHomeClient(args.sim)

    import uvicorn
    print(f"Inspector on http://{args.host}:{args.port}/  (simulator: {args.sim})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
