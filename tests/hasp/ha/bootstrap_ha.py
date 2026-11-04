#!/usr/bin/env python3
"""Prepare a local Home Assistant for the HASP lab308e harness.

    python tests/hasp/ha/bootstrap_ha.py

Idempotent. It (1) installs the Virtual Components integration into
ha-config/custom_components/virtual, (2) starts the container from
docker-compose.yml, (3) completes onboarding when HA is fresh, (4) creates a
long-lived token and (5) writes HA_URL, HA_TOKEN and HA_VIRTUAL_DIR into the
repository .env. The HA login lives in .ha-credentials (gitignored).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import secrets
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx
import websockets

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE.parents[2] / ".env"
CONFIG_DIR = HERE / "ha-config"
VIRTUAL_DIR = CONFIG_DIR / "custom_components" / "virtual"
CREDENTIALS = HERE / ".ha-credentials"
VIRTUAL_ZIP = "https://api.github.com/repos/twrecked/hass-virtual/zipball/{tag}"
CLIENT_ID = "http://localhost:8123/"
TOKEN_NAME = "hasp-bench"


def install_virtual(tag: str) -> None:
    """Copy custom_components/virtual of a hass-virtual release into the HA config."""
    marker = VIRTUAL_DIR / ".installed-tag"
    if VIRTUAL_DIR.exists():
        if marker.exists() and marker.read_text().strip() == tag:
            print(f"Virtual Components {tag} already installed at {VIRTUAL_DIR}")
            return
        shutil.rmtree(VIRTUAL_DIR)
    print(f"Downloading hass-virtual {tag} ...")
    response = httpx.get(VIRTUAL_ZIP.format(tag=tag), follow_redirects=True, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        prefix = next(n for n in archive.namelist() if n.endswith("custom_components/virtual/manifest.json"))
        prefix = prefix[: -len("manifest.json")]
        for name in archive.namelist():
            if name.startswith(prefix) and not name.endswith("/"):
                target = VIRTUAL_DIR / name[len(prefix):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    marker.write_text(tag)
    print(f"Installed into {VIRTUAL_DIR}")


def start_container() -> None:
    """docker compose up -d for this folder's compose file."""
    subprocess.run(["docker", "compose", "-f", str(HERE / "docker-compose.yml"), "up", "-d"], check=True)


def wait_for_ha(base: str, timeout: float) -> None:
    """Block until the HA HTTP server answers."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if httpx.get(f"{base}/api/onboarding", timeout=5).status_code in (200, 404):
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Home Assistant did not answer at {base} within {timeout:.0f}s")
        time.sleep(3)


def load_credentials() -> dict:
    """The saved HA login, or a fresh one (not yet saved)."""
    if CREDENTIALS.exists():
        return json.loads(CREDENTIALS.read_text())
    return {"username": "hasp", "password": secrets.token_urlsafe(16)}


def onboard(base: str, creds: dict) -> str | None:
    """Create the owner account on a fresh HA; returns an access token, or None
    when onboarding was already done."""
    steps = httpx.get(f"{base}/api/onboarding", timeout=10)
    if steps.status_code == 404 or all(step["done"] for step in steps.json()):
        return None
    CREDENTIALS.write_text(json.dumps(creds))
    user = httpx.post(f"{base}/api/onboarding/users", timeout=20, json={
        "client_id": CLIENT_ID, "name": "HASP bench", "username": creds["username"],
        "password": creds["password"], "language": "en"})
    user.raise_for_status()
    token = exchange_code(base, user.json()["auth_code"])
    headers = {"Authorization": f"Bearer {token}"}
    for path, body in (("core_config", {}), ("analytics", {}),
                       ("integration", {"client_id": CLIENT_ID, "redirect_uri": CLIENT_ID})):
        httpx.post(f"{base}/api/onboarding/{path}", headers=headers, json=body, timeout=20)
    return token


def exchange_code(base: str, code: str) -> str:
    """Authorization code -> short-lived access token."""
    response = httpx.post(f"{base}/auth/token", timeout=20, data={
        "grant_type": "authorization_code", "code": code, "client_id": CLIENT_ID})
    response.raise_for_status()
    return response.json()["access_token"]


def login(base: str, creds: dict) -> str:
    """Username/password login flow -> short-lived access token."""
    flow = httpx.post(f"{base}/auth/login_flow", timeout=20, json={
        "client_id": CLIENT_ID, "handler": ["homeassistant", None], "redirect_uri": CLIENT_ID})
    flow.raise_for_status()
    result = httpx.post(f"{base}/auth/login_flow/{flow.json()['flow_id']}", timeout=20, json={
        "client_id": CLIENT_ID, "username": creds["username"], "password": creds["password"]})
    result.raise_for_status()
    if result.json().get("type") != "create_entry":
        raise RuntimeError("HA login failed; check tests/hasp/ha/.ha-credentials")
    return exchange_code(base, result.json()["result"])


async def long_lived_token(base: str, access_token: str) -> str:
    """Create a 10-year long-lived token over the websocket API."""
    async with websockets.connect(base.replace("http", "ws", 1) + "/api/websocket") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": access_token}))
        if json.loads(await ws.recv()).get("type") != "auth_ok":
            raise RuntimeError("Websocket auth failed")
        await ws.send(json.dumps({"id": 1, "type": "auth/long_lived_access_token",
                                  "client_name": f"{TOKEN_NAME}-{int(time.time())}",
                                  "lifespan": 3650}))
        reply = json.loads(await ws.recv())
        if not reply.get("success"):
            raise RuntimeError(f"Token creation failed: {reply}")
        return reply["result"]


def set_env(values: dict) -> None:
    """Replace or append KEY=value lines in the repository .env."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    for key, value in values.items():
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        entry = f"{key}={value}"
        for index, line in enumerate(lines):
            if pattern.match(line):
                lines[index] = entry
                break
        else:
            lines.append(entry)
    ENV_FILE.write_text("\n".join(lines) + "\n")


def main() -> int:
    """Run the bootstrap steps in order."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8123")
    # lab308e.yaml needs climate and media_player platforms, absent before 0.10.
    parser.add_argument("--virtual-tag", default="v0.10.0-alpha.5")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    CONFIG_DIR.mkdir(exist_ok=True)
    install_virtual(args.virtual_tag)
    start_container()
    wait_for_ha(args.url, args.timeout)
    creds = load_credentials()
    access = onboard(args.url, creds)
    if access is None:
        if not CREDENTIALS.exists():
            print("HA is already onboarded but .ha-credentials is missing. Create a token in the HA UI "
                  "(Profile > Security) and put HA_TOKEN in .env.", file=sys.stderr)
            return 1
        access = login(args.url, creds)
    token = asyncio.run(long_lived_token(args.url, access))
    set_env({"HA_URL": args.url.replace("http", "ws", 1) + "/api/websocket", "HA_TOKEN": token,
             "HA_VIRTUAL_DIR": str(VIRTUAL_DIR)})
    # The integration is loaded at HA start; a restart makes sure a first install is seen.
    subprocess.run(["docker", "restart", "homeassistant"], check=True, stdout=subprocess.DEVNULL)
    wait_for_ha(args.url, args.timeout)
    print(f"Done. HA at {args.url}, login in {CREDENTIALS.name}, .env updated "
          f"(HA_URL, HA_TOKEN, HA_VIRTUAL_DIR).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
