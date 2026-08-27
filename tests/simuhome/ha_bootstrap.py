#!/usr/bin/env python3
"""
Headless HomeAssistant onboarding + long-lived token creation.

Automates the interactive first-run flow so a fresh HA instance (e.g. HA Core
in a venv on a cluster head node) becomes harness-ready without a browser:

1. creates the owner user via the onboarding API,
2. exchanges the auth code for a short-lived access token,
3. completes the remaining onboarding steps (core config, analytics,
   integration),
4. mints a long-lived access token over the websocket API and prints it
   (and writes it to --token-file, chmod 600).

Idempotence: if onboarding is already done, pass --username/--password of the
existing owner to log in and mint a fresh long-lived token.

Usage:
    python ha_bootstrap.py --url http://127.0.0.1:8123 \
        --username ami --password <pw> --token-file ~/.ha_token
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

CLIENT_ID_SUFFIX = "/"


async def _onboarding_status(session: aiohttp.ClientSession, url: str):
    async with session.get(f"{url}/api/onboarding") as resp:
        if resp.status == 404:
            return None  # onboarding finished: endpoint gone on older cores
        resp.raise_for_status()
        return await resp.json()


async def _create_owner(session, url, client_id, username, password) -> str:
    payload = {
        "client_id": client_id,
        "name": username,
        "username": username,
        "password": password,
        "language": "en",
    }
    async with session.post(f"{url}/api/onboarding/users", json=payload) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"onboarding/users failed ({resp.status}): {body}")
        return json.loads(body)["auth_code"]


async def _login(session, url, client_id, username, password) -> str:
    """Login flow for an already-onboarded instance."""
    flow = {
        "client_id": client_id,
        "handler": ["homeassistant", None],
        "redirect_uri": client_id,
    }
    async with session.post(f"{url}/auth/login_flow", json=flow) as resp:
        resp.raise_for_status()
        flow_id = (await resp.json())["flow_id"]
    async with session.post(
        f"{url}/auth/login_flow/{flow_id}",
        json={"client_id": client_id, "username": username, "password": password},
    ) as resp:
        body = await resp.json()
        if "result" not in body:
            raise RuntimeError(f"login failed: {body}")
        return body["result"]


async def _exchange_code(session, url, client_id, code) -> str:
    data = {"grant_type": "authorization_code", "code": code, "client_id": client_id}
    async with session.post(f"{url}/auth/token", data=data) as resp:
        body = await resp.json()
        if "access_token" not in body:
            raise RuntimeError(f"token exchange failed: {body}")
        return body["access_token"]


async def _finish_onboarding(session, url, client_id, access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    status = await _onboarding_status(session, url)
    remaining = {s["step"] for s in (status or []) if not s.get("done")}
    if "core_config" in remaining:
        async with session.post(
            f"{url}/api/onboarding/core_config", headers=headers, json={}
        ) as resp:
            if resp.status not in (200, 201):
                print(f"warning: core_config step returned {resp.status}", file=sys.stderr)
    if "analytics" in remaining:
        async with session.post(
            f"{url}/api/onboarding/analytics", headers=headers, json={}
        ) as resp:
            if resp.status not in (200, 201):
                print(f"warning: analytics step returned {resp.status}", file=sys.stderr)
    if "integration" in remaining:
        async with session.post(
            f"{url}/api/onboarding/integration",
            headers=headers,
            json={"client_id": client_id, "redirect_uri": client_id},
        ) as resp:
            if resp.status not in (200, 201):
                print(f"warning: integration step returned {resp.status}", file=sys.stderr)


async def _mint_long_lived(session, url, access_token) -> str:
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    async with session.ws_connect(ws_url) as ws:
        msg = await ws.receive_json()
        assert msg["type"] == "auth_required", msg
        await ws.send_json({"type": "auth", "access_token": access_token})
        msg = await ws.receive_json()
        if msg["type"] != "auth_ok":
            raise RuntimeError(f"websocket auth failed: {msg}")
        await ws.send_json(
            {
                "id": 1,
                "type": "auth/long_lived_access_token",
                "client_name": f"ami-harness-{os.getpid()}",
                "lifespan": 3650,
            }
        )
        msg = await ws.receive_json()
        if not msg.get("success"):
            raise RuntimeError(f"long-lived token creation failed: {msg}")
        return msg["result"]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8123")
    parser.add_argument("--username", default="ami")
    parser.add_argument("--password", required=True)
    parser.add_argument("--token-file", type=Path, default=None)
    args = parser.parse_args()

    url = args.url.rstrip("/")
    client_id = url + CLIENT_ID_SUFFIX

    async with aiohttp.ClientSession() as session:
        status = await _onboarding_status(session, url)
        needs_user = any(
            s["step"] == "user" and not s.get("done") for s in (status or [])
        )
        if needs_user:
            print("onboarding: creating owner user", file=sys.stderr)
            code = await _create_owner(session, url, client_id, args.username, args.password)
        else:
            print("onboarding: already done, logging in", file=sys.stderr)
            code = await _login(session, url, client_id, args.username, args.password)
        access_token = await _exchange_code(session, url, client_id, code)
        await _finish_onboarding(session, url, client_id, access_token)
        token = await _mint_long_lived(session, url, access_token)

    if args.token_file:
        path = args.token_file.expanduser()
        path.write_text(token + "\n")
        path.chmod(0o600)
        print(f"long-lived token written to {path}", file=sys.stderr)
    print(token)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
