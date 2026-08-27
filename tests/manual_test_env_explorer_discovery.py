#!/usr/bin/env python3
"""EnvExplorer discovery and artifact state monitoring.

This script:
1. Starts EnvExplorer agent connected to XMPP
2. Lets it discover the environment via Yggdrasil integration
3. Monitors and prints artifact state every 3 seconds

Prerequisites:
- XMPP server (Prosody) running on localhost:5222
- HomeAssistant Integration Engine running
- Yggdrasil HMAS environment accessible

Usage:
    conda run -n ami-agents python tests/manual_test_env_explorer_discovery.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Load .env file FIRST, before any other imports
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)  # override=True to ensure env vars are set
    print(f"[SETUP] Loaded .env from {env_path}")
else:
    print(f"[SETUP] Warning: .env file not found at {env_path}")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
from ami_agents.agents.env_explorer.experience import build_experience_engine_context_snapshot
from ami_agents.shared.utils.config_loader import ConfigLoader
from ami_agents.shared.utils.logger import LoggerFactory


logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see subscription details
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("EnvExplorerDiscovery")
# Reduce noise from verbose modules
logging.getLogger("slixmpp").setLevel(logging.INFO)
logging.getLogger("aiohttp").setLevel(logging.INFO)


class DummyHMASClient:
    """Dummy HMAS client stub."""
    pass


async def main():
    """Main entry point."""
    import os

    # Use the same pattern as manual_test_full_flow_plan.py:
    # Use SPADE_PASSWORD (with fallback to "password") for all agents
    # This ensures accounts created in one run can be used in subsequent runs
    xmpp_server = os.getenv("SPADE_HOST", "localhost")
    xmpp_port = int(os.getenv("SPADE_PORT", "5222"))
    password = os.getenv("SPADE_PASSWORD", "password")

    logger.info(f"[CONFIG] XMPP Server: {xmpp_server}:{xmpp_port}")
    logger.info(f"[CONFIG] Using SPADE_PASSWORD: {password[:10] if password else 'EMPTY'}")

    # Load configuration
    config_dir = Path(__file__).resolve().parents[1] / "ami_agents" / "config"
    agents_config = ConfigLoader.load_with_env_vars(str(config_dir / "agents.yaml"))
    env_config = ConfigLoader.load_with_env_vars(str(config_dir / "environment.yaml"))

    # Get EnvExplorer JID from config
    env_explorer_config = agents_config.get("env_explorer", {})
    jid = env_explorer_config.get("jid")

    if not jid:
        logger.error(f"[ERROR] Missing JID in agents.yaml")
        return

    if not password:
        logger.error(f"[ERROR] Missing SPADE_PASSWORD environment variable")
        return

    # If the account already exists with a different password, SPADE auth will fail.
    # As a workaround for testing, we can use a fresh account by adding a unique suffix
    # (This is only for this test script; production uses fixed JIDs in agents.yaml)
    import time
    timestamp = int(time.time() * 1000) % 10000  # Last 4 digits of milliseconds
    fresh_jid = f"env_explorer_test_{timestamp}@{jid.split('@')[1]}"
    logger.info(f"[CONFIG] Using fresh test JID: {fresh_jid} (original: {jid})")
    jid = fresh_jid

    # Merge configs (environment config + agents config for logging/timeouts)
    merged_config = env_config.copy()
    merged_config.update(agents_config)

    logger.info(f"Creating EnvExplorer agent with JID={jid}, password={password}")

    agent = EnvExplorerAgent(
        jid=jid,
        password=password,  # Use the SPADE_PASSWORD env var (matching manual_test_full_flow_plan.py)
        config=merged_config,
        hmas_client=DummyHMASClient(),
    )

    logger.info(f"Starting EnvExplorer agent: {agent.jid}")
    logger.info("Connecting to XMPP and discovering environment...")

    try:
        # Start the agent (this connects to XMPP and runs behaviors)
        logger.info("Attempting to connect to XMPP server with auto_register=True...")
        await agent.start(auto_register=True)
        logger.info("Successfully connected to XMPP!")

        # Give discovery time to start
        logger.info("Waiting for discovery to initialize...")
        await asyncio.sleep(2)
        logger.info("Discovery initialized, monitoring for state changes...")
        logger.info("Set logging to INFO level to see events")

        # Keep track of previous states to detect changes
        previous_states = {}
        for artifact_id, artifact in (agent.artifacts or {}).items():
            previous_states[artifact_id] = dict(artifact.current_state or {})

        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MONITORING STARTED")
        print(f"{'='*80}")
        print(f"Discovery complete: {agent.discovery_complete}")
        print(f"Total artifacts: {len(agent.artifacts or {})}")
        print("\nWaiting for state changes... (Press Ctrl+C to stop)")
        print(f"{'='*80}\n")

        # Monitor for state changes
        while True:
            await asyncio.sleep(1)

            if not agent.artifacts:
                continue

            for artifact_id, artifact in agent.artifacts.items():
                current_state = artifact.current_state or {}
                previous_state = previous_states.get(artifact_id, {})

                # Check if state changed
                if current_state != previous_state:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n{'='*80}")
                    print(f"[{timestamp}] STATE CHANGE DETECTED: {artifact.name}")
                    print(f"{'='*80}")
                    print(f"Artifact ID: {artifact_id}")
                    print(f"Type: {artifact.artifact_type.value}")
                    print(f"Workspace: {artifact.workspace_id}")
                    print(f"\nFull Current State ({len(current_state)} properties):")
                    for prop_uri, value in sorted(current_state.items()):
                        print(f"  {prop_uri}: {value}")

                    # Print context for just this artifact (what SHACL validator would see)
                    print(f"\n{'='*80}")
                    print(f"Context Builder Snapshot (for SHACL validation):")
                    print(f"{'='*80}")
                    import json
                    context_for_artifact = {artifact_id: current_state}
                    print(json.dumps(context_for_artifact, indent=2, default=str))

                    # Update the previous state
                    previous_states[artifact_id] = dict(current_state)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info("Stopping EnvExplorer agent...")
        await agent.stop()
        logger.info("Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())
