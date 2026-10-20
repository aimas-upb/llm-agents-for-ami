"""
Main entry point for the AMI Agents system.

Orchestrates initialization and startup of all agents.
"""

import asyncio
import logging
import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from ami_agents.agents.user_assistant.user_assistant_agent import UserAssistantAgent
from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent
from ami_agents.environment.discovery.discovery_service import DiscoveryFactory
from ami_agents.environment.integration.integration_engine import IntegrationEngineFactory
from ami_agents.shared.utils.logger import LoggerFactory
from ami_agents.shared.utils.config_resolver import resolve_yggdrasil_url


# Agent sections of agents.yaml; each one holds that agent's XMPP jid/password.
AGENT_SECTIONS = ("user_assistant", "env_explorer", "interaction_solver")


def load_environment():
    """Load .env file if it exists."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)


class AMIAgentsOrchestrator:
    """
    Orchestrates the AMI agent system.

    Responsible for:
    - Loading configurations
    - Initializing environment connection
    - Starting all agents in correct order
    - Managing agent lifecycle
    - Handling shutdown
    """

    def __init__(self, config_dir: str = "ami_agents/config"):
        """
        Initialize orchestrator.

        Args:
            config_dir: Directory containing configuration files.
        """
        self.config_dir = Path(config_dir)
        self.config = {}
        self.agents = {}
        self.integration_engine = None
        self.discovery_service = None
        self.logger = None
        self.running = False

    async def initialize(self) -> None:
        """
        Initialize the AMI agent system.
        """
        # 1. Setup basic logging first (before config loading)
        import logging
        self.logger = logging.getLogger(__name__)

        # 2. Load configurations
        self.load_configurations()

        # 3. Setup logging with config values
        self.setup_logging()

        # 4. Initialize environment connection
        hmas_url = await self.initialize_environment()

        # 5. Initialize agents with the HMAS URL
        await self.initialize_agents(hmas_url)

        self.logger.info("AMI Agents system initialization complete")

    def setup_logging(self) -> None:
        """
        Setup logging for the system using LoggerFactory.
        """
        log_config = self.config.get("logging", {})

        # Create system logger
        self.logger = LoggerFactory.create_logger(__name__, log_config)

        # Store logging config for agent initialization
        self.logging_config = log_config

        level = log_config.get("level", "INFO")
        file_enabled = log_config.get("file_logging", {}).get("enabled", False)
        console_enabled = log_config.get("console_logging", {}).get("enabled", True)

        self.logger.info("Logging configured: level=%s, file_enabled=%s, console_enabled=%s",
                        level, file_enabled, console_enabled)

    def load_configurations(self) -> None:
        """
        Load all configuration files.
        """
        from ami_agents.shared.utils.config_loader import ConfigLoader

        config_dir = os.path.join(os.path.dirname(__file__), "config")

        # Load all config files
        agents_config = ConfigLoader.load_with_env_vars(os.path.join(config_dir, "agents.yaml"))
        env_config = ConfigLoader.load_environment_config(os.path.join(config_dir, "environment.yaml"))

        # Check if services.yaml exists (optional)
        services_path = os.path.join(config_dir, "services.yaml")
        services_config = {}
        if os.path.exists(services_path):
            services_config = ConfigLoader.load_with_env_vars(services_path)

        # Merge configs
        self.config = ConfigLoader.merge_configs(agents_config, env_config, services_config)

        # Validate the XMPP spec of each agent. There is no global `spade:`
        # section: each agent's own section carries its jid and password,
        # and SPADE takes host/port from the JID.
        self.validate_agent_specs()

        if self.logger:
            self.logger.info("Configuration loaded successfully from %s", config_dir)

    def validate_agent_specs(self) -> None:
        """
        Check that every agent section carries a usable XMPP jid and password.

        Returns None when all three agents can log in to XMPP; raises
        ValueError naming the first missing section or field otherwise. It
        exists so a broken config fails here with a clear message, rather
        than as an XMPP auth error after the first agents have started.
        """
        for section in AGENT_SECTIONS:
            # 1. The agent section itself must be present.
            spec = self.config.get(section)
            if not isinstance(spec, dict):
                raise ValueError(f"Missing required configuration section: {section}")

            # 2. jid and password must be set. ConfigLoader leaves an unset
            #    `${VAR}` (no default) as the literal placeholder, so treat
            #    that as missing and name the env var to export.
            for field in ("jid", "password"):
                value = spec.get(field)
                if not value:
                    raise ValueError(f"Missing '{field}' in configuration section: {section}")
                if isinstance(value, str) and value.startswith("${"):
                    raise ValueError(
                        f"'{section}.{field}' is unresolved ({value}); "
                        f"set the environment variable it references"
                    )


    async def initialize_environment(self) -> str:
        """
        Resolve the HMAS endpoint of the environment the agents start in.

        Returns the Yggdrasil URL of `environments.default`, whose definition
        ConfigLoader.load_environment_config() loaded as the `environment`
        section; YGGDRASIL_URL overrides it. Raises ValueError when no URL
        is configured.
        """
        environment_id = (self.config.get("environments") or {}).get("default")
        environment = self.config.get("environment") or {}
        yggdrasil_url = resolve_yggdrasil_url(self.config)
        self.logger.info("Using environment %s (%s): %s",
                         environment_id, environment.get("name"), yggdrasil_url)
        return yggdrasil_url

    async def initialize_agents(self, hmas_url: str) -> None:
        """
        Initialize all agents.

        Args:
            hmas_url: The HMAS environment endpoint URL.
        """
        from ami_agents.agents.user_assistant.user_assistant_agent import UserAssistantAgent
        from ami_agents.agents.env_explorer.env_explorer_agent import EnvExplorerAgent
        from ami_agents.agents.interaction_solver.interaction_solver_agent import InteractionSolverAgent

        # Get agent configs
        ua_config = self.config.get("user_assistant", {}).copy()
        ee_config = self.config.get("env_explorer", {}).copy()
        is_config = self.config.get("interaction_solver", {}).copy()

        # Add shared config sections to each agent
        shared_configs = {
            "default_llm": self.config.get("default_llm", {}),
            "timeouts": self.config.get("timeouts", {}),
            "environment": self.config.get("environment", {}),
            "bt_execution": self.config.get("bt_execution", {}),
            "http": self.config.get("http", {}),
            "yggdrasil_url": hmas_url
        }

        for agent_config in [ua_config, ee_config, is_config]:
            agent_config.update(shared_configs)

        # Setup target JIDs for inter-agent communication. "user" is the
        # client the UA sends end-of-plan notices to; without it they are
        # dropped, so warn when it is not set.
        target_jids = {
            "user_assistant": ua_config["jid"],
            "explorer": ee_config["jid"],
            "solver": is_config["jid"]
        }
        user_jid = ua_config.get("user_jid")
        if user_jid:
            target_jids["user"] = user_jid
        else:
            self.logger.warning("AMI_USER_JID is not set; end-of-plan notices will be dropped")

        # Merge global logging config with agent-specific overrides
        ua_logging_config = LoggerFactory.merge_configs(self.logging_config, ua_config.get("logging"))
        ee_logging_config = LoggerFactory.merge_configs(self.logging_config, ee_config.get("logging"))
        is_logging_config = LoggerFactory.merge_configs(self.logging_config, is_config.get("logging"))

        ua_config_with_logging = {**ua_config, "logging": ua_logging_config}
        ee_config_with_logging = {**ee_config, "logging": ee_logging_config}
        is_config_with_logging = {**is_config, "logging": is_logging_config}

        # Create agents (but don't start them yet)
        self.agents["user_assistant"] = UserAssistantAgent(
            jid=ua_config["jid"],
            password=ua_config["password"],
            config=ua_config_with_logging,
            target_jids=target_jids
        )

        self.agents["env_explorer"] = EnvExplorerAgent(
            jid=ee_config["jid"],
            password=ee_config["password"],
            config=ee_config_with_logging
        )

        self.agents["interaction_solver"] = InteractionSolverAgent(
            jid=is_config["jid"],
            password=is_config["password"],
            config=is_config_with_logging,
            target_jids=target_jids,
        )

        self.logger.info("All agents initialized: %s", list(self.agents.keys()))

    async def start(self) -> None:
        """
        Start all agents in the correct order.
        """
        # 1. Start EnvExplorer first (discovers environment)
        if "env_explorer" in self.agents:
            await self.agents["env_explorer"].start()
            self.logger.info("EnvExplorer agent started")

        # 2. Start InteractionSolver (handles planning)
        if "interaction_solver" in self.agents:
            await self.agents["interaction_solver"].start()
            self.logger.info("InteractionSolver agent started")

        # 3. Start UserAssistant last (interfaces with users)
        if "user_assistant" in self.agents:
            await self.agents["user_assistant"].start()
            self.logger.info("UserAssistant agent started")

        self.logger.info("All agents started and system is ready")

    async def stop(self) -> None:
        """
        Stop all agents and cleanup.
        """
        # Stop agents in reverse order
        for agent_name in ["user_assistant", "interaction_solver", "env_explorer"]:
            if agent_name in self.agents:
                try:
                    await self.agents[agent_name].stop()
                    self.logger.info("%s agent stopped", agent_name)
                except Exception as e:
                    self.logger.error("Error stopping %s agent: %s", agent_name, e)

        # Cleanup integration engine if exists
        if self.integration_engine:
            try:
                # Add cleanup logic if integration engine supports it
                self.integration_engine = None
            except Exception as e:
                self.logger.error("Error cleaning up integration engine: %s", e)

        self.logger.info("AMI Agents system shutdown complete")

    async def run(self) -> None:
        """
        Main run loop.
        """
        try:
            # 1. Initialize the system
            await self.initialize()

            # 2. Start all agents
            await self.start()

            self.logger.info("AMI Agents system started successfully")

            # 3. Keep system running
            self.running = True
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error("System error: %s", e)
            raise
        finally:
            self.running = False
            await self.stop()


async def main():
    """
    Main entry point.
    """
    # Load environment variables from .env file
    load_environment()

    orchestrator = AMIAgentsOrchestrator()

    # Create a simple logger for system messages
    system_logger = LoggerFactory.get_logger("AMI.System")

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        system_logger.info("Shutting down AMI Agents system...")
        await orchestrator.stop()
    except Exception as e:
        system_logger.error("Error running AMI Agents system: %s", e)
        await orchestrator.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
