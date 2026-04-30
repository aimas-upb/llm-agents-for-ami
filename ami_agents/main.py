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
        Setup logging for the system.
        """
        import logging
        from logging.handlers import RotatingFileHandler

        log_config = self.config.get("logging", {})
        level = log_config.get("level", "INFO")
        log_file = log_config.get("file", "logs/ami_agents.log")
        max_bytes = log_config.get("max_bytes", 10485760)  # 10MB
        backup_count = log_config.get("backup_count", 5)

        # Create logs directory if not exists
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Add file handler if log file is configured
        if log_file:
            handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)

        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging configured: level=%s, file=%s", level, log_file)

    def load_configurations(self) -> None:
        """
        Load all configuration files.
        """
        from ami_agents.shared.utils.config_loader import ConfigLoader

        config_dir = os.path.join(os.path.dirname(__file__), "config")

        # Load all config files
        agents_config = ConfigLoader.load_with_env_vars(os.path.join(config_dir, "agents.yaml"))
        env_config = ConfigLoader.load_with_env_vars(os.path.join(config_dir, "environment.yaml"))

        # Check if services.yaml exists (optional)
        services_path = os.path.join(config_dir, "services.yaml")
        services_config = {}
        if os.path.exists(services_path):
            services_config = ConfigLoader.load_with_env_vars(services_path)

        # Merge configs
        self.config = ConfigLoader.merge_configs(agents_config, env_config, services_config)

        # Validate required fields
        required_sections = ["spade", "user_assistant", "env_explorer", "interaction_solver"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        if self.logger:
            self.logger.info("Configuration loaded successfully from %s", config_dir)


    async def initialize_environment(self) -> str:
        """
        Initialize environment discovery and connection.

        Returns:
            The HMAS environment endpoint URL.
        """
        # Get environment configuration
        env_config = self.config.get("environment", {})
        discovery_method = env_config.get("discovery_method", "mdns")

        # For now, use a simple approach - get URL from environments config
        environments = self.config.get("environments", {})
        if environments:
            # Use the first available environment
            first_env = list(environments.values())[0]
            yggdrasil_url = first_env.get("yggdrasil_url", os.getenv("BASE_WS_URI", "http://localhost:8080"))
            self.logger.info("Using Yggdrasil environment: %s", yggdrasil_url)
            return yggdrasil_url

        # Fallback to default
        default_url = os.getenv("BASE_WS_URI", "http://localhost:8080")
        self.logger.warning("No environment configuration found, using default: %s", default_url)
        return default_url

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
            "llm": self.config.get("llm", {}),
            "timeouts": self.config.get("timeouts", {}),
            "environment": self.config.get("environment", {}),
            "bt_execution": self.config.get("bt_execution", {}),
            "http": self.config.get("http", {}),
            "yggdrasil_url": hmas_url
        }

        for agent_config in [ua_config, ee_config, is_config]:
            agent_config.update(shared_configs)

        # Setup target JIDs for inter-agent communication
        target_jids = {
            "user_assistant": ua_config["jid"],
            "explorer": ee_config["jid"],
            "solver": is_config["jid"]
        }

        # Create agents (but don't start them yet)
        self.agents["user_assistant"] = UserAssistantAgent(
            jid=ua_config["jid"],
            password=ua_config["password"],
            config=ua_config,
            target_jids=target_jids
        )

        self.agents["env_explorer"] = EnvExplorerAgent(
            jid=ee_config["jid"],
            password=ee_config["password"],
            config=ee_config
        )

        self.agents["interaction_solver"] = InteractionSolverAgent(
            jid=is_config["jid"],
            password=is_config["password"],
            config=is_config,
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

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        print("\nShutting down AMI Agents system...")
        await orchestrator.stop()
    except Exception as e:
        print(f"Error running AMI Agents system: {e}")
        await orchestrator.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
