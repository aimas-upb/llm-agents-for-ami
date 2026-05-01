"""Entry point for AMI Agents system."""
import asyncio
import sys
from dotenv import load_dotenv
from .main import AMIAgentsOrchestrator
from .shared.utils.logger import LoggerFactory


def main():
    """Main entry point."""
    # Load .env file
    load_dotenv()

    # Create and run orchestrator
    orchestrator = AMIAgentsOrchestrator()

    # Create a simple logger for system messages
    system_logger = LoggerFactory.get_logger("AMI.System")

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        system_logger.info("System stopped by user")
        sys.exit(0)
    except Exception as e:
        system_logger.error("System failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()