"""Entry point for AMI Agents system."""
import asyncio
import sys
from dotenv import load_dotenv
from .main import AMIAgentsOrchestrator


def main():
    """Main entry point."""
    # Load .env file
    load_dotenv()

    # Create and run orchestrator
    orchestrator = AMIAgentsOrchestrator()

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        print("System stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"System failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()