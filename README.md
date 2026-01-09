# AMI Agents - LLM-Powered Ambient Intelligence

AAMAS 2026 demo for LLM-powered agents in a Hypermedia Multi-Agent System (HMAS), focused on Ambient Intelligence (AmI) applications.

This demonstrator presents an end-to-end interaction workflow that integrates a simulated HomeAssistant smart environment deployment, its hypermedia-based semantic representation, and the AmI HMAS framework agents supporting both explicit and implicit user requests. The HomeAssistant deployment is configured with a smart light, motorized blinds, and indoor / outdoor environmental sensors.

With the HomeAssistant interface running, the mapping engine is launched, automatically translating the HomeAssistant configuration into a TD-based Hypermedia Environment. The resulting RDF model is inspected through a web browser, illustrating how devices, states, and affordances are exposed as navigable semantic resources.

Next, the AmI HMAS agents are started. Logs illustrate agent initialization, exploration of the hypermedia environment, and discovery of available artifacts. User-driven interactions are then demonstrated. The user first queries the state of the smart light, revealing the interaction between UserAssistant and EnvExplorer for state retrieval. An explicit command to adjust the light and blinds is subsequently issued, classified as unambiguous, executed directly, and stored as a signifier linking the user goal to concrete affordances. 

The handling of implicit requests is then showcased. With the room in a state of low light conditions, the user complains about insufficient illumination. In the absence of a prior Signifier, the system explores relevant affordances, proposes a plan to increase brightness, validates it with the user, executes it, and records the resulting Signifier. When a similar, but differently phrased, implicit request is later issued, the previously learned Signifier is reused to recover and execute the plan directly, demonstrating experience-based adaptation without additional environment exploration.

[Reference document](https://docs.google.com/document/d/1JYpx-sBP1SLu42xdmnWWiPL_kFRihx7WUkXE6ZyaVy0/edit?tab=t.0) for the design of frameworks and TODOs.

## Overview

This project implements an agent-based system for goal-driven interaction with smart environments. Users can make natural language inquiries about environment state and express goals (both precise and under-specified) that the system will plan and execute.

### Key Features

- **Natural Language Interface**: Express goals and queries in natural language
- **Intelligent Planning**: LLM-powered planning with behavior trees
- **Environment Discovery**: Automatic discovery and mapping of smart environments
- **Adaptive Execution**: Learn from usage patterns (signifiers) to improve future planning
- **Maintenance Goals**: Support for persistent, triggered goals
- **Multi-Environment Support**: Works with HomeAssistant, Yggdrasil, and other WoT-compliant environments

## Architecture

The system consists of three main agents implemented using SPADE and SPADE_LLM:

### UserAssistant Agent
Dual-purpose agent with:
- **User-facing**: Chat interface with conversation memory and message classification
- **System-facing**: Plan management, execution monitoring, and preference storage

Message types handled:
- ENV_CAPABILITIES: Environment capability queries
- ENV_STATE: Environment state queries
- GOAL_REQUEST: Goal execution requests
- PLAN_MANAGEMENT: Plan lifecycle management
- PREFERENCE_STATEMENT: User preference recording

### EnvExplorer Agent
Classical SPADE agent responsible for:
- Environment discovery using WoT Discovery principles (Direct, Well-Known URI, mDNS)
- Continuous monitoring for environment changes
- Signifier storage (usage experiences)
- Affordance matching using hybrid reasoning (rule-based + LLM)
- Physics-informed modeling for capability matching

### InteractionSolver Agent
Planning and execution agent that:
- Receives goal requests from UserAssistant
- Gathers planning context from EnvExplorer
- Generates behavior tree plans using LLM
- Monitors plan execution with node-level status tracking
- Supports community-based planning (future)

## Project Structure

```
ami_agents/
├── config/                    # YAML configuration files
│   ├── environment.yaml       # Environment discovery and connection
│   ├── agents.yaml           # Agent configurations
│   └── services.yaml         # External services
│
├── agents/                   # Agent implementations
│   ├── user_assistant/       # UserAssistant agent
│   ├── env_explorer/         # EnvExplorer agent
│   └── interaction_solver/   # InteractionSolver agent
│
├── environment/              # Environment connection layer
│   ├── discovery/            # Discovery service implementations
│   ├── connection/           # HMAS client
│   └── integration/          # Integration engines (HomeAssistant, Yggdrasil)
│
├── shared/                   # Shared components
│   ├── models/              # Data models
│   ├── protocols/           # Protocol interfaces
│   ├── memory/              # Memory management
│   └── utils/               # Utilities
│
└── main.py                  # Main entry point
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation.

## Getting Started

### Prerequisites

- Python 3.9+
- SPADE server (XMPP server)
- HomeAssistant instance (optional, for HomeAssistant integration)
- OpenAI or Anthropic API key (for LLM functionality)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-agents-for-ami
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Configure agents:
```bash
# Edit configuration files in ami_agents/config/
# - environment.yaml: Set discovery method and environment connection
# - agents.yaml: Configure agent JIDs, passwords, and LLM settings
# - services.yaml: Configure external services
```

### Running the System

```bash
python -m ami_agents.main
```

## Configuration

### Environment Discovery

Three discovery methods are supported:

1. **Direct**: Provide a known TD Directory URL
2. **Well-Known URI**: Compose URI from known structure or query service
3. **mDNS**: Discover on local network (default: `_ami-hmas._tcp.local`)

### HomeAssistant Integration

The IntegrationEngine converts HomeAssistant deployments to HMAS:

- HomeAssistant instance → "home" workspace
- Floors → Floor workspaces
- Areas → Area workspaces
- Device labels → Logical area workspaces
- Devices → Artifacts with Thing Descriptions

### External Services

External services (calendar, weather, etc.) are mapped to virtual Things:
- API endpoints → Action affordances
- Sensor data → Property affordances
- Organized in user-specific workspaces

## Implementation Status

⚠️ **Current Status**: This is a skeleton implementation with TODO stubs.

All major components have been structured with:
- Interface definitions
- Method signatures
- Detailed TODO comments describing implementation steps
- Logical flow from agent start to shutdown

### What's Implemented
- ✅ Complete folder structure
- ✅ Configuration files (YAML)
- ✅ Data models and type definitions
- ✅ Protocol interfaces
- ✅ Agent class structures with behavior skeletons
- ✅ Environment connection layer interfaces

### What Needs Implementation
- 🔲 SPADE integration and message handling
- 🔲 LLM provider implementations
- 🔲 Database and storage backends
- 🔲 HMAS client implementation
- 🔲 Discovery service implementations
- 🔲 Integration engine implementations
- 🔲 Memory management
- 🔲 Behavior tree execution
- 🔲 Signifier storage and retrieval
- 🔲 Testing infrastructure

## Development

### Code Organization Principles

1. **Separation of Concerns**: Each component has a clear responsibility
2. **Interface-Driven**: Protocol interfaces define contracts between components
3. **Configuration-Driven**: Behavior controlled via YAML configuration
4. **Async-First**: All I/O operations use async/await
5. **Type-Annotated**: All functions have type hints

### Next Steps for Implementation

1. Implement storage backends (SQLite for plans, signifiers, conversations)
2. Implement HMAS client with WoT Thing Description support
3. Implement LLM provider wrappers (OpenAI, Anthropic)
4. Implement discovery services (especially mDNS)
5. Implement HomeAssistant integration engine
6. Implement SPADE behaviors and message handling
7. Implement behavior tree execution engine
8. Add comprehensive testing
9. Add user interface (CLI or web-based)

## Technologies

- **SPADE**: Multi-agent system framework
- **SPADE_LLM**: LLM integration for SPADE agents
- **W3C WoT**: Web of Things Thing Descriptions
- **HMAS**: Hypermedia Multi-Agent Systems
- **OpenAI/Anthropic**: LLM providers for planning and intent extraction
- **HomeAssistant**: Smart home platform integration

## License

[Specify License]

## Contributing

[Specify contribution guidelines]

## References

- [SPADE Documentation](https://spade-mas.readthedocs.io/en/latest/)
- [SPADE_LLM Documentation](https://sosanzma.github.io/spade_llm/)
- [W3C WoT Thing Description](https://www.w3.org/TR/wot-thing-description/)
- [W3C WoT Discovery](https://www.w3.org/TR/wot-discovery/)
