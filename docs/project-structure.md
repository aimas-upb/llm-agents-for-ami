# AMI Agents - Project Structure

This document describes the folder structure and organization of the AMI Agents project.

## Directory Structure

```
ami_agents/
├── config/                          # Configuration files
│   ├── environment.yaml             # Environment discovery and connection config
│   ├── agents.yaml                  # Agent-specific configurations
│   └── services.yaml                # External services configuration
│
├── agents/                          # Agent implementations
│   ├── user_assistant/              # UserAssistant agent
│   │   ├── __init__.py
│   │   └── user_assistant_agent.py  # Main agent + ChatAgent + PlanManager
│   │
│   ├── env_explorer/                # EnvExplorer agent
│   │   ├── __init__.py
│   │   └── env_explorer_agent.py    # Environment discovery and monitoring
│   │
│   └── interaction_solver/          # InteractionSolver agent
│       ├── __init__.py
│       └── interaction_solver_agent.py  # Planning and goal resolution
│
├── environment/                     # Environment connection layer
│   ├── discovery/                   # Environment discovery implementations
│   │   ├── __init__.py
│   │   └── discovery_service.py     # Direct, Well-Known, mDNS discovery
│   │
│   ├── connection/                  # HMAS client
│   │   ├── __init__.py
│   │   └── hmas_client.py          # Client for interacting with HMAS
│   │
│   └── integration/                 # Integration engines
│       ├── __init__.py
│       └── integration_engine.py    # HomeAssistant and Yggdrasil integration
│
├── shared/                          # Shared components
│   ├── models/                      # Data models
│   │   ├── __init__.py
│   │   ├── messages.py              # Message types for inter-agent communication
│   │   ├── plan.py                  # Plan and BehaviorTree models
│   │   └── environment.py           # Environment models (Workspace, Artifact, etc.)
│   │
│   ├── protocols/                   # Protocol interfaces
│   │   ├── __init__.py
│   │   ├── agent_protocol.py        # Base agent and routing interfaces
│   │   └── llm_protocol.py          # LLM integration interfaces
│   │
│   ├── memory/                      # Memory management
│   │   └── __init__.py
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── config_loader.py         # Configuration loading utilities
│       └── logger.py                # Logging utilities
│
└── main.py                          # Main entry point and orchestrator

# Project root files
├── requirements.txt                 # Python dependencies
├── .env.example                     # Example environment variables
├── .gitignore                       # Git ignore patterns
└── README.md                        # Project overview (existing)
```

## Component Responsibilities

### Configuration Layer (`config/`)
- **environment.yaml**: Environment discovery settings, integration engine config, external services
- **agents.yaml**: SPADE settings, agent-specific configurations, LLM settings, memory settings
- **services.yaml**: External service definitions with Thing Description mappings

### Agent Layer (`agents/`)

#### UserAssistant
- **ChatAgent**: User-facing conversation management, message classification, intent extraction
- **PlanManager**: System-facing plan management, execution monitoring, preference storage

#### EnvExplorer
- Environment discovery and crawling
- Change detection and monitoring
- Signifier storage and retrieval
- Affordance matching (hybrid reasoning)

#### InteractionSolver
- Goal request handling
- Planning context gathering
- Behavior tree plan generation
- Plan execution monitoring

### Environment Layer (`environment/`)

#### Discovery
- Direct discovery (configured endpoint)
- Well-known URI discovery
- mDNS discovery on local network

#### Connection
- HMAS client for interacting with Thing Descriptions
- Workspace and artifact operations
- Property read/write, action invocation
- Event subscriptions

#### Integration
- HomeAssistant integration (converts to HMAS)
- Yggdrasil integration (already HMAS)
- TD Directory functionality

### Shared Layer (`shared/`)

#### Models
- Message types and structures
- Plan representations (BehaviorTree, NodeTemplate)
- Environment models (Workspace, Artifact, ThingDescription, Signifier)

#### Protocols
- Agent interfaces (IAgent, IMessageRouter, IMemoryManager)
- LLM interfaces (ILLMProvider, IIntentExtractor, IPlanGenerator)

#### Utilities
- Configuration loading with environment variable substitution
- Logging setup and management

## Data Flow

1. **Startup**:
   - `main.py` orchestrates system initialization
   - Environment discovery → Integration engine → HMAS creation
   - Agents start in order: EnvExplorer → InteractionSolver → UserAssistant

2. **Environment Discovery**:
   - EnvExplorer crawls HMAS environment
   - Builds internal map of workspaces and artifacts
   - Subscribes to change notifications
   - Notifies other agents when complete

3. **Goal Request Flow**:
   - User message → UserAssistant (ChatAgent)
   - Message classification → Route to handler
   - If GOAL_REQUEST:
     - Extract intent
     - Check running/previous goals
     - Send to InteractionSolver
   - InteractionSolver gathers context from EnvExplorer
   - Generates BehaviorTree plan using LLM
   - Returns plan to UserAssistant
   - UserAssistant executes or schedules plan

4. **Plan Execution**:
   - PlanManager monitors execution
   - Invokes affordances via HMAS client
   - Updates node statuses
   - Stores signifiers for successful actions

5. **Change Detection**:
   - EnvExplorer receives change events
   - Analyzes impact on maintenance plans
   - Notifies UserAssistant if plans affected
   - UserAssistant triggers plans if conditions met

## Extension Points

- **New LLM Providers**: Implement `ILLMProvider` interface
- **New Discovery Methods**: Implement `IDiscoveryService` interface
- **New Integration Sources**: Implement `IIntegrationEngine` interface
- **Custom Behaviors**: Add SPADE behaviors to agents
- **External Services**: Add definitions to `services.yaml`
