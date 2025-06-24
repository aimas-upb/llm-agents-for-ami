<!--
SPDX-FileCopyrightText: 2023 Deutsche Telekom AG

SPDX-License-Identifier: CC0-1.0    
-->
# Environment Monitoring Agent

The Environment Monitoring Agent acts as the environment's "central nervous system" for `lab308`. It maintains a live, in-memory model of the state of every artifact and sensor in the simulation. By listening for property update notifications, it ensures that any agent can query it to get the most current and accurate snapshot of the environment's condition.

**Important:** This agent is designed to work within the `lab308` simulation. Please ensure this agent is running before starting the simulation.

## How to Run

### 1. Configure Environment Variables

This agent requires an OpenAI API key to function. Please set the following environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

On Windows, you can use this command:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
```

### 2. Start the Application

You can run the agent using the Gradle wrapper script:

```bash
# On Linux/macOS
./gradlew bootRun

# On Windows
./gradlew.bat bootRun
```

The agent will start on port 8081.

### 3. Chat Interface

You can interact with this agent through a web interface. Once the agent is running, open the following URL in your browser:

```
https://eclipse.dev/lmos/chat/index.html?agentUrl=http://localhost:8081#/chat
```

### 4. Tracing with OpenTelemetry and Phoenix

This agent uses OpenTelemetry for tracing. To visualize traces, you can use Arize Phoenix.

#### Enable Tracing

To enable tracing, edit the `src/main/resources/application.yml` file and ensure the following properties are set:

```yaml
management:
  otlp:
    tracing:
      endpoint: http://localhost:6006/v1/traces
  tracing:
    enabled: true
    sampling:
      probability: 1.0
```

#### Run Phoenix

You can run Phoenix using Docker. Make sure you have Docker installed and running.

```bash
docker run -p 6006:6006 -p 4317:4317 -p 4318:4318 arizephoenix/phoenix:latest
```

Once Phoenix is running, you can access the UI at [http://localhost:6006](http://localhost:6006) to view traces.

## Code of Conduct

This project has adopted the [Contributor Covenant](https://www.contributor-covenant.org/) in version 2.1 as our code of conduct. Please see the details in our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). All contributors must abide by the code of conduct.

By participating in this project, you agree to abide by its [Code of Conduct](./CODE_OF_CONDUCT.md) at all times.

## Licensing

This project follows the [REUSE standard for software licensing](https://reuse.software/).    
Each file contains copyright and license information, and license texts can be found in the [./LICENSES](./LICENSES) folder. For more information visit https://reuse.software/.    
You can find a guide for developers at https://telekom.github.io/reuse-template/.   
