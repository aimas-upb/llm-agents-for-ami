# llm-agents-for-ami
LLM powered agents in a Hypermedia MAS, focused on **smart environment** applications.

This is a demonstrator project for the deployment of the AmI HMAS agent system on a simulated smart environment environment deployed using the Yggdrasil platform.

Note that the demonstrator setup is split per branches in this repository:
 - Yggdrasil Deployment with the simulated Lab308 smart lab environment --> branch lab308
 - Home Assistant Adapter that serves artifacts with the same API as Yggdrasil pulled real time from HomeAssistant --> branch HomeAssistantAdapter
 - Eclipse LMOS agent deployment for the UserAssistant agent --> branch UserAssistantAgent
 - Eclipse LMOS agent deployment for the EnvMonitor agent --> branch EnvMonitor
 - Eclipse LMOS agent deployment for the EnvExplorer agent --> branch EnvExplorer
 - Eclipse LMOS agent deployment for the InteractionSolver agent --> branch InteractionSolver

Follow README indications in each branch to deploy the environment and the agents.
