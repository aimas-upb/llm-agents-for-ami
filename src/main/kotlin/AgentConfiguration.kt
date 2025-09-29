// SPDX-FileCopyrightText: 2025 Deutsche Telekom AG and others
//
// SPDX-License-Identifier: Apache-2.0

package org.eclipse.lmos.arc.app.config

import org.eclipse.lmos.arc.agents.Agent
import org.eclipse.lmos.arc.spring.Agents
import org.slf4j.LoggerFactory
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.env.Environment

@Configuration
open class EnvMonitorAgentConfiguration {

    private val log = LoggerFactory.getLogger(javaClass)

    @Bean
    open fun envMonitorAgent(agents: Agents, environment: Environment): Agent<*, *> {
        val ollamaUrl = environment.getProperty("OLLAMA_URL").orEmpty()
        val openRouterKey = environment.getProperty("OPENROUTER_API_KEY").orEmpty()
        val openAiKey = environment.getProperty("OPENAI_API_KEY").orEmpty()

        val modelId = when {
            ollamaUrl.isNotBlank() -> "test-ollama"
            openRouterKey.isNotBlank() -> "test-openrouter"
            openAiKey.isNotBlank() -> "test-openai"
            else -> error("No LLM provider configured. Set OLLAMA_URL, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
        }

        val providerDetails = when (modelId) {
            "test-ollama" -> {
                val modelName = environment.getProperty("OLLAMA_MODEL_NAME") ?: "llama3.1"
                val url = ollamaUrl.ifBlank { environment.getProperty("OLLAMA_URL") ?: "http://localhost:11434" }
                "Ollama model='$modelName' url='$url'"
            }
            "test-openrouter" -> {
                val modelName = environment.getProperty("OPENROUTER_MODEL_NAME") ?: "openrouter/openai/gpt-4.1-mini"
                val baseUrl = environment.getProperty("OPENROUTER_BASE_URL") ?: "https://openrouter.ai/api/v1"
                "OpenRouter model='$modelName' baseUrl='$baseUrl' apiKeySupplied=true"
            }
            else -> {
                val modelName = environment.getProperty("OPENAI_MODEL_NAME") ?: "gpt-4.1-mini"
                val baseUrl = environment.getProperty("OPENAI_BASE_URL") ?: "https://api.openai.com/v1"
                "OpenAI model='$modelName' baseUrl='$baseUrl' apiKeySupplied=true"
            }
        }
        log.info("Configuring env-monitor-agent to use model client '{}' ({})", modelId, providerDetails)

        return agents {
            name = "env-monitor-agent"
            description = "Provides information about the current state of environment artifacts and can answer questions about them."

            model { modelId }

            tools {
                +"get_artifact_state_rdf"
                +"get_all_artifact_uris"
                +"get_all_artifacts_states"
            }

            prompt {
                """
                You are an environment monitoring assistant for a lab.
                Your primary function is to provide the current state of lab artifacts when asked.
                You have tools available to get the state of specific artifacts ("get_artifact_state_rdf"),
                to list all known artifact URIs ("get_all_artifact_uris"), and to get the complete
                state of all artifacts ("get_all_artifacts_states").

                Instructions:
                - When asked for the state of a specific artifact, use the "get_artifact_state_rdf" tool with the artifact's full URI. Present the RDF state information clearly to the user, perhaps summarizing key properties if appropriate, but always include the metadata comments (timestamp, trigger).
                - When asked which artifacts are known, use the "get_all_artifact_uris" tool.
                - When asked for the state of all artifacts or a comprehensive overview, use the "get_all_artifacts_states" tool.
                - If you don't have information about a specific artifact URI, say so.
                - Answer concisely based on the information from the tools. Do not make up information.
                """
            }
        }
    }
}
