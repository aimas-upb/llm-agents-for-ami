// SPDX-FileCopyrightText: 2025 Deutsche Telekom AG and others
//
// SPDX-License-Identifier: Apache-2.0

package org.eclipse.lmos.arc.app.config

import org.eclipse.lmos.arc.agents.Agent
import org.eclipse.lmos.arc.spring.Agents
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
open class EnvMonitorAgentConfiguration {

    @Bean
    open fun envMonitorAgent(agents: Agents): Agent<*, *> {
        return agents {
            name = "env-monitor-agent"
            description = "Provides information about the current state of environment artifacts and can answer questions about them."

            model { "test-gemini" }

            tools {
                +"get_artifact_state_rdf"
                +"get_all_artifact_uris"
            }

            prompt {
                """
                You are an environment monitoring assistant for a lab.
                Your primary function is to provide the current state of lab artifacts when asked.
                You have tools available to get the state of specific artifacts ("get_artifact_state_rdf")
                and to list all known artifact URIs ("get_all_artifact_uris").

                Instructions:
                - When asked for the state of a specific artifact, use the "get_artifact_state_rdf" tool with the artifact's full URI. Present the RDF state information clearly to the user, perhaps summarizing key properties if appropriate, but always include the metadata comments (timestamp, trigger).
                - When asked which artifacts are known, use the "get_all_artifact_uris" tool.
                - If you don't have information about a specific artifact URI, say so.
                - Answer concisely based on the information from the tools. Do not make up information.
                """
            }
        }
    }
}
