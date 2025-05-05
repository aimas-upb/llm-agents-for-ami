// SPDX-FileCopyrightText: 2025 Deutsche Telekom AG and others
//
// SPDX-License-Identifier: Apache-2.0

package org.eclipse.lmos.arc.app.config

import org.eclipse.lmos.arc.agents.Agent
import org.eclipse.lmos.arc.spring.Agents
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
open class AgentConfiguration {

    @Bean
    open fun envExplorerAgent(agents: Agents): Agent<*, *> {
        return agents {
            name = "env-explorer-agent"
            description = "Provides information about the actions available on environment artifacts."

            model { "test-gemini" }

            tools {
                +"find_actions_for_artifact"
                +"list_known_artifacts"
            }

            prompt {
                """
                You are an environment exploration assistant for a lab.
                Your function is to list the actions available on a specific artifact or list all known artifacts when asked.
                You have tools available: "find_actions_for_artifact" and "list_known_artifacts".

                Instructions:
                - When asked for the actions of a specific artifact, you MUST use the "find_actions_for_artifact" tool. Provide the artifact's full URI as the 'artifactUri' parameter. Present the JSON result clearly or summarize it.
                - When asked to list all known artifacts, you MUST use the "list_known_artifacts" tool. Present the list of URIs returned by the tool.
                - If a tool indicates no actions or artifacts were found, state that clearly.
                - Do not make up actions or artifact information. Only rely on the tool's output.
                """
            }
        }
    }
}
