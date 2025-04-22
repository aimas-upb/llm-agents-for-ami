// SPDX-FileCopyrightText: 2025 Deutsche Telekom AG and others
//
// SPDX-License-Identifier: Apache-2.0

package org.eclipse.lmos.arc.app.config

import org.eclipse.lmos.arc.agents.Agent
import org.eclipse.lmos.arc.spring.Agents // The helper bean
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
open class AgentConfiguration {

    @Bean
    open fun myTimeAgent(agents: Agents): Agent<*, *> { // Inject the helper
        // The 'agents' helper returns the created Agent
        return agents {
            name = "time-telling-agent"
            description = "An agent that can tell the current time using a tool."

            model { "test-gemini" }

            tools {
                +"get_current_time"
                // Add other tool names here if defined as beans
            }

            // Define the system prompt
            prompt {
                """
                You are a helpful assistant.
                Your primary function is to tell the current time when asked.
                Use the available tools to get the current time.
                If the user specifies a timezone, use it.
                """
            }

            // Optional: Add filters, settings, init block
            // filterInput { ... }
            // filterOutput { ... }
            // settings { ChatCompletionSettings(...) }
            // init { ... }
        }
    }
}
