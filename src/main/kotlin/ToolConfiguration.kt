package org.eclipse.lmos.arc.app.config

import org.eclipse.lmos.arc.agents.dsl.get
import org.eclipse.lmos.arc.agents.dsl.types
import org.eclipse.lmos.arc.agents.dsl.string
import org.eclipse.lmos.arc.agents.functions.LLMFunction
import org.eclipse.lmos.arc.agents.functions.LLMFunctionException
import org.eclipse.lmos.arc.spring.Functions
import org.springframework.beans.factory.annotation.Value
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

// Added for Micrometer Tracing
import io.micrometer.tracing.Tracer
import io.micrometer.tracing.Span // Explicit import for Span

@Serializable
data class AgentAskRequest(val query: String, val userId: String = "agent-a")

@Configuration
open class AgentCommunicationToolsConfig {

    @Bean
    open fun ktorHttpClient(): HttpClient {
        return HttpClient(CIO) {
            install(ContentNegotiation) {
                json(Json {
                    prettyPrint = true
                    isLenient = true
                    ignoreUnknownKeys = true
                })
            }
            engine {
                requestTimeout = 180000 // 3 minutes timeout
            }
        }
    }

    @Bean
    open fun agentCommunicationTools(
        functions: Functions,
        httpClient: HttpClient,
        tracer: Tracer, // Tracer injected here
        @Value("\${envmonitoring.agent.url:http://localhost:8081/}")
        monitoringAgentUrl: String,
        @Value("\${envexplorer.agent.url:http://localhost:8082/}")
        explorerAgentUrl: String
    ): List<LLMFunction> {
        return functions(
            name = "query_environment_state",
            description = "Asks the Environment Monitoring agent about the current state of a specific artifact, the overall environment, or all artifacts. Can get comprehensive state reports for all known artifacts.",
            params = types(
                string("monitoring_query", "The question about the environment state (e.g., 'What is the state of Lamp308?', 'Is anyone present in Lab308?', 'Show me all artifact states').")
            )
        ) { params ->
            val queryForMonitor = params[0] as? String ?: "What is the general state?"
            val requestBody = AgentAskRequest(query = queryForMonitor)

            // Create a new span for this tool call
            val toolSpan = tracer.nextSpan().name("tool.query_environment_state").start()

            try {
                tracer.withSpan(toolSpan).use {
                    toolSpan.tag("tool.input.monitoring_query", queryForMonitor)
                    toolSpan.tag("tool.target.agent_url", monitoringAgentUrl)
                    toolSpan.event("Attempting to call EnvMonitoringAgent")

                    val response = httpClient.post("$monitoringAgentUrl/ask") {
                        contentType(ContentType.Application.Json)
                        setBody(requestBody)
                    }
                    val responseText: String = response.body()
                    toolSpan.tag("tool.output.http_status", response.status.value.toString())
                    toolSpan.tag("tool.output.response_length", responseText.length.toString())
                    toolSpan.event("Received response from EnvMonitoringAgent")

                    if (response.status != HttpStatusCode.OK) {
                        toolSpan.tag("error", "true")
                        toolSpan.tag("error.message", "EnvMonitoringAgent responded with status ${response.status}: ${responseText.take(200)}")
                        throw LLMFunctionException("EnvMonitoringAgent responded with status ${response.status}: $responseText")
                    }
                    responseText
                }
            } catch (e: Exception) {
                toolSpan.error(e) // Record the exception on the span
                println("Error calling EnvMonitoringAgent: ${e.message}")
                throw LLMFunctionException("Failed to invoke EnvMonitoringAgent: ${e.message}", e)
            } finally {
                toolSpan.end() // Always end the span
            }
        } +
        functions(
            name = "query_environment_capabilities",
            description = "Asks the Environment Explorer agent about available actions, artifact capabilities, or interaction history (signifiers) relevant to a task. Can get comprehensive capability reports for all indexed artifacts.",
            params = types(
                string("explorer_query", "The question about capabilities or history (e.g., 'What actions can control Lamp308?', 'How was the light used recently?', 'Show me all artifacts capabilities').")
            )
        ) { params ->
            val queryForExplorer = params[0] as? String ?: "What capabilities are available?"
            val requestBody = AgentAskRequest(query = queryForExplorer)
            
            val toolSpan = tracer.nextSpan().name("tool.query_environment_capabilities").start()

            try {
                tracer.withSpan(toolSpan).use {
                    toolSpan.tag("tool.input.explorer_query", queryForExplorer)
                    toolSpan.tag("tool.target.agent_url", explorerAgentUrl)
                    toolSpan.event("Attempting to call EnvExplorerAgent")

                    val response = httpClient.post("$explorerAgentUrl/ask") {
                        contentType(ContentType.Application.Json)
                        setBody(requestBody)
                    }
                    val responseText: String = response.body()
                    toolSpan.tag("tool.output.http_status", response.status.value.toString())
                    toolSpan.tag("tool.output.response_length", responseText.length.toString())
                    toolSpan.event("Received response from EnvExplorerAgent")

                    if (response.status != HttpStatusCode.OK) {
                        toolSpan.tag("error", "true")
                        toolSpan.tag("error.message", "EnvExplorerAgent responded with status ${response.status}: ${responseText.take(200)}")
                        throw LLMFunctionException("EnvExplorerAgent responded with status ${response.status}: $responseText")
                    }
                    responseText
                }
            } catch (e: Exception) {
                toolSpan.error(e)
                println("Error calling EnvExplorerAgent: ${e.message}")
                throw LLMFunctionException("Failed to invoke EnvExplorerAgent: ${e.message}", e)
            } finally {
                toolSpan.end()
            }
        }
    }
}