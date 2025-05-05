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
        }
    }

    @Bean
    open fun agentCommunicationTools(
        functions: Functions,
        httpClient: HttpClient,
        @Value("\${envmonitoring.agent.url:http://localhost:8081/}")
        monitoringAgentUrl: String,
        @Value("\${envexplorer.agent.url:http://localhost:8082/}")
        explorerAgentUrl: String
    ): List<LLMFunction> {
        return functions(
            name = "query_environment_state",
            description = "Asks the Environment Monitoring agent about the current state of a specific artifact or the overall environment.",
            params = types(
                string("monitoring_query", "The question about the environment state (e.g., 'What is the state of Lamp308?', 'Is anyone present in Lab308?').")
            )
        ) { params ->
            val queryForMonitor = params[0] as? String ?: "What is the general state?"
            val requestBody = AgentAskRequest(query = queryForMonitor)

            try {
                val response = httpClient.post("$monitoringAgentUrl/ask") {
                    contentType(ContentType.Application.Json)
                    setBody(requestBody)
                }
                val responseText: String = response.body()
                if (response.status != HttpStatusCode.OK) {
                     throw LLMFunctionException("EnvMonitoringAgent responded with status ${response.status}: $responseText")
                }
                responseText
            } catch (e: Exception) {
                println("Error calling EnvMonitoringAgent: ${e.message}")
                throw LLMFunctionException("Failed to invoke EnvMonitoringAgent: ${e.message}", e)
            }
        } +
        functions(
            name = "query_environment_capabilities",
            description = "Asks the Environment Explorer agent about available actions, artifact capabilities, or interaction history (signifiers) relevant to a task.",
            params = types(
                string("explorer_query", "The question about capabilities or history (e.g., 'What actions can control Lamp308?', 'How was the light used recently?').")
            )
        ) { params ->
            val queryForExplorer = params[0] as? String ?: "What capabilities are available?"
            val requestBody = AgentAskRequest(query = queryForExplorer)

            try {
                val response = httpClient.post("$explorerAgentUrl/ask") {
                    contentType(ContentType.Application.Json)
                    setBody(requestBody)
                }
                val responseText: String = response.body()
                 if (response.status != HttpStatusCode.OK) {
                     throw LLMFunctionException("EnvExplorerAgent responded with status ${response.status}: $responseText")
                }
                responseText
            } catch (e: Exception) {
                println("Error calling EnvExplorerAgent: ${e.message}")
                throw LLMFunctionException("Failed to invoke EnvExplorerAgent: ${e.message}", e)
            }
        }
    }
}