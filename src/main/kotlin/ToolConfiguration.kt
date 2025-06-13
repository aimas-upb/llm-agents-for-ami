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
import org.eclipse.lmos.arc.app.service.PlanCacheService
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.decodeFromString

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
                requestTimeout = 180000
            }
        }
    }

    @Bean
    open fun agentCommunicationTools(
        functions: Functions,
        httpClient: HttpClient,
        planCacheService: PlanCacheService,
        @Value("\${envmonitoring.agent.url:http://localhost:8081/}")
        monitoringAgentUrl: String,
        @Value("\${envexplorer.agent.url:http://localhost:8082/}")
        explorerAgentUrl: String,
        @Value("\${interactionsolver.agent.url:http://localhost:8083/}")
        interactionSolverAgentUrl: String
    ): List<LLMFunction> {
        val jsonParser = Json { ignoreUnknownKeys = true; isLenient = true }

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
        } +
        functions(
            name = "request_interaction_plan",
            description = "Sends user intents to the Interaction Solver agent to generate an executable plan for fulfilling the intents.",
            params = types(
                string("intent_list", "The list of user intents to be fulfilled (e.g., 'User Intent(s): - Increase luminosity - Set temperature to 22C').")
            )
        ) { params ->
            val intentList = params[0] as? String ?: "No intents provided"
            val requestBody = AgentAskRequest(query = intentList)

            try {
                val response = httpClient.post("$interactionSolverAgentUrl/ask") {
                    contentType(ContentType.Application.Json)
                    setBody(requestBody)
                }
                val responseText: String = response.body()
                if (response.status != HttpStatusCode.OK) {
                     throw LLMFunctionException("InteractionSolverAgent responded with status ${response.status}: $responseText")
                }
                responseText
            } catch (e: Exception) {
                println("Error calling InteractionSolverAgent: ${e.message}")
                throw LLMFunctionException("Failed to invoke InteractionSolverAgent: ${e.message}", e)
            }
        } +
        functions(
            name = "store_latest_plan",
            description = "Stores the provided JSON-Plan 1.2 temporarily. Overwrites any previously stored plan for this session.",
            params = types(
                string("plan_json", "The full JSON-Plan 1.2 string to be stored.")
            )
        ) { params ->
            val planJson = params[0] as? String
            if (planJson == null) {
                throw LLMFunctionException("plan_json parameter is missing or not a string for store_latest_plan.")
            }
            planCacheService.storePlan(planJson)
            "Plan stored successfully."
        } +
        functions(
            name = "retrieve_and_clear_latest_plan",
            description = "Retrieves the last temporarily stored JSON-Plan 1.2 for this session and then clears it from storage. Returns the plan JSON or an indication if no plan was found.",
            params = types()
        ) { _ ->
            val retrievedPlan = planCacheService.retrieveAndClearPlan()
            retrievedPlan ?: "No plan found in temporary storage or it was already cleared."
        } +
        functions(
            name = "execute_plan",
            description = "Parses the provided JSON-Plan 1.2 and executes each step by making HTTP requests.",
            params = types(
                string("plan_json", "The full JSON-Plan 1.2 string to be executed.")
            )
        ) { params ->
            val planJsonToExecute = params[0] as? String
                ?: throw LLMFunctionException("plan_json parameter is missing or not a string for execute_plan.")

            val plan: PlanToExecute
            try {
                plan = jsonParser.decodeFromString<PlanToExecute>(planJsonToExecute)
            } catch (e: Exception) {
                println("[UserAssistantAgent - execute_plan tool] Error parsing plan JSON: ${e.message}")
                throw LLMFunctionException("Failed to parse plan_json: ${e.message}", e)
            }

            val executionResults = mutableListOf<String>()
            var allStepsSuccessful = true

            kotlinx.coroutines.runBlocking {
                plan.steps.forEach { step ->
                    val stepResultPrefix = "Step ${step.step_id} ('${step.action_name ?: "unnamed action"}' on '${step.artifact_uri}')"
                    try {
                        println("[UserAssistantAgent - execute_plan tool] Executing $stepResultPrefix: ${step.method} ${step.target}")
                        
                        val response = httpClient.request(step.target) {
                            method = HttpMethod.parse(step.method.uppercase())
                            header("X-Agent-WebID", "http://localhost:8080/agents/alex")
                            contentType(ContentType.parse(step.content_type))
                            if (method == HttpMethod.Post || method == HttpMethod.Put || method == HttpMethod.Patch) {
                                setBody(step.payload)
                            }
                        }
                        val responseBody: String = response.bodyAsText()

                        if (response.status.isSuccess()) {
                            executionResults.add("$stepResultPrefix executed successfully (Status ${response.status}). Response: ${responseBody.take(100)}${if(responseBody.length > 100) "..." else ""}")
                        } else {
                            executionResults.add("$stepResultPrefix failed (Status ${response.status}). Response: ${responseBody.take(100)}${if(responseBody.length > 100) "..." else ""}")
                            allStepsSuccessful = false
                        }
                    } catch (e: Exception) {
                        println("[UserAssistantAgent - execute_plan tool] Exception during $stepResultPrefix: ${e.message}")
                        executionResults.add("$stepResultPrefix failed: ${e.message}")
                        allStepsSuccessful = false
                    }
                }
            }

            val overallStatus = if (allStepsSuccessful) "All steps processed (check individual results)." else "One or more steps failed."
            "Execution summary: $overallStatus\nDetails:\n${executionResults.joinToString("\n")}"
        }
    }
}

@kotlinx.serialization.Serializable
data class PlanExecutionStep(
    val step_id: Int,
    val intent: String? = null, 
    val artifact_uri: String,
    val affordance_uri: String? = null, 
    val action_name: String? = null, 
    val method: String,
    val target: String,
    val content_type: String,
    val payload: JsonElement 
)

@kotlinx.serialization.Serializable
data class PlanToExecute(
    val plan_version: String,
    val steps: List<PlanExecutionStep>
)