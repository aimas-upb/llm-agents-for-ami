package controller

import org.eclipse.lmos.arc.agents.Agent
import org.eclipse.lmos.arc.spring.Agents
import org.eclipse.lmos.arc.agents.AgentProvider
import org.eclipse.lmos.arc.agents.ChatAgent
import org.eclipse.lmos.arc.agents.conversation.Conversation
import org.eclipse.lmos.arc.agents.conversation.UserMessage
import org.eclipse.lmos.arc.agents.AgentFailedException
import org.eclipse.lmos.arc.agents.getAgentByName
import org.eclipse.lmos.arc.agents.User
import com.fasterxml.jackson.databind.ObjectMapper
import org.eclipse.lmos.arc.core.Result
import org.eclipse.lmos.arc.core.Success
import org.eclipse.lmos.arc.core.Failure
import service.ActionIndexService
import service.PlanSignifierService
import org.eclipse.rdf4j.model.impl.SimpleValueFactory
import org.eclipse.rdf4j.rio.RDFFormat
import org.eclipse.rdf4j.rio.Rio
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.http.ResponseEntity
import org.springframework.http.server.reactive.ServerHttpRequest
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController
import reactor.core.publisher.Mono
import org.springframework.web.bind.annotation.*
import kotlinx.coroutines.reactor.mono
import java.io.StringReader
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse

data class NotificationPayload(val uri: String)

@RestController
class EchoController(
    private val actionIndexService: ActionIndexService,
    private val objectMapper: ObjectMapper,
    private val agentProvider: AgentProvider,
    private val planSignifierService: PlanSignifierService
) {

    private val log = LoggerFactory.getLogger(javaClass)
    private val httpClient: HttpClient = HttpClient.newHttpClient()
    private val valueFactory = SimpleValueFactory.getInstance()

    @PostMapping("/")
    fun handleNotification(
        @RequestBody body: Mono<String>,
        request: ServerHttpRequest
    ): Mono<ResponseEntity<String>> {

        val notificationType = request.headers.getFirst("X-Notification-Type")
        if (notificationType != null) {
             log.debug("Ignoring request because X-Notification-Type header is present (value: {})", notificationType)
             return Mono.just(ResponseEntity.ok("Request ignored: specific notification type header present."))
        }

        return body.flatMap { jsonBody ->
            var artifactUriString: String? = null
            try {
                val payload = objectMapper.readValue(jsonBody, NotificationPayload::class.java)
                artifactUriString = payload.uri
                val artifactUri = valueFactory.createIRI(artifactUriString)

                log.info("[Artifact: {}] Received notification (no specific type header). Assuming create/delete.", artifactUriString)

                val httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(artifactUriString))
                    .GET()
                    .header("Accept", "text/turtle")
                    .build()

                Mono.fromCallable { httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString()) }
                    .flatMap { response ->
                        when (response.statusCode()) {
                            200, 201, 204 -> {
                                val turtleBody = response.body()
                                log.info("[Artifact: {}] Fetched description (Status {}).", artifactUriString, response.statusCode())
                                if (!turtleBody.isNullOrBlank()) {
                                    try {
                                        val model = Rio.parse(StringReader(turtleBody), "", RDFFormat.TURTLE)
                                        actionIndexService.indexActionsFromModel(artifactUri, model)
                                        Mono.just(ResponseEntity.ok("Artifact actions indexed successfully for $artifactUriString"))
                                    } catch (e: Exception) {
                                        log.error("[Artifact: {}] Failed to parse/index description: {}", artifactUriString, e.message)
                                        Mono.just(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                                            .body("Failed to process fetched description: ${e.message}"))
                                    }
                                } else {
                                    log.warn("[Artifact: {}] Received empty description body. Clearing index.", artifactUriString)
                                    actionIndexService.removeActionsForArtifact(artifactUri)
                                    Mono.just(ResponseEntity.ok("Received empty description for $artifactUriString. Index cleared."))
                                }
                            }
                            404 -> {
                                log.info("[Artifact: {}] Received 404, assuming DELETE.", artifactUriString)
                                actionIndexService.removeActionsForArtifact(artifactUri)
                                Mono.just(ResponseEntity.ok("Artifact actions removed for $artifactUriString (assumed delete)."))
                            }
                            else -> {
                                log.error("[Artifact: {}] Failed fetch. Status: {}, Body: {}", artifactUriString, response.statusCode(), response.body())
                                Mono.just(ResponseEntity.status(HttpStatus.FAILED_DEPENDENCY)
                                    .body("Failed to fetch artifact description (Status ${response.statusCode()})"))
                            }
                        }
                    }
                    .onErrorResume { error ->
                         log.error("[Artifact: {}] Error fetching artifact description: {}", artifactUriString, error.message)
                         actionIndexService.removeActionsForArtifact(artifactUri)
                         Mono.just(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                             .body("Error communicating with artifact $artifactUriString (assuming delete): ${error.message}"))
                    }

            } catch (e: Exception) {
                val errorPrefix = artifactUriString?.let { "[Artifact: $it] " } ?: "[Invalid Payload] "
                log.error("{}Error processing notification: {}", errorPrefix, e.message, e)
                Mono.just(ResponseEntity.status(HttpStatus.BAD_REQUEST)
                        .body("Error processing notification payload: ${e.message}"))
            }
        }
    }

    data class AskTimeRequest(val query: String, val userId: String = "agent-a-caller")

    @PostMapping("/ask")
    fun askTime(@RequestBody request: AskTimeRequest): Mono<String> {
        return mono {
            val agentB = agentProvider.getAgentByName("env-explorer-agent") as? ChatAgent
                ?: return@mono "Error: Agent 'env-explorer-agent' not found."

            val conversationInput = Conversation(
                user = User(request.userId),
                transcript = listOf(UserMessage(request.query))
            )

            val result: Result<Conversation, AgentFailedException> = agentB.execute(conversationInput)

            when (result) {
                is Success -> {
                    val conversation = result.value
                    conversation.transcript.lastOrNull()?.content ?: "Agent env-explorer-agent provided no content."
                }
                is Failure -> {
                    val error = result.reason
                    "Agent 'env-explorer-agent' failed: ${error.message}"
                }
            }
        }
    }

    @PostMapping("/admin/reset")
    fun resetExplorerState(): Mono<ResponseEntity<String>> = mono {
        val removedArtifacts = actionIndexService.clearAllIndexedArtifacts()
        planSignifierService.clearAllSignifiers()
        ResponseEntity.ok("Explorer state reset: cleared $removedArtifacts indexed artifact(s) and all signifiers.")
    }
}
