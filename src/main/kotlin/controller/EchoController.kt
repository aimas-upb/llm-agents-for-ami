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
import org.eclipse.rdf4j.model.*
import org.eclipse.rdf4j.model.vocabulary.XSD
import org.eclipse.lmos.arc.core.Result
import org.eclipse.lmos.arc.core.Success
import org.eclipse.lmos.arc.core.Failure
import org.springframework.http.HttpStatus
import org.springframework.http.ResponseEntity
import org.springframework.http.server.reactive.ServerHttpRequest
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import reactor.core.publisher.Mono
import org.springframework.web.bind.annotation.*
import kotlinx.coroutines.reactor.mono
import org.eclipse.lmos.arc.app.service.EnvironmentStateService
import controller.UpdatePayload
import java.time.Instant
import java.time.format.DateTimeParseException

@RestController
class EchoController(
    private val objectMapper: ObjectMapper,
    private val environmentStateService: EnvironmentStateService,
    private val agentProvider: AgentProvider
) {
    private companion object {
        const val EXPECTED_NOTIFICATION_TYPE = "ArtifactObsPropertyUpdated"
    }

    @PostMapping("/")
    fun handleUpdate(
        @RequestBody body: Mono<String>,
        request: ServerHttpRequest
    ): Mono<ResponseEntity<String>> {
        return body.flatMap { rawBody ->
            val notificationType = request.headers.getFirst("X-Notification-Type")
            if (notificationType != EXPECTED_NOTIFICATION_TYPE) {
                println("Ignoring request: Missing or incorrect X-Notification-Type header. Expected '$EXPECTED_NOTIFICATION_TYPE', got '$notificationType'. Body: $rawBody")
                Mono.just(ResponseEntity.ok("Request ignored: Invalid or missing notification type header."))
            } else {
                var logContextUri: String? = null
                try {
                    println("[Artifact: $logContextUri] Received property update payload:\n$rawBody")
                    val payload = objectMapper.readValue(rawBody, UpdatePayload::class.java)
                    logContextUri = payload.artifactUri
                    val artifactUri = environmentStateService.createIRI(payload.artifactUri)
                    val propertyUri = environmentStateService.createIRI(payload.propertyUri)
                    val triggerUri = environmentStateService.createIRI(payload.triggerUri)
                    val timestamp = Instant.parse(payload.timestamp)
                    val rdfValue = createRdfValue(payload.value, payload.valueTypeUri)

                    environmentStateService.updateArtifactProperty(
                        artifactUri = artifactUri,
                        propertyUri = propertyUri,
                        newValue = rdfValue,
                        timestamp = timestamp,
                        triggeringUri = triggerUri
                    )

                    println("[Artifact: $logContextUri] Successfully processed property update for '${propertyUri.localName}'")
                    Mono.just(ResponseEntity.ok("Property update processed successfully for $logContextUri"))

                } catch (e: DateTimeParseException) {
                     val errorPrefix = logContextUri?.let { "[Artifact: $it] " } ?: "[RawPayload] "
                     println("${errorPrefix}Invalid timestamp format in update: ${e.message}")
                     Mono.just(ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Invalid timestamp format: ${e.message}"))
                } catch (e: IllegalArgumentException) {
                     val errorPrefix = logContextUri?.let { "[Artifact: $it] " } ?: "[RawPayload] "
                     println("${errorPrefix}Invalid data in update: ${e.message}")
                     Mono.just(ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Invalid data format: ${e.message}"))
                } catch (e: Exception) {
                    val errorPrefix = logContextUri?.let { "[Artifact: $it] " } ?: "[RawPayload] "
                    println("${errorPrefix}Error processing property update: ${e.message}")
                    e.printStackTrace()
                    Mono.just(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                            .body("Internal error processing property update: ${e.message}"))
                }
            }
        }
    }

    private fun createRdfValue(value: Any, typeUriStr: String): Value {
        val typeIRI = environmentStateService.createIRI(typeUriStr)
        return try {
            when (typeIRI) {
                XSD.BOOLEAN -> environmentStateService.createLiteral(value.toString().toBooleanStrict())
                XSD.INTEGER -> environmentStateService.createLiteral(value.toString().toInt())
                XSD.LONG -> environmentStateService.createLiteral(value.toString().toLong())
                XSD.DOUBLE -> environmentStateService.createLiteral(value.toString().toDouble())
                XSD.FLOAT -> environmentStateService.createLiteral(value.toString().toFloat())
                XSD.STRING -> environmentStateService.createLiteral(value.toString())
                XSD.DATETIME -> environmentStateService.createLiteral(value.toString(), typeIRI)
                else -> {
                    if (value is String && (value.startsWith("http:") || value.startsWith("https:"))) {
                        try {
                            environmentStateService.createIRI(value)
                        } catch (e: Exception) {
                            println("Value '$value' looks like URI but failed IRI creation, treating as custom literal with type $typeIRI")
                            environmentStateService.createLiteral(value, typeIRI)
                        }
                    } else {
                        environmentStateService.createLiteral(value.toString(), typeIRI)
                    }
                }
            }
        } catch (e: NumberFormatException) {
            println("Type mismatch: Cannot convert value '$value' to specified numeric type $typeIRI: ${e.message}")
            throw IllegalArgumentException("Cannot convert value '$value' to specified type $typeIRI", e)
        } catch (e: IllegalArgumentException) {
             println("Type mismatch: Cannot convert value '$value' to specified type $typeIRI: ${e.message}")
             throw IllegalArgumentException("Cannot convert value '$value' to specified type $typeIRI", e)
        }
    }

    data class AskTimeRequest(val query: String, val userId: String = "agent-a-caller")

    @PostMapping("/ask")
    fun askTime(@RequestBody request: AskTimeRequest): Mono<String> {
        return mono {
            val agentB = agentProvider.getAgentByName("env-monitor-agent") as? ChatAgent
                ?: return@mono "Error: Agent 'env-monitor-agent' not found."

            val conversationInput = Conversation(
                user = User(request.userId),
                transcript = listOf(UserMessage(request.query))
            )

            val result: Result<Conversation, AgentFailedException> = agentB.execute(conversationInput)

            when (result) {
                is Success -> {
                    val conversation = result.value
                    conversation.transcript.lastOrNull()?.content ?: "Agent env-monitor-agent provided no content."
                }
                is Failure -> {
                    val error = result.reason
                    "Agent 'env-monitor-agent' failed: ${error.message}"
                }
            }
        }
    }
}