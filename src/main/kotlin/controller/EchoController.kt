package controller

import org.eclipse.lmos.arc.agents.AgentProvider
import org.eclipse.lmos.arc.agents.ChatAgent
import org.eclipse.lmos.arc.agents.conversation.Conversation
import org.eclipse.lmos.arc.agents.conversation.UserMessage
import org.eclipse.lmos.arc.agents.AgentFailedException
import org.eclipse.lmos.arc.agents.getAgentByName
import org.eclipse.lmos.arc.agents.User
import org.eclipse.lmos.arc.core.Result
import org.eclipse.lmos.arc.core.Success
import org.eclipse.lmos.arc.core.Failure
import org.springframework.web.bind.annotation.*
import kotlinx.coroutines.reactor.mono
import reactor.core.publisher.Mono

@RestController
class EchoController(
    private val agentProvider: AgentProvider
) {

    data class AskTimeRequest(val query: String, val userId: String = "agent-a-caller")

    @PostMapping("/ask")
    fun askTime(@RequestBody request: AskTimeRequest): Mono<String> {
        return mono {
            val agent = agentProvider.getAgentByName("interaction-solver-agent") as? ChatAgent
                ?: return@mono "Error: Agent 'interaction-solver-agent' not found."

            val conversationInput = Conversation(
                user = User(request.userId),
                transcript = listOf(UserMessage(request.query))
            )

            val result: Result<Conversation, AgentFailedException> = agent.execute(conversationInput)

            when (result) {
                is Success -> {
                    val conversation = result.value
                    conversation.transcript.lastOrNull()?.content ?: "Agent interaction-solver-agent provided no content."
                }
                is Failure -> {
                    val error = result.reason
                    "Agent 'interaction-solver-agent' failed: ${error.message}"
                }
            }
        }
    }
} 