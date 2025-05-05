package org.eclipse.lmos.arc.app.config

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.eclipse.lmos.arc.agents.dsl.types
import org.eclipse.lmos.arc.agents.dsl.string
import org.eclipse.lmos.arc.agents.functions.LLMFunction
import service.ActionIndexService
import org.eclipse.lmos.arc.spring.Functions
import org.slf4j.LoggerFactory
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.io.StringWriter

@Configuration
open class ToolConfiguration {

    private val log = LoggerFactory.getLogger(javaClass)
    private val jsonMapper = jacksonObjectMapper()

    @Bean
    open fun actionExplorerTools(
        functions: Functions,
        actionIndexService: ActionIndexService
    ): List<LLMFunction> {
        return functions(
            name = "find_actions_for_artifact",
            description = "Finds all indexed actions available for a specific artifact URI. Returns results as a JSON string list.",
            params = types(
                string("artifactUri", "The full URI of the artifact.")
            )
        ) { params ->
            val artifactUriString = params[0] as String
            log.info("Tool 'find_actions_for_artifact' called for: {}", artifactUriString)
            val actions = actionIndexService.getActionsForArtifact(artifactUriString)

            if (actions.isNotEmpty()) {
                try {
                    jsonMapper.writeValueAsString(actions)
                } catch (e: Exception) {
                    log.error("Failed to serialize actions for {}: {}", artifactUriString, e.message)
                    "Error: Could not serialize actions."
                }
            } else {
                "No actions found indexed for artifact URI '$artifactUriString'."
            }
        } +
         functions(
             name = "list_known_artifacts",
             description = "Lists the URIs of all artifacts for which actions are currently indexed.",
             params = types()
         ) {
             log.info("Tool 'list_known_artifacts' called.")
             val allActions = actionIndexService.getAllActions()
             val knownArtifactUris = allActions.map { it.artifactUri }.distinct()

             if (knownArtifactUris.isNotEmpty()) {
                 "Known artifacts with indexed actions: ${knownArtifactUris.joinToString(", ")}"
             } else {
                 "No artifacts with indexed actions are currently known."
             }
         }
    }
}