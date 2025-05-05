package org.eclipse.lmos.arc.app.config

import org.eclipse.lmos.arc.agents.dsl.get
import org.eclipse.lmos.arc.agents.dsl.types
import org.eclipse.lmos.arc.agents.dsl.string
import org.eclipse.lmos.arc.agents.functions.LLMFunction
import org.eclipse.lmos.arc.spring.Functions
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.time.LocalTime
import org.eclipse.lmos.arc.app.service.EnvironmentStateService
import org.eclipse.rdf4j.rio.RDFFormat
import org.eclipse.rdf4j.rio.Rio
import java.io.StringWriter

@Configuration
open class ToolConfiguration {

    @Bean
    open fun environmentStateTools(
        functions: Functions,
        environmentStateService: EnvironmentStateService
    ): List<LLMFunction> {
        return functions(
            name = "get_artifact_state_rdf",
            description = "Gets the last known state of a specific artifact as an RDF graph (Turtle format), including metadata comments.",
            params = types(
                string("artifactUri", "The full URI of the artifact.")
            )
        ) { params ->
            val artifactUriString = params[0] as String
            val state = environmentStateService.getArtifactState(artifactUriString)

            if (state != null) {
                 val writer = StringWriter()
                 if (state.rdfModel.isNotEmpty()) {
                     Rio.write(state.rdfModel, writer, RDFFormat.TURTLE)
                 } else {
                     writer.write("# RDF Model is empty.")
                 }
                 val rdfString = writer.toString()

                 """
                 # Artifact State for: ${state.artifactUri.stringValue()}
                 # Last Updated At: ${state.lastUpdatedAt}
                 # Last Triggering Affordance: ${state.lastTriggeringAffordanceUri?.stringValue() ?: "N/A"}
                 # --- RDF Data (Turtle) ---
                 $rdfString
                 """.trimIndent()
            } else {
                "Error: Artifact with URI '$artifactUriString' not found."
            }
        } +
         functions(
             name = "get_all_artifact_uris",
             description = "Gets the list of URIs for all currently known artifacts.",
             params = types()
         ) {
             val allStates = environmentStateService.getAllArtifactStates()
             val uris = allStates.map { it.artifactUri.stringValue() }
             if (uris.isNotEmpty()) {
                 "Known artifact URIs: ${uris.joinToString(", ")}"
             } else {
                 "No artifacts are currently known."
             }
         }
    }
}