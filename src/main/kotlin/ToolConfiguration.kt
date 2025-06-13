package org.eclipse.lmos.arc.app.config

import org.eclipse.lmos.arc.agents.dsl.types
import org.eclipse.lmos.arc.agents.dsl.string
import org.eclipse.lmos.arc.agents.functions.LLMFunction
import service.ActionIndexService
import service.PlanSignifierService
import org.eclipse.lmos.arc.spring.Functions
import org.slf4j.LoggerFactory
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
open class ToolConfiguration {

    private val log = LoggerFactory.getLogger(javaClass)

    @Bean
    open fun actionExplorerTools(
        functions: Functions,
        actionIndexService: ActionIndexService,
        planSignifierService: PlanSignifierService
    ): List<LLMFunction> {
        return functions(
            name = "find_actions_for_artifact",
            description = "Finds all indexed action affordances available for a specific artifact URI. Returns the action affordances as an RDF Turtle string.",
            params = types(
                string("artifactUri", "The full URI of the artifact.")
            )
        ) { params ->
            val artifactUriString = params[0] as String
            log.info("Tool 'find_actions_for_artifact' called for: {}", artifactUriString)
            
            val actionRdf = actionIndexService.getActionsForArtifactAsRdf(artifactUriString)
            
            if (actionRdf != null) {
                log.debug("Found action affordances for artifact {}, returning RDF with {} characters", 
                    artifactUriString, actionRdf.length)
                actionRdf
            } else {
                log.debug("No action affordances found for artifact {}", artifactUriString)
                "# No action affordances found for artifact URI '$artifactUriString'."
            }
        } +
         functions(
             name = "list_known_artifacts",
             description = "Lists the URIs of all artifacts for which action affordances are currently indexed.",
             params = types()
         ) {
             log.info("Tool 'list_known_artifacts' called.")
             val knownArtifactUris = actionIndexService.getAllKnownArtifactUris()

             if (knownArtifactUris.isNotEmpty()) {
                 "Known artifacts with indexed action affordances:\n${knownArtifactUris.joinToString("\n")}"
             } else {
                 "No artifacts with indexed action affordances are currently known."
             }
         } +
         functions(
             name = "get_all_artifacts_capabilities",
             description = "Gets the action affordances (capabilities) of all indexed artifacts as RDF Turtle format.",
             params = types()
         ) {
             log.info("Tool 'get_all_artifacts_capabilities' called.")
             val knownArtifactUris = actionIndexService.getAllKnownArtifactUris()
             
             if (knownArtifactUris.isNotEmpty()) {
                 val allCapabilitiesReport = StringBuilder()
                 
                 knownArtifactUris.forEachIndexed { index, artifactUri ->
                     allCapabilitiesReport.appendLine("# ============= Artifact ${index + 1} =============")
                     allCapabilitiesReport.appendLine("# Artifact URI: $artifactUri")
                     allCapabilitiesReport.appendLine("# --- Action Affordances (RDF Turtle) ---")
                     
                     val actionRdf = actionIndexService.getActionsForArtifactAsRdf(artifactUri)
                     if (actionRdf != null) {
                         allCapabilitiesReport.appendLine(actionRdf)
                     } else {
                         allCapabilitiesReport.appendLine("# No action affordances found for this artifact.")
                     }
                     allCapabilitiesReport.appendLine()
                 }
                 
                 allCapabilitiesReport.toString()
             } else {
                 "# No artifacts with indexed action affordances are currently known."
             }
         } +
         functions(
             name = "add_signifiers_from_plan",
             description = "Parses a JSON-Plan 1.2 and creates cashmere signifiers for each step. Returns the IRIs of created signifiers.",
             params = types(
                 string("plan_json", "The JSON plan exactly as output by the Interaction-Solver agent")
             )
         ) { params ->
             val planJson = params[0] as String
             log.info("Tool 'add_signifiers_from_plan' called with plan of {} characters", planJson.length)

             try {
                 val signifierIris = planSignifierService.addSignifiersFromPlan(planJson)
                 if (signifierIris.isEmpty()) {
                     "No signifiers were created (plan may have had no steps)"
                 } else {
                     "Created ${signifierIris.size} signifier(s):\n${signifierIris.joinToString("\n")}"
                 }
             } catch (e: Exception) {
                 log.error("Failed to create signifiers from plan", e)
                 "Error creating signifiers: ${e.message}"
             }
         } +
         functions(
             name = "get_all_signifiers",
             description = "Retrieves the complete RDF Turtle model of all cashmere signifiers stored in memory.",
             params = types()
         ) {
             log.info("Tool 'get_all_signifiers' called")
             planSignifierService.getAllSignifiersAsTurtle()
         }
    }
}