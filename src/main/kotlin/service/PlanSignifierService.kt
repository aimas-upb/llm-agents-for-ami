package service

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import org.eclipse.rdf4j.model.Model
import org.eclipse.rdf4j.model.impl.DynamicModelFactory
import org.eclipse.rdf4j.model.impl.SimpleValueFactory
import org.eclipse.rdf4j.model.vocabulary.RDF
import org.eclipse.rdf4j.model.vocabulary.XSD
import org.eclipse.rdf4j.rio.RDFFormat
import org.eclipse.rdf4j.rio.RDFWriter
import org.eclipse.rdf4j.rio.Rio
import org.eclipse.rdf4j.rio.helpers.BasicWriterSettings
import org.eclipse.rdf4j.rio.turtle.TurtleWriter
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.io.StringWriter

@Service
class PlanSignifierService {

    private val log = LoggerFactory.getLogger(javaClass)
    private val mapper = jacksonObjectMapper()
    private val vf = SimpleValueFactory.getInstance()

    private val TURTLE_BASE_URI = "http://example.org/signifiers/"
    private val CASHMERE_NS = "https://purl.org/cashmere#"
    private val HMAS_NS = "https://purl.org/hmas/"
    private val SH_NS = "http://www.w3.org/ns/shacl#"
    private val BASE_NS = "http://example.org/signifiers#"

    private val CASHMERE_SIGNIFIER = vf.createIRI(CASHMERE_NS, "Signifier")
    private val CASHMERE_SIGNIFIES = vf.createIRI(CASHMERE_NS, "signifies")
    private val CASHMERE_RECOMMENDS_ABILITY = vf.createIRI(CASHMERE_NS, "recommendsAbility")
    private val CASHMERE_HAS_INTENTION_DESCRIPTION = vf.createIRI(CASHMERE_NS, "hasIntentionDescription")
    private val CASHMERE_HAS_STRUCTURED_DESCRIPTION = vf.createIRI(CASHMERE_NS, "hasStructuredDescription")
    private val CASHMERE_RECOMMENDS_CONTEXT = vf.createIRI(CASHMERE_NS, "recommendsContext")
    private val CASHMERE_HAS_SHACL_CONDITION = vf.createIRI(CASHMERE_NS, "hasShaclCondition")

    private val HMAS_LLM_REASONING_ABILITY = vf.createIRI(HMAS_NS, "LLMReasoningAbility")
    private val CASHMERE_INTENTION_DESCRIPTION = vf.createIRI(CASHMERE_NS, "IntentionDescription")
    private val CASHMERE_INTENT_CONTEXT = vf.createIRI(CASHMERE_NS, "IntentContext")

    private val globalModel: Model = DynamicModelFactory().createEmptyModel().apply {
        setNamespace("cashmere", CASHMERE_NS)
        setNamespace("hmas", HMAS_NS)
        setNamespace("sh", SH_NS)
        setNamespace("xsd", XSD.NAMESPACE)
    }

    fun addSignifiersFromPlan(planJson: String): List<String> {
        log.debug("Received plan JSON: {}", planJson)
        val plan: Plan = try {
            mapper.readValue(planJson)
        } catch (e: Exception) {
            log.error("Failed to parse plan JSON", e)
            throw IllegalArgumentException("Invalid plan JSON: "+e.message)
        }

        val createdOrMerged = mutableListOf<String>()

        plan.steps.forEach { step ->
            val actionName = step.action_name ?: step.affordance_uri.substringAfter('#')
            val targetArtifact = step.artifact_uri.substringAfterLast('/')
            val signifierName = "${actionName}-${targetArtifact}-signifier"
            val signifierIri = vf.createIRI(BASE_NS + signifierName)

            val newContextConstraints = step.reasons.flatMap { it.evidence }
                .filter { it.artifact != null && it.property != null && it.operator != null && it.threshold != null }
                .map { ev ->
                    Constraint(
                        ev.artifact!!,
                        ev.property!!,
                        ev.operator!!,
                        ev.threshold!!
                    )
                }
                .toSet()

            val intent = if (!step.intent.isNullOrBlank()) {
                step.intent
            } else {
                step.reasons.firstOrNull()?.let {
                    val direction = it.direction ?: ""
                    val property = it.property ?: ""
                    "${direction} ${property} in a room".trim()
                } ?: "perform action"
            }

            if (!globalModel.contains(signifierIri, RDF.TYPE, CASHMERE_SIGNIFIER)) {
                buildSignifierWithContext(step, signifierIri, intent, newContextConstraints)
                createdOrMerged += signifierIri.stringValue()
                return@forEach
            }

            mergeSignifierContext(signifierIri, intent, newContextConstraints, step)
            createdOrMerged += signifierIri.stringValue()
        }

        log.info("Processed {} signifier(s); global model now contains {} statements", createdOrMerged.distinct().size, globalModel.size)
        return createdOrMerged.distinct()
    }

    private data class Constraint(
        val artifact: String,
        val property: String,
        val operator: String,
        val threshold: Any
    )

    private fun canonicalOperator(opRaw: String): String =
        when (opRaw.lowercase()) {
            "lessthan" -> "<"
            "lessthanorequal", "lessequal", "<=" -> "<="
            "greaterthan" -> ">"
            "greaterthanorequal", "greaterequal", ">=" -> ">="
            "equals", "=" -> "="
            "<", "<=", ">", ">=", "=" -> opRaw
            else -> opRaw
        }

    private fun constraintSubsumes(a: Constraint, b: Constraint): Boolean {
        log.debug("[MERGE_DEBUG] constraintSubsumes: Checking A={}, B={}", a, b)
        if (a.artifact != b.artifact || a.property != b.property) {
            log.debug("[MERGE_DEBUG] constraintSubsumes: Artifact/property mismatch. Returning false.")
            return false
        }

        val aOp = canonicalOperator(a.operator)
        val bOp = canonicalOperator(b.operator)

        val aNum = (a.threshold as? Number)?.toDouble()
        val bNum = (b.threshold as? Number)?.toDouble()
        log.debug("[MERGE_DEBUG] constraintSubsumes: aOp={}, bOp={}, aNum={}, bNum={}", aOp, bOp, aNum, bNum)

        val result = when (aOp to bOp) {
            "<"  to "<"  -> if (aNum != null && bNum != null) aNum <= bNum else false
            "<=" to "<=" -> if (aNum != null && bNum != null) aNum <= bNum else false
            ">"  to ">"  -> if (aNum != null && bNum != null) aNum >= bNum else false
            ">=" to ">=" -> if (aNum != null && bNum != null) aNum >= bNum else false
            "="  to "="  -> a.threshold == b.threshold

            "="  to "<"  -> if (aNum != null && bNum != null) aNum < bNum else false
            "="  to "<=" -> if (aNum != null && bNum != null) aNum <= bNum else false
            "="  to ">"  -> if (aNum != null && bNum != null) aNum > bNum else false
            "="  to ">=" -> if (aNum != null && bNum != null) aNum >= bNum else false

            "<"  to "="  -> false
            "<=" to "="  -> if (aNum != null && bNum != null && aNum == bNum) (aOp == "<=") else false
            ">"  to "="  -> false
            ">=" to "="  -> if (aNum != null && bNum != null && aNum == bNum) (aOp == ">=") else false

            "<"  to "<=" -> if (aNum != null && bNum != null) aNum <= bNum else false
            "<=" to "<"  -> if (aNum != null && bNum != null) aNum < bNum else false

            ">"  to ">=" -> if (aNum != null && bNum != null) aNum >= bNum else false
            ">=" to ">"  -> if (aNum != null && bNum != null) aNum > bNum else false
            
            else -> false
        }
        log.debug("[MERGE_DEBUG] constraintSubsumes: Returning {}.", result)
        return result
    }

    private fun contextSubsumes(ctxA: Set<Constraint>, ctxB: Set<Constraint>): Boolean {
        log.debug("[MERGE_DEBUG] contextSubsumes: Checking ctxA={}, ctxB={}", ctxA, ctxB)
        for (bInCtxB in ctxB) {
            log.debug("[MERGE_DEBUG] contextSubsumes: Current bInCtxB={}", bInCtxB)
            val relevantAConstraints = ctxA.filter { aInCtxA ->
                aInCtxA.artifact == bInCtxB.artifact && aInCtxA.property == bInCtxB.property
            }
            log.debug("[MERGE_DEBUG] contextSubsumes: relevantAConstraints for bInCtxB={}", relevantAConstraints)

            if (relevantAConstraints.isEmpty()) {
                log.debug("[MERGE_DEBUG] contextSubsumes: relevantAConstraints is empty. ctxA cannot make bInCtxB's constraint true. Returning false.")
                return false
            }

            var bCoveredByARelevantConstraint = false
            for (aRelevant in relevantAConstraints) {
                log.debug("[MERGE_DEBUG] contextSubsumes: Checking aRelevant={} against bInCtxB", aRelevant)
                if (constraintSubsumes(aRelevant, bInCtxB)) {
                    log.debug("[MERGE_DEBUG] contextSubsumes: aRelevant DOES subsume bInCtxB. bCoveredByARelevantConstraint=true.")
                    bCoveredByARelevantConstraint = true
                    break
                }
            }

            if (!bCoveredByARelevantConstraint) {
                log.debug("[MERGE_DEBUG] contextSubsumes: bInCtxB NOT covered by any relevant A constraint. Returning false.")
                return false
            }
            log.debug("[MERGE_DEBUG] contextSubsumes: bInCtxB IS covered. Continuing loop for B.")
        }
        log.debug("[MERGE_DEBUG] contextSubsumes: All constraints in ctxB were covered. Returning true.")
        return true
    }

    private fun mergeSignifierContext(signifierIri: org.eclipse.rdf4j.model.IRI, intent: String, newCtx: Set<Constraint>, step: Step) {
        log.debug("[MERGE_DEBUG] mergeSignifierContext: START for signifierIri={}, intent='{}', newCtx={}", signifierIri, intent, newCtx)
        val contextNodes = globalModel
            .filter(signifierIri, CASHMERE_RECOMMENDS_CONTEXT, null)
            .objects()
            .filterIsInstance<org.eclipse.rdf4j.model.Resource>()
            .toList()
        val ctxToConstraints = contextNodes.associateWith { ctxNode ->
            extractConstraintsFromContext(ctxNode)
        }
        log.debug("[MERGE_DEBUG] mergeSignifierContext: Initial existing ctxToConstraints={}", ctxToConstraints)

        log.debug("[MERGE_DEBUG] mergeSignifierContext: --- Step 1: Deduplication ---")
        ctxToConstraints.forEach { (ctxNode, oldCtx) ->
            log.debug("[MERGE_DEBUG] mergeSignifierContext: Comparing (for dedup) oldCtx={} with newCtx={}", oldCtx, newCtx)
            val oldSubsumesNew = contextSubsumes(oldCtx, newCtx)
            val newSubsumesOld = contextSubsumes(newCtx, oldCtx)
            log.debug("[MERGE_DEBUG] mergeSignifierContext: oldSubsumesNew={}, newSubsumesOld={}", oldSubsumesNew, newSubsumesOld)
            if (oldSubsumesNew && newSubsumesOld) {
                log.debug("[MERGE_DEBUG] mergeSignifierContext: Contexts are identical. Returning.")
                return
            }
        }

        log.debug("[MERGE_DEBUG] mergeSignifierContext: --- Step 2: Prune old contexts based on newCtx ---")
        val surviving = mutableListOf<org.eclipse.rdf4j.model.Resource>()
        ctxToConstraints.forEach { (ctxNode, oldCtx) ->
            log.debug("[MERGE_DEBUG] mergeSignifierContext: Comparing oldCtx={} with newCtx={} for pruning", oldCtx, newCtx)
            val newIsMoreSpecificThanOld = contextSubsumes(newCtx, oldCtx)
            val oldIsMoreSpecificThanNew = contextSubsumes(oldCtx, newCtx)

            val newIsStrictlySpecific = newIsMoreSpecificThanOld && !oldIsMoreSpecificThanNew
            val newIsStrictlyGeneral  = oldIsMoreSpecificThanNew && !newIsMoreSpecificThanOld

            if (newIsStrictlySpecific) {
                log.debug("[MERGE_DEBUG] mergeSignifierContext: newCtx is strictly more specific than oldCtx. Pruning oldCtx: {}.", oldCtx)
                removeContextNode(signifierIri, ctxNode)
            } else if (newIsStrictlyGeneral) {
                log.debug("[MERGE_DEBUG] mergeSignifierContext: newCtx is strictly more general than oldCtx. Pruning oldCtx: {}.", oldCtx)
                removeContextNode(signifierIri, ctxNode)
            } else {
                log.debug("[MERGE_DEBUG] mergeSignifierContext: oldCtx is incomparable or equivalent to newCtx. oldCtx survives: {}.", oldCtx)
                surviving += ctxNode
            }
        }
        log.debug("[MERGE_DEBUG] mergeSignifierContext: Surviving old contexts after pruning: {}", surviving.map { extractConstraintsFromContext(it) })

        log.debug("[MERGE_DEBUG] mergeSignifierContext: --- Step 3: Reject newCtx if redundant under any SURVIVING oldCtx ---")
        for (ctxNode in surviving) {
            val oldCtx = extractConstraintsFromContext(ctxNode)
            log.debug("[MERGE_DEBUG] mergeSignifierContext: Comparing (for newCtx rejection) surviving oldCtx={} with newCtx={}", oldCtx, newCtx)
            val oldActuallySubsumesNew = contextSubsumes(oldCtx, newCtx)
            log.debug("[MERGE_DEBUG] mergeSignifierContext: oldSubsumesNew for rejection check={}", oldActuallySubsumesNew)
            if (oldActuallySubsumesNew) {
                log.debug("[MERGE_DEBUG] mergeSignifierContext: Surviving oldCtx DOES subsume newCtx. newCtx is redundant. Returning.")
                return
            }
        }

        log.debug("[MERGE_DEBUG] mergeSignifierContext: --- Step 4: Adding newCtx as it is distinct and not redundant ---")
        log.debug("[MERGE_DEBUG] mergeSignifierContext: Calling buildContextAndAttach for newCtx={}", newCtx)
        buildContextAndAttach(signifierIri, intent, newCtx, step)
    }

    private fun extractConstraintsFromContext(ctxNode: org.eclipse.rdf4j.model.Resource): Set<Constraint> {
        val constraints = mutableSetOf<Constraint>()
        val shapeNodes = globalModel
            .filter(ctxNode, CASHMERE_HAS_SHACL_CONDITION, null)
            .objects()
            .filterIsInstance<org.eclipse.rdf4j.model.Resource>()
        shapeNodes.forEach { shapeRes ->
            val targetNode = globalModel.filter(shapeRes, vf.createIRI(SH_NS, "targetNode"), null).objects().firstOrNull() as? org.eclipse.rdf4j.model.IRI
            val propBn = globalModel.filter(shapeRes, vf.createIRI(SH_NS, "property"), null).objects().firstOrNull() as? org.eclipse.rdf4j.model.Resource
            if (targetNode == null || propBn == null) return@forEach
            val pathIri = globalModel.filter(propBn, vf.createIRI(SH_NS, "path"), null).objects().firstOrNull() as? org.eclipse.rdf4j.model.IRI ?: return@forEach
            val (artifactUri, propertyName) = pathIri.stringValue().split('#').let { it[0] to it.getOrElse(1) {""} }
            var operator: String? = null
            var thresholdValue: Any? = null

            globalModel.filter(propBn, vf.createIRI(SH_NS, "hasValue"), null).objects().firstOrNull()?.let { lit ->
                operator = "="
                thresholdValue = (lit as org.eclipse.rdf4j.model.Literal).label
            }

            if (operator == null) {
                globalModel.filter(propBn, vf.createIRI(SH_NS, "maxExclusive"), null).objects().firstOrNull()?.let { lit ->
                    operator = "<"
                    thresholdValue = (lit as org.eclipse.rdf4j.model.Literal).label
                } ?: globalModel.filter(propBn, vf.createIRI(SH_NS, "maxInclusive"), null).objects().firstOrNull()?.let { lit ->
                    operator = "<="
                    thresholdValue = (lit as org.eclipse.rdf4j.model.Literal).label
                }
            }
            
            if (operator == null) {
                globalModel.filter(propBn, vf.createIRI(SH_NS, "minExclusive"), null).objects().firstOrNull()?.let { lit ->
                    operator = ">"
                    thresholdValue = (lit as org.eclipse.rdf4j.model.Literal).label
                } ?: globalModel.filter(propBn, vf.createIRI(SH_NS, "minInclusive"), null).objects().firstOrNull()?.let { lit ->
                    operator = ">="
                    thresholdValue = (lit as org.eclipse.rdf4j.model.Literal).label
                }
            }

            if (operator == null || thresholdValue == null) {
                log.warn("[MERGE_DEBUG] Could not determine operator/threshold for property shape subject: {}, property blank node: {}", shapeRes, propBn)
                return@forEach
            }
            
            val finalThreshold = thresholdValue.toString().toDoubleOrNull() ?: thresholdValue.toString()
            constraints += Constraint(artifactUri, propertyName, operator!!, finalThreshold)
        }
        return constraints
    }

    private fun removeContextNode(signifierIri: org.eclipse.rdf4j.model.IRI, ctxNode: org.eclipse.rdf4j.model.Resource) {
        log.debug("[MERGE_DEBUG] Removing context node: {}", ctxNode)
        globalModel.remove(signifierIri, CASHMERE_RECOMMENDS_CONTEXT, ctxNode)

        removeNodeAndItsSubgraph(ctxNode)
    }

    private fun removeNodeAndItsSubgraph(node: org.eclipse.rdf4j.model.Resource) {
        val statementsAboutNode = globalModel.filter(node, null, null).toList()

        statementsAboutNode.forEach { stmt ->
            globalModel.remove(stmt)

            val obj = stmt.`object`
            if (obj is org.eclipse.rdf4j.model.BNode) {
                removeNodeAndItsSubgraph(obj)
            }
        }

        if (node is org.eclipse.rdf4j.model.BNode) {
            val statementsPointingToNode = globalModel.filter(null, null, node).toList()
            statementsPointingToNode.forEach { stmt ->
                globalModel.remove(stmt)
            }
        }
    }

    // ========== helpers to build signifier/context ================================

    private fun buildSignifierWithContext(step: Step, signifierIri: org.eclipse.rdf4j.model.IRI, intent: String, constraints: Set<Constraint>) {
        globalModel.add(signifierIri, RDF.TYPE, CASHMERE_SIGNIFIER)
        globalModel.add(signifierIri, CASHMERE_SIGNIFIES, vf.createIRI(step.affordance_uri))
        val abilityBn = vf.createBNode()
        globalModel.add(signifierIri, CASHMERE_RECOMMENDS_ABILITY, abilityBn)
        globalModel.add(abilityBn, RDF.TYPE, HMAS_LLM_REASONING_ABILITY)

        val intDescBn = vf.createBNode()
        globalModel.add(signifierIri, CASHMERE_HAS_INTENTION_DESCRIPTION, intDescBn)
        globalModel.add(intDescBn, RDF.TYPE, CASHMERE_INTENTION_DESCRIPTION)
        val intentJson = """
        {
            'intent': '${intent}',
        }
        """
        globalModel.add(intDescBn, CASHMERE_HAS_STRUCTURED_DESCRIPTION, vf.createLiteral(intentJson, XSD.STRING))

        buildContextAndAttach(signifierIri, intent, constraints, step)
    }

    private fun buildContextAndAttach(signifierIri: org.eclipse.rdf4j.model.IRI, intent: String, constraints: Set<Constraint>, step: Step) {
        val ctxBn = vf.createBNode()
        globalModel.add(signifierIri, CASHMERE_RECOMMENDS_CONTEXT, ctxBn)
        globalModel.add(ctxBn, RDF.TYPE, CASHMERE_INTENT_CONTEXT)

        val sb = StringBuilder()
        sb.append("""
        {
            'conditions': [""")
        constraints.forEachIndexed { idx, c ->
            sb.append("""
                {
                    'artifact': <${c.artifact}>,
                    'propertyAffordance': <${c.artifact}#${c.property}>,
                    'valueConditions': [
                        {
                            'operator': '${c.operator}',
                            'value': ${formatThresholdValue(c.threshold)}
                        }
                    ]
                }""")
            if (idx < constraints.size - 1) sb.append(",")
        }
        sb.append("""
            ]
        }
        """)
        globalModel.add(ctxBn, CASHMERE_HAS_STRUCTURED_DESCRIPTION, vf.createLiteral(sb.toString(), XSD.STRING))

        constraints.forEach { c ->
            val shapeBn = vf.createBNode()
            globalModel.add(ctxBn, CASHMERE_HAS_SHACL_CONDITION, shapeBn)
            globalModel.add(shapeBn, RDF.TYPE, vf.createIRI(SH_NS, "NodeShape"))
            globalModel.add(shapeBn, vf.createIRI(SH_NS, "targetNode"), vf.createIRI(c.artifact))
            val propConstraintBn = vf.createBNode()
            globalModel.add(shapeBn, vf.createIRI(SH_NS, "property"), propConstraintBn)
            val pathIri = vf.createIRI("${c.artifact}#${c.property}")
            globalModel.add(propConstraintBn, vf.createIRI(SH_NS, "path"), pathIri)
            globalModel.add(propConstraintBn, vf.createIRI(SH_NS, "dataType"), determineDatatype(c.threshold))
            val thrLit = createThresholdLiteral(c.threshold)
            when (canonicalOperator(c.operator)) {
                "<"  -> globalModel.add(propConstraintBn, vf.createIRI(SH_NS, "maxExclusive"), thrLit)
                "<=" -> globalModel.add(propConstraintBn, vf.createIRI(SH_NS, "maxInclusive"), thrLit)
                ">"  -> globalModel.add(propConstraintBn, vf.createIRI(SH_NS, "minExclusive"), thrLit)
                ">=" -> globalModel.add(propConstraintBn, vf.createIRI(SH_NS, "minInclusive"), thrLit)
                "="  -> globalModel.add(propConstraintBn, vf.createIRI(SH_NS, "hasValue"), thrLit)
            }
        }
    }

    fun getSignifierAsTurtle(iri: String): String? {
        val iriObj = try { vf.createIRI(iri) } catch (_: Exception) { return null }
        val filteredModel = globalModel.filter(iriObj, null, null)
        if (filteredModel.isEmpty()) return null

        val sw = StringWriter()
        val rdfWriter: RDFWriter = Rio.createWriter(RDFFormat.TURTLE, sw, TURTLE_BASE_URI)
        rdfWriter.writerConfig.set(BasicWriterSettings.XSD_STRING_TO_PLAIN_LITERAL, false)
        rdfWriter.writerConfig.set(BasicWriterSettings.INLINE_BLANK_NODES, true)

        rdfWriter.startRDF()
        
        globalModel.namespaces.forEach { ns ->
            rdfWriter.handleNamespace(ns.prefix, ns.name)
        }
        
        filteredModel.forEach(rdfWriter::handleStatement)
        rdfWriter.endRDF()
        return sw.toString()
    }

    fun listAllSignifierIris(): List<String> =
        globalModel.filter(null, RDF.TYPE, CASHMERE_SIGNIFIER).subjects().map {
            if (it.stringValue().startsWith(BASE_NS)) {
                "<#" + it.stringValue().substringAfter(BASE_NS) + ">"
            } else {
                it.stringValue()
            }
        }.toList()

    fun getAllSignifiersAsTurtle(): String {
        if (globalModel.isEmpty()) return "# No signifiers have been created yet."
        
        val sw = StringWriter()
        val rdfWriter: RDFWriter = Rio.createWriter(RDFFormat.TURTLE, sw, TURTLE_BASE_URI)
        
        rdfWriter.writerConfig.set(BasicWriterSettings.XSD_STRING_TO_PLAIN_LITERAL, false)
        rdfWriter.writerConfig.set(BasicWriterSettings.INLINE_BLANK_NODES, true)

        rdfWriter.startRDF()
        globalModel.namespaces.forEach { ns ->
            rdfWriter.handleNamespace(ns.prefix, ns.name)
        }
        globalModel.forEach(rdfWriter::handleStatement)
        rdfWriter.endRDF()
        
        return sw.toString()
    }

    fun clearAllSignifiers() {
        globalModel.clear()
        log.info("Cleared all signifiers from global model")
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class Plan(val plan_version: String?, val steps: List<Step>)

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class Step(
        val step_id: Int,
        val artifact_uri: String,
        val affordance_uri: String,
        val action_name: String?,
        val method: String?,
        val target: String?,
        val content_type: String?,
        val payload: Any?,
        val reasons: List<Reason>,
        val intent: String?
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class Reason(
        val property: String?,
        val direction: String?,
        val evidence: List<Evidence>,
        val why: String?
    )

    @JsonIgnoreProperties(ignoreUnknown = true)
    private data class Evidence(
        val artifact: String?,
        val property: String?,
        val operator: String?,
        val threshold: Any?,
        val reading: Any?
    )

    private fun createThresholdLiteral(value: Any?): org.eclipse.rdf4j.model.Literal {
        return when (value) {
            is Int, is Long -> vf.createLiteral(value.toString(), XSD.INTEGER)
            is Float, is Double -> {
                val dbl = (value as Number).toDouble()
                if (dbl % 1.0 == 0.0) {
                    vf.createLiteral(dbl.toInt().toString(), XSD.INTEGER)
                } else {
                    vf.createLiteral(dbl.toString(), XSD.DECIMAL)
                }
            }
            is Boolean -> vf.createLiteral(value.toString(), XSD.BOOLEAN)
            else -> vf.createLiteral(value.toString(), XSD.STRING)
        }
    }

    private fun determineDatatype(value: Any?): org.eclipse.rdf4j.model.IRI {
        return when (value) {
            is Int, is Long -> XSD.INTEGER
            is Float, is Double -> {
                val dbl = (value as Number).toDouble()
                if (dbl % 1.0 == 0.0) XSD.INTEGER else XSD.DECIMAL
            }
            is Boolean -> XSD.BOOLEAN
            else -> XSD.STRING
        }
    }
    
    private fun formatThresholdValue(threshold: Any?): String {
        return when (threshold) {
            is String -> "'${threshold}'"
            is Number -> threshold.toString()
            else -> "'${threshold.toString()}'"
        }
    }
} 