package service 

import org.eclipse.rdf4j.model.IRI
import org.eclipse.rdf4j.model.Model
import org.eclipse.rdf4j.model.Resource
import org.eclipse.rdf4j.model.impl.DynamicModelFactory
import org.eclipse.rdf4j.model.impl.SimpleValueFactory
import org.eclipse.rdf4j.model.vocabulary.RDF
import org.eclipse.rdf4j.model.vocabulary.XSD
import org.eclipse.rdf4j.rio.RDFFormat
import org.eclipse.rdf4j.rio.Rio
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.io.StringWriter
import java.util.concurrent.ConcurrentHashMap

data class IndexedAction(
    val actionAffordanceUri: String,
    val artifactUri: String,
    val actionName: String,
    val formTargetUri: String,
    val formMethod: String,
    val inputSchemaTurtle: String?
)

@Service
class ActionIndexService {

    private val log = LoggerFactory.getLogger(javaClass)
    private val valueFactory = SimpleValueFactory.getInstance()

    private val TD_NAMESPACE = "https://www.w3.org/2019/wot/td#"
    private val HCTL_NAMESPACE = "https://www.w3.org/2019/wot/hypermedia#"
    private val HTV_NAMESPACE = "http://www.w3.org/2011/http#"
    private val JS_NAMESPACE = "https://www.w3.org/2019/wot/json-schema#"

    private val TD_ACTION_AFFORDANCE: IRI = valueFactory.createIRI(TD_NAMESPACE, "ActionAffordance")
    private val TD_NAME: IRI = valueFactory.createIRI(TD_NAMESPACE, "name")
    private val TD_HAS_FORM: IRI = valueFactory.createIRI(TD_NAMESPACE, "hasForm")
    private val TD_HAS_INPUT_SCHEMA: IRI = valueFactory.createIRI(TD_NAMESPACE, "hasInputSchema")

    private val HCTL_HAS_TARGET: IRI = valueFactory.createIRI(HCTL_NAMESPACE, "hasTarget")
    private val HTV_METHOD_NAME: IRI = valueFactory.createIRI(HTV_NAMESPACE, "methodName")
    private val JS_PROPERTIES: IRI = valueFactory.createIRI(JS_NAMESPACE, "properties")
    private val actionIndex = ConcurrentHashMap<String, MutableList<IndexedAction>>()

    fun createIRI(uri: String): IRI = valueFactory.createIRI(uri)

    fun indexActionsFromModel(artifactUri: IRI, model: Model) {
        val artifactUriString = artifactUri.stringValue()
        log.info(">>> INDEXING START for artifact: {}", artifactUriString)
        log.debug("Model size: {}", model.size)

        log.debug("Available types in model: {}", model.filter(null, RDF.TYPE, null).objects())
        log.debug("Statements about artifact URI ({}): {}", artifactUriString, model.filter(artifactUri, null, null).toList())

        val foundActions = mutableListOf<IndexedAction>()

        val actionAffordanceSubjects = model.filter(null, RDF.TYPE, TD_ACTION_AFFORDANCE).subjects()
        log.info("Found {} potential action affordance subjects for {}.", actionAffordanceSubjects.size, artifactUriString)

        if (actionAffordanceSubjects.isEmpty()) {
            log.warn("No subjects with the expected ActionAffordance type found. Check IRI definition and data.")
            model.filter(null, RDF.TYPE, null).forEach { log.warn("Available type statement: {}", it) }
        }

        for (actionSubject in actionAffordanceSubjects) {
            log.debug("Processing action subject: {} (Type: {})", actionSubject, actionSubject::class.simpleName)
            if (actionSubject !is Resource) continue

            val actionAffordanceUriString = if (actionSubject is IRI) actionSubject.stringValue() else actionSubject.toString()

            val actionNameValue = model.filter(actionSubject, TD_NAME, null).firstOrNull()?.`object`
            val actionName = actionNameValue?.stringValue()
            log.debug("  Action Name (td:name): {} (Value: {})", actionName, actionNameValue)

            var formTargetUriString: String? = null
            var formMethodString: String? = null
            var inputSchemaModel: Model? = null

            val formNode = model.filter(actionSubject, TD_HAS_FORM, null).objects().firstOrNull { it is Resource }
            if (formNode != null) {
                 log.debug("  Found Form node: {}", formNode)
                 val resourceNode = formNode as Resource
                 formTargetUriString = model.filter(resourceNode, HCTL_HAS_TARGET, null).firstOrNull()?.`object`?.stringValue()
                 formMethodString = model.filter(resourceNode, HTV_METHOD_NAME, null).firstOrNull()?.`object`?.stringValue()
                 log.debug("    Form Target (hctl:hasTarget): {}", formTargetUriString)
                 log.debug("    Form Method (htv:methodName): {}", formMethodString)
            } else {
                log.warn("  No Form node (td:hasForm) found for action {}", actionAffordanceUriString)
            }

            val schemaNode = model.filter(actionSubject, TD_HAS_INPUT_SCHEMA, null).objects().firstOrNull { it is Resource }
             if (schemaNode != null) {
                 val schemaResource = schemaNode as Resource
                 log.debug("  Found Input Schema node: {}", schemaResource)
                 inputSchemaModel = DynamicModelFactory().createEmptyModel()
                 inputSchemaModel.addAll(model.filter(schemaResource, null, null))
                 model.filter(schemaResource, RDF.TYPE, null).forEach { inputSchemaModel?.add(it) }
                 model.filter(schemaResource, JS_PROPERTIES, null).objects().forEach { propNode ->
                     if (propNode is Resource) {
                         log.debug("    Adding properties from schema property node: {}", propNode)
                         inputSchemaModel.addAll(model.filter(propNode, null, null))
                         model.filter(propNode, RDF.TYPE, null).forEach { inputSchemaModel?.add(it) }
                     }
                 }
                 log.debug("    Input Schema Model size: {}", inputSchemaModel?.size ?: 0)
             } else {
                  log.debug("  No Input Schema (td:hasInputSchema) found for action {}", actionAffordanceUriString)
             }

            if (actionName != null && formTargetUriString != null && formMethodString != null) {
                val inputSchemaTurtleString = inputSchemaModel?.takeIf { it.isNotEmpty() }?.let { schemaModel ->
                    try {
                        val writer = StringWriter()
                        schemaModel.setNamespace("td", TD_NAMESPACE)
                        schemaModel.setNamespace("js", JS_NAMESPACE)
                        schemaModel.setNamespace("xsd", XSD.NAMESPACE)
                        Rio.write(schemaModel, writer, RDFFormat.TURTLE)
                        writer.toString()
                    } catch (e: Exception) {
                        log.error("Failed to serialize input schema for action {}: {}", actionAffordanceUriString, e.message)
                        "# Error serializing schema: ${e.message}"
                    }
                }

                val indexedAction = IndexedAction(
                    actionAffordanceUri = actionAffordanceUriString,
                    artifactUri = artifactUriString,
                    actionName = actionName,
                    formTargetUri = formTargetUriString,
                    formMethod = formMethodString,
                    inputSchemaTurtle = inputSchemaTurtleString
                )

                foundActions.add(indexedAction)
                log.info("   --> Action is VALID for indexing: name='{}'", actionName)
            } else {
                log.warn("   --> Action is SKIPPED (missing mandatory fields): name='{}', target='{}', method='{}'",
                    actionName, formTargetUriString, formMethodString)
            }
            log.debug("--- Finished processing action subject: {}", actionSubject)
        }

        if (foundActions.isNotEmpty()) {
            actionIndex[artifactUriString] = foundActions
            log.info("<<< INDEXING END for {}: Indexed {} valid actions.", artifactUriString, foundActions.size)
        } else {
            actionIndex.remove(artifactUriString)
            log.warn("<<< INDEXING END for {}: No valid actions were found/indexed.", artifactUriString)
        }
    }

    fun removeActionsForArtifact(artifactUri: IRI) {
        val artifactUriString = artifactUri.stringValue()
        val removedActions = actionIndex.remove(artifactUriString)
        log.info("Removed {} indexed actions for artifact {}", removedActions?.size ?: 0, artifactUriString)
    }

    fun getActionsForArtifact(artifactUriString: String): List<IndexedAction> {
        return actionIndex[artifactUriString] ?: emptyList()
    }

    fun getAllActions(): List<IndexedAction> {
        return actionIndex.values.flatten()
    }
}