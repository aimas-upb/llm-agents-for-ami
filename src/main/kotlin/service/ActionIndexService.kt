package service 

import org.eclipse.rdf4j.model.IRI
import org.eclipse.rdf4j.model.Model
import org.eclipse.rdf4j.model.Resource
import org.eclipse.rdf4j.model.Value
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
    private val TD_TITLE: IRI = valueFactory.createIRI(TD_NAMESPACE, "title")
    private val TD_HAS_FORM: IRI = valueFactory.createIRI(TD_NAMESPACE, "hasForm")
    private val TD_HAS_INPUT_SCHEMA: IRI = valueFactory.createIRI(TD_NAMESPACE, "hasInputSchema")
    private val TD_HAS_OUTPUT_SCHEMA: IRI = valueFactory.createIRI(TD_NAMESPACE, "hasOutputSchema")

    private val HCTL_HAS_TARGET: IRI = valueFactory.createIRI(HCTL_NAMESPACE, "hasTarget")
    private val HCTL_FOR_CONTENT_TYPE: IRI = valueFactory.createIRI(HCTL_NAMESPACE, "forContentType")
    private val HCTL_HAS_OPERATION_TYPE: IRI = valueFactory.createIRI(HCTL_NAMESPACE, "hasOperationType")
    private val HCTL_FOR_SUB_PROTOCOL: IRI = valueFactory.createIRI(HCTL_NAMESPACE, "forSubProtocol")
    
    private val HTV_METHOD_NAME: IRI = valueFactory.createIRI(HTV_NAMESPACE, "methodName")
    
    private val JS_PROPERTIES: IRI = valueFactory.createIRI(JS_NAMESPACE, "properties")
    private val JS_PROPERTY_NAME: IRI = valueFactory.createIRI(JS_NAMESPACE, "propertyName")
    private val JS_REQUIRED: IRI = valueFactory.createIRI(JS_NAMESPACE, "required")
    private val JS_ENUM: IRI = valueFactory.createIRI(JS_NAMESPACE, "enum")
    private val JS_OBJECT_SCHEMA: IRI = valueFactory.createIRI(JS_NAMESPACE, "ObjectSchema")
    private val JS_STRING_SCHEMA: IRI = valueFactory.createIRI(JS_NAMESPACE, "StringSchema")
    private val JS_BOOLEAN_SCHEMA: IRI = valueFactory.createIRI(JS_NAMESPACE, "BooleanSchema")
    private val JS_INTEGER_SCHEMA: IRI = valueFactory.createIRI(JS_NAMESPACE, "IntegerSchema")
    
    // Store filtered RDF models containing only action affordances for each artifact
    private val actionIndex = ConcurrentHashMap<String, Model>()

    fun createIRI(uri: String): IRI = valueFactory.createIRI(uri)

    fun indexActionsFromModel(artifactUri: IRI, model: Model) {
        val artifactUriString = artifactUri.stringValue()
        log.info(">>> INDEXING START for artifact: {}", artifactUriString)
        log.debug("Original model size: {}", model.size)

        // Create a new model to store only action affordance-related statements
        val actionAffordanceModel = DynamicModelFactory().createEmptyModel()
        
        // Set up namespaces for the filtered model
        actionAffordanceModel.setNamespace("td", TD_NAMESPACE)
        actionAffordanceModel.setNamespace("hctl", HCTL_NAMESPACE)
        actionAffordanceModel.setNamespace("htv", HTV_NAMESPACE)
        actionAffordanceModel.setNamespace("js", JS_NAMESPACE)
        actionAffordanceModel.setNamespace("xsd", XSD.NAMESPACE)

        // Find the artifact subject - it should be the main Thing
        val artifactSubject = model.filter(null, RDF.TYPE, valueFactory.createIRI(TD_NAMESPACE, "Thing"))
            .subjects().firstOrNull() as? Resource
            
        if (artifactSubject == null) {
            log.warn("No Thing found in the model for artifact {}", artifactUriString)
            actionIndex.remove(artifactUriString)
            return
        }

        log.debug("Found artifact Thing: {}", artifactSubject)

        // Find all action affordances linked to this Thing
        val actionAffordances = model.filter(artifactSubject, valueFactory.createIRI(TD_NAMESPACE, "hasActionAffordance"), null)
            .objects().filterIsInstance<Resource>()

        log.info("Found {} action affordances for artifact {}", actionAffordances.size, artifactUriString)

        // Filter out infrastructure actions by name
        val infrastructureActionNames = setOf(
            "getStatus",
            "getArtifactRepresentation", 
            "updateArtifactRepresentation",
            "deleteArtifactRepresentation",
            "focusArtifact",
            "subscribeToArtifact",
            "unsubscribeFromArtifact"
        )
        
        val filteredActionAffordances = actionAffordances.filter { actionAffordance ->
            val actionName = model.filter(actionAffordance, TD_NAME, null).objects().firstOrNull()?.stringValue()
            val shouldInclude = actionName == null || !infrastructureActionNames.contains(actionName)
            if (!shouldInclude) {
                log.debug("Filtering out infrastructure action: {}", actionName)
            }
            shouldInclude
        }

        log.info("After filtering infrastructure actions: {} remaining action affordances", filteredActionAffordances.size)

        if (filteredActionAffordances.isEmpty()) {
            log.warn("No non-infrastructure action affordances found for artifact {}", artifactUriString)
            actionIndex.remove(artifactUriString)
            return
        }

        // Process each action affordance with simple, readable URIs
        filteredActionAffordances.forEach { actionAffordance ->
            val actionName = model.filter(actionAffordance, TD_NAME, null).objects().firstOrNull()?.stringValue()
            if (actionName != null) {
                processActionAffordanceCompact(model, actionAffordanceModel, actionAffordance, actionName, artifactUriString)
            }
        }

        if (!actionAffordanceModel.isEmpty()) {
            actionIndex[artifactUriString] = actionAffordanceModel
            log.info("<<< INDEXING END for {}: Indexed {} action affordance statements.", 
                artifactUriString, actionAffordanceModel.size)
        } else {
            actionIndex.remove(artifactUriString)
            log.warn("<<< INDEXING END for {}: No action affordance statements found.", artifactUriString)
        }
    }

    private fun processActionAffordanceCompact(
        sourceModel: Model, 
        targetModel: Model, 
        originalActionAffordance: Resource,
        actionName: String,
        artifactUriString: String
    ) {
        // Create simple, readable URIs
        val actionUri = valueFactory.createIRI("${artifactUriString}#${actionName}")
        
        // Add action type and name
        targetModel.add(actionUri, RDF.TYPE, TD_ACTION_AFFORDANCE)
        targetModel.add(actionUri, TD_NAME, valueFactory.createLiteral(actionName))
        
        // Add title if different from name
        val title = sourceModel.filter(originalActionAffordance, TD_TITLE, null).objects().firstOrNull()?.stringValue()
        if (title != null && title != actionName) {
            targetModel.add(actionUri, TD_TITLE, valueFactory.createLiteral(title))
        }

        // Process form - create simple form URI
        sourceModel.filter(originalActionAffordance, TD_HAS_FORM, null).objects().firstOrNull()?.let { formNode ->
            if (formNode is Resource) {
                val formUri = valueFactory.createIRI("${artifactUriString}#${actionName}-form")
                targetModel.add(actionUri, TD_HAS_FORM, formUri)
                
                // Copy essential form properties
                sourceModel.filter(formNode, HTV_METHOD_NAME, null).objects().firstOrNull()?.let { method ->
                    targetModel.add(formUri, HTV_METHOD_NAME, method)
                }
                sourceModel.filter(formNode, HCTL_HAS_TARGET, null).objects().firstOrNull()?.let { target ->
                    targetModel.add(formUri, HCTL_HAS_TARGET, target)
                }
                sourceModel.filter(formNode, HCTL_FOR_CONTENT_TYPE, null).objects().firstOrNull()?.let { contentType ->
                    targetModel.add(formUri, HCTL_FOR_CONTENT_TYPE, contentType)
                }
            }
        }

        // Process input schema if present
        sourceModel.filter(originalActionAffordance, TD_HAS_INPUT_SCHEMA, null).objects().firstOrNull()?.let { schemaNode ->
            if (schemaNode is Resource) {
                val schemaUri = valueFactory.createIRI("${artifactUriString}#${actionName}-input")
                targetModel.add(actionUri, TD_HAS_INPUT_SCHEMA, schemaUri)
                processSchemaCompact(sourceModel, targetModel, schemaNode, schemaUri)
            }
        }

        // Process output schema if present
        sourceModel.filter(originalActionAffordance, TD_HAS_OUTPUT_SCHEMA, null).objects().firstOrNull()?.let { schemaNode ->
            if (schemaNode is Resource) {
                val schemaUri = valueFactory.createIRI("${artifactUriString}#${actionName}-output")
                targetModel.add(actionUri, TD_HAS_OUTPUT_SCHEMA, schemaUri)
                processSchemaCompact(sourceModel, targetModel, schemaNode, schemaUri)
            }
        }
    }

    private fun processSchemaCompact(sourceModel: Model, targetModel: Model, originalSchemaNode: Resource, schemaUri: IRI) {
        // Copy schema type
        sourceModel.filter(originalSchemaNode, RDF.TYPE, null).objects().forEach { type ->
            targetModel.add(schemaUri, RDF.TYPE, type)
        }

        // Copy required fields
        sourceModel.filter(originalSchemaNode, JS_REQUIRED, null).objects().forEach { required ->
            targetModel.add(schemaUri, JS_REQUIRED, required)
        }

        // Process properties with simple URIs
        sourceModel.filter(originalSchemaNode, JS_PROPERTIES, null).objects().forEachIndexed { index, propNode ->
            if (propNode is Resource) {
                val propName = sourceModel.filter(propNode, JS_PROPERTY_NAME, null).objects().firstOrNull()?.stringValue()
                val propUri = if (propName != null) {
                    valueFactory.createIRI("${schemaUri}-${propName}")
                } else {
                    valueFactory.createIRI("${schemaUri}-prop${index + 1}")
                }
                
                targetModel.add(schemaUri, JS_PROPERTIES, propUri)
                
                // Copy all property details
                sourceModel.filter(propNode, null, null).forEach { stmt ->
                    targetModel.add(propUri, stmt.predicate, stmt.`object`)
                }
            }
        }
    }

    fun removeActionsForArtifact(artifactUri: IRI) {
        val artifactUriString = artifactUri.stringValue()
        val removedModel = actionIndex.remove(artifactUriString)
        log.info("Removed indexed action model for artifact {} (had {} statements)", 
            artifactUriString, removedModel?.size ?: 0)
    }

    fun getActionsForArtifactAsRdf(artifactUriString: String): String? {
        val model = actionIndex[artifactUriString]
        return if (model != null && !model.isEmpty()) {
            try {
                val writer = StringWriter()
                Rio.write(model, writer, RDFFormat.TURTLE)
                writer.toString()
            } catch (e: Exception) {
                log.error("Failed to serialize action model for {}: {}", artifactUriString, e.message)
                null
            }
        } else {
            null
        }
    }

    fun getAllKnownArtifactUris(): List<String> {
        return actionIndex.keys.toList()
    }

    fun hasActionsForArtifact(artifactUriString: String): Boolean {
        return actionIndex.containsKey(artifactUriString) && 
               actionIndex[artifactUriString]?.isEmpty() == false
    }
}