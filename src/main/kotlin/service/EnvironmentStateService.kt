package org.eclipse.lmos.arc.app.service

import org.eclipse.lmos.arc.app.model.ArtifactState
import org.eclipse.rdf4j.model.*
import org.eclipse.rdf4j.model.impl.SimpleValueFactory
import org.eclipse.rdf4j.model.vocabulary.*
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.time.Instant
import java.util.concurrent.ConcurrentHashMap

@Service
class EnvironmentStateService {

    private val log = LoggerFactory.getLogger(javaClass)
    private val valueFactory: ValueFactory = SimpleValueFactory.getInstance()

    private val artifactStates = ConcurrentHashMap<String, ArtifactState>()

    fun addOrReplaceArtifact(artifactUri: IRI, initialModel: Model, timestamp: Instant, triggeringUri: IRI?) {
        val uriString = artifactUri.stringValue()
        log.info("Adding/Replacing artifact: {} triggered by {} at {}", uriString, triggeringUri ?: "N/A", timestamp)
        val newState = ArtifactState(
            artifactUri = artifactUri,
            rdfModel = initialModel,
            lastUpdatedAt = timestamp,
            lastTriggeringAffordanceUri = triggeringUri
        )
        setCommonNamespaces(newState.rdfModel)
        artifactStates[uriString] = newState
    }

    fun updateArtifactProperty(
        artifactUri: IRI,
        propertyUri: IRI,
        newValue: Value,
        timestamp: Instant,
        triggeringUri: IRI?
    ) {
        val uriString = artifactUri.stringValue()
        if (triggeringUri == null) {
             log.warn("Received property update for {} without a triggering URI. Metadata might be incomplete.", uriString)
        }
        log.debug("Updating property {} for {} to {} triggered by {} at {}",
            propertyUri.localName, uriString, newValue, triggeringUri ?: "N/A", timestamp)

        artifactStates.compute(uriString) { _, existingState ->
            val currentState = existingState ?: ArtifactState(artifactUri)

            currentState.rdfModel.remove(artifactUri, propertyUri, null)
            currentState.rdfModel.add(artifactUri, propertyUri, newValue)
            currentState.lastUpdatedAt = timestamp
            currentState.lastTriggeringAffordanceUri = triggeringUri
            setCommonNamespaces(currentState.rdfModel)

            currentState
        }
    }

    fun removeArtifact(artifactUri: IRI) {
        val uriString = artifactUri.stringValue()
        log.info("Removing artifact: {}", uriString)
        artifactStates.remove(uriString)
    }

    fun getArtifactState(artifactUriString: String): ArtifactState? {
        return artifactStates[artifactUriString]
    }

    fun getAllArtifactStates(): List<ArtifactState> {
        return artifactStates.values.toList()
    }

    /**
     * Clears all tracked artifact states and returns the number of entries removed.
     */
    fun reset(): Int {
        val removed = artifactStates.size
        log.warn("Resetting environment state; removing {} artifacts", removed)
        artifactStates.clear()
        return removed
    }

    fun createIRI(uri: String): IRI = valueFactory.createIRI(uri)
    fun createLiteral(value: String): Literal = valueFactory.createLiteral(value)
    fun createLiteral(value: Boolean): Literal = valueFactory.createLiteral(value)
    fun createLiteral(value: Int): Literal = valueFactory.createLiteral(value)
    fun createLiteral(value: Long): Literal = valueFactory.createLiteral(value)
    fun createLiteral(value: Double): Literal = valueFactory.createLiteral(value)
    fun createLiteral(value: Float): Literal = valueFactory.createLiteral(value)
    fun createLiteral(value: String, datatype: IRI): Literal = valueFactory.createLiteral(value, datatype)

    private fun setCommonNamespaces(model: Model) {
        if (model.getNamespace("rdf").isEmpty) model.setNamespace("rdf", RDF.NAMESPACE)
        if (model.getNamespace("rdfs").isEmpty) model.setNamespace("rdfs", RDFS.NAMESPACE)
        if (model.getNamespace("xsd").isEmpty) model.setNamespace("xsd", XSD.NAMESPACE)
    }
}
