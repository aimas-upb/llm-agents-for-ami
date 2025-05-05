package org.eclipse.lmos.arc.app.model

import org.eclipse.rdf4j.model.IRI
import org.eclipse.rdf4j.model.Model
import org.eclipse.rdf4j.model.impl.DynamicModelFactory
import java.time.Instant

data class ArtifactState(
    val artifactUri: IRI,
    var rdfModel: Model = DynamicModelFactory().createEmptyModel(),
    var lastUpdatedAt: Instant = Instant.now(),
    var lastTriggeringAffordanceUri: IRI? = null
) {
    val artifactId: String
        get() = artifactUri.localName
}