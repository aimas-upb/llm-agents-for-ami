package controller

import com.fasterxml.jackson.annotation.JsonIgnoreProperties

@JsonIgnoreProperties(ignoreUnknown = true)
data class UpdatePayload(
    val artifactUri: String,
    val propertyUri: String,
    val value: Any,
    val valueTypeUri: String,
    val timestamp: String,
    val triggerUri: String
)