// SPDX-FileCopyrightText: 2025 Deutsche Telekom AG and others
//
// SPDX-License-Identifier: Apache-2.0

plugins {
    kotlin("jvm") version "2.1.10"
    kotlin("plugin.serialization") version "2.1.10"
    kotlin("plugin.spring") version "2.1.10"
    id("org.springframework.boot") version "3.4.3"
    id("io.spring.dependency-management") version "1.1.7"
    //id("org.graalvm.buildtools.native") version "0.10.2"
}

group = "org.eclipse.lmos.app"

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

kotlin {
    compilerOptions {
        freeCompilerArgs.addAll("-Xjsr305=strict", "-Xcontext-receivers")
    }
}

dependencies {
    val arcVersion = "0.122.0-M2"
    val langchain4jVersion = "0.36.2"
    val kotlinXVersion = "1.8.1"
    val ktorVersion = "2.3.12"

    implementation("io.ktor:ktor-client-core:$ktorVersion")
    implementation("io.ktor:ktor-client-cio:$ktorVersion")
    implementation("io.ktor:ktor-client-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")

    // Arc
    implementation("org.eclipse.lmos:arc-langchain4j-client:$arcVersion")
    implementation("org.eclipse.lmos:arc-azure-client:$arcVersion")
    implementation("org.eclipse.lmos:arc-agents:$arcVersion")
    implementation("org.eclipse.lmos:arc-spring-boot-starter:$arcVersion")
    implementation("org.eclipse.lmos:arc-reader-pdf:$arcVersion")
    implementation("org.eclipse.lmos:arc-reader-html:$arcVersion")
    implementation("org.eclipse.lmos:arc-assistants:$arcVersion")
    implementation("org.eclipse.lmos:arc-reader-html:$arcVersion")
    implementation("org.eclipse.lmos:arc-api:$arcVersion")
    implementation("org.eclipse.lmos:arc-graphql-spring-boot-starter:$arcVersion")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-reactor:$kotlinXVersion")

    // Tracing
    implementation(platform("io.micrometer:micrometer-tracing-bom:1.4.5"))
    implementation("io.micrometer:micrometer-tracing")
    implementation("io.micrometer:micrometer-tracing-bridge-otel")
    implementation("io.opentelemetry:opentelemetry-exporter-otlp")

    // Azure
    implementation("com.azure:azure-identity:1.15.4")

    // Spring Boot
    implementation("org.springframework.boot:spring-boot-starter-actuator")
    implementation("org.springframework.boot:spring-boot-starter-webflux")

    // Langchain4j
    implementation("dev.langchain4j:langchain4j-bedrock:$langchain4jVersion")
    implementation("dev.langchain4j:langchain4j-google-ai-gemini:$langchain4jVersion")
    implementation("dev.langchain4j:langchain4j-ollama:$langchain4jVersion")
    implementation("dev.langchain4j:langchain4j-open-ai:$langchain4jVersion")

    // Metrics
    implementation("io.micrometer:micrometer-registry-prometheus")

    // Test
    testImplementation("org.springframework.boot:spring-boot-testcontainers")
    testImplementation("org.testcontainers:mongodb:1.20.6")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
}

repositories {
    mavenLocal()
    mavenCentral()
    maven(url = "https://oss.sonatype.org/content/repositories/snapshots/")
}
