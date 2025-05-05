// SPDX-FileCopyrightText: 2025 Deutsche Telekom AG and others
//
// SPDX-License-Identifier: Apache-2.0


package org.eclipse.lmos.arc.app

import kotlinx.coroutines.runBlocking
import org.slf4j.LoggerFactory
import org.springframework.boot.CommandLineRunner
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication
import org.springframework.context.annotation.Bean

/**
 * Simple Spring Boot application that demonstrates how to use the Arc Agents and test YggdrasilClient.
 */
@SpringBootApplication(scanBasePackages = ["org.eclipse.lmos.arc.app", "client", "controller"])
class ArcAIApplication

fun main(args: Array<String>) {
    runApplication<ArcAIApplication>(*args)
}
