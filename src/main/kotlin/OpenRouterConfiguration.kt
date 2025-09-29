// SPDX-FileCopyrightText: 2025 Deutsche Telekom AG and others
//
// SPDX-License-Identifier: Apache-2.0

package org.eclipse.lmos.arc.app.config

import dev.langchain4j.model.chat.ChatLanguageModel
import dev.langchain4j.model.openai.OpenAiChatModel
import java.time.Duration
import org.eclipse.lmos.arc.spring.AIConfig
import org.springframework.beans.factory.config.BeanPostProcessor
import org.eclipse.lmos.arc.agents.llm.ChatCompletionSettings
import org.eclipse.lmos.arc.client.langchain4j.LangChainClient
import org.eclipse.lmos.arc.client.langchain4j.LangChainConfig
import org.eclipse.lmos.arc.spring.clients.ClientBuilder
import org.slf4j.LoggerFactory
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.stereotype.Component

private const val OPEN_ROUTER_APP_NAME = "ami:UserAssistantAgent"
private const val OPEN_ROUTER_APP_URL = "https://github.com/aimas-upb/llm-agents-for-ami"

@Configuration
open class OpenRouterConfiguration {

    private val log = LoggerFactory.getLogger(javaClass)

    @Bean
    open fun openRouterClientBuilder(): ClientBuilder = ClientBuilder { config, eventPublisher ->
        if (config.client != "openrouter" && config.client != "openai-sdk") {
            return@ClientBuilder null
        }

        val apiKey = sanitize(config.apiKey)
        if (apiKey.isNullOrBlank() || apiKey == "changeme-openrouter") {
            log.info("Skipping OpenRouter client '{}' because no valid API key was provided", config.id)
            return@ClientBuilder null
        }

        val baseUrl = sanitize(config.url) ?: "https://openrouter.ai/api/v1"
        val modelName = sanitize(config.modelName) ?: "openrouter/openai/gpt-4.1-mini"

        log.info(
            "Initializing OpenRouter client '{}' with model '{}' at {} (apiKey=obfuscated key)",
            config.id,
            modelName,
            baseUrl
        )

        val langChainConfig = LangChainConfig(modelName, baseUrl, apiKey, null, null)

        val chatModelSupplier: (LangChainConfig, ChatCompletionSettings?) -> ChatLanguageModel = { _, _ ->
            OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName(modelName)
                .baseUrl(baseUrl)
                .customHeaders(
                    mapOf(
                        "X-Title" to OPEN_ROUTER_APP_NAME,
                        "HTTP-Referer" to OPEN_ROUTER_APP_URL,
                        "Referer" to OPEN_ROUTER_APP_URL
                    )
                )
                .timeout(Duration.ofMinutes(5))
                .build()
        }

        LangChainClient(langChainConfig, chatModelSupplier, eventPublisher)
    }
}

@Component
open class OpenRouterAiConfigPostProcessor : BeanPostProcessor {

    override fun postProcessBeforeInitialization(bean: Any, beanName: String): Any {
        if (bean is AIConfig) {
            val filteredClients = bean.clients.filterNot { clientConfig ->
                val isOpenRouter = clientConfig.client == "openrouter" || clientConfig.client == "openai-sdk"
                val apiKey = clientConfig.apiKey
                isOpenRouter && (apiKey.isNullOrBlank() || apiKey == "changeme-openrouter")
            }
            if (filteredClients.size != bean.clients.size) {
                return bean.copy(clients = filteredClients)
            }
        }
        return bean
    }
}

private fun sanitize(raw: String?): String? {
    if (raw == null) return null
    val trimmed = raw.trim().trim('"', '\'', '“', '”')
    return trimmed.takeIf { it.isNotEmpty() }
}
