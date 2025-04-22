package org.eclipse.lmos.arc.app.config

import org.eclipse.lmos.arc.agents.dsl.get
import org.eclipse.lmos.arc.agents.dsl.types // For defining function parameters
import org.eclipse.lmos.arc.agents.dsl.string // For defining string parameters
import org.eclipse.lmos.arc.agents.functions.LLMFunction
import org.eclipse.lmos.arc.spring.Functions // The helper bean
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.time.LocalTime // Example service dependency

@Configuration
open class ToolConfiguration {

    // Example service that the function might use
    @Bean
    open fun timeService(): TimeService = TimeServiceImpl()

    @Bean
    open fun getTimeFunction(functions: Functions): List<LLMFunction> { // Inject the helper
        // The 'functions' helper returns a List<LLMFunction>
        return functions(
            name = "get_current_time", // The name LLM will use to call the tool
            description = "Gets the current time.",
            params = types( // Define parameters using helpers
                string("timezone", "Optional timezone, e.g., 'Europe/Berlin'")
            ),
            isSensitive = false // Mark true if output might be sensitive
        ) { params ->
            // Implementation lambda. 'this' is DSLContext.
            // 'params' is a List<Any?> corresponding to the 'types' definition.
            val timezone = params[0] as? String // Access parameters by index

            // Access other Spring beans using get<>() from DSLContext
            val timeService = get<TimeService>()
            val currentTime = timeService.getCurrentTime(timezone)

            // The return value is the string sent back to the LLM
            "The current time ${timezone?.let { "in $it " } ?: ""}is $currentTime."
        }
    }
}

// Example service definition
interface TimeService {
    fun getCurrentTime(timezone: String?): String
}

class TimeServiceImpl : TimeService {
    override fun getCurrentTime(timezone: String?): String {
        // Simplified example logic
        return LocalTime.now().toString()
    }
}