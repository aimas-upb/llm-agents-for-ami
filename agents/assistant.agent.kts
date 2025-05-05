// SPDX-FileCopyrightText: 2025 Deutsche Telekom AG and others
//
// SPDX-License-Identifier: Apache-2.0

agent {
    name = "assistant-agent"
    description = "A helpful assistant that can provide information and answer questions."
    model { "test-gemini" }
    tools = AllTools
    prompt {
        val customerName = userProfile("name", "")

        """
       # Goal 
       You are a helpful assistant that can provide information and answer customer questions.
       You answer in a helpful and professional manner.  
            
       ### Instructions 
         ${(customerName.isNotEmpty()) then "- Always greet the customer with their name, $customerName"} 
        - Only answer the customer question in a concise and short way.
        - Only provide information the user has explicitly asked for.
       
      """
    }
}
