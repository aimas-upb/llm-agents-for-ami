"""Code generation prompt template.

This prompt instructs the LLM to generate safe Python code
for executing smart home plans.
"""

CODE_GENERATION_PROMPT = """You are a code generation system for a smart home AI assistant.

Your task is to generate Python code that executes the user's intent by calling device affordances.

INTENT: {intent}

AVAILABLE AFFORDANCES:
{affordances}
{similar_plans}

SAFETY REQUIREMENTS - YOUR CODE MUST:
1. ONLY use these allowed imports: aiohttp, json, datetime, math, networkx, asyncio, typing, re, collections, itertools, functools
2. NEVER use: os, sys, subprocess, eval, exec, open, file, socket, requests, globals, locals, getattr
3. Use aiohttp for all HTTP calls to device endpoints
4. Handle errors gracefully with try/except
5. Include type hints for function parameters
6. Be self-contained and executable

CODE STRUCTURE:
- Define an async main() function
- Use aiohttp.ClientSession for HTTP requests
- Print status messages for debugging
- Return a result dictionary with success status

TEMPLATE:
```python
import asyncio
import json
from typing import Dict, Any
import aiohttp

async def call_affordance(session: aiohttp.ClientSession, url: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    \"\"\"Call a device affordance endpoint.\"\"\"
    try:
        if method.upper() == "GET":
            async with session.get(url) as response:
                return {{"success": response.status < 400, "data": await response.json()}}
        elif method.upper() == "POST":
            async with session.post(url, json=data) as response:
                return {{"success": response.status < 400, "data": await response.json()}}
        elif method.upper() == "PUT":
            async with session.put(url, json=data) as response:
                return {{"success": response.status < 400, "data": await response.json()}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}

async def main() -> Dict[str, Any]:
    \"\"\"Execute the plan.\"\"\"
    results = []

    async with aiohttp.ClientSession() as session:
        # YOUR CODE HERE - call affordances to achieve the intent
        pass

    return {{
        "success": all(r.get("success", False) for r in results),
        "results": results,
        "message": "Plan executed"
    }}

if __name__ == "__main__":
    result = asyncio.run(main())
    print(json.dumps(result, indent=2))
```
{retry_feedback}

Generate the complete Python code to achieve: {intent}

Return ONLY the Python code, wrapped in ```python code blocks."""
