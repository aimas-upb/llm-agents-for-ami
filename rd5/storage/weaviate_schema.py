"""Weaviate schema definitions for RD5 plan storage.

This module defines the schema for storing execution plans
with semantic search capabilities.
"""

from typing import Any, Dict, List

# ExecutionPlan class schema for Weaviate
EXECUTION_PLAN_SCHEMA: Dict[str, Any] = {
    "class": "ExecutionPlan",
    "description": "Stores generated execution plans with semantic search support",
    "vectorizer": "text2vec-transformers",
    "moduleConfig": {
        "text2vec-transformers": {
            "poolingStrategy": "masked_mean",
            "vectorizeClassName": False,
        }
    },
    "properties": [
        {
            "name": "plan_id",
            "dataType": ["text"],
            "description": "Unique identifier for the plan",
            "moduleConfig": {
                "text2vec-transformers": {"skip": True}
            },
        },
        {
            "name": "intent_text",
            "dataType": ["text"],
            "description": "Natural language intent that triggered the plan",
            "moduleConfig": {
                "text2vec-transformers": {"skip": False}
            },
        },
        {
            "name": "extracted_intent",
            "dataType": ["text"],
            "description": "Structured/cleaned intent extracted by LLM",
            "moduleConfig": {
                "text2vec-transformers": {"skip": False}
            },
        },
        {
            "name": "plan_code",
            "dataType": ["text"],
            "description": "Generated Python code for the plan",
            "moduleConfig": {
                "text2vec-transformers": {"skip": True}
            },
        },
        {
            "name": "plan_explanation",
            "dataType": ["text"],
            "description": "Human-readable explanation of what the plan does",
            "moduleConfig": {
                "text2vec-transformers": {"skip": False}
            },
        },
        {
            "name": "affordance_uris",
            "dataType": ["text[]"],
            "description": "URIs of affordances used in the plan",
            "moduleConfig": {
                "text2vec-transformers": {"skip": True}
            },
        },
        {
            "name": "device_types",
            "dataType": ["text[]"],
            "description": "Types of devices involved in the plan",
            "moduleConfig": {
                "text2vec-transformers": {"skip": True}
            },
        },
        {
            "name": "success_count",
            "dataType": ["int"],
            "description": "Number of successful executions",
        },
        {
            "name": "failure_count",
            "dataType": ["int"],
            "description": "Number of failed executions",
        },
        {
            "name": "created_at",
            "dataType": ["date"],
            "description": "Timestamp when plan was created",
        },
        {
            "name": "last_executed_at",
            "dataType": ["date"],
            "description": "Timestamp of last execution",
        },
        {
            "name": "execution_context",
            "dataType": ["text"],
            "description": "JSON string with execution context/parameters",
            "moduleConfig": {
                "text2vec-transformers": {"skip": True}
            },
        },
    ],
}


def get_all_schemas() -> List[Dict[str, Any]]:
    """Get all schema definitions for RD5.

    Returns:
        List of Weaviate class schemas.
    """
    return [EXECUTION_PLAN_SCHEMA]
