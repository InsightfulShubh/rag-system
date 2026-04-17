"""
Tool definition and execution for OpenAI function calling.

Two things are needed for OpenAI tool calling:

1. SEARCH_KB_TOOL  — a JSON schema dict that describes the tool to the LLM.
   Sent in every chat.completions.create() call so the LLM knows it exists.

2. search_kb()     — the actual Python function that runs when the LLM
   decides to call the tool. Returns {"context": str, "sources": list}.

3. execute_tool_call() — dispatcher that routes any tool call from the LLM
   to the correct Python function. Right now only search_kb is supported.
"""

import json
from app.services.retrieval import RetrievalService

# Singleton — one instance shared across all requests
_retrieval_service = RetrievalService()


# ──────────────────────────────────────────────────────────────
# 1. OpenAI Tool Spec
#
# This JSON dict tells the LLM:
#   - The tool exists and is named "search_kb"
#   - What it does (description)
#   - What parameter(s) it expects (query: string, required)
#
# The LLM uses this to decide WHEN to call the tool and
# what argument to pass ("query").
# ──────────────────────────────────────────────────────────────

SEARCH_KB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": (
            "Search the knowledge base for relevant information. "
            "Call this tool whenever the user asks a question that requires "
            "external knowledge from documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents in the knowledge base.",
                }
            },
            "required": ["query"],
        },
    },
}


# ──────────────────────────────────────────────────────────────
# 2. Tool Executor
#
# Called by the LLM service after the LLM returns a tool_call.
# Runs the actual RAG retrieval (Stage 1 + Stage 2, no LLM).
# ──────────────────────────────────────────────────────────────

def search_kb(query: str) -> dict:
    """
    Execute the knowledge base search using the existing two-stage RAG logic.

    Args:
        query: search string from the LLM tool call arguments

    Returns:
        dict with keys:
            - context  (str):        formatted chunk texts joined together
            - sources  (list[str]):  source file names

    Example return value:
        {
            "context": "[Source: FastAPI_Framework.txt]\nFastAPI is...\n\n[Source: Python_Programming.txt]\nPython is...",
            "sources": ["FastAPI_Framework.txt", "Python_Programming.txt"]
        }
    """
    return _retrieval_service.search(query)


# ──────────────────────────────────────────────────────────────
# 3. Dispatcher
#
# When the LLM returns a tool_call object, it contains:
#   - tool_call.function.name      → "search_kb"
#   - tool_call.function.arguments → '{"query": "What is FastAPI?"}'
#
# execute_tool_call() parses the args and routes to the right function.
# Returns the result as a JSON string (required by OpenAI API).
# ──────────────────────────────────────────────────────────────

def execute_tool_call(tool_call) -> str:
    """
    Dispatch an OpenAI tool_call to the correct Python function.

    Args:
        tool_call: the tool_call object from response.choices[0].message.tool_calls[0]

    Returns:
        str: JSON-encoded result of the tool execution.
             This string is sent back to the LLM as a "tool" role message.

    Raises:
        ValueError: if an unknown tool name is requested
    """
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "search_kb":
        result = search_kb(args["query"])
        return json.dumps(result)

    raise ValueError(f"Unknown tool: {name}")
