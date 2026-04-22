"""Tool definition and execution for OpenAI function calling."""

import json
from app.services.retrieval import RetrievalService

_retrieval_service = RetrievalService()


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


def search_kb(query: str) -> dict:
    """Run two-stage RAG retrieval (no LLM). Returns {context, sources}."""
    return _retrieval_service.search(query)


def execute_tool_call(tool_call) -> str:
    """Dispatch an LLM tool_call to the matching Python function. Returns JSON string."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "search_kb":
        result = search_kb(args["query"])
        return json.dumps(result)

    raise ValueError(f"Unknown tool: {name}")
