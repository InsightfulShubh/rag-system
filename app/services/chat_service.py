"""
Chat service — orchestrates the full conversational RAG loop.

Responsibilities:
    1. Save the user message to the DB
    2. Load the last N messages (HISTORY_LIMIT = 6) from the DB
    3. Send history + user message to LLM with the search_kb tool definition
    4. If LLM calls the tool → execute search_kb → send result back to LLM
    5. LLM produces a final answer
    6. Save the assistant message to the DB
    7. Return answer + sources
"""

from app.storage import db
from app.clients.llm_client import get_client
from app.services.tools import SEARCH_KB_TOOL, execute_tool_call
from app.config import settings




class ChatService:

    def chat(self, session_id: str, message: str) -> dict:
        """
        Process one user message and return the assistant's reply.

        Flow:
            1. Persist the user message.
            2. Load the last HISTORY_LIMIT messages as conversation context.
            3. Call LLM (with search_kb tool available).
            4. If the LLM issues a tool call, execute it and send the result
               back to the LLM for a final answer.
            5. Persist the assistant reply.
            6. Return {answer, sources}.

        Args:
            session_id: the active conversation session
            message:    the user's text input

        Returns:
            dict with keys:
                - answer  (str):        LLM-generated answer
                - sources (list[str]):  source files used (empty if tool was not called)
        """
        # ── 1. Persist the user message ───────────────────────────────────
        db.save_message(session_id, "user", message)

        # ── 2. Load conversation history (oldest → newest) ────────────────
        history = db.get_messages(session_id, limit=settings.history_limit)

        # ── 3. Build the messages list for the LLM ────────────────────────
        # System prompt comes first, then the conversation history.
        # The current user message is already the last item in history,
        # so we do NOT append it again.
        messages = [{"role": "system", "content": settings.system_prompt}]
        for row in history:
            messages.append({"role": row["role"], "content": row["content"]})

        # ── 4. First LLM call (tool definitions provided) ─────────────────
        model = (
            settings.azure_deployment_name
            if settings.use_azure_openai
            else settings.llm_model
        )
        client = get_client()

        tool_choice = (
            {"type": "function", "function": {"name": "search_kb"}}
            if settings.force_tool_usage
            else "auto"
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[SEARCH_KB_TOOL],
            tool_choice=tool_choice,
            temperature=settings.llm_temperature,
        )

        assistant_msg = response.choices[0].message
        sources: list[str] = []

        # ── 5. Tool-calling loop ───────────────────────────────────────────
        # The LLM may chain multiple tool calls, so we loop until it stops.
        while assistant_msg.tool_calls:
            # Add the assistant's tool-call request to the message thread
            messages.append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                # Execute the tool and collect sources from the result
                tool_result_json = execute_tool_call(tool_call)

                import json as _json
                tool_result = _json.loads(tool_result_json)
                sources.extend(tool_result.get("sources", []))

                # Feed the tool result back as a "tool" role message
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_json,
                    }
                )

            # messages
            # [system, user, assistant(tool_call), tool(result)]

            # Ask the LLM to continue with the tool results in context
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[SEARCH_KB_TOOL],
                tool_choice="none",
                temperature=settings.llm_temperature,
            )
            assistant_msg = response.choices[0].message

        # ── 6. Extract the final text answer ──────────────────────────────
        answer = assistant_msg.content or ""

        # ── 7. Persist the assistant reply ────────────────────────────────
        db.save_message(session_id, "assistant", answer)

        # Deduplicate sources while preserving order
        seen: set[str] = set()
        unique_sources = [s for s in sources if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

        return {"answer": answer, "sources": unique_sources}
