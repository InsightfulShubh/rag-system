import json as _json

from app.storage import db
from app.clients.llm_client import get_client
from app.services.tools import SEARCH_KB_TOOL, execute_tool_call
from app.config import settings


class ChatService:

    def chat(self, session_id: int, message: str) -> dict:
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

        history = db.get_messages(session_id, limit=settings.history_limit)

        messages = [{"role": "system", "content": settings.system_prompt}]
        for row in history:
            messages.append({"role": row["role"], "content": row["content"]})

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

        while assistant_msg.tool_calls:
            messages.append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                tool_result_json = execute_tool_call(tool_call)
                tool_result = _json.loads(tool_result_json)
                sources.extend(tool_result.get("sources", []))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_json,
                    }
                )

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[SEARCH_KB_TOOL],
                tool_choice="none",
                temperature=settings.llm_temperature,
            )
            assistant_msg = response.choices[0].message

        answer = assistant_msg.content or ""
        db.save_message(session_id, "assistant", answer)

        seen: set[str] = set()
        unique_sources = [s for s in sources if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

        return {"answer": answer, "sources": unique_sources}
