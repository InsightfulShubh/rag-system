from app.config import settings
from app.clients import get_client

client = get_client()
_llm_model = settings.azure_deployment_name if settings.use_azure_openai else settings.llm_model


class LLMService:
    """Wraps OpenAI chat completions for grounded RAG answers."""

    def generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        """Build a grounded prompt from retrieved chunks and return the LLM answer."""
        messages = self._build_messages(query, context_chunks)

        response = client.chat.completions.create(
            model=_llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
        )

        return response.choices[0].message.content

    def _build_messages(self, query: str, context_chunks: list[dict]) -> list[dict]:
        """Construct the [system, user] messages list for the chat completions API."""
        context_block = "\n\n".join(
            f"[Source: {chunk['file_name']}]\n{chunk['text']}"
            for chunk in context_chunks
        )

        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions strictly based on "
                "the provided context. "
                "If the answer cannot be found in the context, say: "
                "'I don't have enough information in the knowledge base to answer this question.' "
                "Do not make up information or use knowledge outside the provided context."
            ),
        }

        user_message = {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Question: {query}"
            ),
        }

        return [system_message, user_message]
