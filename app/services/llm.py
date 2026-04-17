from app.config import settings
from app.clients import get_client

# OpenAI/Azure client — initialized from settings via factory function
# Which client is used depends on USE_AZURE_OPENAI flag in .env
client = get_client()

# Pick the correct LLM model/deployment name based on provider
_llm_model = settings.azure_deployment_name if settings.use_azure_openai else settings.llm_model


class LLMService:
    """
    Handles all OpenAI chat completion interactions.

    Responsibility:
        - Build a structured prompt from the query + retrieved chunks
        - Call gpt-4o-mini (or configured model) via OpenAI chat completions API
        - Return the raw answer string

    Prompt strategy:
        - System message:  instructs the model to answer ONLY from provided context
                           and say "I don't know" if the answer isn't in the context
                           (prevents hallucination / making up answers)
        - User message:    injects context chunks + the actual question
    """

    def generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        """
        Build a prompt from the query and retrieved chunks, then call the LLM.

        Args:
            query: the original user question
            context_chunks: list of chunk dicts with keys:
                - 'text' (str): chunk content to use as context
                - 'file_name' (str): source file name (shown in prompt for attribution)

        Returns:
            str: the LLM-generated answer
        """
        messages = self._build_messages(query, context_chunks)

        response = client.chat.completions.create(
            model=_llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
        )

        # response.choices[0].message.content is the assistant's reply text
        return response.choices[0].message.content

    def _build_messages(self, query: str, context_chunks: list[dict]) -> list[dict]:
        """
        Construct the messages list for the OpenAI chat completions API.

        OpenAI chat completions expects:
            [
                {"role": "system",    "content": "...instructions..."},
                {"role": "user",      "content": "...question + context..."},
            ]

        Why separate system and user messages:
            - System message sets the model's behaviour (grounding, tone, rules)
            - User message contains the actual data (context + question)
            - This is the standard pattern for RAG prompting

        Why temperature=0 in generate_answer:
            - RAG answers should be factual and consistent
            - Higher temperature adds creativity/randomness — undesirable here
            - Same query on the same KB should always produce the same answer

        Args:
            query: the user's question
            context_chunks: top-K retrieved chunks from Stage 2

        Returns:
            list of message dicts ready for client.chat.completions.create()
        """
        # Build context block — each chunk labelled with its source file
        # Format:
        #   [Source: report.txt]
        #   <chunk text>
        #
        #   [Source: policy.txt]
        #   <chunk text>
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
