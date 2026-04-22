from app.config import settings
from app.clients import get_client

client = get_client()
_embedding_model = settings.azure_embedding_model if settings.use_azure_openai else settings.embedding_model


def get_embedding(text: str) -> list[float]:
    """Generate a single embedding vector for the given text."""
    response = client.embeddings.create(
        model=_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single API call. Returns embeddings in input order."""
    if not texts:
        return []

    response = client.embeddings.create(
        model=_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]
