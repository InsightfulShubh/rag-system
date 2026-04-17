from app.config import settings
from app.clients import get_client

# OpenAI/Azure client is initialized once at module level.
# Which client is used depends on USE_AZURE_OPENAI flag in .env
client = get_client()

# Pick the correct embedding model name based on provider
_embedding_model = settings.azure_embedding_model if settings.use_azure_openai else settings.embedding_model


def get_embedding(text: str) -> list[float]:
    """
    Generate a single embedding vector for the given text using OpenAI API.
    Used during the query pipeline to embed the user's question.

    How it works:
        - Sends the text to OpenAI's embeddings endpoint
        - Returns a vector of 1536 floats (for text-embedding-3-small)
        - Each float represents a dimension of semantic meaning
        - Similar texts produce vectors pointing in similar directions

    Args:
        text: input text to embed (e.g., a user query)

    Returns:
        list[float]: embedding vector of length 1536
    """
    response = client.embeddings.create(
        model=_embedding_model,
        input=text,
    )
    # response.data is a list; [0] gets the first (and only) embedding result
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embedding vectors for a list of texts in a single API call.
    Used during ingestion to embed all chunks of a file at once.

    Why batch instead of calling get_embedding() in a loop:
        - 1 API call instead of N calls → faster, fewer round trips
        - Lower latency for large files
        - OpenAI supports up to 2048 inputs per batch call

    Args:
        texts: list of text chunks to embed (e.g., all chunks of one file)

    Returns:
        list[list[float]]: list of embedding vectors, same order as input texts

    Example:
        texts = ["chunk 1 text", "chunk 2 text", "chunk 3 text"]
        embeddings = get_embeddings_batch(texts)
        # embeddings[0] → vector for "chunk 1 text"
        # embeddings[1] → vector for "chunk 2 text"
        # embeddings[2] → vector for "chunk 3 text"
    """
    if not texts:
        return []

    response = client.embeddings.create(
        model=_embedding_model,
        input=texts,
    )
    # response.data is sorted by index, preserving input order
    return [item.embedding for item in response.data]
