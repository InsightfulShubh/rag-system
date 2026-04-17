from app.config import settings


def split_into_chunks(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_size characters.

    chunk_size and overlap default to values from config/env (CHUNK_SIZE, CHUNK_OVERLAP)
    so they can be tuned without code changes.

    This is a sliding window approach:
    - Move forward by (chunk_size - overlap) each time
    - This creates overlap between consecutive chunks

    Args:
        text: full document text
        chunk_size: target size of each chunk in characters (default: settings.chunk_size)
        overlap: characters to overlap between consecutive chunks (default: settings.chunk_overlap)

    Returns:
        list[str]: list of text chunks

    Why overlap matters for RAG:
    - Prevents important semantic information from being split across boundaries
    - Without overlap: "The defendant was found" | "guilty of theft"
    - With overlap:    "The defendant was found" | "was found guilty of theft"
      → "was found" bridge preserves context across the boundary
    """
    # Use config defaults if not explicitly overridden by caller
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    if not text or chunk_size <= 0:
        return []

    chunks = []
    step = chunk_size - overlap  # advance by this many chars each iteration

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == len(text):
            break

    return chunks
