from app.config import settings


def split_into_chunks(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> list[str]:
    """Split text into overlapping chunks. Defaults from config (CHUNK_SIZE, CHUNK_OVERLAP)."""
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    if not text or chunk_size <= 0:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == len(text):
            break

    return chunks
