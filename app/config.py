import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from the project root (parent of the app/ directory),
# regardless of which directory uvicorn is launched from.
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


class Settings:
    # LLM Provider Selection
    use_azure_openai: bool = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_api_url: str = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    # Azure OpenAI (DIAL) Configuration
    azure_api_key: str = os.getenv("AZURE_API_KEY", "")
    azure_endpoint: str = os.getenv("AZURE_ENDPOINT", "https://ai-proxy.lab.epam.com")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-02-01")
    azure_deployment_name: str = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4")
    azure_embedding_model: str = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-small-1")

    # LLM temperature:
    #   0.0 → deterministic, factual (recommended for RAG)  ← our default
    #   0.7 → creative, varies between calls
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Chunking settings:
    #   chunk_size    → target character length per chunk
    #   chunk_overlap → characters shared between consecutive chunks
    #                   overlap prevents losing context at chunk boundaries
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()
