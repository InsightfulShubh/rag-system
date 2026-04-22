import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


class Settings:
    use_azure_openai: bool = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_api_url: str = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    # Azure OpenAI (DIAL)
    azure_api_key: str = os.getenv("AZURE_API_KEY", "")
    azure_endpoint: str = os.getenv("AZURE_ENDPOINT", "https://ai-proxy.lab.epam.com")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-02-01")
    azure_deployment_name: str = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4")
    azure_embedding_model: str = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-small-1")

    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    history_limit: int = int(os.getenv("HISTORY_LIMIT", "6"))
    force_tool_usage: bool = os.getenv("FORCE_TOOL_USAGE", "true").lower() == "true"
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful assistant with access to a knowledge base.\n"
        "- Use the search_kb tool whenever the user's question requires external knowledge from documents.\n"
        "- If the answer is not found in the context returned by search_kb, say \"I don't know\".\n"
        "- Be concise and accurate.",
    )


settings = Settings()
