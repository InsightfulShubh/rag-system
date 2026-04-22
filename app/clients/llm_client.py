from openai import OpenAI, AzureOpenAI
from app.config import settings


def get_client() -> OpenAI:
    """Return an OpenAI or AzureOpenAI client based on USE_AZURE_OPENAI env var."""
    if settings.use_azure_openai:
        if not settings.azure_api_key:
            raise ValueError("USE_AZURE_OPENAI=true but AZURE_API_KEY is not set")
        return AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_endpoint,
        )

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_url,
    )


__all__ = ["get_client"]
