"""
LLM Client Factory

Provides a unified `get_client()` function that returns either:
  - OpenAI client (default)
  - AzureOpenAI client (DIAL from EPAM) when enabled via USE_AZURE_OPENAI flag

The actual client type is determined by settings.use_azure_openai.
This allows switching providers via .env without code changes.
"""

from openai import OpenAI
from openai import AzureOpenAI
from app.config import settings


def get_client():
    """
    Factory function to get the appropriate LLM client.
    
    Returns:
        OpenAI or AzureOpenAI: LLM client configured based on settings.use_azure_openai flag
    
    Raises:
        ValueError: If Azure is enabled but azure_api_key is not configured
    """
    if settings.use_azure_openai:
        # Use Azure OpenAI (DIAL)
        if not settings.azure_api_key:
            raise ValueError(
                "Azure OpenAI is enabled (USE_AZURE_OPENAI=true) but AZURE_API_KEY is not set in .env"
            )
        
        print("🔷 Using Azure OpenAI (DIAL) client")
        return AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_endpoint,
        )
    else:
        # Use standard OpenAI
        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI is enabled but OPENAI_API_KEY is not set in .env"
            )
        
        print("🔵 Using OpenAI client")
        return OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_url,
        )


__all__ = ["get_client"]
