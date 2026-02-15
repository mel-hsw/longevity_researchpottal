"""Wrapper around OpenAI text-embedding-3-small."""

from langchain_openai import OpenAIEmbeddings

from src.config import Config


def get_embeddings_model(config: Config | None = None) -> OpenAIEmbeddings:
    """Return the configured OpenAI embeddings model."""
    config = config or Config()
    return OpenAIEmbeddings(
        model=config.embedding_model,
        openai_api_key=config.openai_api_key,
    )
