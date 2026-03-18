"""OpenAI embedding wrapper for creating text embeddings."""

from openai import OpenAI


class OpenAIEmbedder:
    """Wrapper around OpenAI's embedding API."""

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536):
        self.model = model
        self.dimensions = dimensions
        self.client = OpenAI()

    def embed_text(self, text: str) -> list[float]:
        """Create an embedding for a single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for multiple texts in a single API call."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [item.embedding for item in response.data]
