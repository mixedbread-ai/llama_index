import logging
from typing import Any, List, Optional

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from mixedbread_ai import TruncationStrategy
from mixedbread_ai.client import MixedbreadAI, AsyncMixedbreadAI
from mixedbread_ai.types import EncodingFormat

logger = logging.getLogger(__name__)


class MixedbreadAIEmbedding(BaseEmbedding):
    """Class for Mixedbread AI embeddings.

    Args:
        model_name (str): Model for embedding. Defaults to "mixedbread-ai/mxbai-embed-large-v1".
        api_key (Optional[str]): Mixedbread API key. Defaults to None.
        encoding_format (EncodingFormat): Encoding formats. Defaults to EncodingFormat.FLOAT.
        truncation_strategy (TruncationStrategy): Truncation strategy. Defaults to TruncationStrategy.START.
        normalized (bool): Whether to normalize the embeddings. Defaults to True.
        dimensions (Optional[int]): Number of dimensions for embeddings. Only applicable for Matryoshka-based models.
        prompt(Optional[str]): An optional prompt to provide context to the model.
    """

    # Instance variables initialized via Pydantic's mechanism
    api_key: str = Field(description="The Mixedbread AI API key.")
    model_name: str = Field(description="The model name to use for embedding.")
    encoding_format: EncodingFormat = Field(
        description="Encoding format for the embeddings."
    )
    truncation_strategy: TruncationStrategy = Field(
        description="Truncation strategy for input text."
    )
    normalized: bool = Field(
        default=True, description="Whether to normalize the embeddings."
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Number of dimensions for embeddings. Only applicable "
        "for Matryoshka-based models.",
    )
    prompt: Optional[str] = Field(
        default=None, description="An optional prompt to provide context to the model."
    )

    _client: MixedbreadAI = PrivateAttr()
    _async_client: AsyncMixedbreadAI = PrivateAttr()
    _timeout: Optional[float] = PrivateAttr()
    _httpx_client: Optional[httpx.Client] = PrivateAttr()
    _httpx_async_client: Optional[httpx.AsyncClient] = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        encoding_format: EncodingFormat = EncodingFormat.FLOAT,
        truncation_strategy: TruncationStrategy = TruncationStrategy.START,
        normalized: bool = True,
        dimensions: Optional[int] = None,
        prompt: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        timeout: Optional[float] = None,
        httpx_client: Optional[httpx.Client] = None,
        httpx_async_client: Optional[httpx.AsyncClient] = None,
        **kwargs: Any,
    ):
        if embed_batch_size is None:
            embed_batch_size = 32  # Default batch size for Mixedbread AI

        super().__init__(
            api_key=api_key,
            model_name=model_name,
            encoding_format=encoding_format,
            truncation_strategy=truncation_strategy,
            normalized=normalized,
            dimensions=dimensions,
            prompt=prompt,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

        self._client = MixedbreadAI(
            api_key=api_key, timeout=timeout, httpx_client=httpx_client
        )
        self._async_client = AsyncMixedbreadAI(
            api_key=api_key, timeout=timeout, httpx_client=httpx_async_client
        )

    @classmethod
    def class_name(cls) -> str:
        return "MixedbreadAIEmbedding"

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings.
        """
        response = self._client.embeddings(
            model=self.model_name,
            input=texts,
            encoding_format=self.encoding_format,
            normalized=self.normalized,
            truncation_strategy=self.truncation_strategy,
            dimensions=self.dimensions,
            prompt=self.prompt,
        )
        return [item.embedding for item in response.data]

    async def _aget_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously get embeddings for a list of texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings.
        """
        response = await self._async_client.embeddings(
            model=self.model_name,
            input=texts,
            encoding_format=self.encoding_format,
            normalized=self.normalized,
            truncation_strategy=self.truncation_strategy,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query.

        Args:
            query (str): Query text.

        Returns:
            List[float]: Embedding for the query.
        """
        return self._get_embedding([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        Asynchronously get embedding for a query.

        Args:
            query (str): Query text.

        Returns:
            List[float]: Embedding for the query.
        """
        r = await self._aget_embedding([query])
        return r[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: Embedding for the text.
        """
        return self._get_embedding([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        Asynchronously get embedding for a text.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: Embedding for the text.
        """
        r = await self._aget_embedding([text])
        return r[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings.
        """
        return self._get_embedding(texts)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously get embeddings for multiple texts.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings.
        """
        return await self._aget_embedding(texts)
