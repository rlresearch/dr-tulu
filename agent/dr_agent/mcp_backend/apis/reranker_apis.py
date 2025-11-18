from typing import Any, Dict, List

import cohere
from pydantic import BaseModel

from ..cache import cached


class Document(BaseModel):
    """Document model containing the text content."""

    text: str


class RerankResultItem(BaseModel):
    """Reranker result item containing index, relevance score, and document."""

    index: int
    relevance_score: float
    document: Document


class RerankerResult(BaseModel):
    """Reranker result containing a list of RerankResultItem objects."""

    method: str
    model_name: str
    results: List[RerankResultItem]


def vllm_hosted_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
) -> List[RerankResultItem]:
    """
    VLLM hosted reranker using Cohere client API format.

    Args:
        query: The query string to rank documents against
        documents: List of document strings to rerank
        top_n: Number of top documents to return (-1 returns all)
        model_name: Name of the reranker model
        api_url: Base URL for the VLLM reranker API

    Returns:
        List of RerankResultItem objects with index, relevance_score, and document
    """

    if top_n == -1:
        top_n = len(documents)

    # Initialize Cohere client with fake key for VLLM hosted service
    client = cohere.ClientV2("sk-fake-key", base_url=api_url)

    # Call the rerank endpoint
    rerank_result = client.rerank(model=model_name, query=query, documents=documents)

    # Sort results by relevance score in descending order
    sorted_results = sorted(
        rerank_result.results, key=lambda x: x.relevance_score, reverse=True
    )

    # Create RerankResultItem objects for top N results
    top_results = [
        RerankResultItem(
            index=result.index,
            relevance_score=result.relevance_score,
            document=Document(text=result.document["text"]),
        )
        for result in sorted_results[:top_n]
    ]

    return RerankerResult(
        method="vllm_hosted",
        model_name=model_name,
        results=top_results,
    )
