"""
Massive-Serve API Client

This module provides a client interface for the massive-serve search API, following
the same patterns as other APIs in this package (e.g., serper_apis.py).

The massive-serve API provides document retrieval capabilities using dense passage
retrieval with various embedding models and indices.

Basic Usage:
    from dr_agent.mcp_backend.apis.massive_serve_apis import search_massive_serve, parse_massive_serve_results

    # Search for documents
    response = search_massive_serve(
        query="What is machine learning?",
        n_docs=5,
        domains="dpr_wiki_contriever_ivfpq"
    )

    # Parse results into structured format
    results = parse_massive_serve_results(response)

    for result in results:
        print(f"Score: {result.score}")
        print(f"Passage: {result.passage}")
        print(f"Doc ID: {result.doc_id}")

Advanced Usage:
    # Use cached version to avoid repeated API calls
    from dr_agent.mcp_backend.apis.massive_serve_apis import search_massive_serve_cached

    response = search_massive_serve_cached(
        query="Einstein relativity theory",
        n_docs=10,
        domains="dpr_wiki_contriever_ivfpq",
        base_url="custom-server:8080",  # Override default server
        nprobe=128,  # Custom search parameters
        timeout=60   # Custom timeout
    )

Configuration:
    - Default server: rulin@h200-082-157:40991
    - Default timeout: 30 seconds (configurable via API_TIMEOUT env var)
    - Default domain: dpr_wiki_contriever_ivfpq

Environment Variables:
    - API_TIMEOUT: Request timeout in seconds (default: 30)
"""

import json
import os
from typing import Dict, List, Optional, Union

import dotenv
import requests
from pydantic import BaseModel
from typing_extensions import TypedDict

from ..cache import cached

# Load environment variables
dotenv.load_dotenv()

# Default configuration - can be overridden
DEFAULT_MASSIVE_SERVE_BASE_URL = "rulin@h200-082-157:40991"
TIMEOUT = int(os.getenv("API_TIMEOUT", 30))


class MassiveServeSearchResult(BaseModel):
    """Individual search result from massive-serve API."""

    passage: str
    score: float
    doc_id: List[int]


class MassiveServeResponse(TypedDict, total=False):
    """Response structure from massive-serve search API."""

    message: str
    query: str
    n_docs: int
    nprobe: Optional[int]
    results: Dict[str, Union[List[List[List[int]]], List[List[str]], List[List[float]]]]


def search_massive_serve(
    query: str,
    n_docs: int = 10,
    domains: str = "dpr_wiki_contriever_ivfpq",
    base_url: str = None,
    nprobe: Optional[int] = None,
    timeout: int = TIMEOUT,
) -> MassiveServeResponse:
    """
    Search using the massive-serve API for document retrieval.

    Args:
        query: Search query string
        n_docs: Number of documents to return (default: 10)
        domains: Domain/index to search in (default: "dpr_wiki_contriever_ivfpq")
        base_url: Base URL for the massive-serve API (default: uses DEFAULT_MASSIVE_SERVE_BASE_URL)
        nprobe: Number of probes for search (optional)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        MassiveServeResponse containing:
        - message: Status message
        - query: The original search query
        - n_docs: Number of documents requested
        - nprobe: Number of probes used (if specified)
        - results: Dictionary containing:
            - IDs: List of document IDs
            - passages: List of retrieved text passages
            - scores: List of relevance scores

    Raises:
        ValueError: If required parameters are missing
        Exception: If API request fails or returns an error
    """
    if not query:
        raise ValueError("Query parameter is required")

    if not base_url:
        base_url = DEFAULT_MASSIVE_SERVE_BASE_URL

    # Construct the full URL
    url = f"http://{base_url}/search"

    # Prepare the payload
    payload = {
        "query": query,
        "n_docs": n_docs,
        "domains": domains,
    }

    if nprobe is not None:
        payload["nprobe"] = nprobe

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=timeout
        )

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing massive-serve search: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing API response: {str(e)}")


def parse_massive_serve_results(
    response: MassiveServeResponse,
) -> List[MassiveServeSearchResult]:
    """
    Parse the raw massive-serve response into structured search results.

    Args:
        response: Raw response from massive-serve API

    Returns:
        List of MassiveServeSearchResult objects with parsed data

    Raises:
        KeyError: If response structure is unexpected
        ValueError: If response data is malformed
    """
    try:
        results = response["results"]
        ids = results["IDs"][0] if results["IDs"] else []
        passages = results["passages"][0] if results["passages"] else []
        scores = results["scores"][0] if results["scores"] else []

        if len(ids) != len(passages) or len(passages) != len(scores):
            raise ValueError("Mismatched lengths in API response arrays")

        parsed_results = []
        for doc_id, passage, score in zip(ids, passages, scores):
            parsed_results.append(
                MassiveServeSearchResult(passage=passage, score=score, doc_id=doc_id)
            )

        return parsed_results

    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Error parsing massive-serve response structure: {str(e)}")


@cached()
def search_massive_serve_cached(
    query: str,
    n_docs: int = 10,
    domains: str = "dpr_wiki_contriever_ivfpq",
    base_url: str = None,
    nprobe: Optional[int] = None,
    timeout: int = TIMEOUT,
) -> MassiveServeResponse:
    """
    Cached version of search_massive_serve function.

    Same parameters and return type as search_massive_serve, but results are cached
    to avoid repeated API calls for identical queries.
    """
    return search_massive_serve(
        query=query,
        n_docs=n_docs,
        domains=domains,
        base_url=base_url,
        nprobe=nprobe,
        timeout=timeout,
    )


# Example usage and testing
if __name__ == "__main__":
    try:
        # Test the search functionality
        print("Testing massive-serve search API...")

        results = search_massive_serve(
            query="Tell me more about the stories of Einstein.",
            n_docs=3,
            domains="dpr_wiki_contriever_ivfpq",
        )

        print(f"Raw API Response:")
        print(f"Message: {results['message']}")
        print(f"Query: {results['query']}")
        print(f"Number of docs: {results['n_docs']}")
        print()

        # Parse and display structured results
        parsed_results = parse_massive_serve_results(results)
        print(f"Found {len(parsed_results)} results:")

        for i, result in enumerate(parsed_results, 1):
            print(f"\nResult {i}:")
            print(f"  Score: {result.score:.4f}")
            print(f"  Doc ID: {result.doc_id}")
            print(f"  Passage: {result.passage[:200]}...")

    except Exception as e:
        print(f"Error testing massive-serve API: {e}")
