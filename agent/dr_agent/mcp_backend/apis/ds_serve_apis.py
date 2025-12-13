"""
DS Serve API Client

This module provides a client interface for the DS Serve search API, following
the same patterns as other APIs in this package (e.g., massive_serve_apis.py).

The DS Serve API provides document retrieval capabilities using CompactDS (2B passages)
with dual ANN backends (DiskANN + IVFPQ).

Basic Usage:
    from dr_agent.mcp_backend.apis.ds_serve_apis import search_ds_serve, parse_ds_serve_results

    # Search for documents
    response = search_ds_serve(
        query="What is machine learning?",
        n_docs=5,
        backend="diskann"
    )

    # Parse results into structured format
    results = parse_ds_serve_results(response)

    for result in results:
        print(f"Score: {result.score}")
        print(f"Passage: {result.text}")
        print(f"Source: {result.source}")

Advanced Usage:
    # Use cached version to avoid repeated API calls
    from dr_agent.mcp_backend.apis.ds_serve_apis import search_ds_serve_cached

    response = search_ds_serve_cached(
        query="Einstein relativity theory",
        n_docs=10,
        backend="ivfpq",
        nprobe=256,
        timeout=60
    )

Configuration:
    - Default server: api.ds-serve.org:30888
    - Default timeout: 30 seconds (configurable via API_TIMEOUT env var)
    - Default backend: diskann

Environment Variables:
    - API_TIMEOUT: Request timeout in seconds (default: 30)
    - DS_SERVE_BASE_URL: Base URL for DS Serve API (default: api.ds-serve.org:30888)
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
DEFAULT_DS_SERVE_BASE_URL = "api.ds-serve.org:30888"
TIMEOUT = int(os.getenv("API_TIMEOUT", 30))


class DSServeSearchResult(BaseModel):
    """Individual search result from DS Serve API."""

    text: str
    center_text: str
    score: float
    filename: str
    index_id: int
    passage_id: int
    position: int
    source: str
    raw_query: str


class DSServeResponse(TypedDict, total=False):
    """Response structure from DS Serve search API."""

    message: str
    query: str
    n_docs: int
    nprobe: Optional[int]
    expand_index_id: Optional[int]
    expand_offset: Optional[int]
    results: Dict[str, Union[List, Dict]]


def search_ds_serve(
    query: str,
    n_docs: int = 10,
    backend: str = "diskann",
    base_url: str = None,
    nprobe: Optional[int] = None,
    exact_search: bool = False,
    diverse_search: bool = False,
    lambda_param: float = 0.5,
    diskann_L: int = 500,
    diskann_W: int = 8,
    diskann_threads: Optional[int] = None,
    min_words: int = 10,
    timeout: int = TIMEOUT,
) -> DSServeResponse:
    """
    Search using the DS Serve API for document retrieval.

    Args:
        query: Search query string
        n_docs: Number of documents to return (default: 10, max: 1000)
        backend: Backend to use - "diskann" or "ivfpq" (default: "diskann")
        base_url: Base URL for the DS Serve API (default: uses DEFAULT_DS_SERVE_BASE_URL)
        nprobe: Number of IVFPQ clusters to scan (ignored for DiskANN)
        exact_search: Brute-force rerank after ANN (default: False)
        diverse_search: Penalize near-duplicate passages (default: False)
        lambda_param: Diversity tradeoff used with diverse_search (default: 0.5)
        diskann_L: DiskANN candidate list size (>= n_docs, default: 500)
        diskann_W: DiskANN beam width / I/O fan-out (default: 8)
        diskann_threads: Override worker thread count (default: None, uses server default)
        min_words: Minimum passage length filter (default: 10)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        DSServeResponse containing:
        - message: Status message
        - query: The original search query
        - n_docs: Number of documents requested
        - nprobe: Number of probes used (if specified)
        - results: Dictionary containing:
            - passages: List of lists with passage objects
            - scores: List of lists with relevance scores
            - backend_timings_ms: Backend-specific timing information
            - timings_ms: Overall timing information

    Raises:
        ValueError: If required parameters are missing or invalid
        Exception: If API request fails or returns an error
    """
    if not query:
        raise ValueError("Query parameter is required")

    if backend not in ["diskann", "ivfpq"]:
        raise ValueError(f"Backend must be 'diskann' or 'ivfpq', got '{backend}'")

    if not base_url:
        base_url = DEFAULT_DS_SERVE_BASE_URL

    # Construct the full URL
    url = f"http://{base_url}/search"

    # Prepare the payload
    payload = {
        "query": query,
        "n_docs": n_docs,
        "backend": backend,
    }

    # Add optional parameters
    if nprobe is not None:
        payload["nprobe"] = nprobe
    if exact_search:
        payload["exact_search"] = exact_search
    if diverse_search:
        payload["diverse_search"] = diverse_search
        payload["lambda"] = lambda_param
    if backend == "diskann":
        payload["diskann_L"] = diskann_L
        payload["diskann_W"] = diskann_W
        if diskann_threads is not None:
            payload["diskann_threads"] = diskann_threads
        payload["min_words"] = min_words

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
        raise Exception(f"Error performing DS Serve search: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing API response: {str(e)}")


def parse_ds_serve_results(
    response: DSServeResponse,
) -> List[DSServeSearchResult]:
    """
    Parse the raw DS Serve response into structured search results.

    Args:
        response: Raw response from DS Serve API

    Returns:
        List of DSServeSearchResult objects with parsed data

    Raises:
        KeyError: If response structure is unexpected
        ValueError: If response data is malformed
    """
    try:
        results = response["results"]
        passages_list = results.get("passages", [])
        scores_list = results.get("scores", [])

        if not passages_list or not scores_list:
            return []

        # Passages and scores are nested lists (one query -> list of passages/scores)
        passages = passages_list[0] if passages_list else []
        scores = scores_list[0] if scores_list else []

        if len(passages) != len(scores):
            raise ValueError("Mismatched lengths in API response arrays")

        parsed_results = []
        for passage_obj, score in zip(passages, scores):
            parsed_results.append(
                DSServeSearchResult(
                    text=passage_obj.get("text", ""),
                    center_text=passage_obj.get("center_text", ""),
                    score=float(score),
                    filename=passage_obj.get("filename", ""),
                    index_id=passage_obj.get("index_id", 0),
                    passage_id=passage_obj.get("passage_id", 0),
                    position=passage_obj.get("position", 0),
                    source=passage_obj.get("source", ""),
                    raw_query=passage_obj.get("raw_query", ""),
                )
            )

        return parsed_results

    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Error parsing DS Serve response structure: {str(e)}")


@cached()
def search_ds_serve_cached(
    query: str,
    n_docs: int = 10,
    backend: str = "diskann",
    base_url: str = None,
    nprobe: Optional[int] = None,
    exact_search: bool = False,
    diverse_search: bool = False,
    lambda_param: float = 0.5,
    diskann_L: int = 500,
    diskann_W: int = 8,
    diskann_threads: Optional[int] = None,
    min_words: int = 10,
    timeout: int = TIMEOUT,
) -> DSServeResponse:
    """
    Cached version of search_ds_serve function.

    Same parameters and return type as search_ds_serve, but results are cached
    to avoid repeated API calls for identical queries.
    """
    return search_ds_serve(
        query=query,
        n_docs=n_docs,
        backend=backend,
        base_url=base_url,
        nprobe=nprobe,
        exact_search=exact_search,
        diverse_search=diverse_search,
        lambda_param=lambda_param,
        diskann_L=diskann_L,
        diskann_W=diskann_W,
        diskann_threads=diskann_threads,
        min_words=min_words,
        timeout=timeout,
    )


# Example usage and testing
if __name__ == "__main__":
    try:
        # Test the search functionality
        print("Testing DS Serve search API...")

        # Test 1: Default DiskANN
        print("\n=== Test 1: DiskANN Backend ===")
        results = search_ds_serve(
            query="Tell me more about Albert Einstein",
            n_docs=3,
            backend="diskann",
        )

        print(f"Raw API Response:")
        print(f"Message: {results['message']}")
        print(f"Query: {results['query']}")
        print(f"Number of docs: {results['n_docs']}")
        print()

        # Parse and display structured results
        parsed_results = parse_ds_serve_results(results)
        print(f"Found {len(parsed_results)} results:")

        for i, result in enumerate(parsed_results, 1):
            print(f"\nResult {i}:")
            print(f"  Score: {result.score:.4f}")
            print(f"  Source: {result.source}")
            print(f"  Text: {result.text[:200]}...")

        # Test 2: IVFPQ Backend
        print("\n=== Test 2: IVFPQ Backend ===")
        results2 = search_ds_serve(
            query="Explain the basics of quantum physics",
            n_docs=2,
            backend="ivfpq",
            nprobe=64,
        )

        parsed_results2 = parse_ds_serve_results(results2)
        print(f"Found {len(parsed_results2)} results:")

        for i, result in enumerate(parsed_results2, 1):
            print(f"\nResult {i}:")
            print(f"  Score: {result.score:.4f}")
            print(f"  Source: {result.source}")
            print(f"  Text: {result.text[:200]}...")

    except Exception as e:
        print(f"Error testing DS Serve API: {e}")

