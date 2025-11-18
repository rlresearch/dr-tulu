import argparse
import asyncio
import os
from typing import TYPE_CHECKING, Annotated, List, Optional

import aiohttp
import dotenv
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from .apis.data_model import Crawl4aiApiResult
from .apis.massive_serve_apis import parse_massive_serve_results, search_massive_serve
from .apis.pubmed_apis import search_pubmed
from .apis.reranker_apis import RerankerResult
from .apis.semantic_scholar_apis import (
    SemanticScholarSearchQueryParams,
    SemanticScholarSnippetSearchQueryParams,
    search_semantic_scholar_keywords,
    search_semantic_scholar_snippets,
)
from .apis.serper_apis import (
    ScholarResponse,
    SearchResponse,
    WebpageContentResponse,
    fetch_webpage_content,
    search_serper,
    search_serper_scholar,
)
from .apis.jina_apis import JinaWebpageResponse, fetch_webpage_content_jina
from .cache import set_cache_enabled
from .local.crawl4ai_fetcher import Crawl4AiResult

dotenv.load_dotenv()

mcp = FastMCP(
    "RL-RAG MCP",
    include_tags=os.environ.get("MCP_INCLUDE_TAGS", "search,browse,rerank").split(","),
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """
    Check if the MCP server is running.
    curl http://127.0.0.1:8000/health
    """
    return PlainTextResponse("OK")


@mcp.tool(tags={"search", "necessary"})
def semantic_scholar_search(
    query: Annotated[str, "Search query string"],
    year: Annotated[
        Optional[str], "Year range filter (e.g., '2015-2020', '2015-', '-2015')"
    ] = None,
    min_citation_count: Annotated[Optional[int], "Minimum number of citations"] = None,
    sort: Annotated[
        Optional[str], "Sort order (e.g., 'citationCount:asc', 'publicationDate:desc')"
    ] = None,
    venue: Annotated[Optional[str], "Venue filter (e.g., 'ACL', 'EMNLP')"] = None,
    limit: Annotated[int, "Maximum number of results to return (max: 100)"] = 25,
) -> dict:
    """
    Search for academic papers using Semantic Scholar API.

    Returns:
        Dictionary containing search results
    """
    query_params = SemanticScholarSearchQueryParams(
        query=query,
        year=year,
        minCitationCount=min_citation_count,
        sort=sort,
        venue=venue,
    )

    results = search_semantic_scholar_keywords(
        query_params=query_params,
        limit=min(limit, 100),  # Ensure limit doesn't exceed API maximum
    )

    return results


@mcp.tool(tags={"search"})
def semantic_scholar_snippet_search(
    query: Annotated[str, "Search query string to find within paper content"],
    year: Annotated[
        Optional[str],
        "Publication year filter - single number (e.g., '2024') or range (e.g., '2022-2025', '2020-', '-2023')",
    ] = None,
    paper_ids: Annotated[
        Optional[str], "Comma-separated list of specific paper IDs to search within"
    ] = None,
    venue: Annotated[Optional[str], "Venue filter (e.g., 'ACL', 'EMNLP')"] = None,
    limit: Annotated[int, "Number of snippets to retrieve"] = 10,
) -> dict:
    """
    Focused snippet retrieval from scientific papers using Semantic Scholar API.

    Purpose: Search for specific text snippets within academic papers to find relevant passages, quotes,
    or mentions from scientific literature. Returns focused snippets from existing papers rather than
    full paper metadata.

    Returns:
        Dictionary containing snippets from existing papers with text passages and their source papers.
        Each snippet includes the relevant text passage and metadata about the source paper.

    Example:
        Search for LLM evaluation snippets published between 2021-2025 in CS/Medicine:
        query="large language model retrieval evaluation", year="2021-2025", limit=8
    """
    # Convert comma-separated string to list if provided
    paper_ids_list = None
    if paper_ids:
        paper_ids_list = [pid.strip() for pid in paper_ids.split(",")]

    query_params = SemanticScholarSnippetSearchQueryParams(
        query=query,
        year=year,
        paperIds=paper_ids_list,
        venue=venue,
    )

    results = search_semantic_scholar_snippets(
        query_params=query_params,
        limit=limit,
    )

    return results


@mcp.tool(tags={"search"})
def pubmed_search(
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """
    Search for medical and scientific papers using PubMed API.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 10)
        offset: Starting position for pagination (default: 0)

    Returns:
        Dictionary containing search results with the following fields:
        - total: Total number of results
        - offset: Current offset
        - next: Next offset for pagination
        - data: List of paper details including:
            - paperId: PubMed ID
            - title: Paper title
            - authors: List of authors
            - abstract: Paper abstract
            - year: Publication year
            - venue: Journal name
            - url: Link to PubMed page
            - citationCount: Number of citations (if available from Semantic Scholar)
    """
    results = search_pubmed(
        keywords=query,
        limit=limit,
        offset=offset,
    )

    return results


@mcp.tool(tags={"necessary", "rerank"})
def vllm_hosted_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
) -> RerankerResult:
    """
    Rerank a list of documents based on their relevance to the query using VLLM hosted reranker.

    Args:
        query: Search query string
        documents: List of document texts to rank
        top_n: Number of top documents to return
        model_name: Name of the reranker model (default: "BAAI/bge-reranker-v2-m3")
        api_url: Base URL for the VLLM reranker API (default: "http://localhost:30002")

    Returns:
        RerankerResult containing reranker results with method, model_name, and ranked results
    """
    from dr_agent.mcp_backend.apis.reranker_apis import vllm_hosted_reranker

    results = vllm_hosted_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
    )

    return results


@mcp.tool(tags={"search", "necessary"})
def massive_serve_search(
    query: str,
    n_docs: int = 10,
    domains: str = "dpr_wiki_contriever_ivfpq",
    base_url: Optional[str] = None,
    nprobe: Optional[int] = None,
) -> dict:
    """
    Search for documents using massive-serve API for dense passage retrieval.

    This tool provides access to large-scale document collections using dense passage
    retrieval with various embedding models and indices.

    Args:
        query: Search query string
        n_docs: Number of documents to return (default: 10)
        domains: Domain/index to search in (default: "dpr_wiki_contriever_ivfpq")
        base_url: Base URL for the massive-serve API (optional, uses default if not provided)
        nprobe: Number of probes for search (optional, uses API default)

    Returns:
        Dictionary containing search results with the following fields:
        - message: Status message
        - query: The original search query
        - n_docs: Number of documents requested
        - results: Dictionary with IDs, passages, and scores
        - data: Parsed list of search results with passage text, scores, and doc IDs
    """
    # Call the massive-serve API
    response = search_massive_serve(
        query=query,
        n_docs=n_docs,
        domains=domains,
        base_url=base_url,
        nprobe=nprobe,
    )

    # Parse the results for easier consumption
    parsed_results = parse_massive_serve_results(response)

    # Add parsed data to the response for convenience
    response["data"] = [
        {
            "passage": result.passage,
            "score": result.score,
            "doc_id": result.doc_id,
        }
        for result in parsed_results
    ]

    return response


@mcp.tool(tags={"search", "necessary"})
def serper_google_webpage_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
    gl: Annotated[
        str,
        "Geolocation - country code to boost search results whose country of origin matches the parameter value",
    ] = "us",
    hl: Annotated[str, "Host language of user interface"] = "en",
):
    """
    General web search using Google Search (based on Serper.dev API). Perform general web search to find relevant webpages, articles, and online resources.

    Returns:
        Dictionary containing web search snippets with the following fields:
        - organic: List of organic search results with title, link, and snippet
        - knowledgeGraph: Knowledge graph information (if available)
        - peopleAlsoAsk: List of related questions
        - relatedSearches: List of related searches
    """
    results = search_serper(
        query=query, num_results=num_results, search_type="search", gl=gl, hl=hl
    )

    return results


@mcp.tool(tags={"browse", "necessary"})
def serper_fetch_webpage_content(
    webpage_url: Annotated[str, "The URL of the webpage to fetch"],
    include_markdown: Annotated[
        bool, "Whether to include markdown formatting in the response"
    ] = True,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Serper.dev API.

    Returns:
        Dictionary containing the webpage content with the following fields:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
        - url: The original URL that was fetched
        - success: Boolean indicating if the fetch was successful
    """
    try:
        result = fetch_webpage_content(
            url=webpage_url,
            include_markdown=include_markdown,
        )

        return {
            **result,
            "success": True,
        }
    except Exception as e:
        return {
            "text": "",
            "markdown": "",
            "metadata": {},
            "url": webpage_url,
            "success": False,
            "error": str(e),
        }


@mcp.tool(tags={"browse"})
def jina_fetch_webpage_content(
    webpage_url: Annotated[str, "The URL of the webpage to fetch"],
    timeout: Annotated[int, "Request timeout in seconds"] = 30,
) -> JinaWebpageResponse:
    """
    Fetch the content of a webpage using Jina Reader API with timeout support.

    Args:
        webpage_url: The URL of the webpage to fetch
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing the webpage content with the following fields:
        - url: The original URL that was fetched
        - title: Page title
        - content: The webpage content as clean text/markdown
        - description: Page description (if available)
        - publishedTime: Published time (if available)
        - metadata: Additional metadata (lang, viewport, etc.)
        - success: Boolean indicating if the fetch was successful
        - error: Error message if fetch failed
    """
    result = fetch_webpage_content_jina(url=webpage_url, timeout=timeout)
    return result


@mcp.tool(tags={"search", "necessary"})
def serper_google_scholar_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
) -> ScholarResponse:
    """
    Search for academic papers using google scholar (based on Serper.dev API).

    Returns:
        Dictionary containing search results with the following fields:
        - organic: List of organic search results
    """
    results = search_serper_scholar(
        query=query,
        num_results=num_results,
    )

    return results


@mcp.tool(tags={"browse", "necessary"})
async def crawl4ai_fetch_webpage_content(
    url: Annotated[str, "URL to fetch and extract content from"],
    ignore_links: Annotated[bool, "If True, remove hyperlinks in markdown"] = True,
    use_pruning: Annotated[
        bool,
        "Apply pruning content filter to extract main content (used when bm25_query is not provided)",
    ] = False,
    bm25_query: Annotated[
        Optional[str],
        "Optional query to enable BM25-based content filtering for focused extraction",
    ] = None,
    bypass_cache: Annotated[bool, "If True, bypass Crawl4AI cache"] = True,
    timeout_ms: Annotated[int, "Per-page timeout in milliseconds"] = 80000,
    include_html: Annotated[
        bool, "Whether to include raw HTML in the response"
    ] = False,
) -> Crawl4AiResult:
    """
    Open a specific URL and extract readable page text as snippets using Crawl4AI.

    Purpose: Fetch and parse webpage content (typically URLs returned from google_search) to extract clean, readable text.
    This tool is useful for opening articles, documentation, and webpages to read their full content.

    Returns:
        Crawl4AiResult with extracted webpage content including markdown-formatted text
    """

    from dr_agent.mcp_backend.local.crawl4ai_fetcher import fetch_markdown

    result = await fetch_markdown(
        url=url,
        query=bm25_query,
        ignore_links=ignore_links,
        use_pruning=use_pruning,
        bypass_cache=bypass_cache,
        headless=True,
        timeout_ms=timeout_ms,
        include_html=include_html,
    )
    return result


@mcp.tool(tags={"browse", "necessary"})
async def crawl4ai_docker_fetch_webpage_content(
    url: Annotated[str, "Target URL to crawl and extract content from"],
    base_url: Annotated[
        Optional[str],
        "Base URL for the Crawl4AI Docker API (e.g., 'http://localhost:8000')",
    ] = None,
    api_key: Annotated[Optional[str], "API key for authentication"] = None,
    use_ai2_config: Annotated[
        bool,
        "If True, use AI2 bot configuration with blocklist (requires CRAWL4AI_BLOCKLIST_PATH env var)",
    ] = False,
    bypass_cache: Annotated[bool, "If True, bypass Crawl4AI cache"] = True,
    ignore_links: Annotated[bool, "If True, remove hyperlinks in markdown"] = True,
    use_pruning: Annotated[
        bool,
        "Apply pruning content filter to extract main content (used when bm25_query is not provided)",
    ] = False,
    bm25_query: Annotated[
        Optional[str],
        "Optional query to enable BM25-based content filtering for focused extraction",
    ] = None,
    timeout_ms: Annotated[int, "Per-page timeout in milliseconds"] = 80000,
    include_html: Annotated[
        bool, "Whether to include raw HTML in the response"
    ] = False,
) -> Crawl4aiApiResult:
    """
    Open a specific URL and extract readable page text as snippets using Crawl4AI Docker API.

    Purpose: Fetch and parse webpage content (typically URLs returned from google_search) to extract clean, readable text.
    This tool is useful for opening articles, documentation, and webpages to read their full content.

    Returns:
        Crawl4aiApiResult with url, success, markdown-formatted text, and optional fit_markdown/html/error fields
    """
    from dr_agent.mcp_backend.apis.crawl4ai_docker_api import crawl_url_docker

    result = await crawl_url_docker(
        url=url,
        base_url=base_url,
        api_key=api_key,
        bypass_cache=bypass_cache,
        include_html=include_html,
        use_ai2_config=use_ai2_config,
        query=bm25_query,
        ignore_links=ignore_links,
        use_pruning=use_pruning,
        timeout_ms=timeout_ms,
    )
    return result


@mcp.tool(tags={"browse"})
def webthinker_fetch_webpage_content(
    url: str,
    snippet: Optional[str] = None,
    keep_links: bool = False,
) -> dict:
    """
    Extract text content from a single URL (webpage or PDF) using advanced web parsing.

    Args:
        url: URL to extract text from
        snippet: Optional snippet to search for and extract context around
        keep_links: Whether to preserve links in the extracted text (default: False)

    Returns:
        Dictionary containing the URL and extracted text content
    """
    from dr_agent.mcp_backend.local.webparsers.webthinker import extract_text_from_url

    text = extract_text_from_url(
        url=url,
        snippet=snippet,
        keep_links=keep_links,
    )

    return {"url": url, "text": text}


@mcp.tool(tags={"browse"})
async def webthinker_fetch_webpage_content_async(
    url: str,
    snippet: Optional[str] = None,
    keep_links: bool = False,
) -> dict:
    """
    Asynchronously extract text content from a single URL (webpage or PDF) using advanced web parsing.

    Args:
        url: URL to extract text from
        snippet: Optional snippet to search for and extract context around
        keep_links: Whether to preserve links in the extracted text (default: False)

    Returns:
        Dictionary containing the URL and extracted text content
    """
    from dr_agent.mcp_backend.local.webparsers.webthinker import (
        extract_text_from_url_async,
    )

    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=240)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.36",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=headers
    ) as session:
        text = await extract_text_from_url_async(
            url=url,
            session=session,
            snippet=snippet,
            keep_links=keep_links,
        )

    return {"url": url, "text": text}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument(
        "--transport",
        type=str,
        default="http",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol to use (default: stdio for local, http for web)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (for HTTP transports)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (for HTTP transports)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp",
        help="Path for the HTTP endpoint (default: /mcp)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level for the server",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable API response caching",
    )

    args = parser.parse_args()

    # Set cache enabled/disabled based on argument
    if args.no_cache:
        set_cache_enabled(False)
    else:
        set_cache_enabled(True)

    # Run the server with the provided arguments
    if args.transport == "stdio":
        # stdio transport doesn't accept host/port/path arguments
        # For stdio, we can omit the transport argument since it's the default
        mcp.run(transport="stdio")
    else:
        # HTTP-based transports accept host/port/path/log_level arguments
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path,
            log_level=args.log_level,
        )
