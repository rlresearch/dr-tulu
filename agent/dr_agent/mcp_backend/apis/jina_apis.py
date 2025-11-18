import os
from typing import Dict, Optional

import dotenv
import requests
from typing_extensions import TypedDict

from ..cache import cached

dotenv.load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
TIMEOUT = int(os.getenv("API_TIMEOUT", 30))


class JinaMetadata(TypedDict, total=False):
    lang: str
    viewport: str


class JinaWebpageResponse(TypedDict, total=False):
    url: str
    title: str
    content: str
    description: str
    publishedTime: str
    metadata: JinaMetadata
    success: bool
    error: str


@cached()
def fetch_webpage_content_jina(
    url: str,
    api_key: str = None,
    timeout: int = TIMEOUT,
) -> JinaWebpageResponse:
    """
    Fetch webpage content using Jina Reader API with JSON format.

    Args:
        url: The URL of the webpage to fetch
        api_key: Jina API key (if not provided, will use JINA_API_KEY env var)
        timeout: Request timeout in seconds (if not provided, will use TIMEOUT env var or default 30)

    Returns:
        JinaWebpageResponse containing:
        - url: The original URL that was fetched
        - title: The webpage title
        - content: The webpage content as clean text/markdown
        - description: The webpage description (if available)
        - publishedTime: Publication timestamp (if available)
        - metadata: Additional metadata (lang, viewport, etc.)
        - success: Boolean indicating if the fetch was successful
        - error: Error message if fetch failed
    """
    if not api_key:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError(
                "JINA_API_KEY environment variable is not set or api_key parameter not provided"
            )

    if timeout is None:
        timeout = TIMEOUT

    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    response = requests.get(jina_url, headers=headers, timeout=timeout)

    if response.status_code != 200:
        raise Exception(
            f"API request failed with status {response.status_code}: {response.text}"
        )

    json_response = response.json()

    # Extract data from JSON response
    data = json_response.get("data", {})

    return {
        "url": data.get("url", url),
        "title": data.get("title", ""),
        "content": data.get("content", ""),
        "description": data.get("description", ""),
        "publishedTime": data.get("publishedTime", ""),
        "metadata": data.get("metadata", {}),
        "success": True,
    }
