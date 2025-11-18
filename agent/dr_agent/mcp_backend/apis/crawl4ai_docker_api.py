import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

from crawl4ai import BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.docker_client import Crawl4aiDockerClient

from ..cache import cached
from .data_model import Crawl4aiApiResult


@dataclass(frozen=True)
class Ai2BotConfig:
    """AI2 bot configuration for Crawl4AI with blocklist and custom settings."""

    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("CRAWL4AI_API_URL")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("CRAWL4AI_API_KEY")
    )
    blocklist_path: Optional[str] = field(
        default_factory=lambda: os.getenv("CRAWL4AI_BLOCKLIST_PATH")
    )

    user_agent: str = (
        "Mozilla/5.0 (compatible) AI2Bot-DeepResearchEval (+https://www.allenai.org/crawler)"
    )
    headless: bool = True
    browser_mode: str = "dedicated"
    use_managed_browser: bool = False
    user_agent_mode: str = ""
    user_agent_generator_config: dict = field(default_factory=lambda: {})
    extra_args: list = field(default_factory=lambda: [])
    enable_stealth: bool = False
    check_robots_txt: bool = True
    semaphore_count: int = 50

    def get_exclude_domains(self) -> list:
        if self.blocklist_path is None:
            raise ValueError(
                "CRAWL4AI_BLOCKLIST_PATH is not set; "
                "download the latest from https://github.com/allenai/crawler-rules/blob/main/blocklist.txt"
            )
        if not os.path.exists(self.blocklist_path):
            raise FileNotFoundError(f"Blocklist file not found: {self.blocklist_path}")
        with open(self.blocklist_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def get_browser_config(self, *args, **kwargs) -> BrowserConfig:
        return BrowserConfig(
            *args,
            headless=self.headless,
            user_agent=self.user_agent,
            browser_mode=self.browser_mode,
            use_managed_browser=self.use_managed_browser,
            user_agent_mode=self.user_agent_mode,
            user_agent_generator_config=self.user_agent_generator_config,
            extra_args=self.extra_args,
            enable_stealth=self.enable_stealth,
            **kwargs,
        )

    def get_crawler_config(self, *args, **kwargs) -> CrawlerRunConfig:
        return CrawlerRunConfig(
            *args,
            check_robots_txt=self.check_robots_txt,
            exclude_domains=self.get_exclude_domains(),
            geolocation=None,
            timezone_id=None,
            locale=None,
            simulate_user=False,
            semaphore_count=self.semaphore_count,
            user_agent=self.user_agent,
            user_agent_mode=self.user_agent_mode,
            user_agent_generator_config=self.user_agent_generator_config,
            **kwargs,
        )

    def get_base_url(self) -> str:
        if self.base_url is None:
            raise ValueError("CRAWL4AI_API_URL is not set")
        return self.base_url

    def get_api_key(self) -> str:
        if self.api_key is None:
            raise ValueError("CRAWL4AI_API_KEY is not set")
        return self.api_key


class Crawl4aiApiClient(Crawl4aiDockerClient):
    """Extended Docker client with custom path handling and authentication."""

    def __init__(self, base_url: str, *args, **kwargs):
        super().__init__(base_url=base_url, *args, **kwargs)
        self._path_url = urlparse(base_url).path

    async def _check_server(self) -> None:
        """Check if server is reachable."""
        await self._http_client.get(f"{self.base_url}{self._path_url}/health")
        self.logger.success(f"Connected to {self.base_url}", tag="READY")

    async def authenticate(self, api_key: str) -> None:
        """Set API key in headers."""
        self._http_client.headers["x-api-key"] = api_key

    def _request(self, method: str, endpoint: str, **kwargs):
        """Override request to add path prefix."""
        return super()._request(method, self._path_url + endpoint, **kwargs)


@cached()
async def crawl_url_docker(
    url: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    bypass_cache: bool = True,
    include_html: bool = False,
    use_ai2_config: bool = False,
    query: Optional[str] = None,
    ignore_links: bool = True,
    use_pruning: bool = False,
    timeout_ms: int = 60000,
) -> Crawl4aiApiResult:
    """
    Crawl a single URL using Crawl4AI Docker API.

    Args:
        url: Target URL to crawl
        base_url: Base URL for the Crawl4AI Docker API
        api_key: API key for authentication (optional)
        bypass_cache: If True, bypass Crawl4AI cache
        include_html: Whether to include raw HTML in the response (default: False)
        use_ai2_config: If True, use AI2 bot configuration with blocklist and custom settings (default: False)
        query: Optional BM25 query to keep relevant content
        ignore_links: If True, remove hyperlinks in markdown (default: True)
        use_pruning: If True (and no query), apply pruning content filter (default: False)

    Returns:
        Crawl4aiApiResult with url, success, markdown, and optional fit_markdown/html/error fields
    """
    from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    # Set up content filter
    content_filter = None
    if query:
        content_filter = BM25ContentFilter(
            user_query=query, bm25_threshold=1.2, language="english"
        )
    elif use_pruning:
        content_filter = PruningContentFilter(
            threshold=0.5, threshold_type="fixed", min_word_threshold=50
        )

    # Set up markdown generator with content filter and options
    md_generator = (
        DefaultMarkdownGenerator(options={"ignore_links": ignore_links})
        if content_filter is None
        else DefaultMarkdownGenerator(
            content_filter=content_filter,
            options={"ignore_links": ignore_links},
        )
    )

    if use_ai2_config:
        ai2_config = Ai2BotConfig()
        api_key = ai2_config.get_api_key()
        base_url = ai2_config.get_base_url()
        browser_config = ai2_config.get_browser_config()
        crawler_config = ai2_config.get_crawler_config(
            cache_mode=CacheMode.BYPASS if bypass_cache else CacheMode.ENABLED,
            markdown_generator=md_generator,
            page_timeout=timeout_ms,
        )
    else:
        browser_config = BrowserConfig(headless=True)
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS if bypass_cache else CacheMode.ENABLED,
            page_timeout=timeout_ms,
            markdown_generator=md_generator,
            exclude_social_media_links=True,
            excluded_tags=["form", "header", "footer", "nav"],
            exclude_domains=["ads.com", "spammytrackers.net"],
            word_count_threshold=10,
        )

    async with Crawl4aiApiClient(base_url=base_url, verbose=False) as client:
        if api_key:
            await client.authenticate(api_key)

        # Crawl the URL
        results = await client.crawl(
            [url], browser_config=browser_config, crawler_config=crawler_config
        )

        if not results:
            return Crawl4aiApiResult(
                url=url, success=False, markdown="", error="Crawl returned no results"
            )

        # Handle single result
        result = results[0] if isinstance(results, list) else results

        if not result.success:
            return Crawl4aiApiResult(
                url=result.url,
                success=False,
                markdown="",
                error=result.error_message,
            )

        response_data = {
            "url": result.url,
            "success": True,
            "markdown": result.markdown,
            "fit_markdown": result.markdown.fit_markdown,
        }

        if include_html:
            response_data["html"] = result.html

        return Crawl4aiApiResult(**response_data)
