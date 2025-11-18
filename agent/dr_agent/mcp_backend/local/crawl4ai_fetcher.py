import asyncio
from typing import Dict, Optional

from pydantic import BaseModel, Field

from ..cache import cached


class Crawl4AiResult(BaseModel):
    """Result structure returned by crawl4ai fetch functions."""

    url: str
    success: bool
    markdown: str
    fit_markdown: Optional[str] = Field(
        None, description="Only present when content filtering is used"
    )
    html: Optional[str] = Field(
        None, description="Only present when include_html=True or markdown is empty"
    )
    error: Optional[str] = Field(None, description="Only present when success=False")


@cached()
async def fetch_markdown(
    url: str,
    query: Optional[str] = None,
    ignore_links: bool = True,
    use_pruning: bool = False,
    bypass_cache: bool = True,
    headless: bool = True,
    timeout_ms: int = 60000,
    include_html: bool = False,
) -> Crawl4AiResult:
    """
    Fetch webpage content using Crawl4AI and return markdown (optionally HTML).

    Args:
            url: Target URL
            query: Optional BM25 query to keep relevant content
            ignore_links: If True, remove hyperlinks in markdown
            use_pruning: If True (and no query), apply pruning content filter
            bypass_cache: If True, bypass crawler cache
            headless: Run browser in headless mode
            timeout_ms: Per-page timeout in milliseconds
            include_html: Whether to include raw HTML in the response

    Returns:
            Dictionary with fields: url, success, markdown, fit_markdown (optional), html (optional), error (on failure)
    """
    try:
        # Lazy imports to avoid hard dependency at module import time
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import (
            BM25ContentFilter,
            PruningContentFilter,
        )
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        content_filter = None
        if query:
            try:
                content_filter = BM25ContentFilter(
                    user_query=query, bm25_threshold=1.2, language="english"
                )
            except Exception:
                content_filter = None
        elif use_pruning:
            try:
                content_filter = PruningContentFilter(
                    threshold=0.5, threshold_type="fixed", min_word_threshold=50
                )
            except Exception:
                content_filter = None

        md_generator = (
            DefaultMarkdownGenerator(options={"ignore_links": ignore_links})
            if content_filter is None
            else DefaultMarkdownGenerator(
                content_filter=content_filter,
                options={"ignore_links": ignore_links},
            )
        )

        browser_conf = BrowserConfig(headless=headless)
        run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS if bypass_cache else CacheMode.ENABLED,
            page_timeout=timeout_ms,
            markdown_generator=md_generator,
            exclude_social_media_links=True,
            excluded_tags=["form", "header", "footer", "nav"],
            exclude_domains=["ads.com", "spammytrackers.net"],
            word_count_threshold=10,
        )

        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url=url, config=run_conf)

        if not getattr(result, "success", True):
            return Crawl4AiResult(
                url=getattr(result, "url", url),
                success=False,
                markdown="",
                html=getattr(result, "html", ""),
                error=getattr(result, "error_message", "Unknown error"),
            )

        md_value = ""
        fit_markdown_value = None
        md_obj = getattr(result, "markdown", None)
        if isinstance(md_obj, str):
            md_value = md_obj
        else:
            fit_markdown_value = getattr(md_obj, "fit_markdown", None)
            raw_markdown = getattr(md_obj, "raw_markdown", None)
            if fit_markdown_value:
                md_value = fit_markdown_value
            elif raw_markdown:
                md_value = raw_markdown
            else:
                md_value = str(md_obj) if md_obj is not None else ""

        response_data = {
            "url": getattr(result, "url", url),
            "success": True,
            "markdown": md_value,
        }
        if fit_markdown_value:
            response_data["fit_markdown"] = fit_markdown_value
        if include_html or not md_value:
            response_data["html"] = getattr(result, "html", "")

        return Crawl4AiResult(**response_data)
    except Exception as e:
        return Crawl4AiResult(
            url=url,
            success=False,
            markdown="",
            html="",
            error=str(e),
        )


@cached()
def fetch_markdown_sync(
    url: str,
    query: Optional[str] = None,
    ignore_links: bool = True,
    use_pruning: bool = False,
    bypass_cache: bool = True,
    headless: bool = True,
    timeout_ms: int = 80000,
    include_html: bool = False,
) -> Crawl4AiResult:
    """
    Synchronous wrapper around fetch_markdown. Safe to call from environments with or without an existing event loop.
    """
    try:
        # If already inside a running loop, dispatch to a separate loop in a thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            from threading import Thread

            result_holder: Dict[str, Crawl4AiResult] = {}

            def _runner():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result_holder["result"] = new_loop.run_until_complete(
                        fetch_markdown(
                            url=url,
                            query=query,
                            ignore_links=ignore_links,
                            use_pruning=use_pruning,
                            bypass_cache=bypass_cache,
                            headless=headless,
                            timeout_ms=timeout_ms,
                            include_html=include_html,
                        )
                    )
                finally:
                    new_loop.close()

            th = Thread(target=_runner)
            th.start()
            th.join()
            return result_holder.get(
                "result",
                Crawl4AiResult(
                    url=url,
                    success=False,
                    markdown="",
                    html="",
                    error="Unknown error",
                ),
            )
        else:
            return asyncio.run(
                fetch_markdown(
                    url=url,
                    query=query,
                    ignore_links=ignore_links,
                    use_pruning=use_pruning,
                    bypass_cache=bypass_cache,
                    headless=headless,
                    timeout_ms=timeout_ms,
                    include_html=include_html,
                )
            )
    except Exception as e:
        return Crawl4AiResult(
            url=url,
            success=False,
            markdown="",
            html="",
            error=str(e),
        )
