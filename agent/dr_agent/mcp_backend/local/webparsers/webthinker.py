# Based on https://github.com/RUC-NLPIR/WebThinker/blob/77eeb37b70918e857a03dac3f02f726c09e54688/scripts/search/bing_search.py#L1

import asyncio
import concurrent.futures
import os
import re
import string
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
import chardet
import pdfplumber
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

# ----------------------- Custom Headers -----------------------
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

# Initialize session
session = requests.Session()
session.headers.update(headers)

# Error indicators for detecting failed page loads
ERROR_INDICATORS = [
    "limit exceeded",
    "Error fetching",
    "Account balance not enough",
    "Invalid bearer token",
    "HTTP error occurred",
    "Error: Connection error occurred",
    "Error: Request timed out",
    "Unexpected error",
    "Please turn on Javascript",
    "Enable JavaScript",
    "port=443",
    "Please enable cookies",
]


class WebParserClient:
    """
    Web parser client for fallback parsing when direct parsing fails
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def parse_urls(self, urls: List[str], timeout: int = 120) -> List[Dict[str, str]]:
        """
        Send URL list to parsing server and get parsed results
        """
        endpoint = urljoin(self.base_url, "/parse_urls")
        response = requests.post(endpoint, json={"urls": urls}, timeout=timeout)
        response.raise_for_status()
        return response.json()["results"]


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text"""
    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate F1 score between two sets of words"""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


def extract_snippet_with_context(
    full_text: str, snippet: str, context_chars: int = 3000
) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its surrounding context
    """
    try:
        full_text = full_text[:100000]
        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2
        sentences = sent_tokenize(full_text)

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            return False, full_text[: context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


def extract_pdf_text(url: str) -> str:
    """
    Extract text from a PDF URL
    """
    try:
        response = session.get(url, timeout=20)
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"

        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text

        return full_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


def extract_text_from_url(
    url: str,
    snippet: Optional[str] = None,
    keep_links: bool = False,
    webparser_url: Optional[str] = None,
) -> str:
    """
    Extract text from a URL (webpage or PDF)

    Args:
        url: URL to extract text from
        snippet: Optional snippet to search for and extract context around
        keep_links: Whether to preserve links in the extracted text
        webparser_url: Optional WebParserClient URL for fallback parsing

    Returns:
        Extracted text or error message
    """
    try:
        if "pdf" in url.lower():
            return extract_pdf_text(url)

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()

            # Handle encoding
            if response.encoding.lower() == "iso-8859-1":
                response.encoding = response.apparent_encoding

            try:
                soup = BeautifulSoup(response.text, "lxml")
            except Exception:
                soup = BeautifulSoup(response.text, "html.parser")

            # Check for error indicators
            has_error = (
                any(
                    indicator.lower() in response.text.lower()
                    for indicator in ERROR_INDICATORS
                )
                and len(response.text.split()) < 64
            ) or response.text == ""

            if has_error and webparser_url:
                # Use WebParserClient as fallback
                client = WebParserClient(webparser_url)
                results = client.parse_urls([url])
                if results and results[0]["success"]:
                    text = results[0]["content"]
                else:
                    error_msg = (
                        results[0].get("error", "Unknown error")
                        if results
                        else "No results returned"
                    )
                    return f"WebParserClient error: {error_msg}"
            else:
                if keep_links:
                    # Remove script and style elements
                    for element in soup.find_all(["script", "style", "meta", "link"]):
                        element.decompose()

                    # Extract text with links preserved
                    text_parts = []
                    for element in (
                        soup.body.descendants if soup.body else soup.descendants
                    ):
                        if isinstance(element, str) and element.strip():
                            cleaned_text = " ".join(element.strip().split())
                            if cleaned_text:
                                text_parts.append(cleaned_text)
                        elif element.name == "a" and element.get("href"):
                            href = element.get("href")
                            link_text = element.get_text(strip=True)
                            if href and link_text:
                                # Handle relative URLs
                                if href.startswith("/"):
                                    base_url = "/".join(url.split("/")[:3])
                                    href = base_url + href
                                elif not href.startswith(("http://", "https://")):
                                    href = url.rstrip("/") + "/" + href
                                text_parts.append(f"[{link_text}]({href})")

                    text = " ".join(text_parts)
                    text = " ".join(text.split())
                else:
                    text = soup.get_text(separator=" ", strip=True)

        except Exception as e:
            if webparser_url:
                # Try WebParserClient on any extraction failure
                client = WebParserClient(webparser_url)
                results = client.parse_urls([url])
                if results and results[0]["success"]:
                    text = results[0]["content"]
                else:
                    error_msg = (
                        results[0].get("error", "Unknown error")
                        if results
                        else "No results returned"
                    )
                    return f"WebParserClient error: {error_msg}"
            else:
                return f"Error extracting content: {str(e)}"

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            return context if success else text
        else:
            return text[:20000]

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 30 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def fetch_page_content(
    urls: List[str],
    max_workers: int = 32,
    snippets: Optional[Dict[str, str]] = None,
    keep_links: bool = False,
    webparser_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    Concurrently fetch content from multiple URLs

    Args:
        urls: List of URLs to fetch
        max_workers: Maximum number of concurrent threads
        snippets: Optional dictionary mapping URLs to snippets to extract
        keep_links: Whether to preserve links in extracted text
        webparser_url: Optional WebParserClient URL for fallback parsing

    Returns:
        Dictionary mapping URLs to extracted content
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_text_from_url,
                url,
                snippets.get(url) if snippets else None,
                keep_links,
                webparser_url,
            ): url
            for url in urls
        }

        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"

    return results


# ----------------------- Async versions -----------------------


async def extract_pdf_text_async(url: str, session: aiohttp.ClientSession) -> str:
    """
    Asynchronously extract text from a PDF
    """
    try:
        async with session.get(url, timeout=30) as response:
            if response.status != 200:
                return (
                    f"Error: Unable to retrieve the PDF (status code {response.status})"
                )

            content = await response.read()

            with pdfplumber.open(BytesIO(content)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text

            return full_text
    except asyncio.TimeoutError:
        return "Error: Request timed out after 30 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


async def extract_text_from_url_async(
    url: str,
    session: aiohttp.ClientSession,
    snippet: Optional[str] = None,
    keep_links: bool = False,
    webparser_url: Optional[str] = None,
) -> str:
    """
    Async version of extract_text_from_url
    """
    try:
        if "pdf" in url.lower():
            text = await extract_pdf_text_async(url, session)
            return text[:10000]

        async with session.get(url) as response:
            # Handle encoding
            content_type = response.headers.get("content-type", "").lower()
            if "charset" in content_type:
                charset = content_type.split("charset=")[-1]
                html = await response.text(encoding=charset)
            else:
                content = await response.read()
                detected = chardet.detect(content)
                encoding = detected["encoding"] if detected["encoding"] else "utf-8"
                html = content.decode(encoding, errors="replace")

            # Check for errors
            has_error = (
                (
                    any(
                        indicator.lower() in html.lower()
                        for indicator in ERROR_INDICATORS
                    )
                    and len(html.split()) < 64
                )
                or len(html) < 50
                or len(html.split()) < 20
            )

            if has_error and webparser_url:
                # Use WebParserClient as fallback
                client = WebParserClient(webparser_url)
                results = client.parse_urls([url])
                if results and results[0]["success"]:
                    text = results[0]["content"]
                else:
                    error_msg = (
                        results[0].get("error", "Unknown error")
                        if results
                        else "No results returned"
                    )
                    return f"WebParserClient error: {error_msg}"
            else:
                try:
                    soup = BeautifulSoup(html, "lxml")
                except Exception:
                    soup = BeautifulSoup(html, "html.parser")

                if keep_links:
                    # Similar link handling as sync version
                    for element in soup.find_all(["script", "style", "meta", "link"]):
                        element.decompose()

                    text_parts = []
                    for element in (
                        soup.body.descendants if soup.body else soup.descendants
                    ):
                        if isinstance(element, str) and element.strip():
                            cleaned_text = " ".join(element.strip().split())
                            if cleaned_text:
                                text_parts.append(cleaned_text)
                        elif element.name == "a" and element.get("href"):
                            href = element.get("href")
                            link_text = element.get_text(strip=True)
                            if href and link_text:
                                if href.startswith("/"):
                                    base_url = "/".join(url.split("/")[:3])
                                    href = base_url + href
                                elif not href.startswith(("http://", "https://")):
                                    href = url.rstrip("/") + "/" + href
                                text_parts.append(f"[{link_text}]({href})")

                    text = " ".join(text_parts)
                    text = " ".join(text.split())
                else:
                    text = soup.get_text(separator=" ", strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            return context if success else text
        else:
            return text[:50000]

    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


async def fetch_page_content_async(
    urls: List[str],
    snippets: Optional[Dict[str, str]] = None,
    keep_links: bool = False,
    max_concurrent: int = 32,
    webparser_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    Asynchronously fetch content from multiple URLs
    """

    async def process_urls():
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=240)
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers
        ) as session:
            tasks = []
            for url in urls:
                task = extract_text_from_url_async(
                    url,
                    session,
                    snippets.get(url) if snippets else None,
                    keep_links,
                    webparser_url,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return {url: result for url, result in zip(urls, results)}

    return await process_urls()
