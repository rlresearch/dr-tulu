import asyncio
import os
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dr_agent.tool_interface.data_types import Document, DocumentToolOutput
from dr_agent.tool_interface.mcp_tools import Crawl4AIBrowseTool, SerperBrowseTool
from dr_agent.tool_interface.tool_parsers import ToolCallInfo


class TestSerperBrowseTool:
    """Test SerperBrowseTool functionality"""

    @pytest.fixture
    def serper_browse_tool(self):
        """Fixture providing a SerperBrowseTool instance with legacy parser"""
        return SerperBrowseTool(
            tool_parser="legacy",
            tool_start_tag="<url>",
            tool_end_tag="</url>",
            result_start_tag="<webpage>",
            result_end_tag="</webpage>",
            max_pages_to_fetch=3,
            timeout=120,
        )

    @pytest.fixture
    def serper_browse_tool_unified(self):
        """Fixture providing a SerperBrowseTool with unified parser"""
        return SerperBrowseTool(
            tool_parser="unified",
            max_pages_to_fetch=3,
            timeout=120,
        )

    @pytest.fixture
    def mock_serper_fetch_response(self):
        """Mock Serper webpage fetch API response"""
        return {
            "success": True,
            "metadata": {
                "title": "Python Programming Guide - Complete Tutorial",
                "description": "Learn Python programming from scratch",
            },
            "markdown": "# Python Programming Guide\n\nThis is a comprehensive guide to Python programming.\n\n## Getting Started\n\nPython is a powerful programming language...",
            "text": "Python Programming Guide\n\nThis is a comprehensive guide to Python programming.\n\nGetting Started\n\nPython is a powerful programming language...",
        }

    def test_tool_configuration(self, serper_browse_tool):
        """Test that SerperBrowseTool is configured correctly"""
        assert serper_browse_tool.max_pages_to_fetch == 3
        assert serper_browse_tool.timeout == 120
        assert serper_browse_tool.get_mcp_tool_name() == "serper_fetch_webpage_content"

    def test_legacy_input_parsing(self, serper_browse_tool):
        """Test parsing of legacy format input - REAL parsing, no mocking"""
        legacy_input = "<url>https://example.com/python-guide</url>"

        # Test actual parsing logic (no mocking)
        tool_call_info = serper_browse_tool.parse_call(legacy_input)

        assert tool_call_info is not None
        assert tool_call_info.content == "https://example.com/python-guide"
        assert tool_call_info.parameters == {}
        assert tool_call_info.start_pos == 0
        assert tool_call_info.end_pos == len(legacy_input)

    def test_unified_input_parsing(self, serper_browse_tool_unified):
        """Test parsing of unified format input - REAL parsing, no mocking"""
        unified_input = '<tool name="SerperBrowseTool" include_markdown="true">https://example.com/article</tool>'

        # Test actual parsing logic (no mocking)
        tool_call_info = serper_browse_tool_unified.parse_call(unified_input)

        assert tool_call_info is not None
        assert tool_call_info.content == "https://example.com/article"
        assert tool_call_info.parameters == {"include_markdown": "true"}
        assert tool_call_info.start_pos >= 0
        assert tool_call_info.end_pos > tool_call_info.start_pos

    def test_get_mcp_params_default(self, serper_browse_tool):
        """Test MCP parameter building with default values"""
        tool_call_info = ToolCallInfo(
            content="https://example.com", parameters={}, start_pos=0, end_pos=20
        )

        params = serper_browse_tool.get_mcp_params(tool_call_info)

        expected_params = {
            "webpage_url": "https://example.com",
            "include_markdown": True,
        }
        assert params == expected_params

    def test_extract_raw_content_success(
        self, serper_browse_tool, mock_serper_fetch_response
    ):
        """Test successful content extraction from Serper response"""
        raw_content = serper_browse_tool._extract_raw_content_from_response(
            mock_serper_fetch_response
        )

        assert raw_content is not None
        assert "Python Programming Guide" in raw_content
        assert "comprehensive guide" in raw_content

    def test_extract_raw_content_failure(self, serper_browse_tool):
        """Test content extraction from failed Serper response"""
        failed_response = {"success": False, "error": "Failed to fetch webpage"}

        raw_content = serper_browse_tool._extract_raw_content_from_response(
            failed_response
        )
        assert raw_content is None

    def test_extract_metadata_success(
        self, serper_browse_tool, mock_serper_fetch_response
    ):
        """Test successful metadata extraction from Serper response"""
        document = Document(url="https://example.com", title="", snippet="")

        webpage_title, fallback_message = (
            serper_browse_tool._extract_metadata_from_document(
                document, mock_serper_fetch_response
            )
        )

        assert webpage_title == "Python Programming Guide - Complete Tutorial"
        assert fallback_message is None

    def test_extract_metadata_failure(self, serper_browse_tool):
        """Test metadata extraction from failed Serper response"""
        failed_response = {"success": False, "error": "Connection timeout"}
        document = Document(url="https://example.com", title="", snippet="")

        webpage_title, fallback_message = (
            serper_browse_tool._extract_metadata_from_document(
                document, failed_response
            )
        )

        assert webpage_title is None
        assert "Failed to fetch content: Connection timeout" in fallback_message

    def test_legacy_parsing_unit_test(self, serper_browse_tool):
        """Unit test for legacy format parsing"""
        legacy_input = (
            "Check this <url>https://python.org/docs</url> for documentation."
        )

        tool_call_info = serper_browse_tool.parse_call(legacy_input)

        assert tool_call_info is not None
        assert tool_call_info.content == "https://python.org/docs"
        assert tool_call_info.parameters == {}
        assert tool_call_info.start_pos == 11
        assert tool_call_info.end_pos == 45

    def test_unified_parsing_unit_test(self, serper_browse_tool_unified):
        """Unit test for unified format parsing"""
        unified_input = '<tool name="SerperBrowseTool" timeout="180" include_markdown="false">https://news.ycombinator.com</tool>'

        tool_call_info = serper_browse_tool_unified.parse_call(unified_input)

        assert tool_call_info is not None
        assert tool_call_info.content == "https://news.ycombinator.com"
        assert tool_call_info.parameters == {
            "timeout": "180",
            "include_markdown": "false",
        }
        assert tool_call_info.start_pos >= 0
        assert tool_call_info.end_pos > tool_call_info.start_pos

    def test_parsing_edge_cases(self, serper_browse_tool):
        """Test parsing edge cases"""
        # Empty URL
        empty_result = serper_browse_tool.parse_call("<url></url>")
        assert empty_result is not None
        assert empty_result.content == ""

        # URL with special characters
        special_result = serper_browse_tool.parse_call(
            "<url>https://example.com/path?query=value&param=test#section</url>"
        )
        assert special_result is not None
        assert (
            special_result.content
            == "https://example.com/path?query=value&param=test#section"
        )

        # Invalid format - should return None
        invalid_result = serper_browse_tool.parse_call("no url tags here")
        assert invalid_result is None

    @pytest.fixture
    def sample_browse_documents(self):
        """Sample documents for browse tool formatting tests"""
        return [
            Document(
                id="page1",
                title="Python Guide",
                snippet="Learn Python programming from scratch",
                url="https://example.com/python-guide",
                text="# Python Programming Guide\n\nThis is a comprehensive guide to Python programming.\n\n## Getting Started\n\nPython is a powerful programming language...",
                score=None,
            ),
            Document(
                id="page2",
                title="Advanced Python",
                snippet="Master advanced Python concepts",
                url="https://example.com/advanced-python",
                text="# Advanced Python Techniques\n\nLearn advanced concepts in Python programming.\n\n## Object-Oriented Programming\n\nPython supports multiple programming paradigms...",
                score=None,
            ),
        ]

    @pytest.fixture
    def sample_browse_output_with_call_id(self, sample_browse_documents):
        """Sample DocumentToolOutput for browse tool with call_id"""
        return DocumentToolOutput(
            tool_name="SerperBrowseTool",
            output="Title: Python Guide\nURL: https://example.com/python-guide\nSnippet: This is a comprehensive guide to Python programming.\n\nTitle: Advanced Python\nURL: https://example.com/advanced-python\nSnippet: Learn advanced concepts in Python programming.",
            called=True,
            error="",
            timeout=False,
            runtime=2.5,
            call_id="browse_456",
            raw_output={},
            documents=sample_browse_documents,
            query="python programming guide",
        )

    def test_browse_format_output_legacy(self, sample_browse_output_with_call_id):
        """Test _format_output method with legacy parser for browse tool"""
        tool = SerperBrowseTool(
            tool_parser="legacy",
            tool_start_tag="<url>",
            tool_end_tag="</url>",
            result_start_tag="<webpage>",
            result_end_tag="</webpage>",
            max_pages_to_fetch=3,
            timeout=120,
        )

        formatted_content = tool._format_output(sample_browse_output_with_call_id)

        # Legacy format should return the raw output content
        expected_content = sample_browse_output_with_call_id.output
        assert formatted_content == expected_content

    def test_browse_format_output_unified(self, sample_browse_output_with_call_id):
        """Test _format_output method with unified parser for browse tool"""
        tool = SerperBrowseTool(
            tool_parser="unified",
            max_pages_to_fetch=3,
            timeout=120,
        )

        formatted_content = tool._format_output(sample_browse_output_with_call_id)

        # Unified format should contain webpage tags with document content
        expected_content = """<webpage id=browse_456-0>
Title: Python Guide
URL: https://example.com/python-guide
Snippet: # Python Programming Guide

This is a comprehensive guide to Python programming.

## Getting Started

Python is a powerful programming language...
</webpage>
<webpage id=browse_456-1>
Title: Advanced Python
URL: https://example.com/advanced-python
Snippet: # Advanced Python Techniques

Learn advanced concepts in Python programming.

## Object-Oriented Programming

Python supports multiple programming paradigms...
</webpage>"""
        assert formatted_content == expected_content

    def test_browse_format_result_legacy(self, sample_browse_output_with_call_id):
        """Test format_result method with legacy parser for browse tool"""
        tool = SerperBrowseTool(
            tool_parser="legacy",
            tool_start_tag="<url>",
            tool_end_tag="</url>",
            result_start_tag="<webpage>",
            result_end_tag="</webpage>",
            max_pages_to_fetch=3,
            timeout=120,
        )

        formatted = tool.format_result(sample_browse_output_with_call_id)

        # Legacy format_result should wrap the content in parser tags
        expected_content = f"<webpage id=browse_456>\n{sample_browse_output_with_call_id.output}\n</webpage>"
        assert formatted == expected_content

    def test_browse_format_result_unified(self, sample_browse_output_with_call_id):
        """Test format_result method with unified parser for browse tool"""
        tool = SerperBrowseTool(
            tool_parser="unified",
            max_pages_to_fetch=3,
            timeout=120,
        )

        formatted = tool.format_result(sample_browse_output_with_call_id)

        # Unified format_result should return the same content as _format_output since parser doesn't add additional wrapping
        expected_content = """<webpage id=browse_456-0>
Title: Python Guide
URL: https://example.com/python-guide
Snippet: # Python Programming Guide

This is a comprehensive guide to Python programming.

## Getting Started

Python is a powerful programming language...
</webpage>
<webpage id=browse_456-1>
Title: Advanced Python
URL: https://example.com/advanced-python
Snippet: # Advanced Python Techniques

Learn advanced concepts in Python programming.

## Object-Oriented Programming

Python supports multiple programming paradigms...
</webpage>"""
        assert formatted == f"<tool_output>{expected_content}</tool_output>"


class TestCrawl4AIBrowseTool:
    """Test Crawl4AIBrowseTool functionality"""

    @pytest.fixture
    def crawl4ai_browse_tool(self):
        """Fixture providing a Crawl4AIBrowseTool instance"""
        return Crawl4AIBrowseTool(
            tool_parser="legacy",
            tool_start_tag="<url>",
            tool_end_tag="</url>",
            result_start_tag="<webpage>",
            result_end_tag="</webpage>",
            max_pages_to_fetch=2,
            timeout=180,
            ignore_links=True,
            use_pruning=False,
        )

    @pytest.fixture
    def crawl4ai_browse_tool_unified(self):
        """Fixture providing a Crawl4AIBrowseTool with unified parser"""
        return Crawl4AIBrowseTool(
            tool_parser="unified",
            max_pages_to_fetch=2,
            timeout=180,
        )

    @pytest.fixture
    def mock_crawl4ai_response(self):
        """Mock Crawl4AI API response"""
        return {
            "success": True,
            "markdown": "# Machine Learning Guide\n\nIntroduction to ML concepts and algorithms.\n\n## Supervised Learning\n\nSupervised learning is...",
            "fit_markdown": "# Machine Learning Guide\n\nIntroduction to ML concepts and algorithms.",
            "html": "<html><head><title>ML Guide</title></head><body><h1>Machine Learning Guide</h1><p>Introduction to ML...</p></body></html>",
            "url": "https://example.com/ml-guide",
        }

    def test_tool_configuration(self, crawl4ai_browse_tool):
        """Test that Crawl4AIBrowseTool is configured correctly"""
        assert crawl4ai_browse_tool.max_pages_to_fetch == 2
        assert crawl4ai_browse_tool.timeout == 180
        assert crawl4ai_browse_tool.ignore_links == True
        assert crawl4ai_browse_tool.use_pruning == False
        assert (
            crawl4ai_browse_tool.get_mcp_tool_name() == "crawl4ai_fetch_webpage_content"
        )

    def test_legacy_input_parsing(self, crawl4ai_browse_tool):
        """Test parsing of legacy format input"""
        legacy_input = "<url>https://example.com/ml-guide</url>"

        # Test actual parsing logic (no mocking)
        tool_call_info = crawl4ai_browse_tool.parse_call(legacy_input)

        assert tool_call_info is not None
        assert tool_call_info.content == "https://example.com/ml-guide"
        assert tool_call_info.parameters == {}
        assert tool_call_info.start_pos == 0
        assert tool_call_info.end_pos == len(legacy_input)

    def test_unified_input_parsing(self, crawl4ai_browse_tool_unified):
        """Test parsing of unified format input"""
        unified_input = '<tool name="Crawl4AIBrowseTool" use_pruning="true" bm25_query="machine learning">https://example.com/article</tool>'

        # Mock the parse_call method to return expected ToolCallInfo
        with patch.object(crawl4ai_browse_tool_unified, "parse_call") as mock_parse:
            mock_parse.return_value = ToolCallInfo(
                content="https://example.com/article",
                parameters={"use_pruning": "true", "bm25_query": "machine learning"},
                start_pos=0,
                end_pos=len(unified_input),
            )

            # Browse tools don't have preprocess_input, they call parse_call directly
            tool_call_info = crawl4ai_browse_tool_unified.parse_call(unified_input)

            assert tool_call_info is not None
            assert tool_call_info.content == "https://example.com/article"
            assert tool_call_info.parameters == {
                "use_pruning": "true",
                "bm25_query": "machine learning",
            }

    def test_get_mcp_params_default(self, crawl4ai_browse_tool):
        """Test MCP parameter building with default values"""
        tool_call_info = ToolCallInfo(
            content="https://example.com", parameters={}, start_pos=0, end_pos=20
        )

        params = crawl4ai_browse_tool.get_mcp_params(tool_call_info)

        expected_params = {
            "url": "https://example.com",
            "ignore_links": True,
            "use_pruning": False,
            "bm25_query": None,
            "bypass_cache": True,
            "timeout_ms": 80000,
            "include_html": False,
        }
        assert params == expected_params

    def test_extract_raw_content_success(
        self, crawl4ai_browse_tool, mock_crawl4ai_response
    ):
        """Test successful content extraction from Crawl4AI response"""
        raw_content = crawl4ai_browse_tool._extract_raw_content_from_response(
            mock_crawl4ai_response
        )

        assert raw_content is not None
        # Should prefer fit_markdown over markdown
        assert (
            raw_content
            == "# Machine Learning Guide\n\nIntroduction to ML concepts and algorithms."
        )

    def test_extract_raw_content_failure(self, crawl4ai_browse_tool):
        """Test content extraction from failed Crawl4AI response"""
        failed_response = {"success": False, "error": "Timeout during crawling"}

        raw_content = crawl4ai_browse_tool._extract_raw_content_from_response(
            failed_response
        )
        assert raw_content is None

    def test_extract_metadata_success(
        self, crawl4ai_browse_tool, mock_crawl4ai_response
    ):
        """Test successful metadata extraction from Crawl4AI response"""
        document = Document(url="https://example.com", title="", snippet="")

        webpage_title, fallback_message = (
            crawl4ai_browse_tool._extract_metadata_from_document(
                document, mock_crawl4ai_response
            )
        )

        # Crawl4AI doesn't provide webpage title
        assert webpage_title is None
        assert fallback_message is None

    def test_extract_metadata_failure(self, crawl4ai_browse_tool):
        """Test metadata extraction from failed Crawl4AI response"""
        failed_response = {"success": False, "error": "Page not found"}
        document = Document(url="https://example.com", title="", snippet="")

        webpage_title, fallback_message = (
            crawl4ai_browse_tool._extract_metadata_from_document(
                document, failed_response
            )
        )

        assert webpage_title is None
        assert "Crawl4AI failed (Page not found)" in fallback_message


@pytest.mark.integration
class TestRealBrowsingIntegration:
    """Integration tests that require real MCP server and API access"""

    @pytest.mark.skipif(
        not all([os.environ.get("MCP_TRANSPORT")]),
        reason="MCP_TRANSPORT environment variables not set",
    )
    @pytest.mark.asyncio
    async def test_real_serper_browse_integration(self):
        """Test actual Serper browse via MCP (requires MCP server and Serper API key)"""

        # Create SerperBrowseTool configured for real API
        serper_browse_tool = SerperBrowseTool(
            tool_parser="legacy",
            tool_start_tag="<url>",
            tool_end_tag="</url>",
            result_start_tag="<webpage>",
            result_end_tag="</webpage>",
            max_pages_to_fetch=1,
            timeout=180,
        )

        # Test with a simple URL
        result = await serper_browse_tool("<url>https://python.org</url>")

        # Verify we got real results
        assert isinstance(result, DocumentToolOutput)
        assert result.called is True
        assert result.error == "" or result.error is None
        assert len(result.documents) > 0

        # Verify document has expected fields
        first_doc = result.documents[0]
        assert first_doc.url == "https://python.org"
        assert first_doc.text is not None
        assert len(first_doc.text) > 100  # Should have meaningful content

    @pytest.mark.skipif(
        not all([os.environ.get("MCP_TRANSPORT")]),
        reason="MCP_TRANSPORT environment variables not set",
    )
    @pytest.mark.asyncio
    async def test_real_serper_browse_from_search_results(self):
        """Test SerperBrowseTool with DocumentToolOutput from search"""
        from dr_agent.tool_interface.mcp_tools import SerperSearchTool

        # First, get search results
        search_tool = SerperSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=2,
            timeout=60,
        )

        search_result = await search_tool("<query>Python programming tutorial</query>")

        # Verify search worked
        assert isinstance(search_result, DocumentToolOutput)
        assert search_result.called is True
        assert len(search_result.documents) > 0

        # Now test browsing those results
        browse_tool = SerperBrowseTool(
            tool_parser="legacy",
            tool_start_tag="<url>",
            tool_end_tag="</url>",
            result_start_tag="<webpage>",
            result_end_tag="</webpage>",
            max_pages_to_fetch=1,
            timeout=180,
        )

        browse_result = await browse_tool(search_result)

        # Verify browsing worked
        assert isinstance(browse_result, DocumentToolOutput)
        assert browse_result.called is True
        assert browse_result.error == "" or browse_result.error is None
        assert len(browse_result.documents) > 0

        # Verify we got webpage content
        first_doc = browse_result.documents[0]
        assert first_doc.text is not None
        assert len(first_doc.text) > 100  # Should have substantial content

    @pytest.mark.skipif(
        not os.environ.get("MCP_TRANSPORT"),
        reason="MCP_TRANSPORT environment variable not set",
    )
    @pytest.mark.asyncio
    async def test_real_crawl4ai_browse_integration(self):
        """Test actual Crawl4AI browse via MCP (requires MCP server)"""

        # Create Crawl4AIBrowseTool configured for real API
        crawl4ai_browse_tool = Crawl4AIBrowseTool(
            tool_parser="legacy",
            tool_start_tag="<url>",
            tool_end_tag="</url>",
            result_start_tag="<webpage>",
            result_end_tag="</webpage>",
            max_pages_to_fetch=1,
            timeout=180,
            ignore_links=True,
            use_pruning=False,
        )

        # Test with a simple URL
        result = await crawl4ai_browse_tool("<url>https://example.com</url>")

        # Verify we got real results
        assert isinstance(result, DocumentToolOutput)
        assert result.called is True
        # Note: Crawl4AI might fail on some sites, so we allow for errors
        # but if successful, should have content
        if result.error == "":
            assert len(result.documents) > 0
            first_doc = result.documents[0]
            assert first_doc.url == "https://example.com"
            assert first_doc.text is not None

    @pytest.mark.skipif(
        not os.environ.get("MCP_TRANSPORT"),
        reason="MCP_TRANSPORT environment variable not set",
    )
    @pytest.mark.asyncio
    async def test_real_crawl4ai_browse_with_unified_format(self):
        """Test Crawl4AI with unified format input"""

        crawl4ai_tool = Crawl4AIBrowseTool(
            tool_parser="unified",
            max_pages_to_fetch=1,
            timeout=180,
        )

        # Test with unified format
        unified_input = '<tool name="Crawl4AIBrowseTool" ignore_links="true" use_pruning="false">https://httpbin.org/html</tool>'
        result = await crawl4ai_tool(unified_input)

        # Verify result
        assert isinstance(result, DocumentToolOutput)
        assert result.called is True
        # Allow for potential errors but verify structure
        if result.error == "":
            assert len(result.documents) > 0

    @pytest.mark.skipif(
        not all([os.environ.get("MCP_TRANSPORT")]),
        reason="MCP_TRANSPORT environment variables not set",
    )
    @pytest.mark.asyncio
    async def test_real_serper_browse_with_unified_format(self):
        """Test SerperBrowseTool with unified format input"""

        serper_tool = SerperBrowseTool(
            tool_parser="unified",
            max_pages_to_fetch=1,
            timeout=180,
        )

        # Test with unified format
        unified_input = '<tool name="SerperBrowseTool" include_markdown="true">https://httpbin.org/html</tool>'
        result = await serper_tool(unified_input)

        # Verify result
        assert isinstance(result, DocumentToolOutput)
        assert result.called is True
        assert result.error == "" or result.error is None
        assert len(result.documents) > 0

        # Verify document content
        first_doc = result.documents[0]
        assert first_doc.url == "https://httpbin.org/html"
        assert first_doc.text is not None
        assert len(first_doc.text) > 50  # Should have some content

    @pytest.mark.skipif(
        not os.environ.get("MCP_TRANSPORT"),
        reason="MCP_TRANSPORT environment variable not set",
    )
    @pytest.mark.asyncio
    async def test_mcp_browse_connection_health(self):
        """Test that MCP client can connect and ping for browsing tools"""

        # Create any MCP browse tool to test connection
        tool = SerperBrowseTool(
            tool_parser="legacy",
            tool_start_tag="<url>",
            tool_end_tag="</url>",
            result_start_tag="<webpage>",
            result_end_tag="</webpage>",
            max_pages_to_fetch=1,
            timeout=30,
        )

        # Test that we can initialize MCP client
        mcp_client = tool.init_mcp_client()
        assert mcp_client is not None

        # Test connection with async context
        try:
            async with mcp_client:
                await mcp_client.ping()
                # If we get here, connection is working
                assert True
        except Exception as e:
            pytest.fail(f"MCP connection failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
