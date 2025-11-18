import asyncio
import os
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dr_agent.tool_interface.data_types import Document, DocumentToolOutput
from dr_agent.tool_interface.mcp_tools import (
    SemanticScholarSnippetSearchTool,
    SerperSearchTool,
)
from dr_agent.tool_interface.tool_parsers import ToolCallInfo


class TestSerperSearchTool:
    """Test SerperSearchTool functionality"""

    @pytest.fixture
    def serper_tool(self):
        """Fixture providing a SerperSearchTool instance"""
        return SerperSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=5,
            timeout=30,
        )

    @pytest.fixture
    def serper_tool_unified(self):
        """Fixture providing a SerperSearchTool with unified parser"""
        return SerperSearchTool(
            tool_parser="unified", number_documents_to_search=5, timeout=30
        )

    @pytest.fixture
    def mock_serper_response(self):
        """Mock Serper API response"""
        return {
            "organic": [
                {
                    "title": "Python Programming Guide",
                    "link": "https://example.com/python-guide",
                    "snippet": "Learn Python programming from scratch with this comprehensive guide.",
                },
                {
                    "title": "Advanced Python Techniques",
                    "link": "https://example.com/advanced-python",
                    "snippet": "Master advanced Python concepts like decorators, generators, and metaclasses.",
                },
                {
                    "title": "Python Best Practices",
                    "link": "https://example.com/python-best-practices",
                    "snippet": "Follow these Python best practices to write clean, maintainable code.",
                },
            ],
            "SearchParameters": {"q": "python programming", "num": 5, "type": "search"},
        }

    def test_tool_configuration(self, serper_tool):
        """Test that SerperSearchTool is configured correctly"""
        assert serper_tool.number_documents_to_search == 5
        assert serper_tool.timeout == 30
        assert serper_tool.get_mcp_tool_name() == "serper_google_webpage_search"

    def test_legacy_input_parsing(self, serper_tool):
        """Test parsing of legacy format input"""
        legacy_input = "<query>python programming best practices</query>"

        # Mock the parse_call method to return expected ToolCallInfo
        with patch.object(serper_tool, "parse_call") as mock_parse:
            mock_parse.return_value = ToolCallInfo(
                content="python programming best practices",
                parameters={},
                start_pos=7,
                end_pos=39,
            )

            tool_call_info = serper_tool.preprocess_input(legacy_input)

            assert tool_call_info is not None
            assert tool_call_info.content == "python programming best practices"
            assert tool_call_info.parameters == {}
            mock_parse.assert_called_once_with(legacy_input)

    def test_unified_input_parsing(self, serper_tool_unified):
        """Test parsing of unified format input"""
        unified_input = '<tool name="SerperSearchTool" num_results="3">machine learning algorithms</tool>'

        # Mock the parse_call method to return expected ToolCallInfo
        with patch.object(serper_tool_unified, "parse_call") as mock_parse:
            mock_parse.return_value = ToolCallInfo(
                content="machine learning algorithms",
                parameters={"num_results": "3"},
                start_pos=0,
                end_pos=len(unified_input),
            )

            tool_call_info = serper_tool_unified.preprocess_input(unified_input)

            assert tool_call_info is not None
            assert tool_call_info.content == "machine learning algorithms"
            assert tool_call_info.parameters == {"num_results": "3"}

    def test_get_mcp_params_default(self, serper_tool):
        """Test MCP parameter building with default values"""
        tool_call_info = ToolCallInfo(
            content="test query", parameters={}, start_pos=0, end_pos=10
        )

        params = serper_tool.get_mcp_params(tool_call_info)

        expected_params = {
            "query": "test query",
            "num_results": 5,  # Default from tool configuration
        }
        assert params == expected_params

    def test_get_mcp_params_with_override(self, serper_tool):
        """Test MCP parameter building with parameter override"""
        tool_call_info = ToolCallInfo(
            content="test query",
            parameters={"num_results": "10"},
            start_pos=0,
            end_pos=10,
        )

        params = serper_tool.get_mcp_params(tool_call_info)

        expected_params = {
            "query": "test query",
            "num_results": 10,  # Override from parameters
        }
        assert params == expected_params

    def test_get_mcp_params_invalid_override(self, serper_tool):
        """Test MCP parameter building with invalid parameter override"""
        tool_call_info = ToolCallInfo(
            content="test query",
            parameters={"num_results": "invalid"},
            start_pos=0,
            end_pos=10,
        )

        params = serper_tool.get_mcp_params(tool_call_info)

        expected_params = {
            "query": "test query",
            "num_results": 5,  # Falls back to default
        }
        assert params == expected_params

    def test_extract_documents_success(self, serper_tool, mock_serper_response):
        """Test successful document extraction from Serper response"""
        documents = serper_tool.extract_documents(mock_serper_response)

        assert len(documents) == 3

        # Check first document
        doc1 = documents[0]
        assert doc1.title == "Python Programming Guide"
        assert doc1.url == "https://example.com/python-guide"
        assert (
            doc1.snippet
            == "Learn Python programming from scratch with this comprehensive guide."
        )
        assert doc1.text is None
        assert doc1.score is None

    def test_extract_documents_empty_response(self, serper_tool):
        """Test document extraction from empty response"""
        empty_response = {"organic": []}
        documents = serper_tool.extract_documents(empty_response)

        assert len(documents) == 0

    def test_extract_documents_malformed_response(self, serper_tool):
        """Test document extraction from malformed response"""
        malformed_response = {"organic": [{"invalid": "data"}]}
        documents = serper_tool.extract_documents(malformed_response)

        # Should skip documents with no title, snippet, or URL
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_successful_search_call_real_parsing(
        self, serper_tool, mock_serper_response
    ):
        """Test successful search execution with real parsing"""
        with patch.object(
            serper_tool, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            # Only mock the MCP call, test real parsing
            mock_execute.return_value = mock_serper_response

            # Execute tool with real parsing
            result = await serper_tool("<query>python programming</query>")

            # Verify result
            assert isinstance(result, DocumentToolOutput)
            assert result.called is True
            assert result.error == ""
            assert len(result.documents) == 3
            assert "Python Programming Guide" in result.output
            assert result.query == "python programming"

            # Verify MCP call was made correctly with real parsed content
            mock_execute.assert_called_once_with(
                "serper_google_webpage_search",
                {"query": "python programming", "num_results": 5},
            )

    @pytest.mark.asyncio
    async def test_successful_search_call_unified_parsing(
        self, serper_tool_unified, mock_serper_response
    ):
        """Test successful search with unified format parsing"""
        with patch.object(
            serper_tool_unified, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            # Only mock the MCP call, test real parsing
            mock_execute.return_value = mock_serper_response

            # Test unified format parsing
            unified_input = (
                '<tool name="SerperSearchTool" num_results="3">machine learning</tool>'
            )
            result = await serper_tool_unified(unified_input)

            # Verify result - this tests real parsing of unified format
            assert isinstance(result, DocumentToolOutput)
            assert result.called is True
            assert result.error == ""
            assert result.query == "machine learning"

            # Verify MCP call was made with parsed parameters
            mock_execute.assert_called_once_with(
                "serper_google_webpage_search",
                {
                    "query": "machine learning",
                    "num_results": 3,
                },  # num_results should be parsed from parameters
            )

    def test_legacy_parsing_unit_test(self, serper_tool):
        """Unit test for legacy format parsing"""
        # Test that parse_call actually works for legacy format
        legacy_input = "abcdefg<query>deep learning algorithms</query>"

        tool_call_info = serper_tool.parse_call(legacy_input)

        assert tool_call_info is not None
        assert tool_call_info.content == "deep learning algorithms"
        assert tool_call_info.parameters == {}
        assert tool_call_info.start_pos == 7
        assert tool_call_info.end_pos == 46

    def test_unified_parsing_unit_test(self, serper_tool_unified):
        """Unit test for unified format parsing"""
        # Test that parse_call actually works for unified format
        unified_input = '<tool name="SerperSearchTool" num_results="5" location="US">natural language processing</tool>'

        tool_call_info = serper_tool_unified.parse_call(unified_input)

        assert tool_call_info is not None
        assert tool_call_info.content == "natural language processing"
        assert tool_call_info.parameters == {"num_results": "5", "location": "US"}
        assert tool_call_info.start_pos >= 0
        assert tool_call_info.end_pos > tool_call_info.start_pos

    def test_parsing_edge_cases(self, serper_tool):
        """Test parsing edge cases"""
        # Empty query
        empty_result = serper_tool.parse_call("<query></query>")
        assert empty_result is not None
        assert empty_result.content == ""

        # Query with special characters
        special_result = serper_tool.parse_call(
            "<query>Python & AI: 2024 trends!</query>"
        )
        assert special_result is not None
        assert special_result.content == "Python & AI: 2024 trends!"

        # Invalid format - should return None
        invalid_result = serper_tool.parse_call("no query tags here")
        assert invalid_result is None

    def test_parameter_parsing_incorrect_tool_name(self, serper_tool_unified):
        """Test unified format parameter parsing edge cases"""
        # Multiple parameters
        multi_param_input = '<tool name="search" num_results="10" country="UK" safe_search="true">AI research</tool>'
        result = serper_tool_unified.parse_call(multi_param_input)

        assert result is None

    def test_parameter_parsing_edge_cases(self, serper_tool_unified):
        """Test unified format parameter parsing edge cases"""
        # Multiple parameters
        multi_param_input = '<tool name="SerperSearchTool" num_results="10" country="UK" safe_search="true">AI research</tool>'
        result = serper_tool_unified.parse_call(multi_param_input)

        assert result is not None
        assert result.content == "AI research"
        assert result.parameters["num_results"] == "10"
        assert result.parameters["country"] == "UK"
        assert result.parameters["safe_search"] == "true"

    @pytest.mark.asyncio
    async def test_search_call_with_error(self, serper_tool):
        """Test search execution with MCP error"""
        with patch.object(
            serper_tool, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            # Setup mock - only mock the MCP call, test real parsing
            mock_execute.return_value = {"error": "API timeout"}

            # Execute tool with real parsing
            result = await serper_tool("<query>test query</query>")

            # Verify error handling
            assert isinstance(result, DocumentToolOutput)
            assert result.called is True
            assert "Query failed: API timeout" in result.error
            assert len(result.documents) == 0

    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, serper_tool):
        """Test handling of invalid input"""
        # Test non-string input - should raise ValueError
        with pytest.raises(ValueError, match="must be a string or dict"):
            await serper_tool(123)

    @pytest.mark.asyncio
    async def test_no_query_found(self, serper_tool):
        """Test handling when no query is found in input"""
        # Test with actual invalid input (no mocking needed)
        result = await serper_tool("invalid input without query tags")

        assert isinstance(result, DocumentToolOutput)
        assert result.called is True
        assert "No valid query found" in result.error
        assert len(result.documents) == 0


class TestSemanticScholarSnippetSearchTool:
    """Test SemanticScholarSnippetSearchTool functionality"""

    @pytest.fixture
    def semantic_scholar_tool(self):
        """Fixture providing a SemanticScholarSnippetSearchTool instance"""
        return SemanticScholarSnippetSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=3,
            timeout=30,
        )

    @pytest.fixture
    def mock_semantic_scholar_response(self):
        """Mock Semantic Scholar API response"""
        return {
            "data": [
                {
                    "snippet": {
                        "text": "Deep learning has revolutionized machine learning by enabling automatic feature extraction."
                    },
                    "paper": {
                        "title": "Deep Learning: A Comprehensive Review",
                        "authors": ["John Doe", "Jane Smith"],
                    },
                    "score": 0.95,
                },
                {
                    "snippet": {
                        "text": "Neural networks are the foundation of modern deep learning architectures."
                    },
                    "paper": {
                        "title": "Neural Networks and Deep Learning",
                        "authors": ["Alice Johnson"],
                    },
                    "score": 0.87,
                },
            ]
        }

    def test_tool_configuration(self, semantic_scholar_tool):
        """Test that SemanticScholarSnippetSearchTool is configured correctly"""
        assert semantic_scholar_tool.number_documents_to_search == 3
        assert semantic_scholar_tool.timeout == 30
        assert (
            semantic_scholar_tool.get_mcp_tool_name()
            == "semantic_scholar_snippet_search"
        )

    def test_get_mcp_params_default(self, semantic_scholar_tool):
        """Test MCP parameter building for Semantic Scholar"""
        tool_call_info = ToolCallInfo(
            content="deep learning", parameters={}, start_pos=0, end_pos=13
        )

        params = semantic_scholar_tool.get_mcp_params(tool_call_info)

        expected_params = {
            "query": "deep learning",
            "limit": 3,  # Default from tool configuration
        }
        assert params == expected_params

    def test_extract_documents_success(
        self, semantic_scholar_tool, mock_semantic_scholar_response
    ):
        """Test successful document extraction from Semantic Scholar response"""
        documents = semantic_scholar_tool.extract_documents(
            mock_semantic_scholar_response
        )

        assert len(documents) == 2

        # Check first document
        doc1 = documents[0]
        assert doc1.title == "Deep Learning: A Comprehensive Review"
        assert (
            doc1.snippet
            == "Deep learning has revolutionized machine learning by enabling automatic feature extraction."
        )
        assert doc1.url == ""  # Semantic Scholar doesn't provide URLs in snippet search
        assert doc1.score == 0.95

    def test_extract_documents_empty_response(self, semantic_scholar_tool):
        """Test document extraction from empty Semantic Scholar response"""
        empty_response = {"data": []}
        documents = semantic_scholar_tool.extract_documents(empty_response)

        assert len(documents) == 0

    def test_extract_documents_fallback_format(self, semantic_scholar_tool):
        """Test document extraction with fallback snippet format"""
        fallback_response = {
            "data": [
                {
                    "snippet": {
                        "text": "This is a standalone snippet without paper info."
                    }
                }
            ]
        }

        documents = semantic_scholar_tool.extract_documents(fallback_response)

        assert len(documents) == 1
        doc = documents[0]
        assert doc.snippet == "This is a standalone snippet without paper info."
        assert doc.title == ""
        assert doc.score is None

    @pytest.mark.asyncio
    async def test_successful_search_execution(
        self, semantic_scholar_tool, mock_semantic_scholar_response
    ):
        """Test successful Semantic Scholar search execution"""
        with patch.object(
            semantic_scholar_tool, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            # Only mock the MCP call, test real parsing
            mock_execute.return_value = mock_semantic_scholar_response

            # Execute tool with real parsing
            result = await semantic_scholar_tool("<query>deep learning</query>")

            # Verify result
            assert isinstance(result, DocumentToolOutput)
            assert result.called is True
            assert result.error == ""
            assert len(result.documents) == 2
            assert "Deep Learning: A Comprehensive Review" in result.output
            assert result.query == "deep learning"

            # Verify MCP call was made correctly
            mock_execute.assert_called_once_with(
                "semantic_scholar_snippet_search",
                {"query": "deep learning", "limit": 3},
            )

    def test_semantic_scholar_parsing_unit_test(self, semantic_scholar_tool):
        """Unit test for Semantic Scholar parsing"""
        # Test that parse_call works for academic queries
        academic_input = (
            "<query>transformer neural networks attention mechanism</query>"
        )

        tool_call_info = semantic_scholar_tool.parse_call(academic_input)

        assert tool_call_info is not None
        assert (
            tool_call_info.content == "transformer neural networks attention mechanism"
        )
        assert tool_call_info.parameters == {}
        assert tool_call_info.start_pos == 0
        assert tool_call_info.end_pos == 62


class TestMCPSearchToolFormatting:
    """Test format_result functionality for MCP search tools"""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for formatting tests"""
        return [
            Document(
                id="doc1",
                title="Python Tutorial",
                snippet="Learn Python programming basics",
                url="https://example.com/python",
                score=0.95,
            ),
            Document(
                id="doc2",
                title="Advanced Python",
                snippet="Master advanced Python concepts",
                url="https://example.com/advanced-python",
                score=0.87,
            ),
        ]

    @pytest.fixture
    def sample_search_output_with_call_id(self, sample_documents):
        """Sample DocumentToolOutput with call_id"""
        return DocumentToolOutput(
            tool_name="SerperSearchTool",
            output="Title: Python Tutorial\nURL: https://example.com/python\nSnippet: Learn Python programming basics\n\nTitle: Advanced Python\nURL: https://example.com/advanced-python\nSnippet: Master advanced Python concepts",
            called=True,
            error="",
            timeout=False,
            runtime=1.5,
            call_id="search_123",
            raw_output={},
            documents=sample_documents,
            query="python programming",
        )

    @pytest.fixture
    def sample_search_output_no_call_id(self, sample_documents):
        """Sample DocumentToolOutput without call_id"""
        return DocumentToolOutput(
            tool_name="SerperSearchTool",
            output="Title: Python Tutorial\nURL: https://example.com/python\nSnippet: Learn Python programming basics\n\nTitle: Advanced Python\nURL: https://example.com/advanced-python\nSnippet: Master advanced Python concepts",
            called=True,
            error="",
            timeout=False,
            runtime=1.5,
            call_id="",  # Empty call_id
            raw_output={},
            documents=sample_documents,
            query="python programming",
        )

    def test_format_output_legacy(self, sample_search_output_with_call_id):
        """Test _format_output method with legacy parser"""
        tool = SerperSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=5,
            timeout=30,
        )

        formatted_content = tool._format_output(sample_search_output_with_call_id)

        # Legacy format should return the raw output content
        expected_content = sample_search_output_with_call_id.output
        assert formatted_content == expected_content

    def test_format_result_legacy(self, sample_search_output_with_call_id):
        """Test format_result method with legacy parser"""
        tool = SerperSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=5,
            timeout=30,
        )

        formatted = tool.format_result(sample_search_output_with_call_id)

        # Legacy format_result should wrap the content in parser tags
        expected_content = f"<snippet id=search_123>\n{sample_search_output_with_call_id.output}\n</snippet>"
        assert formatted == expected_content

    def test_format_output_unified(self, sample_search_output_with_call_id):
        """Test _format_output method with unified parser"""
        tool = SerperSearchTool(
            tool_parser="unified", number_documents_to_search=5, timeout=30
        )

        formatted_content = tool._format_output(sample_search_output_with_call_id)

        # Unified format should contain snippet tags with document content
        expected_content = """<snippet id=search_123-0>
Title: Python Tutorial
URL: https://example.com/python
Snippet: Learn Python programming basics
</snippet>
<snippet id=search_123-1>
Title: Advanced Python
URL: https://example.com/advanced-python
Snippet: Master advanced Python concepts
</snippet>"""
        assert formatted_content == expected_content

    def test_format_result_unified(self, sample_search_output_with_call_id):
        """Test format_result method with unified parser"""
        tool = SerperSearchTool(
            tool_parser="unified", number_documents_to_search=5, timeout=30
        )

        formatted = tool.format_result(sample_search_output_with_call_id)

        # Unified format_result should return the same content as _format_output since parser doesn't add additional wrapping
        expected_content = """<snippet id=search_123-0>
Title: Python Tutorial
URL: https://example.com/python
Snippet: Learn Python programming basics
</snippet>
<snippet id=search_123-1>
Title: Advanced Python
URL: https://example.com/advanced-python
Snippet: Master advanced Python concepts
</snippet>"""
        assert formatted == f"<tool_output>{expected_content}</tool_output>"

    def test_legacy_format_with_call_id(self, sample_search_output_with_call_id):
        """Test legacy format_result with call_id"""
        tool = SerperSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=5,
            timeout=30,
        )

        formatted = tool.format_result(sample_search_output_with_call_id)
        expected_content = sample_search_output_with_call_id.output

        assert formatted == f"<snippet id=search_123>\n{expected_content}\n</snippet>"
        assert "search_123" in formatted
        assert expected_content in formatted

    def test_legacy_format_without_call_id(self, sample_search_output_no_call_id):
        """Test legacy format_result without call_id"""
        tool = SerperSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=5,
            timeout=30,
        )

        formatted = tool.format_result(sample_search_output_no_call_id)
        expected_content = sample_search_output_no_call_id.output

        assert formatted == f"<snippet>\n{expected_content}\n</snippet>"
        assert " id=" not in formatted  # No ID when call_id is empty
        assert expected_content in formatted

    def test_unified_format_with_call_id(
        self, sample_search_output_with_call_id, sample_documents
    ):
        """Test unified format_result with call_id and multiple documents"""
        tool = SerperSearchTool(
            tool_parser="unified", number_documents_to_search=5, timeout=30
        )

        formatted = tool.format_result(sample_search_output_with_call_id)

        # Should have multiple snippets, one for each document
        assert "<snippet id=search_123-0>" in formatted
        assert "<snippet id=search_123-1>" in formatted
        assert formatted.count("<snippet") == 2
        assert formatted.count("</snippet>") == 2

        # Should contain document content
        assert "Python Tutorial" in formatted
        assert "Advanced Python" in formatted
        assert "https://example.com/python" in formatted

    def test_unified_format_single_document(self):
        """Test unified format with single document"""
        tool = SerperSearchTool(
            tool_parser="unified", number_documents_to_search=1, timeout=30
        )

        single_doc = Document(
            id="single_doc",
            title="Single Result",
            snippet="This is a single search result",
            url="https://example.com/single",
            score=0.9,
        )

        output = DocumentToolOutput(
            tool_name="SerperSearchTool",
            output="Single Result\nThis is a single search result\nhttps://example.com/single",
            called=True,
            error="",
            timeout=False,
            runtime=1.0,
            call_id="single_123",
            raw_output={},
            documents=[single_doc],
            query="single test",
        )

        formatted = tool.format_result(output)

        # Should have only one snippet
        assert "<snippet id=single_123-0>" in formatted
        assert formatted.count("<snippet") == 1
        assert formatted.count("</snippet>") == 1
        assert "Single Result" in formatted

    def test_semantic_scholar_legacy_formatting(self):
        """Test SemanticScholarSnippetSearchTool legacy formatting"""
        tool = SemanticScholarSnippetSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=3,
            timeout=30,
        )

        # Create academic-style document
        academic_doc = Document(
            id="paper1",
            title="Deep Learning: A Comprehensive Review",
            snippet="Deep learning has revolutionized machine learning by enabling automatic feature extraction from raw data.",
            url="",  # S2 doesn't provide URLs in snippet search
            score=0.95,
        )

        output = DocumentToolOutput(
            tool_name="SemanticScholarSnippetSearchTool",
            output=academic_doc.stringify(),
            called=True,
            error="",
            timeout=False,
            runtime=2.0,
            call_id="s2_456",
            raw_output={},
            documents=[academic_doc],
            query="deep learning",
        )

        formatted = tool.format_result(output)

        assert (
            formatted == f"<snippet id=s2_456>\n{academic_doc.stringify()}\n</snippet>"
        )
        assert "Deep Learning: A Comprehensive Review" in formatted
        assert "s2_456" in formatted

    def test_semantic_scholar_unified_formatting(self):
        """Test SemanticScholarSnippetSearchTool unified formatting"""
        tool = SemanticScholarSnippetSearchTool(
            tool_parser="unified", number_documents_to_search=2, timeout=30
        )

        # Create multiple academic documents
        academic_docs = [
            Document(
                id="paper1",
                title="Neural Networks and Deep Learning",
                snippet="Neural networks are computational models inspired by biological neural networks.",
                url="",
                score=0.92,
            ),
            Document(
                id="paper2",
                title="Attention Is All You Need",
                snippet="The Transformer architecture relies entirely on attention mechanisms.",
                url="",
                score=0.89,
            ),
        ]

        output = DocumentToolOutput(
            tool_name="SemanticScholarSnippetSearchTool",
            output="\n\n".join(doc.stringify() for doc in academic_docs),
            called=True,
            error="",
            timeout=False,
            runtime=1.8,
            call_id="s2_multi_789",
            raw_output={},
            documents=academic_docs,
            query="neural networks",
        )

        formatted = tool.format_result(output)

        # Should have separate snippets for each paper
        assert "<snippet id=s2_multi_789-0>" in formatted
        assert "<snippet id=s2_multi_789-1>" in formatted
        assert formatted.count("<snippet") == 2
        assert "Neural Networks and Deep Learning" in formatted
        assert "Attention Is All You Need" in formatted

    def test_format_empty_documents_list(self):
        """Test formatting when documents list is empty"""
        tool = SerperSearchTool(
            tool_parser="unified", number_documents_to_search=5, timeout=30
        )

        empty_output = DocumentToolOutput(
            tool_name="SerperSearchTool",
            output="",
            called=True,
            error="No results found",
            timeout=False,
            runtime=1.0,
            call_id="empty_123",
            raw_output={},
            documents=[],  # Empty documents
            query="nonexistent query",
        )

        formatted = tool.format_result(empty_output)

        # Should use the error message
        assert formatted == f"<tool_output>No results found</tool_output>"

    def test_format_documents_without_stringify_content(self):
        """Test formatting with documents that have minimal content"""
        tool = SerperSearchTool(
            tool_parser="unified", number_documents_to_search=1, timeout=30
        )

        minimal_doc = Document(
            id="minimal",
            title="",  # Empty title
            snippet="",  # Empty snippet
            url="https://example.com/minimal",  # Only URL
            score=None,
        )

        output = DocumentToolOutput(
            tool_name="SerperSearchTool",
            output="https://example.com/minimal",
            called=True,
            error="",
            timeout=False,
            runtime=0.5,
            call_id="minimal_456",
            raw_output={},
            documents=[minimal_doc],
            query="minimal test",
        )

        formatted = tool.format_result(output)

        # Should still create snippet even with minimal content
        assert "<snippet id=minimal_456-0>" in formatted
        assert "</snippet>" in formatted
        assert "https://example.com/minimal" in formatted


@pytest.mark.integration
class TestRealMCPIntegration:
    """Integration tests that require real MCP server and API access"""

    @pytest.mark.skipif(
        not all([os.environ.get("MCP_TRANSPORT")]),
        reason="MCP_TRANSPORT environment variables not set",
    )
    @pytest.mark.asyncio
    async def test_real_serper_search_integration(self):
        """Test actual Serper search via MCP (requires MCP server and Serper API key)"""

        # Create SerperSearchTool configured for real API
        serper_tool = SerperSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=3,
            timeout=60,
        )

        # Test with a simple query
        result = await serper_tool("<query>Python programming tutorial</query>")

        # Verify we got real results
        assert isinstance(result, DocumentToolOutput)
        assert result.called is True
        assert result.error == "" or result.error is None
        assert len(result.documents) > 0

        # Verify documents have expected fields
        first_doc = result.documents[0]
        assert first_doc.url.startswith("http")
        assert len(first_doc.title) > 0 or len(first_doc.snippet) > 0

    @pytest.mark.skipif(
        not all([os.environ.get("MCP_TRANSPORT")]),
        reason="MCP_TRANSPORT environment variable not set",
    )
    @pytest.mark.asyncio
    async def test_real_semantic_scholar_integration(self):
        """Test actual Semantic Scholar search via MCP (requires MCP server)"""

        # Create SemanticScholarSnippetSearchTool for real API
        scholar_tool = SemanticScholarSnippetSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=2,
            timeout=60,
        )

        # Test with an academic query
        result = await scholar_tool("<query>machine learning neural networks</query>")

        # Verify we got real results
        assert isinstance(result, DocumentToolOutput)
        assert result.called is True
        assert result.error == "" or result.error is None
        assert len(result.documents) > 0

        # Verify documents have academic content
        first_doc = result.documents[0]
        assert len(first_doc.snippet) > 20  # Should have meaningful content
        assert first_doc.score is not None  # S2 provides relevance scores

    @pytest.mark.skipif(
        not all([os.environ.get("MCP_TRANSPORT"), os.environ.get("SERPER_API_KEY")]),
        reason="MCP_TRANSPORT and SERPER_API_KEY environment variables not set",
    )
    @pytest.mark.asyncio
    async def test_real_serper_browse_integration(self):
        """Test actual Serper search + browse via MCP (requires MCP server and Serper API key)"""
        from dr_agent.tool_interface.mcp_tools import SerperBrowseTool

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

        search_result = await search_tool("<query>OpenAI GPT-4</query>")

        # Verify search worked
        assert isinstance(search_result, DocumentToolOutput)
        assert search_result.called is True
        assert len(search_result.documents) > 0

        # Now test browsing those results
        browse_tool = SerperBrowseTool(
            tool_parser="legacy", max_pages_to_fetch=1, timeout=120
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
    async def test_mcp_connection_health(self):
        """Test that MCP client can connect and ping successfully"""

        # Create any MCP tool to test connection
        tool = SerperSearchTool(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            number_documents_to_search=1,
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
