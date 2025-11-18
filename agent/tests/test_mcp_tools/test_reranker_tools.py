import asyncio
import os
import warnings
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dr_agent.tool_interface.data_types import Document, DocumentToolOutput
from dr_agent.tool_interface.mcp_tools import VllmHostedRerankerTool
from dr_agent.tool_interface.tool_parsers import LegacyToolCallParser


class TestVllmHostedRerankerTool:
    """Test VllmHostedRerankerTool functionality"""

    @pytest.fixture
    def sample_documents(self):
        """Fixture providing sample documents for testing"""
        return [
            Document(
                id="doc-1",
                title="Python Programming Guide",
                snippet="Learn Python programming from scratch with comprehensive examples.",
                url="https://example.com/python-guide",
                text=None,
                score=0.85,
            ),
            Document(
                id="doc-2",
                title="Advanced Python Techniques",
                snippet="Master advanced concepts like decorators and metaclasses in Python.",
                url="https://example.com/advanced-python",
                text=None,
                score=0.72,
            ),
            Document(
                id="doc-3",
                title="Python Best Practices",
                snippet="Follow these Python best practices for clean, maintainable code.",
                url="https://example.com/python-best-practices",
                text=None,
                score=0.68,
            ),
        ]

    @pytest.fixture
    def sample_document_tool_output(self, sample_documents):
        """Fixture providing DocumentToolOutput with sample documents"""
        return DocumentToolOutput(
            tool_name="SerperSearchTool",
            output="Search results for python programming",
            called=True,
            error="",
            timeout=False,
            runtime=1.5,
            call_id="search-123",
            raw_output={
                "organic": [{"title": "Python Guide", "snippet": "Learn Python"}],
                "SearchParameters": {"q": "python programming"},
            },
            documents=sample_documents,
            query="python programming",
        )

    @pytest.fixture
    def reranker_tool_legacy(self):
        """Fixture providing a VllmHostedRerankerTool with legacy parser"""
        return VllmHostedRerankerTool(
            model_name="BAAI/bge-reranker-v2-m3",
            api_url="http://localhost:8000",
            tool_parser="legacy",
            tool_start_tag="<rerank>",
            tool_end_tag="</rerank>",
            result_start_tag="<reranked>",
            result_end_tag="</reranked>",
            top_n=3,
            timeout=60,
        )

    @pytest.fixture
    def reranker_tool_unified(self):
        """Fixture providing a VllmHostedRerankerTool with unified parser"""
        return VllmHostedRerankerTool(
            model_name="BAAI/bge-reranker-v2-m3",
            api_url="http://localhost:8000",
            tool_parser="unified",
            top_n=3,
            timeout=60,
        )

    @pytest.fixture
    def mock_reranker_response(self):
        """Mock VLLM reranker API response"""
        return {
            "results": [
                {"index": 1, "relevance_score": 0.95},  # Advanced Python (was index 1)
                {"index": 0, "relevance_score": 0.88},  # Python Guide (was index 0)
                {"index": 2, "relevance_score": 0.75},  # Best Practices (was index 2)
            ]
        }

    def test_tool_configuration(self, reranker_tool_legacy):
        """Test that VllmHostedRerankerTool is configured correctly"""
        assert reranker_tool_legacy.model_name == "BAAI/bge-reranker-v2-m3"
        assert reranker_tool_legacy.api_url == "http://localhost:8000"
        assert reranker_tool_legacy.top_n == 3
        assert reranker_tool_legacy.timeout == 60
        assert reranker_tool_legacy.get_mcp_tool_name() == "vllm_hosted_reranker"

    def test_mcp_params_generation(self, reranker_tool_legacy):
        """Test MCP parameter generation"""
        query = "python programming"
        documents = ["Learn Python", "Advanced Python", "Python practices"]
        top_n = 2

        params = reranker_tool_legacy.get_mcp_params(query, documents, top_n)

        expected_params = {
            "query": "python programming",
            "documents": ["Learn Python", "Advanced Python", "Python practices"],
            "top_n": 2,
            "model_name": "BAAI/bge-reranker-v2-m3",
            "api_url": "http://localhost:8000",
        }

        assert params == expected_params

    def test_query_extraction_from_document_tool_output(
        self, reranker_tool_legacy, sample_document_tool_output
    ):
        """Test query extraction from DocumentToolOutput"""
        query = reranker_tool_legacy._extract_query_from_input(
            sample_document_tool_output
        )
        assert query == "python programming"

    def test_query_extraction_from_raw_output_search_parameters(
        self, reranker_tool_legacy
    ):
        """Test query extraction from raw_output SearchParameters"""
        tool_output = DocumentToolOutput(
            tool_name="SerperSearchTool",
            output="Search results",
            called=True,
            error="",
            timeout=False,
            runtime=1.0,
            call_id="test-123",
            raw_output={"SearchParameters": {"q": "machine learning"}},
            documents=[],
            query=None,  # No direct query
        )

        query = reranker_tool_legacy._extract_query_from_input(tool_output)
        assert query == "machine learning"

    def test_query_extraction_fallback_methods(self, reranker_tool_legacy):
        """Test various fallback methods for query extraction"""
        # Test with different query field names
        tool_output = DocumentToolOutput(
            tool_name="SearchTool",
            output="Results",
            called=True,
            error="",
            timeout=False,
            runtime=1.0,
            call_id="test-123",
            raw_output={"search_query": "deep learning"},
            documents=[],
            query=None,
        )

        query = reranker_tool_legacy._extract_query_from_input(tool_output)
        assert query == "deep learning"

    def test_query_extraction_no_query_found(self, reranker_tool_legacy):
        """Test when no query can be extracted"""
        tool_output = DocumentToolOutput(
            tool_name="SearchTool",
            output="Results",
            called=True,
            error="",
            timeout=False,
            runtime=1.0,
            call_id="test-123",
            raw_output={"no_query_field": "value"},
            documents=[],
            query=None,
        )

        query = reranker_tool_legacy._extract_query_from_input(tool_output)
        assert query is None

    @pytest.mark.asyncio
    async def test_successful_reranking_legacy(
        self, reranker_tool_legacy, sample_document_tool_output, mock_reranker_response
    ):
        """Test successful reranking with legacy format"""
        with patch.object(
            reranker_tool_legacy, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = mock_reranker_response

            result = await reranker_tool_legacy(sample_document_tool_output)

            # Verify result structure
            assert isinstance(result, DocumentToolOutput)
            assert result.called is True
            assert result.error == ""
            assert len(result.documents) == 3
            assert result.query == "python programming"

            # Verify documents are reranked by score
            sorted_docs = sorted(result.documents, key=lambda x: x.score, reverse=True)
            assert sorted_docs[0].title == "Advanced Python Techniques"  # Highest score
            assert sorted_docs[0].score == 0.95
            assert sorted_docs[1].title == "Python Programming Guide"
            assert sorted_docs[1].score == 0.88
            assert sorted_docs[2].title == "Python Best Practices"
            assert sorted_docs[2].score == 0.75

            # Verify MCP call parameters
            expected_documents = [
                "Title: Python Programming Guide\nURL: https://example.com/python-guide\nSearch Snippet: Learn Python programming from scratch with comprehensive examples.",
                "Title: Advanced Python Techniques\nURL: https://example.com/advanced-python\nSearch Snippet: Master advanced concepts like decorators and metaclasses in Python.",
                "Title: Python Best Practices\nURL: https://example.com/python-best-practices\nSearch Snippet: Follow these Python best practices for clean, maintainable code.",
            ]
            mock_execute.assert_called_once_with(
                "vllm_hosted_reranker",
                {
                    "query": "python programming",
                    "documents": expected_documents,
                    "top_n": 3,
                    "model_name": "BAAI/bge-reranker-v2-m3",
                    "api_url": "http://localhost:8000",
                },
            )

    @pytest.mark.asyncio
    async def test_successful_reranking_unified(
        self, reranker_tool_unified, sample_document_tool_output, mock_reranker_response
    ):
        """Test successful reranking with unified format"""
        with patch.object(
            reranker_tool_unified, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = mock_reranker_response

            result = await reranker_tool_unified(sample_document_tool_output)

            # Verify result structure
            assert isinstance(result, DocumentToolOutput)
            assert result.called is True
            assert result.error == ""
            assert len(result.documents) == 3
            assert result.query == "python programming"

            # Verify documents are reranked
            sorted_docs = sorted(result.documents, key=lambda x: x.score, reverse=True)
            assert sorted_docs[0].score == 0.95  # Highest rerank score

    @pytest.mark.asyncio
    async def test_top_n_limiting(
        self, reranker_tool_legacy, sample_document_tool_output, mock_reranker_response
    ):
        """Test that top_n parameter limits results correctly"""
        # Create a reranker tool with top_n=2
        limited_reranker = VllmHostedRerankerTool(
            model_name="BAAI/bge-reranker-v2-m3",
            api_url="http://localhost:30001",
            tool_parser="legacy",
            tool_start_tag="<rerank>",
            tool_end_tag="</rerank>",
            result_start_tag="<reranked>",
            result_end_tag="</reranked>",
            top_n=2,  # Limit to 2 results
        )

        with patch.object(
            limited_reranker, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            # Mock response with only 2 results
            mock_execute.return_value = {
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.88},
                ]
            }

            result = await limited_reranker(sample_document_tool_output)

            # Should only have 2 documents
            assert len(result.documents) == 2

            # Verify the top_n parameter was passed correctly
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args.args[-1]  # Get keyword arguments
            assert call_args["top_n"] == 2

    @pytest.mark.asyncio
    async def test_auto_top_n_when_negative(
        self, sample_document_tool_output, mock_reranker_response
    ):
        """Test that negative top_n uses all documents"""
        # Create reranker with top_n=-1 (use all documents)
        auto_reranker = VllmHostedRerankerTool(
            model_name="BAAI/bge-reranker-v2-m3",
            api_url="http://localhost:8000",
            top_n=-1,  # Use all documents
        )

        with patch.object(
            auto_reranker, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = mock_reranker_response

            result = await auto_reranker(sample_document_tool_output)

            # Should use all 3 documents
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args.args[-1]
            assert call_args["top_n"] == 3  # Should be set to number of documents

    @pytest.mark.asyncio
    async def test_documents_with_text_content(
        self, reranker_tool_legacy, mock_reranker_response
    ):
        """Test reranking with documents that have text content instead of snippets"""
        documents_with_text = [
            Document(
                id="doc-1",
                title="Python Guide",
                snippet="",  # Empty snippet
                url="https://example.com/python",
                text="This is the full text content of the Python programming guide...",
                score=0.8,
            ),
            Document(
                id="doc-2",
                title="Python Advanced",
                snippet=None,  # No snippet
                url="https://example.com/advanced",
                text="Advanced Python techniques including decorators and generators...",
                score=0.7,
            ),
        ]

        tool_output = DocumentToolOutput(
            tool_name="BrowseTool",
            output="Browsed content",
            called=True,
            error="",
            timeout=False,
            runtime=2.0,
            call_id="browse-123",
            raw_output=None,
            documents=documents_with_text,
            query="python programming",
        )

        with patch.object(
            reranker_tool_legacy, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = {
                "results": [
                    {"index": 1, "relevance_score": 0.92},
                    {"index": 0, "relevance_score": 0.85},
                ]
            }

            result = await reranker_tool_legacy(tool_output)

            # Should use text content for reranking since snippets are empty/None
            expected_documents = [
                "Title: Python Guide\nURL: https://example.com/python\nFull Text: This is the full text content of the Python programming guide...",
                "Title: Python Advanced\nURL: https://example.com/advanced\nFull Text: Advanced Python techniques including decorators and generators...",
            ]
            mock_execute.assert_called_once_with(
                "vllm_hosted_reranker",
                {
                    "query": "python programming",
                    "documents": expected_documents,
                    "top_n": 2,
                    "model_name": "BAAI/bge-reranker-v2-m3",
                    "api_url": "http://localhost:8000",
                },
            )

    @pytest.mark.asyncio
    async def test_invalid_input_type(self, reranker_tool_legacy):
        """Test handling of invalid input types"""
        # Test with string input (should fail)
        with pytest.raises(ValueError, match="expects DocumentToolOutput as input"):
            await reranker_tool_legacy("invalid string input")

        # Test with regular ToolOutput (should fail)
        from dr_agent.tool_interface.data_types import ToolOutput

        regular_output = ToolOutput(
            tool_name="VllmHostedRerankerTool",
            output="some output",
            called=True,
            error="",
            timeout=False,
            runtime=1.0,
        )

        with pytest.raises(ValueError, match="expects DocumentToolOutput as input"):
            await reranker_tool_legacy(regular_output)

    @pytest.mark.asyncio
    async def test_no_documents_in_input(self, reranker_tool_legacy):
        """Test handling when DocumentToolOutput has no documents"""
        empty_tool_output = DocumentToolOutput(
            tool_name="SearchTool",
            output="No results",
            called=True,
            error="",
            timeout=False,
            runtime=1.0,
            call_id="empty-123",
            raw_output=None,
            documents=[],  # Empty documents list
            query="test query",
        )

        with pytest.raises(ValueError, match="does not contain documents to rerank"):
            await reranker_tool_legacy(empty_tool_output)

    @pytest.mark.asyncio
    async def test_no_query_extraction_warning(
        self, reranker_tool_legacy, sample_documents
    ):
        """Test warning when query cannot be extracted"""
        tool_output_no_query = DocumentToolOutput(
            tool_name="SearchTool",
            output="Results",
            called=True,
            error="",
            timeout=False,
            runtime=1.0,
            call_id="no-query-123",
            raw_output={},  # No query information
            documents=sample_documents,
            query=None,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = await reranker_tool_legacy(tool_output_no_query)

            # Should have issued a warning
            assert len(w) >= 1
            assert "Could not extract query from input for reranking" in str(
                w[0].message
            )

        # Should return error
        assert "Could not extract query from input" in result.error

    # @pytest.mark.asyncio
    # async def test_no_valid_document_texts(self, reranker_tool_legacy):
    #     """Test handling when documents have no valid text content"""
    #     empty_text_documents = [
    #         Document(
    #             id="doc-1",
    #             title="Empty Doc",
    #             snippet="",  # Empty
    #             url="https://example.com",
    #             text=None,  # No text
    #             score=0.8,
    #         ),
    #         Document(
    #             id="doc-2",
    #             title="Whitespace Doc",
    #             snippet="   ",  # Only whitespace
    #             url="https://example.com",
    #             text="",  # Empty text
    #             score=0.7,
    #         ),
    #     ]

    #     tool_output = DocumentToolOutput(
    #         tool_name="SearchTool",
    #         output="Results",
    #         called=True,
    #         error="",
    #         timeout=False,
    #         runtime=1.0,
    #         call_id="empty-text-123",
    #         raw_output=None,
    #         documents=empty_text_documents,
    #         query="test query",
    #     )

    #     result = await reranker_tool_legacy(tool_output)

    #     assert isinstance(result, DocumentToolOutput)
    #     assert "No valid document texts found for reranking" in result.error

    @pytest.mark.asyncio
    async def test_mcp_call_error(
        self, reranker_tool_legacy, sample_document_tool_output
    ):
        """Test handling of MCP call errors"""
        with patch.object(
            reranker_tool_legacy, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = {"error": "Reranker service unavailable"}

            result = await reranker_tool_legacy(sample_document_tool_output)

            # Verify error handling
            assert isinstance(result, DocumentToolOutput)
            assert result.called is True
            assert "Reranking failed: Reranker service unavailable" in result.error
            assert len(result.documents) == 3  # Should preserve original documents

    @pytest.mark.asyncio
    async def test_empty_reranker_results(
        self, reranker_tool_legacy, sample_document_tool_output
    ):
        """Test handling when reranker returns empty results"""
        with patch.object(
            reranker_tool_legacy, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = {"results": []}  # Empty results

            result = await reranker_tool_legacy(sample_document_tool_output)

            assert isinstance(result, DocumentToolOutput)
            assert "No reranked results returned from reranker" in result.error

    @pytest.mark.asyncio
    async def test_malformed_reranker_results(
        self, reranker_tool_legacy, sample_document_tool_output
    ):
        """Test handling of malformed reranker response"""
        with patch.object(
            reranker_tool_legacy, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            # Mock malformed response (missing index or score)
            mock_execute.return_value = {
                "results": [
                    {"relevance_score": 0.95},  # Missing index
                    {"index": 1},  # Missing relevance_score
                    {"index": "invalid", "relevance_score": 0.75},  # Invalid index
                ]
            }

            result = await reranker_tool_legacy(sample_document_tool_output)

            # Should handle gracefully and return whatever valid documents it can
            assert isinstance(result, DocumentToolOutput)
            # The exact behavior depends on implementation, but should not crash

    @pytest.mark.asyncio
    async def test_reranker_index_out_of_bounds(
        self, reranker_tool_legacy, sample_document_tool_output
    ):
        """Test handling when reranker returns invalid document indices"""
        with patch.object(
            reranker_tool_legacy, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            # Mock response with out-of-bounds indices
            mock_execute.return_value = {
                "results": [
                    {"index": 0, "relevance_score": 0.95},  # Valid
                    {
                        "index": 5,
                        "relevance_score": 0.88,
                    },  # Out of bounds (only 3 docs)
                    {"index": -1, "relevance_score": 0.75},  # Negative index
                ]
            }

            result = await reranker_tool_legacy(sample_document_tool_output)

            # Should only include valid documents
            assert isinstance(result, DocumentToolOutput)
            assert len(result.documents) == 1  # Only the valid index=0 document

    def test_output_formatting_legacy(self, reranker_tool_legacy, sample_documents):
        """Test output formatting for legacy format"""
        # Create a simple result to test formatting
        result_output = DocumentToolOutput(
            tool_name="VllmHostedRerankerTool",
            output="",
            called=True,
            error="",
            timeout=False,
            runtime=1.0,
            call_id="format-test-123",
            raw_output=None,
            documents=sample_documents,
            query="test query",
        )

        formatted = reranker_tool_legacy._format_output(result_output)

        # For legacy format, should just return the output directly
        assert formatted == result_output.output

    def test_output_formatting_unified(self, reranker_tool_unified, sample_documents):
        """Test output formatting for unified format"""
        # Create a simple result to test formatting
        result_output = DocumentToolOutput(
            tool_name="VllmHostedRerankerTool",
            output="Formatted output content",
            called=True,
            error="",
            timeout=False,
            runtime=1.0,
            call_id="format-test-123",
            raw_output=None,
            documents=sample_documents,
            query="test query",
        )

        formatted = reranker_tool_unified._format_output(result_output)

        # For unified format, should return the output directly (since reranker doesn't use tags)
        assert formatted == "Formatted output content"

    def test_document_id_preservation(self, reranker_tool_legacy):
        """Test that document IDs are preserved through reranking"""
        original_docs = [
            Document(
                id="custom-id-1", title="Doc 1", snippet="Content 1", url="", score=0.5
            ),
            Document(
                id="custom-id-2", title="Doc 2", snippet="Content 2", url="", score=0.6
            ),
        ]

        # Test the _process_reranker_results method directly
        mock_response = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
            ]
        }

        reranked_docs = reranker_tool_legacy._process_reranker_results(
            mock_response, original_docs, "test query"
        )

        assert len(reranked_docs) == 2
        assert reranked_docs[0].id == "custom-id-2"  # Index 1 from original
        assert reranked_docs[0].score == 0.95
        assert reranked_docs[1].id == "custom-id-1"  # Index 0 from original
        assert reranked_docs[1].score == 0.85

    @pytest.mark.asyncio
    async def test_exception_handling(
        self, reranker_tool_legacy, sample_document_tool_output
    ):
        """Test general exception handling in the reranker"""
        with patch.object(
            reranker_tool_legacy, "_execute_mcp_call", new_callable=AsyncMock
        ) as mock_execute:
            # Mock an exception during MCP call
            mock_execute.side_effect = Exception("Connection error")

            result = await reranker_tool_legacy(sample_document_tool_output)

            assert isinstance(result, DocumentToolOutput)
            assert result.called is True
            assert "Error processing VllmHostedRerankerTool" in result.error
            assert "Connection error" in result.error
