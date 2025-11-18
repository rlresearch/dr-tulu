import asyncio
import os
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dr_agent.client import GenerateWithToolsOutput, LLMToolClient
from dr_agent.tool_interface.base import BaseTool
from dr_agent.tool_interface.data_types import ToolOutput


class MockLegacySearchTool(BaseTool):
    """Mock legacy search tool for testing without actual API calls"""

    def __init__(self, timeout: int = 10):
        super().__init__(
            tool_parser="legacy",
            tool_start_tag="<query>",
            tool_end_tag="</query>",
            result_start_tag="<snippet>",
            result_end_tag="</snippet>",
            timeout=timeout,
        )
        self.name = "MockLegacySearchTool"

    async def __call__(self, tool_input) -> ToolOutput:
        """Mock tool execution"""
        if isinstance(tool_input, str):
            call_info = self.parse_call(tool_input)
            if not call_info:
                return ToolOutput(
                    tool_name=self.name,
                    output="",
                    called=False,
                    error="No query found",
                    runtime=0.1,
                )
            query = call_info.content.strip()
        else:
            return ToolOutput(
                tool_name=self.name,
                output="",
                called=False,
                error="Invalid input type",
                runtime=0.1,
            )

        # Mock search results in legacy format
        mock_result = f"""<snippet id="1">Legacy mock result for: {query}. This is test data.\n\nAdditional legacy information about {query} from mock API.</snippet>"""

        return ToolOutput(
            output=mock_result, tool_name=self.name, called=True, runtime=0.1
        )

    def _format_output(self, output: ToolOutput) -> str:
        """Format the tool output into a string representation"""
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }


class MockUnifiedSearchTool(BaseTool):
    """Mock unified search tool for testing without actual API calls"""

    def __init__(self, timeout: int = 10):
        super().__init__(
            tool_parser="unified",
            timeout=timeout,
        )
        self.name = "MockUnifiedSearchTool"

    async def __call__(self, tool_input) -> ToolOutput:
        """Mock tool execution for unified parser"""
        if isinstance(tool_input, str):
            call_info = self.parse_call(tool_input)
            if not call_info:
                return ToolOutput(
                    tool_name=self.name,
                    output="",
                    called=False,
                    error="No tool call found",
                    runtime=0.1,
                )
            query = call_info.content.strip()
            params = call_info.parameters
        else:
            return ToolOutput(
                tool_name=self.name,
                output="",
                called=False,
                error="Invalid input type",
                runtime=0.1,
            )

        # Mock search results in unified format with parameter handling
        param_info = f" with params: {params}" if params else ""
        mock_result = f"""<snippet id="1">Unified mock result for: {query}{param_info}. This is test data.</snippet>\n<snippet id="2">Additional unified information about {query} from mock API.</snippet>"""

        return ToolOutput(
            output=mock_result, tool_name=self.name, called=True, runtime=0.1
        )

    def _format_output(self, output: ToolOutput) -> str:
        """Format the tool output into a string representation"""
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }


@pytest.fixture
def mock_legacy_search_tool():
    """Fixture providing a mock legacy search tool"""
    return MockLegacySearchTool()


@pytest.fixture
def mock_unified_search_tool():
    """Fixture providing a mock unified search tool"""
    return MockUnifiedSearchTool()


@pytest.fixture
def hosted_client_legacy(mock_legacy_search_tool):
    """Fixture for hosted LLM client (vLLM) with legacy parser"""
    return LLMToolClient(
        model_name="Qwen3-8B",
        tokenizer_name="Qwen/Qwen3-8B",
        base_url="http://localhost:30002/v1",
        api_key="dummy-key",
        tools=[mock_legacy_search_tool],
    )


@pytest.fixture
def hosted_client_unified(mock_unified_search_tool):
    """Fixture for hosted LLM client (vLLM) with unified parser"""
    return LLMToolClient(
        model_name="Qwen3-8B",
        tokenizer_name="Qwen/Qwen3-8B",
        base_url="http://localhost:30002/v1",
        api_key="dummy-key",
        tools=[mock_unified_search_tool],
    )


@pytest.fixture
def commercial_client_legacy(mock_legacy_search_tool):
    """Fixture for commercial LLM client (OpenAI) with legacy parser"""
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
    return LLMToolClient(
        model_name="gpt-4o",
        api_key=api_key,
        tools=[mock_legacy_search_tool],
    )


@pytest.fixture
def commercial_client_unified(mock_unified_search_tool):
    """Fixture for commercial LLM client (OpenAI) with unified parser"""
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key")
    return LLMToolClient(
        model_name="gpt-4o",
        api_key=api_key,
        tools=[mock_unified_search_tool],
    )


class TestLLMClientInitialization:
    """Test LLM client initialization and configuration"""

    def test_hosted_client_legacy_setup(self, hosted_client_legacy):
        """Test hosted client with legacy parser is configured correctly"""
        assert hosted_client_legacy.model_name == "Qwen3-8B"
        assert hosted_client_legacy.base_url == "http://localhost:30002/v1"
        assert hosted_client_legacy.api_key == "dummy-key"
        assert len(hosted_client_legacy.tools) == 1
        assert not hosted_client_legacy.is_commercial_api_model

    def test_hosted_client_unified_setup(self, hosted_client_unified):
        """Test hosted client with unified parser is configured correctly"""
        assert hosted_client_unified.model_name == "Qwen3-8B"
        assert hosted_client_unified.base_url == "http://localhost:30002/v1"
        assert hosted_client_unified.api_key == "dummy-key"
        assert len(hosted_client_unified.tools) == 1
        assert not hosted_client_unified.is_commercial_api_model

    def test_commercial_client_legacy_setup(self, commercial_client_legacy):
        """Test commercial client with legacy parser is configured correctly"""
        assert commercial_client_legacy.model_name == "gpt-4o"
        assert commercial_client_legacy.api_key is not None
        assert len(commercial_client_legacy.tools) == 1
        assert commercial_client_legacy.is_commercial_api_model

    def test_commercial_client_unified_setup(self, commercial_client_unified):
        """Test commercial client with unified parser is configured correctly"""
        assert commercial_client_unified.model_name == "gpt-4o"
        assert commercial_client_unified.api_key is not None
        assert len(commercial_client_unified.tools) == 1
        assert commercial_client_unified.is_commercial_api_model

    def test_legacy_tool_stop_sequences(self, hosted_client_legacy):
        """Test that legacy tool stop sequences are properly configured"""
        stop_sequences = hosted_client_legacy._get_all_stop_sequences()
        assert "</query>" in stop_sequences

    def test_unified_tool_stop_sequences(self, hosted_client_unified):
        """Test that unified tool stop sequences are properly configured"""
        stop_sequences = hosted_client_unified._get_all_stop_sequences()
        assert "</tool>" in stop_sequences


class TestLegacyToolCalling:
    """Test legacy tool calling functionality"""

    def test_find_first_tool_call(self, hosted_client_legacy):
        """Test legacy tool call detection in text"""
        text_with_tool = (
            "I need to search for <query>python programming</query> information."
        )
        tool = hosted_client_legacy._find_first_tool_call(text_with_tool)

        assert tool is not None
        assert tool.name == "MockLegacySearchTool"

        # Get call info separately
        call_info = tool.parse_call(text_with_tool)
        assert call_info is not None
        assert "python programming" in call_info.content
        assert call_info.start_pos >= 0
        assert call_info.end_pos > call_info.start_pos

    def test_no_tool_call_found(self, hosted_client_legacy):
        """Test when no legacy tool calls are present"""
        text_without_tool = "This is just regular text without any tool calls."
        tool = hosted_client_legacy._find_first_tool_call(text_without_tool)
        assert tool is None

    @pytest.mark.asyncio
    async def test_mock_legacy_tool_execution(self, mock_legacy_search_tool):
        """Test mock legacy tool execution"""
        test_input = "Please <query>machine learning basics</query> for me."
        result = await mock_legacy_search_tool(test_input)

        assert result.called
        assert "machine learning basics" in result.output
        assert "Legacy mock result" in result.output
        assert result.runtime > 0
        assert not result.error


class TestUnifiedToolCalling:
    """Test unified tool calling functionality"""

    def test_find_first_tool_call(self, hosted_client_unified):
        """Test unified tool call detection in text"""
        text_with_tool = 'I need to search for <tool name="MockUnifiedSearchTool">python programming</tool> information.'
        tool = hosted_client_unified._find_first_tool_call(text_with_tool)

        assert tool is not None
        assert tool.name == "MockUnifiedSearchTool"

        # Get call info separately
        call_info = tool.parse_call(text_with_tool)
        assert call_info is not None
        assert "python programming" in call_info.content
        assert call_info.start_pos >= 0
        assert call_info.end_pos > call_info.start_pos

    def test_unified_tool_with_parameters(self, hosted_client_unified):
        """Test unified tool call detection with parameters"""
        text_with_tool = 'Search for <tool name="MockUnifiedSearchTool" lang="en" mode="strict">artificial intelligence</tool> please.'
        tool = hosted_client_unified._find_first_tool_call(text_with_tool)

        assert tool is not None
        assert tool.name == "MockUnifiedSearchTool"

        call_info = tool.parse_call(text_with_tool)
        assert call_info is not None
        assert "artificial intelligence" in call_info.content
        assert call_info.parameters == {"lang": "en", "mode": "strict"}

    def test_no_tool_call_found(self, hosted_client_unified):
        """Test when no unified tool calls are present"""
        text_without_tool = "This is just regular text without any tool calls."
        tool = hosted_client_unified._find_first_tool_call(text_without_tool)
        assert tool is None

    @pytest.mark.asyncio
    async def test_mock_unified_tool_execution(self, mock_unified_search_tool):
        """Test mock unified tool execution"""
        test_input = '<tool name="MockUnifiedSearchTool">machine learning basics</tool>'
        result = await mock_unified_search_tool(test_input)

        assert result.called
        assert "machine learning basics" in result.output
        assert "Unified mock result" in result.output
        assert result.runtime > 0
        assert not result.error

    @pytest.mark.asyncio
    async def test_mock_unified_tool_with_params(self, mock_unified_search_tool):
        """Test mock unified tool execution with parameters"""
        test_input = '<tool name="MockUnifiedSearchTool" lang="en" timeout="30">deep learning</tool>'
        result = await mock_unified_search_tool(test_input)

        assert result.called
        assert "deep learning" in result.output
        assert "with params: {'lang': 'en', 'timeout': '30'}" in result.output
        assert result.runtime > 0
        assert not result.error


class TestMixedParserTypes:
    """Test error handling when mixing different parser types"""

    def test_mixed_parser_validation_error(self):
        """Test that mixing legacy and unified parsers raises an error"""
        legacy_tool = MockLegacySearchTool()
        unified_tool = MockUnifiedSearchTool()

        with pytest.raises(ValueError) as exc_info:
            LLMToolClient(
                model_name="gpt-4o",
                api_key="dummy-key",
                tools=[legacy_tool, unified_tool],
            )

        assert "All tools must use the same parser type" in str(exc_info.value)
        assert "LegacyToolCallParser" in str(exc_info.value)
        assert "UnifiedToolCallParser" in str(exc_info.value)

    def test_single_tool_no_validation_error(self):
        """Test that single tools don't trigger validation errors"""
        # Should work with single legacy tool
        client_legacy = LLMToolClient(
            model_name="gpt-4o",
            api_key="dummy-key",
            tools=[MockLegacySearchTool()],
        )
        assert len(client_legacy.tools) == 1

        # Should work with single unified tool
        client_unified = LLMToolClient(
            model_name="gpt-4o",
            api_key="dummy-key",
            tools=[MockUnifiedSearchTool()],
        )
        assert len(client_unified.tools) == 1

    def test_no_tools_no_validation_error(self):
        """Test that clients with no tools don't trigger validation errors"""
        client = LLMToolClient(
            model_name="gpt-4o",
            api_key="dummy-key",
            tools=[],
        )
        assert len(client.tools) == 0

    def test_same_parser_type_no_error(self):
        """Test that using multiple tools of the same parser type works"""
        # Multiple legacy tools should work
        legacy_tool1 = MockLegacySearchTool()
        legacy_tool2 = MockLegacySearchTool()
        legacy_tool2.name = "MockLegacySearchTool2"

        client_legacy = LLMToolClient(
            model_name="gpt-4o",
            api_key="dummy-key",
            tools=[legacy_tool1, legacy_tool2],
        )
        assert len(client_legacy.tools) == 2

        # Multiple unified tools should work
        unified_tool1 = MockUnifiedSearchTool()
        unified_tool2 = MockUnifiedSearchTool()
        unified_tool2.name = "MockUnifiedSearchTool2"

        client_unified = LLMToolClient(
            model_name="gpt-4o",
            api_key="dummy-key",
            tools=[unified_tool1, unified_tool2],
        )
        assert len(client_unified.tools) == 2


class TestHostedLLMIntegration:
    """Test hosted LLM (vLLM) integration"""

    @pytest.mark.asyncio
    async def test_hosted_llm_legacy_with_mock_response(self, hosted_client_legacy):
        """Test hosted LLM with legacy parser and mocked API response"""

        # Mock the vLLM API response with legacy format
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].text = (
            "I'll search for that. <query>test query</query>"
        )

        with patch(
            "litellm.atext_completion", new_callable=AsyncMock
        ) as mock_completion:
            mock_completion.return_value = mock_response

            messages = [{"role": "user", "content": "Tell me about AI"}]

            result = await hosted_client_legacy.generate_with_tools(
                messages, max_tool_calls=1, max_tokens=100
            )

            assert isinstance(result, GenerateWithToolsOutput)
            assert result.tool_call_count >= 0
            assert result.total_tokens > 0
            assert mock_completion.called

    @pytest.mark.asyncio
    async def test_hosted_llm_unified_with_mock_response(self, hosted_client_unified):
        """Test hosted LLM with unified parser and mocked API response"""

        # Mock the vLLM API response with unified format
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].text = (
            'I\'ll search for that. <tool name="MockUnifiedSearchTool">test query</tool>'
        )

        with patch(
            "litellm.atext_completion", new_callable=AsyncMock
        ) as mock_completion:
            mock_completion.return_value = mock_response

            messages = [{"role": "user", "content": "Tell me about AI"}]

            result = await hosted_client_unified.generate_with_tools(
                messages, max_tool_calls=1, max_tokens=100
            )

            assert isinstance(result, GenerateWithToolsOutput)
            assert result.tool_call_count >= 0
            assert result.total_tokens > 0
            assert mock_completion.called

    def test_hosted_model_detection(self):
        """Test that hosted models are correctly identified"""
        test_cases = [
            ("Qwen/Qwen3-8B", False),
            ("meta-llama/Llama-2-7b-chat", False),
            ("mistralai/Mixtral-8x7B-Instruct-v0.1", False),
        ]

        for model_name, expected_commercial in test_cases:
            client = LLMToolClient(model_name=model_name, api_key="dummy")
            assert client.is_commercial_api_model == expected_commercial


class TestCommercialLLMIntegration:
    """Test commercial LLM integration"""

    @pytest.mark.asyncio
    async def test_commercial_llm_legacy_with_mock_response(
        self, commercial_client_legacy
    ):
        """Test commercial LLM with legacy parser and mocked API response"""

        # Mock the commercial API response with legacy format
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = (
            "I'll help you with that. <query>AI research</query>"
        )
        mock_response.choices[0].finish_reason = "stop"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            messages = [{"role": "user", "content": "Tell me about AI"}]

            result = await commercial_client_legacy.generate_with_tools(
                messages, max_tool_calls=1, max_tokens=100
            )

            assert isinstance(result, GenerateWithToolsOutput)
            assert result.tool_call_count >= 0
            assert result.total_tokens > 0
            assert mock_completion.called

    @pytest.mark.asyncio
    async def test_commercial_llm_unified_with_mock_response(
        self, commercial_client_unified
    ):
        """Test commercial LLM with unified parser and mocked API response"""

        # Mock the commercial API response with unified format
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = (
            'I\'ll help you with that. <tool name="MockUnifiedSearchTool">AI research</tool>'
        )
        mock_response.choices[0].finish_reason = "stop"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            messages = [{"role": "user", "content": "Tell me about AI"}]

            result = await commercial_client_unified.generate_with_tools(
                messages, max_tool_calls=1, max_tokens=100
            )

            assert isinstance(result, GenerateWithToolsOutput)
            assert result.tool_call_count >= 0
            assert result.total_tokens > 0
            assert mock_completion.called

    def test_commercial_model_detection(self):
        """Test that commercial models are correctly identified"""
        test_cases = [
            ("gpt-4", True),
            ("gpt-3.5-turbo", True),
            ("gpt-4o-mini", True),
            ("claude-3-opus-20240229", True),
            ("claude-3-5-sonnet-20241022", True),
            ("gemini-pro", True),
        ]

        for model_name, expected_commercial in test_cases:
            client = LLMToolClient(model_name=model_name, api_key="dummy")
            assert client.is_commercial_api_model == expected_commercial


class TestTokenCounting:
    """Test token counting functionality"""

    def test_token_counting_fallback_legacy(self, hosted_client_legacy):
        """Test fallback token counting method with legacy parser"""
        test_text = "This is a test sentence with some words."
        token_count = hosted_client_legacy._count_tokens(test_text)

        # Should return a reasonable estimate (characters / 4)
        expected_estimate = len(test_text) // 4
        assert token_count >= expected_estimate - 5  # Allow some variance
        assert token_count <= expected_estimate + 5

    def test_token_counting_fallback_unified(self, hosted_client_unified):
        """Test fallback token counting method with unified parser"""
        test_text = "This is a test sentence with some words."
        token_count = hosted_client_unified._count_tokens(test_text)

        # Should return a reasonable estimate (characters / 4)
        expected_estimate = len(test_text) // 4
        assert token_count >= expected_estimate - 5  # Allow some variance
        assert token_count <= expected_estimate + 5

    def test_message_token_counting_legacy(self, commercial_client_legacy):
        """Test token counting for message format with legacy parser"""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        token_count = commercial_client_legacy._count_tokens_messages(messages)
        assert token_count > 0

    def test_message_token_counting_unified(self, commercial_client_unified):
        """Test token counting for message format with unified parser"""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        token_count = commercial_client_unified._count_tokens_messages(messages)
        assert token_count > 0


@pytest.mark.integration
class TestRealIntegration:
    """Integration tests that require real API access (marked as integration)"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_real_openai_integration_legacy(self):
        """Test actual OpenAI API call with legacy parser (requires API key)"""

        mock_tool = MockLegacySearchTool()

        client = LLMToolClient(
            model_name="gpt-4o",  # Use cheaper model for testing
            api_key=os.getenv("OPENAI_API_KEY"),
            tools=[mock_tool],
        )

        result = await client.generate_with_tools(
            [
                {
                    "role": "user",
                    "content": "Hello, can you search for <query>AI</query>?",
                }
            ],
            max_tool_calls=1,
            max_tokens=50,
        )

        assert isinstance(result, GenerateWithToolsOutput)
        assert result.total_tokens > 0

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_real_openai_integration_unified(self):
        """Test actual OpenAI API call with unified parser (requires API key)"""

        mock_tool = MockUnifiedSearchTool()

        client = LLMToolClient(
            model_name="gpt-4o",  # Use cheaper model for testing
            api_key=os.getenv("OPENAI_API_KEY"),
            tools=[mock_tool],
        )

        result = await client.generate_with_tools(
            [
                {
                    "role": "user",
                    "content": 'Hello, can you search for <tool name="MockUnifiedSearchTool">AI</tool>?',
                }
            ],
            max_tool_calls=1,
            max_tokens=50,
        )

        assert isinstance(result, GenerateWithToolsOutput)
        assert result.total_tokens > 0

    @pytest.mark.skipif(
        not all([os.getenv("HOSTED_LLM_URL"), os.getenv("HOSTED_LLM_API_KEY")]),
        reason="Hosted LLM environment variables not set",
    )
    @pytest.mark.asyncio
    async def test_real_hosted_llm_integration_legacy(self):
        """Test actual hosted LLM call with legacy parser (requires setup)"""

        mock_tool = MockLegacySearchTool()

        client = LLMToolClient(
            model_name="Qwen3-8B",
            tokenizer_name="Qwen/Qwen3-8B",
            base_url=os.getenv("HOSTED_LLM_URL"),
            api_key=os.getenv("HOSTED_LLM_API_KEY"),
            tools=[mock_tool],
        )

        result = await client.generate_with_tools(
            [
                {
                    "role": "user",
                    "content": "Hello, please search for <query>test</query>",
                }
            ],
            max_tool_calls=1,
            max_tokens=50,
        )

        assert isinstance(result, GenerateWithToolsOutput)
        assert result.total_tokens > 0

    @pytest.mark.skipif(
        not all([os.getenv("HOSTED_LLM_URL"), os.getenv("HOSTED_LLM_API_KEY")]),
        reason="Hosted LLM environment variables not set",
    )
    @pytest.mark.asyncio
    async def test_real_hosted_llm_integration_unified(self):
        """Test actual hosted LLM call with unified parser (requires setup)"""

        mock_tool = MockUnifiedSearchTool()

        client = LLMToolClient(
            model_name="Qwen3-8B",
            tokenizer_name="Qwen/Qwen3-8B",
            base_url=os.getenv("HOSTED_LLM_URL"),
            api_key=os.getenv("HOSTED_LLM_API_KEY"),
            tools=[mock_tool],
        )

        result = await client.generate_with_tools(
            [
                {
                    "role": "user",
                    "content": 'Hello, please search for <tool name="MockUnifiedSearchTool">test</tool>',
                }
            ],
            max_tool_calls=1,
            max_tokens=50,
        )

        assert isinstance(result, GenerateWithToolsOutput)
        assert result.total_tokens > 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
