from typing import Union

import pytest

from dr_agent.tool_interface.base import BaseTool
from dr_agent.tool_interface.data_types import ToolInput, ToolOutput
from dr_agent.tool_interface.tool_parsers import (
    ToolCallInfo,
    create_tool_parser,
    get_registered_parsers,
)


class LegacySearchTool(BaseTool):
    """Test implementation of legacy tool for testing"""

    def __init__(self):
        super().__init__(
            tool_parser="legacy",
            tool_start_tag="<search>",
            tool_end_tag="</search>",
            result_start_tag="<result>",
            result_end_tag="</result>",
        )
        self.name = "legacy_search"

    def __call__(self, tool_input: Union[str, ToolInput, ToolOutput]) -> ToolOutput:
        if isinstance(tool_input, str):
            call_info = self.parse_call(tool_input)

            if call_info is None:
                return self._create_error_output(
                    f"No tool call found in input: {tool_input}",
                    call_id="dummy_call_id",
                    runtime=0.1,
                )

            return ToolOutput(
                tool_name=self.name,
                output=call_info.content,
                called=True,
                runtime=0.1,
            )

        return self._create_error_output(
            f"Unsupported input type: {type(tool_input)}",
            call_id="dummy_call_id",
            runtime=0.1,
        )

    def _format_output(self, output: ToolOutput) -> str:
        return f"{output.output}"

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }


class UnifiedSearchTool(BaseTool):
    """Test implementation of unified tool for testing"""

    def __init__(self):
        super().__init__(tool_parser="unified")
        self.name = "search"

    def __call__(self, tool_input: Union[str, ToolInput, ToolOutput]) -> ToolOutput:
        if isinstance(tool_input, dict):
            # Unified tool with parameters
            content = tool_input.get("content", "")
            params = {k: v for k, v in tool_input.items() if k != "content"}
            output = f"Unified search executed with content: '{content}'"
            if params:
                output += f" and parameters: {params}"
        elif isinstance(tool_input, str):
            # Simple content only
            output = f"Unified search executed with: {tool_input}"
        else:
            return self._create_error_output(
                f"Unsupported input type: {type(tool_input)}",
                call_id="dummy_call_id",
                runtime=0.1,
            )

        return ToolOutput(
            tool_name=self.name,
            output=output,
            called=True,
            runtime=0.1,
        )

    def _format_output(self, output: ToolOutput) -> str:
        return f"{output.output}"

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }


@pytest.fixture
def legacy_tool():
    """Fixture providing a legacy search tool"""
    return LegacySearchTool()


@pytest.fixture
def unified_tool():
    """Fixture providing a unified search tool"""
    return UnifiedSearchTool()


class TestLegacyToolFunctionality:
    """Test legacy tool functionality"""

    def test_legacy_tool_initialization(self, legacy_tool):
        """Test legacy tool is initialized correctly"""
        assert legacy_tool.name == "legacy_search"
        assert legacy_tool.tool_parser.tool_start_tag == "<search>"
        assert legacy_tool.tool_parser.tool_end_tag == "</search>"
        assert legacy_tool.tool_parser.result_start_tag == "<result>"
        assert legacy_tool.tool_parser.result_end_tag == "</result>"

        assert legacy_tool.stop_sequences == ["</search>"]

    def test_legacy_has_calls_positive(self, legacy_tool):
        """Test legacy tool correctly detects tool calls"""
        test_cases = [
            "I need to search for <search>machine learning</search>",
            "<search>python programming</search> please",
            "Find <search>artificial intelligence</search> information",
            "Multiple calls: <search>AI</search> and then <search>ML</search>",
        ]

        for test_input in test_cases:
            assert legacy_tool.has_calls(
                test_input
            ), f"Should detect call in: {test_input}"

    def test_legacy_has_calls_negative(self, legacy_tool):
        """Test legacy tool correctly rejects non-tool text"""
        test_cases = [
            "I need to search for machine learning",  # No tags
            "I need to search for <search>machine learning",  # Missing end tag
            "machine learning</search>",  # Missing start tag
            "<other>not a search</other>",  # Wrong tool
            "",  # Empty string
        ]

        for test_input in test_cases:
            assert not legacy_tool.has_calls(
                test_input
            ), f"Should not detect call in: {test_input}"

    def test_legacy_parse_call_success(self, legacy_tool):
        """Test legacy tool successfully parses tool calls"""
        test_cases = [
            {
                "input": "I need to search for <search>machine learning</search>",
                "expected_content": "machine learning",
                "expected_params": {},
            },
            {
                "input": "<search>python programming basics</search> for beginners",
                "expected_content": "python programming basics",
                "expected_params": {},
            },
            {
                "input": "Find <search>  artificial intelligence  </search> please",
                "expected_content": "artificial intelligence",  # Should be stripped
                "expected_params": {},
            },
        ]

        for test_case in test_cases:
            call_info = legacy_tool.parse_call(test_case["input"])
            assert call_info is not None, f"Should parse: {test_case['input']}"
            assert isinstance(call_info, ToolCallInfo)
            assert call_info.content == test_case["expected_content"]
            assert call_info.parameters == test_case["expected_params"]
            assert call_info.start_pos >= 0
            assert call_info.end_pos > call_info.start_pos

    def test_legacy_parse_call_failure(self, legacy_tool):
        """Test legacy tool handles invalid inputs correctly"""
        test_cases = [
            "I need to search for machine learning",  # No tags
            "I need to search for <search>machine learning",  # Missing end tag
            "machine learning</search>",  # Missing start tag
            "<other>not a search</other>",  # Wrong tool
            "",  # Empty string
        ]

        for test_input in test_cases:
            call_info = legacy_tool.parse_call(test_input)
            assert call_info is None, f"Should not parse: {test_input}"

    def test_legacy_tool_execution_success(self, legacy_tool):
        """Test successful legacy tool execution"""
        test_input = "I need to search for <search>machine learning</search>"
        result = legacy_tool(test_input)

        assert result.called
        assert result.error is None
        assert result.output == "machine learning"
        assert result.tool_name == "legacy_search"
        assert result.runtime > 0

    def test_legacy_tool_execution_error(self, legacy_tool):
        """Test legacy tool execution with invalid input"""
        test_cases = [
            "I need to search for machine learning",  # No tags
            "I need to search for <search>machine learning",  # Missing end tag
            "",  # Empty string
        ]

        for test_input in test_cases:
            result = legacy_tool(test_input)
            assert not result.called
            assert result.error is not None
            assert "No tool call found" in result.error

    def test_legacy_format_result(self, legacy_tool):
        """Test legacy tool result formatting"""
        mock_output = ToolOutput(
            tool_name="test", output="test result", called=True, runtime=0.1
        )

        formatted = legacy_tool.format_result(mock_output)
        assert formatted == "<result>\ntest result\n</result>"


class TestUnifiedToolFunctionality:
    """Test unified tool functionality"""

    def test_unified_tool_initialization(self, unified_tool):
        """Test unified tool is initialized correctly"""
        assert unified_tool.name == "search"
        # Unified parser uses different output format, no result tags
        assert unified_tool.stop_sequences == ["</tool>"]

    def test_unified_has_calls_positive(self, unified_tool):
        """Test unified tool correctly detects tool calls"""
        test_cases = [
            '<tool name="search">python programming</tool>',
            'Please <tool name="search" lang="en">machine learning</tool> for me',
            '<tool name="search" mode="strict" lang="en">artificial intelligence</tool>',
            'Multiple: <tool name="search">AI</tool> and <tool name="other">ML</tool>',
        ]

        for test_input in test_cases:
            assert unified_tool.has_calls(
                test_input
            ), f"Should detect call in: {test_input}"

    def test_unified_has_calls_negative(self, unified_tool):
        """Test unified tool correctly rejects non-matching text"""
        test_cases = [
            '<tool name="other_tool">not for search</tool>',  # Wrong tool name
            "Please search for python programming",  # No tags
            '<tool name="search">incomplete',  # Missing end tag
            "<search>legacy format</search>",  # Legacy format
            "",  # Empty string
        ]

        for test_input in test_cases:
            assert not unified_tool.has_calls(
                test_input
            ), f"Should not detect call in: {test_input}"

    def test_unified_parse_call_success(self, unified_tool):
        """Test unified tool successfully parses tool calls with parameters"""
        test_cases = [
            {
                "input": '<tool name="search">python programming</tool>',
                "expected_content": "python programming",
                "expected_params": {},
            },
            {
                "input": '<tool name="search" lang="en">machine learning</tool>',
                "expected_content": "machine learning",
                "expected_params": {"lang": "en"},
            },
            {
                "input": '<tool name="search" lang="en" mode="strict">AI research</tool>',
                "expected_content": "AI research",
                "expected_params": {"lang": "en", "mode": "strict"},
            },
            {
                "input": 'Please <tool name="search" timeout="30">  deep learning  </tool> thanks',
                "expected_content": "deep learning",  # Should be stripped
                "expected_params": {"timeout": "30"},
            },
        ]

        for test_case in test_cases:
            call_info = unified_tool.parse_call(test_case["input"])
            assert call_info is not None, f"Should parse: {test_case['input']}"
            assert isinstance(call_info, ToolCallInfo)
            assert call_info.content == test_case["expected_content"]
            assert call_info.parameters == test_case["expected_params"]
            assert call_info.start_pos >= 0
            assert call_info.end_pos > call_info.start_pos

    def test_unified_parse_call_failure(self, unified_tool):
        """Test unified tool handles invalid inputs correctly"""
        test_cases = [
            '<tool name="other_tool">not for search</tool>',  # Wrong tool name
            "Please search for python programming",  # No tags
            '<tool name="search">incomplete',  # Missing end tag
            "<search>legacy format</search>",  # Legacy format
            "",  # Empty string
        ]

        for test_input in test_cases:
            call_info = unified_tool.parse_call(test_input)
            assert call_info is None, f"Should not parse: {test_input}"

    def test_unified_tool_execution_simple(self, unified_tool):
        """Test unified tool execution with simple string input"""
        test_input = "machine learning"
        result = unified_tool(test_input)

        assert result.called
        assert result.error is None
        assert result.output == "Unified search executed with: machine learning"
        assert result.tool_name == "search"
        assert result.runtime > 0

    def test_unified_tool_execution_with_parameters(self, unified_tool):
        """Test unified tool execution with parameters"""
        test_input = {"content": "machine learning", "lang": "en", "mode": "strict"}
        result = unified_tool(test_input)

        assert result.called
        assert result.error is None
        assert "machine learning" in result.output
        assert "lang" in result.output
        assert "strict" in result.output
        assert result.tool_name == "search"

    def test_unified_tool_execution_content_only_dict(self, unified_tool):
        """Test unified tool execution with dict containing only content"""
        test_input = {"content": "python programming"}
        result = unified_tool(test_input)

        assert result.called
        assert result.error is None
        assert "python programming" in result.output
        assert "parameters" not in result.output  # No additional parameters
        assert result.tool_name == "search"

    def test_unified_format_result(self, unified_tool):
        """Test unified tool result formatting"""
        mock_output = ToolOutput(
            tool_name="test", output="test result", called=True, runtime=0.1
        )

        formatted = unified_tool.format_result(mock_output)
        # Unified parser uses more complex XML format
        assert "<tool_output>" in formatted
        assert "test result" in formatted
        assert "</tool_output>" in formatted


class TestParserRegistry:
    """Test the new parser registry system"""

    def test_registry_contains_expected_parsers(self):
        """Test that the parser registry contains expected parsers"""
        registered = get_registered_parsers()
        assert "legacy" in registered
        assert "unified" in registered
        assert "null" in registered

    def test_create_parser_with_registry(self):
        """Test creating parsers using the registry system"""
        # Test legacy parser creation
        legacy_parser = create_tool_parser(
            "legacy",
            tool_start_tag="<test>",
            tool_end_tag="</test>",
            result_start_tag="<output>",
            result_end_tag="</output>",
        )
        assert legacy_parser.tool_start_tag == "<test>"
        assert legacy_parser.tool_end_tag == "</test>"

        # Test unified parser creation
        unified_parser = create_tool_parser("unified")
        assert unified_parser.stop_sequences == ["</tool>"]

        # Test null parser creation
        null_parser = create_tool_parser("null")
        assert null_parser.stop_sequences == []


class TestToolCompatibility:
    """Test compatibility and edge cases between tool types"""

    def test_multiple_tool_calls_in_text(self, unified_tool):
        """Test parsing when multiple tool calls exist"""
        test_input = """
        First search: <tool name="search">python</tool>
        Then another: <tool name="search" lang="en">machine learning</tool>
        And wrong tool: <tool name="other">ignore this</tool>
        """

        # Should detect calls
        assert unified_tool.has_calls(test_input)

        # Should parse first matching call
        call_info = unified_tool.parse_call(test_input)
        assert call_info is not None
        assert call_info.content == "python"
        assert call_info.parameters == {}

    def test_edge_case_empty_content(self, unified_tool):
        """Test handling of empty content"""
        test_input = '<tool name="search"></tool>'

        assert unified_tool.has_calls(test_input)
        call_info = unified_tool.parse_call(test_input)
        assert call_info is not None
        assert call_info.content == ""
        assert call_info.parameters == {}

    def test_edge_case_special_characters(self, unified_tool):
        """Test handling of special characters in content and parameters"""
        test_input = '<tool name="search" query="test & query" mode="strict">content with "quotes" and &amp;</tool>'

        call_info = unified_tool.parse_call(test_input)
        assert call_info is not None
        assert "quotes" in call_info.content
        assert call_info.parameters["query"] == "test & query"
        assert call_info.parameters["mode"] == "strict"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
