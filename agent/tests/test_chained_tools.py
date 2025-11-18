import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dr_agent.tool_interface.base import BaseTool
from dr_agent.tool_interface.chained_tool import ChainedTool
from dr_agent.tool_interface.data_types import (
    Document,
    DocumentToolOutput,
    ToolOutput,
)
from dr_agent.tool_interface.tool_parsers import ToolCallInfo


class MockTool(BaseTool):
    """Mock tool for testing tool composition"""

    def __init__(
        self, name="MockTool", output_prefix="", should_error=False, timeout=10
    ):
        super().__init__(
            tool_parser="legacy",
            tool_start_tag="<input>",
            tool_end_tag="</input>",
            result_start_tag="<output>",
            result_end_tag="</output>",
            timeout=timeout,
        )
        self._tool_name = name
        self.output_prefix = output_prefix
        self.should_error = should_error

    async def __call__(self, tool_input) -> ToolOutput:
        """Mock tool execution"""
        call_id = self._generate_call_id()
        start_time = time.time()

        if self.should_error:
            return ToolOutput(
                tool_name=self.name,
                output="",
                called=True,
                error=f"Mock error from {self.name}",
                runtime=0.1,
                call_id=call_id,
            )

        # Extract input based on type
        if isinstance(tool_input, ToolOutput):
            input_text = tool_input.output
        elif isinstance(tool_input, str):
            # Try to parse tool call or use directly
            call_info = self.parse_call(tool_input)
            input_text = call_info.content if call_info else tool_input
        else:
            input_text = str(tool_input)

        result = f"{self.output_prefix}Processed by {self.name}: {input_text}".strip()

        return ToolOutput(
            tool_name=self.name,
            output=result,
            called=True,
            error="",
            runtime=time.time() - start_time,
            call_id=call_id,
        )

    def _format_output(self, output: ToolOutput) -> str:
        """Format the tool output into a string representation"""
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Input text"}},
            "required": ["input"],
        }


class TestChainedTool:
    """Test ChainedTool functionality"""

    @pytest.fixture
    def simple_tools(self):
        """Fixture providing simple mock tools for chaining"""
        tool1 = MockTool(name="Tool1", output_prefix="[Step1] ")
        tool2 = MockTool(name="Tool2", output_prefix="[Step2] ")
        tool3 = MockTool(name="Tool3", output_prefix="[Step3] ")
        return [tool1, tool2, tool3]

    @pytest.fixture
    def chained_tool_legacy(self, simple_tools):
        """Fixture providing ChainedTool with legacy parser"""
        return ChainedTool(
            tools=simple_tools,
            tool_parser="legacy",
            tool_start_tag="<input>",
            tool_end_tag="</input>",
            result_start_tag="<output>",
            result_end_tag="</output>",
            output_formatting="last",
        )

    @pytest.fixture
    def chained_tool_unified(self, simple_tools):
        """Fixture providing ChainedTool with unified parser"""
        return ChainedTool(
            tools=simple_tools,
            tool_parser="unified",
            output_formatting="combine",
        )

    def test_chained_tool_initialization(self, simple_tools):
        """Test ChainedTool initialization"""
        chained_tool = ChainedTool(tools=simple_tools, output_formatting="last")

        assert len(chained_tool.tools) == 3
        assert chained_tool.output_formatting == "last"
        # Verify tools were copied with null parsers
        for tool in chained_tool.tools:
            assert tool.tool_parser.__class__.__name__ == "NullToolCallParser"

    def test_chained_tool_name(self, simple_tools):
        """Test ChainedTool name property"""
        chained_tool = ChainedTool(tools=simple_tools)
        assert chained_tool.name == "ChainedTool"

        # Test custom name
        chained_tool = ChainedTool(tools=simple_tools, name="CustomChain")
        assert chained_tool.name == "CustomChain"

    @pytest.mark.asyncio
    async def test_sequential_execution_last_output(self, chained_tool_legacy):
        """Test sequential execution with 'last' output formatting"""
        result = await chained_tool_legacy("<input>initial data</input>")

        assert isinstance(result, ToolOutput)
        assert result.called is True
        assert result.error == ""
        # Should contain output from all three tools in sequence
        expected = "[Step3] Processed by Tool3: [Step2] Processed by Tool2: [Step1] Processed by Tool1: initial data"
        assert result.output == expected

    @pytest.mark.asyncio
    async def test_sequential_execution_combine_output(self, chained_tool_unified):
        """Test sequential execution with 'combine' output formatting"""
        result = await chained_tool_unified(
            '<tool name="ChainedTool">test input</tool>'
        )

        assert isinstance(result, ToolOutput)
        assert result.called is True
        assert result.error == ""

        # Should combine all tool outputs
        assert "=== MockTool Output ===" in result.output
        assert "[Step1] Processed by Tool1: test input" in result.output

    @pytest.mark.asyncio
    async def test_custom_output_formatting(self, simple_tools):
        """Test custom output formatting function"""

        def custom_formatter(outputs):
            return " -> ".join([output.output for output in outputs])

        chained_tool = ChainedTool(
            tools=simple_tools,
            output_formatting=custom_formatter,
        )

        result = await chained_tool("test data")

        assert isinstance(result, ToolOutput)
        assert " -> " in result.output
        assert result.output.count(" -> ") == 2  # Two arrows for three tools

    @pytest.mark.asyncio
    async def test_error_propagation(self, simple_tools):
        """Test error propagation in tool chain"""
        # Make the second tool error
        simple_tools[1].should_error = True

        chained_tool = ChainedTool(tools=simple_tools)
        result = await chained_tool("test input")

        assert isinstance(result, ToolOutput)
        assert result.called is False
        assert "Tool 2 (MockTool) failed" in result.error
        assert "Mock error from Tool2" in result.error

    @pytest.mark.asyncio
    async def test_tool_output_as_input(self, chained_tool_legacy):
        """Test using ToolOutput as input to chained tool"""
        input_output = ToolOutput(
            tool_name="ExternalTool",
            output="preprocessed data",
            called=True,
            error="",
            runtime=0.5,
        )

        result = await chained_tool_legacy(input_output)

        assert isinstance(result, ToolOutput)
        assert result.called is True
        assert result.error == ""
        assert "preprocessed data" in result.output

    @pytest.mark.asyncio
    async def test_empty_tools_list(self):
        """Test ChainedTool with empty tools list"""
        with pytest.raises(ValueError, match="ChainedTool requires at least one tool"):
            chained_tool = ChainedTool(tools=[])

    @pytest.mark.asyncio
    async def test_single_tool_chain(self):
        """Test ChainedTool with single tool"""
        single_tool = MockTool(name="SingleTool", output_prefix="[Only] ")
        chained_tool = ChainedTool(tools=[single_tool])

        result = await chained_tool("solo input")

        assert isinstance(result, ToolOutput)
        assert result.called is True
        assert result.error == ""
        assert result.output == "[Only] Processed by SingleTool: solo input"

    def test_preprocess_input_string(self, chained_tool_legacy):
        """Test preprocessing string input"""
        processed = chained_tool_legacy.preprocess_input("simple string input")
        assert processed == "simple string input"

    def test_preprocess_input_tool_output(self, chained_tool_legacy):
        """Test preprocessing ToolOutput input"""
        tool_output = ToolOutput(
            tool_name="TestTool",
            output="output from previous tool",
            called=True,
            error="",
            runtime=1.0,
        )

        processed = chained_tool_legacy.preprocess_input(tool_output)
        assert processed == ToolOutput(
            tool_name="TestTool",
            output="output from previous tool",
            called=True,
            error="",
            runtime=1.0,
        )

    @pytest.mark.asyncio
    async def test_output_formatting_error_handling(self, simple_tools):
        """Test error handling in output formatting"""

        def broken_formatter(outputs):
            raise ValueError("Formatter error")

        chained_tool = ChainedTool(
            tools=simple_tools,
            output_formatting=broken_formatter,
        )

        with pytest.raises(ValueError, match="Formatter error"):
            await chained_tool("test input")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
