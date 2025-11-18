import asyncio
import copy
import time
from typing import Callable, List, Literal, Optional, Union

from .base import BaseTool
from .data_types import ToolInput, ToolOutput
from .tool_parsers import NullToolCallParser, ToolCallParser


class ChainedTool(BaseTool):
    """
    A tool that composes multiple tools together in a pipeline.
    Supports sequential composition where output of one tool feeds into the next.
    """

    def __init__(
        self,
        tools: List[BaseTool],
        error_handling_strategy: Literal[
            "keep_progress",
            "stop_on_error",
            "continue",
        ] = "stop_on_error",
        tool_parser: Optional[ToolCallParser] = None,
        output_formatting: Union[str, Callable[[List[ToolOutput]], str]] = "last",
        **kwargs,
    ):
        """
        Initialize a composed tool.

        Args:
            tools: List of tools to compose
            tool_parser: Optional tool parser instance (defaults to NullToolCallParser)
            output_formatting: How to format the final output. Can be:
                - "last": Use only the last tool's output
                - "combine": Use default combination of all outputs
                - Callable: Custom function that takes List[ToolOutput] and returns str
            error_handling: How to handle tool failures. Can be:
                - "keep_progress": Stop at first error, keep outputs from successful tools
                - "stop_on_error": Stop and return error immediately (original behavior)
                - "continue": Continue executing all tools regardless of errors
            **kwargs: Additional arguments passed to BaseTool
        """
        if len(tools) == 0:
            raise ValueError("ChainedTool requires at least one tool")

        super().__init__(
            tool_parser=tool_parser,
            **kwargs,
        )

        # Remove start and end tags from all tools to prevent conflicts
        # In this case, all the contained tools should not need to be tagged tools anymore.
        self.tools = []
        for tool in tools:
            # Create a copy of the tool with null parser
            tool_copy = copy.copy(tool)
            tool_copy.tool_parser = NullToolCallParser()
            self.tools.append(tool_copy)

        self.output_formatting = output_formatting
        self.error_handling = error_handling_strategy

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> ToolOutput:
        """Execute the composed tool pipeline"""
        call_id = self._generate_call_id()
        start_time = time.time()

        return await self._execute_sequential(tool_input, call_id, start_time)

    def preprocess_input(self, tool_input: Union[str, ToolInput, ToolOutput]) -> str:
        """Preprocess the input for the composed tool"""
        tool_input = self.extract_tool_input(tool_input)

        if isinstance(tool_input, ToolOutput):
            return tool_input
        elif isinstance(tool_input, str):
            return tool_input
        elif isinstance(tool_input, dict):
            return tool_input
        else:
            raise ValueError(
                f"ChainedTool input must be a string or ToolOutput, got {type(tool_input)}"
            )

    async def _execute_sequential(
        self,
        tool_input: Union[str, ToolInput, ToolOutput],
        call_id: str,
        start_time: float,
    ) -> ToolOutput:
        """Execute tools sequentially, passing output to next tool"""
        current_input = self.preprocess_input(tool_input)
        all_outputs = []
        failed_at = None

        for i, tool in enumerate(self.tools):
            # Execute current tool
            output = await tool(current_input)
            all_outputs.append(output)

            # Check for errors
            if output.error:
                if self.error_handling == "stop_on_error":
                    return self._create_error_output(
                        f"Tool {i+1} ({tool.__class__.__name__}) failed: {output.error}",
                        call_id,
                        time.time() - start_time,
                        raw_output={"failed_at": i, "outputs": all_outputs},
                    )
                elif self.error_handling == "keep_progress":
                    failed_at = i
                    break
                # For "continue" mode, just record the failure but keep going
                elif failed_at is None:  # Record first failure
                    failed_at = i

            # Pass the output string directly to the next tool
            if i < len(self.tools) - 1:
                current_input = output

        # Format final output based on output_formatting parameter
        if failed_at is None or self.error_handling in ["keep_progress", "continue"]:
            if callable(self.output_formatting):
                # Use custom function
                final_output = self.output_formatting(all_outputs)
            elif self.output_formatting == "combine":
                # Use default combination
                final_output = self._combine_outputs(all_outputs)
            else:
                # Default to "last" - use only the last tool's output
                final_output = all_outputs[-1].output

        # Add completion info based on error handling strategy
        if failed_at is not None:
            if self.error_handling == "keep_progress":
                failed_tool_name = self.tools[failed_at].__class__.__name__
                final_output += f"\n\n[Note: Chain partially completed - {failed_tool_name} failed at step {failed_at + 1}]"
            elif self.error_handling == "continue":
                failed_tools = [
                    i for i, output in enumerate(all_outputs) if output.error
                ]
                failed_names = [self.tools[i].__class__.__name__ for i in failed_tools]
                final_output += f"\n\n[Note: Chain completed with failures at steps: {', '.join(f'{name} (step {i+1})' for i, name in zip(failed_tools, failed_names))}]"

        # Calculate tool call statistics
        total_tool_calls = len(all_outputs)
        total_failed_tool_calls = len(
            [output for output in all_outputs if output.error]
        )

        return ToolOutput(
            output=final_output,
            called=True,
            error="",
            timeout=False,
            runtime=time.time() - start_time,
            call_id=call_id,
            raw_output={
                "tool_outputs": all_outputs,
                "failed_at": failed_at,
                "expected_tool_calls": len(self.tools),
                "total_tool_calls": total_tool_calls,
                "total_failed_tool_calls": total_failed_tool_calls,
            },
            tool_name=self.name,
        )

    def _combine_outputs(self, outputs: List[ToolOutput]) -> str:
        """Combine outputs from multiple tools"""
        combined = []
        for i, output in enumerate(outputs):
            tool_name = self.tools[i].__class__.__name__
            combined.append(f"=== {tool_name} Output ===\n{output.output}")
        return "\n\n".join(combined)

    def _format_output(self, output: ToolOutput) -> str:
        """Format chained tool output into string representation"""
        return output.output

    def _generate_tool_schema(self):
        """Generate parameters schema for chained tool - use first tool's schema"""
        return self.tools[0].tool_tool_schema
