import inspect
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .data_types import Document, DocumentToolOutput, ToolInput, ToolOutput
from .tool_parsers import (
    NullToolCallParser,
    ToolCallInfo,
    ToolCallParser,
    create_tool_parser,
)


class BaseTool(ABC):
    """Base class for all tools with integrated specification and execution"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        timeout: int = 60,
        name: Optional[str] = None,
        create_string_output: bool = True,
        description: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the tool with parser configuration.

        Args:
            tool_parser: Pre-configured parser instance (takes precedence)
            timeout: Tool execution timeout in seconds
            name: Optional custom name for the tool (defaults to class name)
            description: Optional description for OpenAI tool schema
            **kwargs: Additional arguments passed to parser constructor
        """
        if isinstance(tool_parser, str):
            self.tool_parser = create_tool_parser(tool_parser, **(kwargs or {}))
        elif isinstance(tool_parser, ToolCallParser):
            self.tool_parser = tool_parser
        elif tool_parser is None:
            self.tool_parser = NullToolCallParser()
        else:
            raise ValueError(f"Invalid tool parser type: {type(tool_parser)}")

        self.timeout = timeout
        self.create_string_output = create_string_output
        self._description = description

        # Set custom name if provided
        if name is not None:
            self._tool_name = name

    @property
    def name(self) -> str:
        """Get the name of the tool"""
        if hasattr(self, "_tool_name"):
            return self._tool_name
        else:
            return self.__class__.__name__

    @name.setter
    def name(self, name: str):
        """Set the name of the tool"""
        self._tool_name = name

    @property
    def description(self) -> str:
        """Get the description of the tool"""
        if self._description is None:
            self._description = self._tool_description()
        return self._description

    @property
    def tool_tool_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for the tool"""
        return self._generate_tool_schema()

    def _generate_call_id(self) -> str:
        """Generate a unique call ID for tool execution"""
        return str(uuid.uuid4())[:8]

    def _create_error_output(
        self,
        error_msg: str,
        call_id: str,
        runtime: float,
        output: str = "",
        raw_output: Optional[Dict[str, Any]] = None,
    ) -> ToolOutput:
        """Create a standardized error output"""
        return ToolOutput(
            output=output,
            error=error_msg,
            called=False,
            timeout=False,
            runtime=runtime,
            call_id=call_id,
            raw_output=raw_output,
            tool_name=self.name,
        )

    @abstractmethod
    def __call__(self, tool_input: Union[str, ToolInput, ToolOutput]) -> ToolOutput:
        """
        Execute the tool with the given input.

        Args:
            tool_input: Can be one of:
                - str: A prompt string
                - ToolInput (Dict[str, Any]): Dictionary with input parameters
                - ToolOutput: Output from another tool (for tool chaining)

        Returns:
            ToolOutput: The result of tool execution
        """
        pass

    @abstractmethod
    def _format_output(self, output: ToolOutput) -> str:
        """
        Format the tool output into a string representation.
        This method handles tool-specific formatting logic.

        Args:
            output: The tool output to format

        Returns:
            String representation of the output
        """
        pass

    # Tool calling interface - delegates to parser
    def has_calls(self, text: str) -> bool:
        """Check if this tool has any calls in the given text"""
        return self.tool_parser.has_calls(text, self.name)

    def parse_call(self, text: str) -> Optional[ToolCallInfo]:
        """Parse the first tool call for this tool in the text, including parameters and position"""
        return self.tool_parser.parse_call(text, self.name)

    def format_result(self, output: ToolOutput) -> str:
        """Format the tool output using the parser's format"""
        formatted_content = self._format_output(output)
        return self.tool_parser.format_result(formatted_content, output)

    def extract_tool_input(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> Optional[str]:
        """
        Extract tool call content from input string using the parser.
        For non-string inputs or when no tool call is found, returns the input as-is.

        Args:
            tool_input: The input containing a tool call

        Returns:
            Extracted tool call content, or original input if no valid tool call found
        """
        if not isinstance(tool_input, str):
            return tool_input

        # Use parse_call to extract the content from the tool call
        call_info = self.parse_call(tool_input)
        return call_info.content if call_info else tool_input

    @property
    def stop_sequences(self) -> List[str]:
        """Get stop sequences for this tool format"""
        return self.tool_parser.stop_sequences

    def filter_no_parser_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter kwargs to remove those required by the tool parser's constructor.

        Args:
            kwargs: Dictionary of keyword arguments to filter

        Returns:
            Filtered dictionary with parser-specific kwargs removed
        """
        if not hasattr(self.tool_parser, "__init__"):
            return kwargs

        # Get the signature of the parser's __init__ method
        parser_signature = inspect.signature(self.tool_parser.__init__)

        # Get parameter names (excluding 'self')
        parser_params = set(parser_signature.parameters.keys()) - {"self"}

        # Filter kwargs to exclude those the parser accepts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in parser_params}

        return filtered_kwargs

    def _tool_description(self) -> str:
        """
        Extract description from the tool's docstring or class docstring.

        Returns:
            Description string for the tool
        """
        # Try to get description from __call__ method docstring
        if self.__call__.__doc__:
            # Get first line of docstring
            return self.__call__.__doc__.strip().split("\n")[0]

        # Fallback to class docstring
        if self.__class__.__doc__:
            return self.__class__.__doc__.strip().split("\n")[0]

        # Default fallback
        return f"Tool for {self.name}"

    @abstractmethod
    def _generate_tool_schema(self) -> Dict[str, Any]:
        """
        Generate OpenAI-compatible parameters schema from tool signature.
        Subclasses must implement this method to define their parameter schema.

        Returns:
            Parameters schema dict with type, properties, and required fields
        """
        pass

    def to_openai_tool_schema(self) -> Dict[str, Any]:
        """
        Generate OpenAI-compatible tool schema for native tool calling.

        Returns:
            Tool schema dict with type, function name, description, and parameters
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.tool_tool_schema,
            },
        }
