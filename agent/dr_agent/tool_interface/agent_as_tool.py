import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from .base import BaseTool
from .data_types import ToolInput, ToolOutput
from .tool_parsers import ToolCallParser

if TYPE_CHECKING:
    from ..agent_interface import BaseAgent


class AgentAsTool(BaseTool):
    """Tool wrapper that makes a BaseAgent usable as a Tool"""

    def __init__(
        self,
        agent: "BaseAgent",
        tool_parser: Optional[ToolCallParser] = None,
        timeout: int = 60,
        **kwargs,
    ):
        """
        Initialize AgentTool wrapper.

        Args:
            agent: BaseAgent instance to wrap
            tool_parser: Optional tool parser configuration (str type, instance, or None for null parser)
            timeout: Timeout for agent execution in seconds
            **default_kwargs: Default parameters to pass to agent calls
        """
        super().__init__(tool_parser=tool_parser, timeout=timeout, **kwargs)
        self.agent = agent
        self.default_kwargs = self.filter_no_parser_kwargs(kwargs)

    @property
    def name(self) -> str:
        """Get the name of the tool - uses agent class name if not set"""
        if hasattr(self, "_tool_name"):
            return self._tool_name
        else:
            return self.agent.__class__.__name__

    def _default_preprocess_input(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> str:
        """
        Preprocess and extract input for agent execution.
        Handles parsing tool call strings and applies agent-specific preprocessing.

        Args:
            tool_input: Raw input to the tool

        Returns:
            Processed input string ready for agent execution
        """
        # First get the base processed input
        tool_input = self.extract_tool_input(tool_input)

        if isinstance(tool_input, ToolOutput):
            # Use the output field from ToolOutput as the input
            base_query = tool_input.output
        elif isinstance(tool_input, str):
            # Parse tool call string to extract query
            base_query = tool_input
        else:
            raise ValueError(
                f"AgentAsTool input must be a string or ToolOutput, got {type(tool_input)}"
            )

        return base_query

    def preprocess_input(self, tool_input: Union[str, ToolInput, ToolOutput]) -> str:

        # Apply agent-specific preprocessing if available
        if hasattr(self.agent, "preprocess_input"):
            return self.agent.preprocess_input(tool_input)
        else:
            return self._default_preprocess_input(tool_input)

    def _default_postprocess_output(self, result: Dict[str, Any]) -> str:
        """Default postprocess the output of the agent"""
        return result.generated_text

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        """Postprocess the output of the agent"""
        if hasattr(self.agent, "postprocess_output"):
            return self.agent.postprocess_output(result)
        else:
            return self._default_postprocess_output(result)

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> ToolOutput:
        """Execute the agent with the given input"""
        call_id = self._generate_call_id()
        start_time = time.time()

        try:
            # Handle different input types using the new preprocess_input method
            if not isinstance(tool_input, (str, ToolOutput)):
                # For ToolInput (dict), we'd need more complex handling - for now, raise an error
                return self._create_error_output(
                    f"Unsupported tool_input type: {type(tool_input)}. Expected str or ToolOutput.",
                    call_id,
                    time.time() - start_time,
                )

            # Extract and preprocess query using the new method
            processed_agent_input = self.preprocess_input(tool_input)
            if not processed_agent_input:
                return self._create_error_output(
                    "No valid query found in agent tool call",
                    call_id,
                    time.time() - start_time,
                )

            # Determine how to pass the query to the agent based on its prompt template
            agent_params = self.agent.required_params
            agent_kwargs = {}

            if isinstance(processed_agent_input, str) and len(agent_params) == 1:
                agent_kwargs[agent_params[0]] = processed_agent_input
            elif isinstance(processed_agent_input, dict):
                agent_kwargs.update(processed_agent_input)

            agent_kwargs.update(self.default_kwargs)

            # Execute the agent with timeout
            try:
                result = await asyncio.wait_for(
                    self.agent(**agent_kwargs), timeout=self.timeout
                )
                output = self.postprocess_output(result)

                return ToolOutput(
                    tool_name=self.name,
                    output=output,
                    called=True,
                    error="",
                    timeout=False,
                    runtime=time.time() - start_time,
                    call_id=call_id,
                    raw_output=result.model_dump(),  # make sure it's a dictionary so pydantic doesn't fail
                    # TODO: @shannons can we allow a pydantic input?
                )

            except asyncio.TimeoutError:
                return ToolOutput(
                    output="",
                    called=True,
                    error=f"Agent execution timed out after {self.timeout} seconds",
                    timeout=True,
                    runtime=time.time() - start_time,
                    call_id=call_id,
                    raw_output=None,
                    tool_name=self.name,
                )

        except Exception as e:
            return ToolOutput(
                output="",
                called=True,
                error=f"Error executing agent: {str(e)}",
                timeout=False,
                runtime=time.time() - start_time,
                call_id=call_id,
                raw_output=None,
                tool_name=self.name,
            )

    def _format_output(self, output: ToolOutput) -> str:
        """Format agent output into string representation"""
        if hasattr(self.agent, "_format_output"):
            return self.agent._format_output(output)
        else:
            return output.output

    def _generate_tool_schema(self):
        """Generate parameters schema for agent tool based on agent's required parameters"""
        required_params = self.agent.required_params

        properties = {}
        for param in required_params:
            properties[param] = {
                "type": "string",
                "description": f"Input for parameter: {param}",
            }

        return {
            "type": "object",
            "properties": properties,
            "required": required_params,
        }
