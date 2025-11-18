import asyncio
import copy
import inspect
import re
import string
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, TypeVar, Union

from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from .client import _llm_tool_client_context
from .tool_interface import AgentAsTool

T = TypeVar("T")

if TYPE_CHECKING:
    from .client import LLMToolClient
    from .tool_interface.base import BaseTool


def _get_reserved_kwargs(class_name: str):
    # Collect all parameter names from LLMToolClient.generate_with_tools
    reserved = set()
    for method_name in ["generate_with_tools"]:
        method = getattr(class_name, method_name, None)
        if method:
            sig = inspect.signature(method)
            for param in sig.parameters.values():
                if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                    reserved.add(param.name)
    return reserved


@dataclass
class BaseAgent:
    """Base class for prompted LLM clients"""

    client: Optional["LLMToolClient"] = None
    prompt: Optional[Union[str, callable]] = None
    system_prompt: Optional[str] = None
    tools: Optional[List["BaseTool"]] = field(default=None)

    def __post_init__(self):
        """Validate the prompt template after initialization"""

        if self.prompt is None:
            if self.__class__.prompt is None:
                raise ValueError("No prompt defined for this client")
            else:
                # Bind the method to this instance if it's a callable
                class_prompt = self.__class__.prompt
                if callable(class_prompt) and not hasattr(class_prompt, "__self__"):
                    # This is an unbound method, bind it to self
                    self.prompt = class_prompt.__get__(self, self.__class__)
                else:
                    self.prompt = class_prompt

        for variable_name in ["system_prompt", "tools"]:
            if getattr(self.__class__, variable_name) is not None:
                setattr(self, variable_name, getattr(self.__class__, variable_name))

        if self.client is None:
            self.client = _llm_tool_client_context.get()
            if self.client is None:
                raise ValueError(
                    "No LLM client provided. Either pass a client directly or use the 'using_client' context manager:\n"
                    "  with using_client(your_client):\n"
                    "      agent = YourAgent()"
                )

        if self.tools is not None:
            # Create a copy of the client with modified tools

            # Replace the tools list with a new list
            self.client = copy.copy(self.client)
            self.client.tools = self.tools.copy()

        self._reserved_kwargs = _get_reserved_kwargs(self.client.__class__)

        if isinstance(self.prompt, str):
            # Check string template for conflicts
            template_vars = self._extract_variables(self.prompt)
            conflicts = set(template_vars) & self._reserved_kwargs

            if conflicts:
                raise ValueError(
                    f"Prompt template contains reserved variable names: {conflicts}. "
                    f"These are reserved for generation parameters. "
                    f"Please rename these variables in your template."
                )

        elif callable(self.prompt):
            # Check method signature for conflicts
            import inspect

            try:
                sig = inspect.signature(self.prompt)
                method_params = set(sig.parameters.keys())

                # Remove 'self' if it's there (for bound methods)
                method_params.discard("self")

                conflicts = method_params & self._reserved_kwargs

                if conflicts:
                    raise ValueError(
                        f"Prompt method parameters conflict with reserved names: {conflicts}. "
                        f"These are reserved for generation parameters. "
                        f"Please rename these parameters in your prompt method."
                    )
            except Exception as e:
                # If we can't inspect the signature, skip validation
                # (some callable objects might not have inspectable signatures)
                pass

    @property
    def required_params(self) -> List[str]:
        """Get the required parameters for the agent's prompt"""
        if not hasattr(self, "_required_params"):
            if isinstance(self.prompt, str):
                self._required_params = self._extract_variables(self.prompt)
            elif callable(self.prompt):
                self._required_params = list(
                    inspect.signature(self.prompt).parameters.keys()
                )
            else:
                raise TypeError(
                    f"Prompt must be string or callable, got {type(self.prompt)}"
                )
        return self._required_params

    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from a format string template, including positional and named fields"""

        formatter = string.Formatter()
        # Collect all field names (including positional indices as strings)
        required_vars = set()
        for _, field_name, _, _ in formatter.parse(template):
            if field_name is not None:
                required_vars.add(field_name)
        return list(required_vars)

    def _prepare_prompt(self, **kwargs) -> str:
        """Prepare the prompt based on its type (string or callable)"""

        if isinstance(self.prompt, str):
            # Extract required variables from template
            required_vars = self._extract_variables(self.prompt)

            # Get only the required variables from kwargs
            prompt_vars = {var: kwargs[var] for var in required_vars if var in kwargs}

            # Check for missing variables
            missing_vars = set(required_vars) - set(prompt_vars.keys())
            if missing_vars:
                raise ValueError(f"Missing required variables: {missing_vars}")

            return self.prompt.format(**prompt_vars)

        elif callable(self.prompt):
            # If it's a method, call it with kwargs
            # Filter kwargs to only include parameters the method accepts
            sig = inspect.signature(self.prompt)
            valid_params = sig.parameters.keys()

            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return self.prompt(**filtered_kwargs)

        else:
            raise TypeError(
                f"Prompt must be string or callable, got {type(self.prompt)}"
            )

    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Generate response using the prompt template and provided variables.

        All kwargs are passed through:
        - Variables for prompt formatting (will be extracted as needed)
        - Generation parameters (temperature, max_tokens, etc.)
        - Tool parameters (max_tool_calls, verbose, etc.)
        """
        # Separate generation parameters from prompt variables
        generation_params = {
            "temperature",
            "top_p",
            "max_tokens",
            "top_k",
            "repetition_penalty",
            "seed",
            "max_tool_calls",
            "include_tool_results",
            "verbose",
            "generation_prefix",
            "stop",
            "tool_calling_mode",  # Support for native vs parser tool calling
        }

        # Extract generation parameters
        gen_kwargs = {k: v for k, v in kwargs.items() if k in generation_params}

        # Prepare the prompt with remaining kwargs (prompt variables)
        formatted_prompt = self._prepare_prompt(**kwargs)

        if isinstance(formatted_prompt, list):
            messages = formatted_prompt
        else:
            # Create messages
            messages = []
            if "system_prompt" in kwargs and kwargs["system_prompt"] is not None:
                messages.append({"role": "system", "content": kwargs["system_prompt"]})
            elif self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": formatted_prompt})

        # Call the underlying client
        return await self.client.generate_with_tools(
            prompt_or_messages=messages, **gen_kwargs
        )

    def _parse_args(self, item: T, **kwargs):
        """Parse arguments from item based on its type, prioritizing item values."""

        if isinstance(item, dict):
            merged_kwargs = {**kwargs, **item}
            return (), merged_kwargs
        else:
            raise ValueError(f"Unsupported item type: {type(item)}")
        # elif isinstance(item, (list, tuple)):
        #     return item, kwargs
        # else:
        #     return (item,), kwargs

    async def _execute_with_semaphore(
        self, semaphore: asyncio.Semaphore, item: T, **kwargs
    ) -> Dict[str, Any]:
        """Execute the agent's __call__ method with semaphore control."""
        async with semaphore:
            parsed_args, parsed_kwargs = self._parse_args(item, **kwargs)
            return await self.__call__(*parsed_args, **parsed_kwargs)

    async def map(
        self,
        items: Iterable[T],
        *,
        max_concurrent_tasks: int = 5,
        progress_desc: Optional[str] = None,
        return_exceptions: bool = False,
        **kwargs,
    ) -> List[Union[Dict[str, Any], Exception]]:
        """
        Execute the agent in parallel across multiple items.

        Args:
            items: Iterable of items to process
            *args: Additional positional arguments to pass to __call__
            max_concurrent_tasks: Maximum number of concurrent tasks
            **kwargs: Additional keyword arguments to pass to __call__

        Returns:
            List of results from __call__ method, in the same order as input items
        """
        if max_concurrent_tasks == 1:
            results = []
            for item in tqdm(items, desc=progress_desc):
                parsed_args, parsed_kwargs = self._parse_args(item, **kwargs)
                result = await self.__call__(*parsed_args, **parsed_kwargs)
                results.append(result)

            return results

        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            self._execute_with_semaphore(semaphore, item, **kwargs) for item in items
        ]

        # Execute with optional progress bar
        results = await tqdm_async.gather(*tasks)
        # TODO: tqdm_async won't allow return_exceptions=True
        # https://github.com/tqdm/tqdm/issues/1286

        return results

    def as_tool(
        self,
        tool_start_tag: Optional[str] = None,
        tool_end_tag: Optional[str] = None,
        result_start_tag: Optional[str] = None,
        result_end_tag: Optional[str] = None,
        **kwargs,
    ) -> "AgentAsTool":
        """Convert the agent to an AgentTool"""

        return AgentAsTool(
            agent=self,
            tool_start_tag=tool_start_tag,
            tool_end_tag=tool_end_tag,
            result_start_tag=result_start_tag,
            result_end_tag=result_end_tag,
            **kwargs,
        )
