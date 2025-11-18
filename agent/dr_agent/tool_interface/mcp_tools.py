import asyncio
import copy
import json
import logging
import os
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from anyio.streams.memory import BrokenResourceError
from fastmcp import Client
from fastmcp.exceptions import FastMCPError, ResourceError
from fastmcp.utilities.exceptions import McpError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseTool
from .data_types import Document, DocumentToolOutput, ToolInput, ToolOutput
from .tool_parsers import LegacyToolCallParser, ToolCallInfo, ToolCallParser
from .utils import extract_snippet_with_context

SERPER_MAX_QUERY_LENGTH = 2048

import dr_agent

RAG_MCP_PATH = Path(dr_agent.__file__).parent / "mcp_backend" / "main.py"
DEFAULT_MCP_MAX_CONCURRENT_CALLS = 20


class MCPErrorHandlingMode(Enum):
    """Error handling strategies for MCP calls"""

    RETURN_ERROR = "return_error"  # Always return error dict, never raise exceptions
    RAISE_EXCEPT_TIMEOUT = "raise_except_timeout"  # Raise except timeout errors, return dict for timeout errors
    RAISE_ALL = "raise_all"  # Raise all exceptions


DEFAULT_MCP_ERROR_MODE = MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT

logger = logging.getLogger(__name__)


class MCPMixin:
    """Mixin class that provides MCP (Model Context Protocol) functionality to tools"""

    # Global semaphore for controlling concurrent MCP calls
    _global_semaphore = None
    _max_concurrent_calls = None
    _error_handling_mode = None

    def __init__(
        self,
        *args,
        timeout: int = 60,
        name: Optional[str] = None,
        excluded_arguments: Optional[List[str]] = None,
        **kwargs,
    ):
        # Initialize global semaphore if not already done
        if MCPMixin._global_semaphore is None:
            MCPMixin._max_concurrent_calls = int(
                os.environ.get(
                    "MCP_MAX_CONCURRENT_CALLS", DEFAULT_MCP_MAX_CONCURRENT_CALLS
                )
            )
            MCPMixin._global_semaphore = asyncio.Semaphore(
                MCPMixin._max_concurrent_calls
            )

        # Initialize global error handling mode if not already done
        if MCPMixin._error_handling_mode is None:
            # Allow setting via kwargs or environment variable
            error_mode_str = kwargs.pop(
                "mcp_error_handling_mode", os.environ.get("MCP_ERROR_HANDLING_MODE")
            )
            if error_mode_str is not None:
                try:
                    MCPMixin._error_handling_mode = MCPErrorHandlingMode(error_mode_str)
                except ValueError:
                    logger.warning(
                        f"Invalid MCP_ERROR_HANDLING_MODE: {error_mode_str}. "
                        f"Using default: {DEFAULT_MCP_ERROR_MODE.value}"
                    )
                    MCPMixin._error_handling_mode = DEFAULT_MCP_ERROR_MODE
            else:
                MCPMixin._error_handling_mode = DEFAULT_MCP_ERROR_MODE

        # Fetch needed MCP arguments before calling super().__init__
        self.transport_type = kwargs.pop("transport_type", None) or os.environ.get(
            "MCP_TRANSPORT", "StreamableHttpTransport"
        )
        self.mcp_executable = kwargs.pop("mcp_executable", None) or os.environ.get(
            "MCP_EXECUTABLE", RAG_MCP_PATH
        )
        self.mcp_port = kwargs.pop("mcp_port", None) or os.environ.get(
            "MCP_TRANSPORT_PORT", 8000
        )
        self.mcp_host = kwargs.pop("mcp_host", None) or os.environ.get(
            "MCP_TRANSPORT_HOST", "localhost"
        )
        # Call super().__init__ to ensure proper MRO handling
        super().__init__(*args, timeout=timeout, name=name, **kwargs)
        self.timeout = timeout
        self.mcp_client_config = kwargs
        self.pinged = False
        self.excluded_arguments = excluded_arguments or []

        # Cache for MCP tool schema (lazy loaded)
        self._mcp_tool_schema = None

    def init_mcp_client(self):
        """Initialize MCP client based on environment variables"""
        if not Client:
            raise ImportError(
                "MCP client not available. Please install the MCP client library."
            )

        transport_type = self.transport_type

        if transport_type == "StreamableHttpTransport":
            logger.debug(
                f"Using MCP transport: {transport_type}, port: {self.mcp_port}"
            )
            return Client(
                f"http://{self.mcp_host}:{self.mcp_port}/mcp", timeout=self.timeout
            )
        elif transport_type == "FastMCPTransport":
            if not self.mcp_executable:
                raise ValueError(
                    "MCP_EXECUTABLE environment variable not set for FastMCPTransport"
                )
            logger.debug(
                f"Using MCP transport: {transport_type}, executable: {self.mcp_executable}"
            )
            return Client(self.mcp_executable, timeout=self.timeout)
        else:
            raise ValueError(f"Invalid MCP transport: {transport_type}")

    @retry(
        retry=retry_if_exception_type(
            (
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
                McpError,
                FastMCPError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _execute_mcp_call(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an MCP tool call with proper error handling and global concurrency control"""
        if MCPMixin._global_semaphore is None:
            MCPMixin._max_concurrent_calls = int(
                os.environ.get(
                    "MCP_MAX_CONCURRENT_CALLS", DEFAULT_MCP_MAX_CONCURRENT_CALLS
                )
            )
            MCPMixin._global_semaphore = asyncio.Semaphore(
                MCPMixin._max_concurrent_calls
            )
        async with MCPMixin._global_semaphore:
            try:
                mcp_client = self.init_mcp_client()
                async with mcp_client:
                    if not self.pinged:
                        await mcp_client.ping()
                        self.pinged = True
                    result = await mcp_client.call_tool(tool_name, params)

                    # Handle different response formats
                    if hasattr(result, "content") and result.content:
                        if isinstance(result.content[0], dict):
                            return result.content[0]
                        elif hasattr(result.content[0], "text"):
                            return json.loads(result.content[0].text)
                        else:
                            return {"data": str(result.content[0])}
                    else:
                        return {"error": "No content in response", "data": []}

            except (asyncio.TimeoutError, TimeoutError) as e:
                error_msg = f"MCP call timed out after {self.timeout} seconds"
                print(f"Error: {error_msg}")
                if MCPMixin._error_handling_mode in [
                    MCPErrorHandlingMode.RETURN_ERROR,
                    MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT,
                ]:
                    return {"error": error_msg, "data": []}
                else:  # raise_all
                    raise e
            except McpError as e:
                # Check if this is actually a timeout error disguised as McpError
                if "Timed out while waiting for response" in str(e):
                    error_msg = f"MCP call timed out: {str(e)}"
                    print(f"Error: {error_msg}")
                    if MCPMixin._error_handling_mode in [
                        MCPErrorHandlingMode.RETURN_ERROR,
                        MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT,
                    ]:
                        return {"error": error_msg, "data": []}
                    else:  # raise_all
                        raise e
                else:
                    error_msg = f"MCP error: {str(e)}"
                    print(f"Error: {error_msg}")
                    if (
                        MCPMixin._error_handling_mode
                        == MCPErrorHandlingMode.RETURN_ERROR
                    ):
                        return {"error": error_msg, "data": []}
                    else:  # raise_timeout or raise_all
                        raise e
            except (ConnectionError, FastMCPError) as e:
                error_msg = f"MCP call failed: {str(e)}"
                print(f"Error: {error_msg}")
                if MCPMixin._error_handling_mode == MCPErrorHandlingMode.RETURN_ERROR:
                    return {"error": error_msg, "data": []}
                else:  # raise_all
                    raise e
            except (BrokenResourceError, ResourceError) as e:
                error_msg = f"MCP call failed: {str(e)}"

                print(f"Error: {error_msg}")
                # For most of this error, it's caused by the MCP server had encountered some bugs
                #
                if MCPMixin._error_handling_mode in [
                    MCPErrorHandlingMode.RETURN_ERROR,
                    MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT,
                ]:
                    return {"error": error_msg, "data": []}
                else:  # raise_all
                    raise e
            except Exception as e:
                # Check if this is a timeout-related error
                error_str = str(e).lower()
                is_timeout_error = any(
                    timeout_keyword in error_str
                    for timeout_keyword in ["timeout", "timed out", "time out"]
                )

                if is_timeout_error:
                    error_msg = f"MCP call timed out: {str(e)}"
                    print(f"Error: {error_msg}")
                    if MCPMixin._error_handling_mode in [
                        MCPErrorHandlingMode.RETURN_ERROR,
                        MCPErrorHandlingMode.RAISE_EXCEPT_TIMEOUT,
                    ]:
                        return {"error": error_msg, "data": []}
                    else:  # raise_all
                        raise e
                else:
                    error_msg = f"Unexpected error: {str(e)}"
                    print(f"Error: {error_msg}")
                    if (
                        MCPMixin._error_handling_mode
                        == MCPErrorHandlingMode.RETURN_ERROR
                    ):
                        return {"error": error_msg, "data": []}
                    else:  # raise_timeout or raise_all
                        raise e

    async def _fetch_mcp_tool_schema(self) -> Optional[Dict[str, Any]]:
        """
        Fetch tool schema from MCP server by calling list_tools().
        Caches the result for future use.

        Returns:
            Tool schema dict with description and inputSchema, or None if not found
        """
        if self._mcp_tool_schema is not None:
            return self._mcp_tool_schema

        try:
            mcp_client = self.init_mcp_client()
            async with mcp_client:
                # List all available tools from the MCP server
                tools = await mcp_client.list_tools()

                # Find the tool that matches our MCP tool name
                mcp_tool_name = self.get_mcp_tool_name()
                for tool in tools:
                    if tool.name == mcp_tool_name:
                        # Get the input schema
                        input_schema = (
                            tool.inputSchema if hasattr(tool, "inputSchema") else None
                        )

                        # Filter out 'title' fields from properties if present
                        if input_schema and "properties" in input_schema:
                            input_schema = copy.deepcopy(input_schema)
                            for prop_name, prop_schema in input_schema[
                                "properties"
                            ].items():
                                if (
                                    isinstance(prop_schema, dict)
                                    and "title" in prop_schema
                                ):
                                    prop_schema.pop("title")

                        # Cache the schema
                        self._mcp_tool_schema = {
                            "name": tool.name,
                            "description": (
                                tool.description
                                if hasattr(tool, "description")
                                else None
                            ),
                            "inputSchema": input_schema,
                        }
                        return self._mcp_tool_schema

                # Tool not found
                logger.warning(
                    f"MCP tool '{mcp_tool_name}' not found in server's tool list"
                )
                return None

        except Exception as e:
            logger.warning(f"Failed to fetch MCP tool schema: {e}")
            return None

    def _get_mcp_description(self) -> Optional[str]:
        """
        Get description from cached MCP tool schema.
        Returns None if schema hasn't been fetched yet.
        """
        if self._mcp_tool_schema and self._mcp_tool_schema.get("description"):
            return self._mcp_tool_schema["description"]
        return None

    def _get_mcp_tool_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get parameters schema from cached MCP tool schema.
        Returns None if schema hasn't been fetched yet.
        Filters out excluded arguments from the schema.
        """
        if self._mcp_tool_schema and self._mcp_tool_schema.get("inputSchema"):
            schema = self._mcp_tool_schema["inputSchema"]

            # If no excluded arguments, return schema as-is
            if not self.excluded_arguments:
                return schema

            # Create a deep copy to avoid modifying the cached schema
            filtered_schema = copy.deepcopy(schema)

            # Remove excluded arguments from properties
            if "properties" in filtered_schema:
                for arg in self.excluded_arguments:
                    filtered_schema["properties"].pop(arg, None)

            # Remove excluded arguments from required list
            if "required" in filtered_schema:
                filtered_schema["required"] = [
                    req
                    for req in filtered_schema["required"]
                    if req not in self.excluded_arguments
                ]

            return filtered_schema
        return None

    def _tool_description(self) -> str:
        """
        Extract description from MCP schema if available, otherwise use fallback.
        Priority: MCP server > docstring (user-provided description handled by property)
        """
        # Try to get from MCP schema
        mcp_description = self._get_mcp_description()
        if mcp_description:
            return mcp_description

        # Fallback to parent implementation (docstring)
        return super()._tool_description()

    @abstractmethod
    def get_mcp_tool_name(self) -> str:
        """Return the MCP tool name for this browse tool"""
        pass

    def _generate_tool_schema(self) -> Dict[str, Any]:
        """
        Generate parameters schema for search tools.
        Uses MCP schema if available, otherwise returns fallback schema.
        """
        # Try to get from MCP schema first
        mcp_schema = self._get_mcp_tool_schema()
        if mcp_schema:
            return mcp_schema

    @abstractmethod
    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """
        Build parameters for MCP tool call.

        Args:
            tool_call_info: ToolCallInfo object containing content and parameters

        Returns:
            Dictionary of parameters for MCP tool
        """
        pass


class MCPSearchTool(MCPMixin, BaseTool, ABC):
    """Base class for MCP search tools with shared pipeline logic"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        number_documents_to_search: int = 10,
        timeout: int = 60,
        name: Optional[str] = None,
        create_string_output: bool = True,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            timeout=timeout,
            name=name,
            create_string_output=create_string_output,
            **kwargs,
        )
        self.number_documents_to_search = number_documents_to_search

    @abstractmethod
    def extract_documents(self, raw_output: Dict[str, Any]) -> List[Document]:
        """
        Extract documents from raw MCP response.
        This should return a list of Document objects with title, snippet, url, and score.

        Args:
            raw_output: Raw response from MCP tool

        Returns:
            List of Document objects
        """
        pass

    # ===== Optional methods for subclasses to override =====

    def _create_error_output(
        self,
        error_msg: str,
        call_id: str,
        runtime: float,
        output: str = "",
        raw_output: Optional[Dict[str, Any]] = None,
    ) -> DocumentToolOutput:
        """Create a standardized error output for search tools"""
        return DocumentToolOutput(
            output=output,
            error=error_msg,
            called=True,
            timeout=False,
            runtime=runtime,
            call_id=call_id,
            raw_output=raw_output,
            tool_name=self.name,
            documents=[],
        )

    def preprocess_input(
        self, tool_input: Union[str, Dict[str, Any], ToolInput, ToolOutput]
    ) -> Optional[ToolCallInfo]:
        """
        Preprocess and extract input for MCP search execution.
        Uses the tool parser system to extract content and parameters from tool calls.

        Args:
            tool_input: Raw input to the tool (string for parser mode, dict for native mode)

        Returns:
            ToolCallInfo object with content and parameters, or None if invalid
        """
        if isinstance(tool_input, str):
            return self.parse_call(tool_input)
        elif isinstance(tool_input, dict):
            # Native mode: tool_input is a dict with query and optional parameters
            query = tool_input.get("query", "")
            if not query:
                return None
            # Create ToolCallInfo from dict parameters
            return ToolCallInfo(
                content=query,
                parameters={k: v for k, v in tool_input.items() if k != "query"},
                start_pos=0,
                end_pos=len(query),
            )
        else:
            raise ValueError(
                f"MCP Search Tool input must be a string or dict, got {type(tool_input)}"
            )

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        """Shared pipeline for all search tools"""
        # print(f"{self.__class__.__name__} called with tool_input: {tool_input}")

        # Fetch MCP schema on first call (cached for subsequent calls)
        if self._mcp_tool_schema is None:
            await self._fetch_mcp_tool_schema()

        call_id = self._generate_call_id()
        start_time = time.time()

        # Step 1: Preprocess input
        tool_call_info = self.preprocess_input(tool_input)
        if not tool_call_info:
            return self._create_error_output(
                "No valid query found in tool call.",
                call_id,
                time.time() - start_time,
            )

        # Step 2: Validate call parameters
        # try:
        #     validated_params = self.validate_and_convert_parameters(
        #         tool_call_info.parameters
        #     )
        #     # Update tool_call_info parameters with validated ones
        #     tool_call_info.parameters = validated_params
        # except ValueError as e:
        #     return self._create_error_output(
        #         f"Parameter validation failed: {e}",
        #         call_id,
        #         time.time() - start_time,
        #     )

        # Step 3: Build MCP parameters
        params = self.get_mcp_params(tool_call_info)

        # Step 4: Execute MCP call
        raw_output = await self._execute_mcp_call(self.get_mcp_tool_name(), params)

        # Step 5: Check for execution errors
        if error := raw_output.get("error"):
            return self._create_error_output(
                f"Query failed: {error}",
                call_id,
                time.time() - start_time,
                raw_output=raw_output,
            )

        # Step 6: Extract documents for structured output
        documents = self.extract_documents(raw_output)

        if not documents:
            return self._create_error_output(
                "No results found for the query.",
                call_id,
                time.time() - start_time,
                raw_output=raw_output,
            )

        # Step 7: Create content from documents using stringify
        content_parts = []
        for doc in documents:
            content_parts.append(doc.stringify())
        content = "\n\n".join(content_parts)

        return DocumentToolOutput(
            tool_name=self.name,
            output=content if self.create_string_output else "",
            called=True,
            error="",
            timeout=False,
            runtime=time.time() - start_time,
            call_id=call_id,
            raw_output=raw_output,
            documents=documents,
            query=tool_call_info.content,  # Save the original query
        )

    def _format_output(self, output: Union[ToolOutput, DocumentToolOutput]) -> str:
        """Format the search results into string representation"""
        if output.error:
            return output.error
        else:
            if isinstance(self.tool_parser, LegacyToolCallParser):
                content_parts = []
                for doc in output.documents:
                    content_parts.append(doc.stringify())
                content = "\n\n".join(content_parts)
                return content
            else:
                combined_snippet_text = []
                for index, doc in enumerate(output.documents):
                    combined_snippet_text.append(
                        f"<snippet id={output.call_id}-{index}>\n{doc.stringify()}\n</snippet>"
                    )
                return "\n".join(combined_snippet_text)


class SemanticScholarSnippetSearchTool(MCPSearchTool):
    """Tool for searching academic papers using Semantic Scholar via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        number_documents_to_search: int = 3,
        timeout: int = 60,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            number_documents_to_search=number_documents_to_search,
            timeout=timeout,
            name=name,
            description=description,
            **kwargs,
        )

    def get_mcp_tool_name(self) -> str:
        return "semantic_scholar_snippet_search"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        # Start with default parameters
        params = {
            "query": tool_call_info.content,
            "limit": self.number_documents_to_search,
        }

        # Override with validated parameters from tool call
        if "limit" in tool_call_info.parameters:
            try:
                params["limit"] = int(tool_call_info.parameters["limit"])
            except ValueError:
                pass  # Keep default if conversion fails

        return params

    def extract_documents(self, raw_output: Dict[str, Any]) -> List[Document]:
        """Extract documents from Semantic Scholar response"""
        data = raw_output.get("data", [])
        documents = []

        for item in data:
            if isinstance(item, dict):
                # Handle structured response with snippet and paper info
                if "snippet" in item and "paper" in item:
                    snippet_info = item.get("snippet", {})
                    paper_info = item.get("paper", {})

                    if snippet_info.get("snippetKind") == "title":
                        snippet_text = ""
                    else:
                        snippet_text = snippet_info.get("text", "").strip()

                    doc = Document(
                        title=paper_info.get("title", "").strip(),
                        snippet=snippet_text,
                        url="",  # Semantic Scholar doesn't provide direct URLs in snippet search
                        text="",  # No full text content from search
                        score=item.get("score"),
                    )

                    if doc.title or doc.snippet:
                        documents.append(doc)

                # Handle direct snippet text (fallback case)
                elif "snippet" in item:
                    snippet_text = item["snippet"].get("text", "").strip()
                    if snippet_text:
                        doc = Document(
                            title="", snippet=snippet_text, url="", text="", score=None
                        )
                        documents.append(doc)

        return documents


class SerperSearchTool(MCPSearchTool):
    """Tool for web search using Serper Google API via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        number_documents_to_search: int = 10,
        timeout: int = 60,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            number_documents_to_search=number_documents_to_search,
            timeout=timeout,
            name=name,
            description=description,
            **kwargs,
        )

    def get_mcp_tool_name(self) -> str:
        return "serper_google_webpage_search"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """Build parameters for Serper API"""
        # Start with default parameters
        params = {
            "query": tool_call_info.content[:SERPER_MAX_QUERY_LENGTH],
            "num_results": self.number_documents_to_search,
        }

        # Override with validated parameters from tool call
        if "num_results" in tool_call_info.parameters:
            try:
                params["num_results"] = int(tool_call_info.parameters["num_results"])
            except ValueError:
                pass  # Keep default if conversion fails

        return params

    def extract_documents(self, raw_output: Dict[str, Any]) -> List[Document]:
        """Extract documents from Serper response"""
        organic_results = raw_output.get("organic", [])
        documents = []

        for result in organic_results:
            if isinstance(result, dict):
                doc = Document(
                    title=result.get("title", "").strip(),
                    url=result.get("link", "").strip(),
                    snippet=result.get("snippet", "").strip(),
                    text=None,
                    score=None,
                )
                if doc.title or doc.snippet or doc.url:
                    documents.append(doc)

        return documents


class MassiveServeSearchTool(MCPSearchTool):
    """Tool for searching documents using massive-serve API via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        number_documents_to_search: int = 10,
        timeout: int = 60,
        base_url: str = None,
        domains: str = "dpr_wiki_contriever_ivfpq",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            number_documents_to_search=number_documents_to_search,
            timeout=timeout,
            name=name,
            **kwargs,
        )
        self.base_url = base_url
        self.domains = domains

    def get_mcp_tool_name(self) -> str:
        return "massive_serve_search"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """Build parameters for massive-serve API"""
        params = {
            "query": tool_call_info.content,
            "n_docs": self.number_documents_to_search,
            "domains": self.domains,
        }
        if self.base_url:
            params["base_url"] = self.base_url
        return params

    def extract_documents(self, raw_output: Dict[str, Any]) -> List[Document]:
        """Extract documents from massive-serve response"""
        data = raw_output.get("data", [])
        documents = []

        for item in data:
            if isinstance(item, dict) and "passage" in item:
                doc = Document(
                    title="",  # massive-serve doesn't provide titles
                    snippet=item["passage"].strip(),
                    url="",  # No URLs from this service
                    text=None,  # No full text content from search
                    score=item.get("score"),
                )

                if doc.snippet:
                    documents.append(doc)

        return documents


class MCPBrowseTool(MCPMixin, BaseTool, ABC):
    """Base class for MCP browse tools that fetch webpage content from URLs in search results"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        max_pages_to_fetch: int = 5,
        timeout: int = 120,
        use_localized_snippets: bool = True,
        context_chars: int = 2000,
        name: Optional[str] = None,
        create_string_output: bool = True,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            timeout=timeout,
            name=name,
            create_string_output=create_string_output,
            **kwargs,
        )
        self.max_pages_to_fetch = max_pages_to_fetch
        self.use_localized_snippets = use_localized_snippets
        self.context_chars = context_chars

    def _create_error_output(
        self,
        error_msg: str,
        call_id: str,
        runtime: float,
        output: str = "",
        raw_output: Optional[Dict[str, Any]] = None,
    ) -> DocumentToolOutput:
        """Create a standardized error output for browse tools"""
        return DocumentToolOutput(
            output=output,
            error=error_msg,
            called=True,
            timeout=False,
            runtime=runtime,
            call_id=call_id,
            raw_output=raw_output,
            tool_name=self.name,
            documents=[],
        )

    def extract_urls(self, raw_output: Dict[str, Any]) -> List[str]:
        """Extract URLs from Serper search response"""
        urls = []

        # Handle organic search results
        organic_results = raw_output.get("organic", [])
        for result in organic_results:
            if isinstance(result, dict) and "link" in result:
                urls.append(result["link"])

        return urls

    def extract_urls_and_snippets(
        self, raw_output: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract URLs, titles, and snippets from Serper search response"""
        results = []

        # Handle organic search results
        organic_results = raw_output.get("organic", [])
        for result in organic_results:
            if isinstance(result, dict) and "link" in result:
                results.append(
                    {
                        "url": result["link"],
                        "title": result.get("title", "").strip(),
                        "snippet": result.get("snippet", "").strip(),
                    }
                )

        return results

    @abstractmethod
    def _extract_raw_content_from_response(
        self, raw_output: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract raw text content from webpage fetch response.
        This method handles tool-specific response formats.

        Args:
            raw_output: Raw response from MCP webpage fetch tool

        Returns:
            Raw text content from webpage, or None if extraction failed
        """
        pass

    @abstractmethod
    def _extract_metadata_from_document(
        self, document: Document, raw_output: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract metadata for display purposes from document and raw response.

        Args:
            document: Document object being processed
            raw_output: Raw response from MCP webpage fetch tool

        Returns:
            Tuple of (webpage_title, fallback_message)
            - webpage_title: Title extracted from webpage, if available
            - fallback_message: Error or informational message, if any
        """
        pass

    async def _fetch_single_webpage(
        self,
        url: str,
        **kwargs,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Fetch content from a single webpage.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (url, raw_output_from_mcp)
        """

        # Create a minimal ToolCallInfo for URL fetching
        url_tool_call = ToolCallInfo(
            content=url, parameters=kwargs, start_pos=0, end_pos=len(url)
        )
        params = self.get_mcp_params(url_tool_call)
        raw_output = await self._execute_mcp_call(self.get_mcp_tool_name(), params)
        return url, raw_output

    async def _fetch_webpages_parallel(
        self, documents: List[Document], **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple webpages in parallel using document objects.

        Args:
            documents: List of Document objects with URLs to fetch

        Returns:
            Dictionary mapping document IDs to fetch results
        """
        # Limit the number of documents to fetch
        docs_with_urls = [doc for doc in documents if doc.url]
        docs_to_fetch = docs_with_urls[: self.max_pages_to_fetch]

        async def fetch_document(doc):
            url, raw_output = await self._fetch_single_webpage(doc.url, **kwargs)
            return doc.id, raw_output

        tasks = [fetch_document(doc) for doc in docs_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = {}
        for result in results:
            if isinstance(result, Exception):
                # For exceptions, we don't have the doc_id, so we need to handle this differently
                # We'll skip the exception case or handle it in the calling code
                continue
            else:
                doc_id, raw_output = result
                processed_results[doc_id] = raw_output

        return processed_results

    async def __call__(
        self, tool_input: Union[str, Dict[str, Any], ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        """Browse webpages from URLs found in search tool output or from direct URL string/dict"""
        # Fetch MCP schema on first call (cached for subsequent calls)
        if self._mcp_tool_schema is None:
            await self._fetch_mcp_tool_schema()

        call_id = self._generate_call_id()
        start_time = time.time()

        # Step 1: Handle different input types
        input_documents = []

        original_query = None

        if isinstance(tool_input, dict):
            # Native mode: dict with url parameter
            url = tool_input.get("url", "")
            if not url:
                return self._create_error_output(
                    "No URL provided in tool call.",
                    call_id,
                    time.time() - start_time,
                )

            doc = Document(
                title="",
                snippet="",
                url=url.strip(),
                text=None,
                score=None,
            )
            input_documents = [doc]

        elif isinstance(tool_input, str):
            # Direct URL string input - use tool parser to extract URL
            tool_call_info = self.parse_call(tool_input)
            if not tool_call_info:
                return self._create_error_output(
                    "No valid URL found in tool call.",
                    call_id,
                    time.time() - start_time,
                )

            doc = Document(
                title="",
                snippet="",
                url=tool_call_info.content.strip(),
                text=None,
                score=None,
            )
            input_documents = [doc]

        elif isinstance(tool_input, (ToolOutput, DocumentToolOutput)):
            # Original logic for ToolOutput/DocumentToolOutput
            if not tool_input.raw_output:
                return self._create_error_output(
                    "ToolOutput does not contain raw_output to extract URLs from.",
                    call_id,
                    time.time() - start_time,
                )

            # Get documents from input (prioritize DocumentToolOutput, fallback to raw_output)
            if isinstance(tool_input, DocumentToolOutput) and tool_input.documents:
                input_documents = tool_input.documents
                original_query = tool_input.query
            else:
                # Fallback: extract from raw_output for backward compatibility
                url_snippet_pairs = self.extract_urls_and_snippets(
                    tool_input.raw_output
                )
                for pair in url_snippet_pairs:
                    doc = Document(
                        title=pair.get("title", ""),
                        snippet=pair.get("snippet", ""),
                        url=pair["url"],
                    )
                    input_documents.append(doc)
        else:
            # return self._create_error_output(
            #     "MCPBrowseTool expects ToolOutput from search tool or URL string as input.",
            #     call_id,
            #     time.time() - start_time,
            # )
            raise ValueError(
                "MCPBrowseTool expects ToolOutput from search tool or URL string as input."
            )

        if not input_documents:
            raise ValueError("No documents with URLs found to browse.")

        logger.debug(
            f"Found {len(input_documents)} documents to browse, fetching up to {self.max_pages_to_fetch}"
        )

        # Step 2: Create copies of input documents to enrich with fetched content
        enriched_documents = []
        for doc in input_documents[: self.max_pages_to_fetch]:
            # Create a copy of the document that we'll enrich with text content
            enriched_doc = Document(
                id=doc.id,
                title=doc.title,
                snippet=doc.snippet,
                url=doc.url,
                text=None,  # Will be populated with fetched content
                score=doc.score,
            )
            enriched_documents.append(enriched_doc)

        # Create mapping for easy lookup
        docs_map = {doc.id: doc for doc in enriched_documents}

        # Step 3: Fetch webpages in parallel
        additional_params = {"query": original_query} if original_query else {}
        fetch_results = await self._fetch_webpages_parallel(
            enriched_documents, **additional_params
        )

        # Step 4: Process fetch results and update documents directly
        for doc_id, raw_output in fetch_results.items():
            document = docs_map.get(doc_id, None)
            if error := raw_output.get("error"):
                # Find document to get URL for error message
                document.error = error
            else:
                # Extract raw content directly (tool-specific logic)
                raw_content = self._extract_raw_content_from_response(raw_output)

                # Store the raw content directly in the document's text attribute
                document.text = raw_content

        # Step 6: Create output content using stringify on enriched documents
        webpage_contents = []
        for doc in enriched_documents:
            # Extract tool-specific metadata for display
            webpage_title, fallback_message = self._extract_metadata_from_document(
                doc, fetch_results.get(doc.id, {})
            )

            content = doc.stringify(
                webpage_title=webpage_title,
                use_localized_snippets=self.use_localized_snippets,
                context_chars=self.context_chars,
                fallback_message=fallback_message,
            )

            if not doc.title.strip():
                if webpage_title and webpage_title.strip():
                    doc.title = webpage_title.strip()

            if content:
                webpage_contents.append(content)

        # Combine all webpage contents
        final_content = "\n\n".join(webpage_contents)

        return DocumentToolOutput(
            tool_name=self.name,
            output=final_content if self.create_string_output else "",
            called=True,
            error="",
            timeout=False,
            runtime=time.time() - start_time,
            call_id=call_id,
            # raw_output=[
            #     {
            #         "doc_id": doc_id,
            #         "url": next(
            #             (doc.url for doc in docs_to_fetch if doc.id == doc_id),
            #             "unknown",
            #         ),
            #         "raw_output": raw_output,
            #     }
            #     for doc_id, raw_output in fetch_results.items()
            # ],
            raw_output=None,
            documents=enriched_documents,
            query=getattr(tool_input, "query", None),  # Copy query from input
        )

    def _format_output(self, output: Union[ToolOutput, DocumentToolOutput]) -> str:
        """Format the browse results into string representation"""
        if isinstance(self.tool_parser, LegacyToolCallParser):
            return output.output
        else:
            combined_webpage_text = []
            for index, doc in enumerate(output.documents):
                combined_webpage_text.append(
                    f"<webpage id={output.call_id}-{index}>\n{doc.stringify()}\n</webpage>"
                )
            return "\n".join(combined_webpage_text)


class SerperBrowseTool(MCPBrowseTool):
    """Tool for fetching webpage content using Serper API via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        max_pages_to_fetch: int = 5,
        timeout: int = 120,
        use_localized_snippets: bool = True,
        context_chars: int = 2000,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            max_pages_to_fetch=max_pages_to_fetch,
            timeout=timeout,
            use_localized_snippets=use_localized_snippets,
            context_chars=context_chars,
            name=name,
            **kwargs,
        )

    def get_mcp_tool_name(self) -> str:
        return "serper_fetch_webpage_content"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """Build parameters for Serper webpage fetch API"""
        return {"webpage_url": tool_call_info.content, "include_markdown": True}

    def _extract_raw_content_from_response(
        self, raw_output: Dict[str, Any]
    ) -> Optional[str]:
        """Extract raw text content from Serper response"""
        # Check if webpage fetching failed
        if raw_output.get("success") is False:
            return None

        # Extract content from Serper response
        markdown_content = raw_output.get("markdown", "")
        text_content = raw_output.get("text", "")
        return markdown_content or text_content

    def _extract_metadata_from_document(
        self, document: Document, raw_output: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract metadata for display from Serper response"""
        # Check if webpage fetching failed
        if raw_output.get("success") is False:
            error_message = (
                f"Failed to fetch content: {raw_output.get('error', 'Unknown error')}"
            )
            return None, error_message

        # Extract title from Serper response
        metadata = raw_output.get("metadata", {})
        webpage_title = metadata.get("title", "").strip() if metadata else ""

        return webpage_title, None


class JinaBrowseTool(MCPBrowseTool):
    """Tool for fetching webpage content using Jina Reader API via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        max_pages_to_fetch: int = 5,
        timeout: int = 120,
        request_timeout: int = 30,
        use_localized_snippets: bool = True,
        context_chars: int = 2000,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            max_pages_to_fetch=max_pages_to_fetch,
            timeout=timeout,
            use_localized_snippets=use_localized_snippets,
            context_chars=context_chars,
            name=name,
            **kwargs,
        )
        self.request_timeout = request_timeout

    def get_mcp_tool_name(self) -> str:
        return "jina_fetch_webpage_content"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """Build parameters for Jina Reader API"""
        return {
            "webpage_url": tool_call_info.content,
            "timeout": self.request_timeout,
        }

    def _extract_raw_content_from_response(
        self, raw_output: Dict[str, Any]
    ) -> Optional[str]:
        """Extract raw text content from Jina response"""
        # Check if webpage fetching failed
        if raw_output.get("success") is False:
            return None

        # Extract content from Jina JSON response
        content = raw_output.get("content", "")
        return content

    def _extract_metadata_from_document(
        self, document: Document, raw_output: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract metadata for display from Jina response"""
        # Check if webpage fetching failed
        if raw_output.get("success") is False:
            error_message = (
                f"Failed to fetch content: {raw_output.get('error', 'Unknown error')}"
            )
            return None, error_message

        # Extract title from Jina JSON response
        webpage_title = raw_output.get("title", "").strip()

        return webpage_title, None


class WebThinkerBrowseTool(MCPBrowseTool):
    """Tool for fetching webpage content using WebThinker web parser via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        max_pages_to_fetch: int = 5,
        timeout: int = 180,  # WebThinker might take longer
        keep_links: bool = False,
        use_localized_snippets: bool = True,
        context_chars: int = 2000,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            max_pages_to_fetch=max_pages_to_fetch,
            timeout=timeout,
            use_localized_snippets=use_localized_snippets,
            context_chars=context_chars,
            name=name,
            **kwargs,
        )
        self.keep_links = keep_links

    def get_mcp_tool_name(self) -> str:
        return "webthinker_fetch_webpage_content_async"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """Build parameters for WebThinker API"""
        return {
            "url": tool_call_info.content,
            "snippet": None,  # Could be made configurable
            "keep_links": self.keep_links,
        }

    def _extract_raw_content_from_response(
        self, raw_output: Dict[str, Any]
    ) -> Optional[str]:
        """Extract raw text content from WebThinker response"""
        return raw_output.get("text", "")

    def _extract_metadata_from_document(
        self, document: Document, raw_output: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract metadata for display from WebThinker response"""
        # WebThinker doesn't provide webpage title
        webpage_title = None

        # Handle case where no content was extracted
        text_content = raw_output.get("text", "")
        if not text_content:
            fallback_message = (
                "Note: WebThinker failed to extract content, using search snippet"
            )
            return webpage_title, fallback_message

        return webpage_title, None


class Crawl4AIBrowseTool(MCPBrowseTool):
    """Tool for fetching webpage content using Crawl4AI via MCP"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        max_pages_to_fetch: int = 5,
        timeout: int = 180,  # Crawl4AI might take longer
        ignore_links: bool = True,
        use_pruning: bool = False,
        bm25_query: Optional[str] = None,
        bypass_cache: bool = True,
        include_html: bool = False,
        use_localized_snippets: bool = True,
        context_chars: int = 2000,
        name: Optional[str] = None,
        use_docker_version: bool = False,
        use_ai2_config: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            max_pages_to_fetch=max_pages_to_fetch,
            timeout=timeout,
            use_localized_snippets=use_localized_snippets,
            context_chars=context_chars,
            name=name,
            description=description,
            **kwargs,
        )
        self.ignore_links = ignore_links
        self.use_pruning = use_pruning
        self.bm25_query = bm25_query
        self.bypass_cache = bypass_cache
        self.timeout_ms = (
            max(0, (timeout - 1)) * 1000
        )  # make sure crawl4ai timeout is shorter than MCP timeout
        self.include_html = include_html
        self.use_docker_version = use_docker_version
        self.use_ai2_config = use_ai2_config
        self.base_url = base_url
        self.api_key = api_key

    def get_mcp_tool_name(self) -> str:
        if self.use_docker_version:
            return "crawl4ai_docker_fetch_webpage_content"
        return "crawl4ai_fetch_webpage_content"

    def get_mcp_params(self, tool_call_info: ToolCallInfo) -> Dict[str, Any]:
        """Build parameters for Crawl4AI API"""
        use_pruning = self.use_pruning
        bm25_query = tool_call_info.parameters.get("query", self.bm25_query)

        # If use_pruning is True and bm25_query is None, set use_pruning to True and bm25_query to None
        if self.use_pruning and bm25_query is None:
            use_pruning = True
            bm25_query = None
        else:
            use_pruning = False

        input_params = {
            "url": tool_call_info.content,
            "ignore_links": self.ignore_links,
            "use_pruning": use_pruning,
            "bm25_query": bm25_query,
            "bypass_cache": self.bypass_cache,
            "timeout_ms": self.timeout_ms,
            "include_html": self.include_html,
        }

        # Add docker-specific parameters
        if self.use_docker_version:
            if self.base_url is not None:
                input_params["base_url"] = self.base_url
            if self.api_key is not None:
                input_params["api_key"] = self.api_key
            input_params["use_ai2_config"] = self.use_ai2_config

        # print(input_params)
        return input_params

    def _extract_raw_content_from_response(
        self, raw_output: Dict[str, Any]
    ) -> Optional[str]:
        """Extract raw text content from Crawl4AI response"""
        # Check if crawling was successful
        if not raw_output.get("success", False):
            return None

        # Extract content from Crawl4AI response
        markdown_content = raw_output.get("markdown", "")
        fit_markdown_content = raw_output.get("fit_markdown", "")
        html_content = raw_output.get("html", "")

        # Prefer fit_markdown if available, then markdown, then html
        return fit_markdown_content or markdown_content or html_content

    def _extract_metadata_from_document(
        self, document: Document, raw_output: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract metadata for display from Crawl4AI response"""
        # Check if crawling was successful
        if not raw_output.get("success", False):
            error_msg = raw_output.get("error", "Unknown error")
            fallback_message = (
                f"Note: Crawl4AI failed ({error_msg}), using search snippet"
            )
            return None, fallback_message

        # Crawl4AI doesn't provide webpage title
        webpage_title = None

        # Handle case where no content was extracted
        markdown_content = raw_output.get("markdown", "")
        fit_markdown_content = raw_output.get("fit_markdown", "")
        html_content = raw_output.get("html", "")
        full_content = fit_markdown_content or markdown_content or html_content

        if not full_content:
            fallback_message = (
                "Note: No content extracted by Crawl4AI, using search snippet"
            )
            return webpage_title, fallback_message

        return webpage_title, None


class MCPRerankerTool(MCPMixin, BaseTool, ABC):
    """Base class for MCP reranker tools with shared pipeline logic"""

    def __init__(
        self,
        tool_parser: Optional[ToolCallParser | str] = None,
        timeout: int = 120,
        top_n: int = -1,
        name: Optional[str] = None,
        score_threshold: Optional[float] = None,
        create_string_output: bool = True,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            timeout=timeout,
            name=name,
            create_string_output=create_string_output,
            **kwargs,
        )
        self.top_n = top_n
        self.score_threshold = score_threshold

    def _create_error_output(
        self,
        error_msg: str,
        call_id: str,
        runtime: float,
        input_documents: List[Document] = None,
        raw_output: Optional[Dict[str, Any]] = None,
    ) -> DocumentToolOutput:
        """Create a standardized error output for reranker tools"""
        return DocumentToolOutput(
            output="",
            error=error_msg,
            called=True,
            timeout=False,
            runtime=runtime,
            call_id=call_id,
            raw_output=raw_output,
            tool_name=self.name,
            documents=input_documents or [],
        )

    def _extract_query_from_input(
        self, tool_input: DocumentToolOutput
    ) -> Optional[str]:
        """
        Extract query from DocumentToolOutput.
        This can be overridden by subclasses for custom query extraction logic.
        """
        # First priority: check the query field directly
        if tool_input.query:
            return tool_input.query

        # Second priority: try to get query from raw_output if available
        if tool_input.raw_output and isinstance(tool_input.raw_output, dict):
            # For search results, query might be in the raw output
            if "query" in tool_input.raw_output:
                return tool_input.raw_output["query"]

            # Try Serper search format: SearchParameters.q
            if "SearchParameters" in tool_input.raw_output:
                search_params = tool_input.raw_output["SearchParameters"]
                if isinstance(search_params, dict) and "q" in search_params:
                    return search_params["q"]

            # Try other common query field names
            for query_field in ["search_query", "original_query", "q"]:
                if query_field in tool_input.raw_output:
                    return tool_input.raw_output[query_field]

        # Fallback: extract from output text (this is a simple heuristic)
        # Subclasses should override this method for better query extraction
        return None

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        """Shared pipeline for all reranker tools"""
        call_id = self._generate_call_id()
        start_time = time.time()

        # Step 1: Validate input is DocumentToolOutput with documents
        if not isinstance(tool_input, DocumentToolOutput):
            raise ValueError("MCPRerankerTool expects DocumentToolOutput as input.")

        if not tool_input.documents:
            raise ValueError("DocumentToolOutput does not contain documents to rerank.")

        # Step 2: Extract query from input
        query = self._extract_query_from_input(tool_input)
        if not query:
            warnings.warn(
                "Could not extract query from input for reranking.", UserWarning
            )
            return self._create_error_output(
                "Could not extract query from input for reranking.",
                call_id,
                time.time() - start_time,
                input_documents=tool_input.documents,
            )

        # Step 3: Extract document texts for reranking
        document_texts = []
        for doc in tool_input.documents:
            # Use snippet if available, otherwise use text, otherwise skip
            text = doc.simple_stringify(prioritize_summary=True)

            if text.strip():
                document_texts.append(text.strip())
            else:
                document_texts.append(f"{doc.title}\n{doc.url}\n{doc.snippet}")

        if not document_texts:
            raise ValueError("No valid document texts found for reranking.")

        # Step 4: Determine top_n
        effective_top_n = self.top_n if self.top_n > 0 else len(document_texts)
        effective_top_n = min(effective_top_n, len(document_texts))

        # Step 5: Build parameters and execute MCP call
        params = self.get_mcp_params(query, document_texts, effective_top_n)
        raw_output = await self._execute_mcp_call(self.get_mcp_tool_name(), params)

        # Step 6: Check for execution errors
        if error := raw_output.get("error"):
            return self._create_error_output(
                f"Reranking failed: {error}",
                call_id,
                time.time() - start_time,
                input_documents=tool_input.documents,
                raw_output=raw_output,
            )

        # Step 7: Process reranker results and update documents
        reranked_documents = self._process_reranker_results(
            raw_output, tool_input.documents, query
        )

        if not reranked_documents:
            return self._create_error_output(
                "No reranked results returned from reranker.",
                call_id,
                time.time() - start_time,
                input_documents=tool_input.documents,
                raw_output=raw_output,
            )

        # Step 8: Format output content
        output_content = self._format_reranked_output(reranked_documents, query)

        return DocumentToolOutput(
            tool_name=self.name,
            output=output_content if self.create_string_output else "",
            called=True,
            error="",
            timeout=False,
            runtime=time.time() - start_time,
            call_id=call_id,
            raw_output=raw_output,
            documents=reranked_documents,
            query=query,  # Preserve the query for downstream tools
        )

    @abstractmethod
    def _process_reranker_results(
        self, raw_output: Dict[str, Any], original_documents: List[Document], query: str
    ) -> List[Document]:
        """
        Process reranker results and create updated documents with new scores.
        Uses index-based mapping for reliable document matching.
        """
        pass

    def _format_reranked_output(self, documents: List[Document], query: str) -> str:
        """Format the reranked documents into output text"""
        output_parts = []

        # sort documents by score and save as a new list
        sorted_documents = sorted(documents, key=lambda x: x.score, reverse=True)

        for i, doc in enumerate(sorted_documents):
            # Use document stringify method to format the document
            output_parts.append(doc.stringify())

        return "\n\n".join(output_parts)

    # def format_result(self, output: Union[ToolOutput, DocumentToolOutput]) -> str:
    #     """Format the final result with reranker tags"""
    #     if output.call_id:
    #         return f"<reranked id={output.call_id}>\n{output.output}\n</reranked>"
    #     else:
    #         return f"<reranked>\n{output.output}\n</reranked>"

    def _format_output(self, output: Union[ToolOutput, DocumentToolOutput]) -> str:
        """Format the reranker results into string representation"""
        return output.output


class VllmHostedRerankerTool(MCPRerankerTool):
    """Tool for reranking documents using VLLM hosted reranker via MCP"""

    def __init__(
        self,
        model_name: str,
        api_url: str,
        tool_parser: Optional[ToolCallParser | str] = None,
        top_n: int = -1,
        timeout: int = 120,
        name: Optional[str] = None,
        score_threshold: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            tool_parser=tool_parser,
            timeout=timeout,
            top_n=top_n,
            name=name,
            score_threshold=score_threshold,
            **kwargs,
        )
        self.model_name = model_name
        self.api_url = api_url

    def get_mcp_tool_name(self) -> str:
        return "vllm_hosted_reranker"

    def get_mcp_params(
        self, query: str, documents: List[str], top_n: int
    ) -> Dict[str, Any]:
        """Build parameters for VLLM hosted reranker API"""
        return {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "model_name": self.model_name,
            "api_url": self.api_url,
        }

    def _process_reranker_results(
        self, raw_output: Dict[str, Any], original_documents: List[Document], query: str
    ) -> List[Document]:
        """
        Process reranker results and create updated documents with new scores.
        Uses index-based mapping for reliable document matching.
        """
        # Handle RerankerResult format with list of RerankResultItem objects
        reranked_results = raw_output.get("results", [])
        if not reranked_results:
            return []

        reranked_documents = []

        # Process reranked results using index-based mapping
        for result in reranked_results:
            if isinstance(result, dict):
                # Extract index and score from reranker result
                doc_index = int(result.get("index"))
                rerank_score = result.get("relevance_score", 0.0)

                # Get the original document by index
                if doc_index is not None and 0 <= doc_index < len(original_documents):
                    original_doc = original_documents[doc_index]

                    # Create updated document with new rerank score
                    reranked_doc = Document(
                        id=original_doc.id,
                        title=original_doc.title,
                        snippet=original_doc.snippet,
                        summary=original_doc.summary,
                        url=original_doc.url,
                        text=original_doc.text,
                        score=rerank_score,  # Update with rerank score
                    )
                    reranked_documents.append(reranked_doc)

        if self.score_threshold is not None:
            reranked_documents = [
                doc for doc in reranked_documents if doc.score >= self.score_threshold
            ]

        # sort documents by score
        reranked_documents = sorted(
            reranked_documents, key=lambda x: x.score, reverse=True
        )

        return reranked_documents
