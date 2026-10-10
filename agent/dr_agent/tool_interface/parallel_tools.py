"""Optional, anonymous web search and browsing through Parallel Search MCP."""

from contextvars import ContextVar
from functools import wraps
from inspect import signature
from uuid import uuid4

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from .data_types import Document
from .mcp_tools import MCPBrowseTool, MCPSearchTool

PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
PARALLEL_USER_AGENT = "dr-tulu (+https://github.com/rlresearch/dr-tulu)"
_session_id = ContextVar("parallel_mcp_session_id", default=None)


def with_parallel_session(function):
    """Use a supplied conversation ID without mixing concurrent workflow calls."""
    call_signature = signature(function)

    @wraps(function)
    async def wrapped(*args, **kwargs):
        arguments = call_signature.bind(*args, **kwargs).arguments
        session_id = arguments.get("session_id")
        if session_id is not None and not 1 <= len(session_id) <= 100:
            raise ValueError("session_id must contain 1 to 100 characters")
        # History does not identify a conversation uniquely. Without a caller's
        # ID, omit the optional metadata rather than changing it between turns.
        if session_id is None:
            session_id = "" if arguments.get("messages") else str(uuid4())
        token = _session_id.set(session_id)
        try:
            return await function(*args, **kwargs)
        finally:
            _session_id.reset(token)

    return wrapped


class _ParallelMCPMixin:
    def __init__(self, *args, model_name=None, session_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_model_name = model_name
        self.parallel_session_id = session_id or str(uuid4())

    def init_mcp_client(self):
        # Explicit transport bypasses local MCP settings and saved authentication.
        return Client(
            StreamableHttpTransport(
                PARALLEL_MCP_URL,
                headers={"User-Agent": PARALLEL_USER_AGENT},
            ),
            timeout=self.timeout,
        )

    def _parallel_metadata(self):
        session_id = _session_id.get()
        if session_id is None:
            session_id = self.parallel_session_id
        params = {"session_id": session_id} if session_id else {}
        if self.parallel_model_name:
            params["model_name"] = self.parallel_model_name
        return params


class ParallelSearchTool(_ParallelMCPMixin, MCPSearchTool):
    """General web search with Parallel's anonymous Fast mode."""

    def get_mcp_tool_name(self):
        return "web_search"

    def get_mcp_params(self, tool_call_info):
        query = tool_call_info.content.strip()
        return {
            "objective": query,
            "search_queries": [query],
            **self._parallel_metadata(),
        }

    def _generate_tool_schema(self):
        # Keep DR Tulu's query interface even after remote schema discovery.
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Web search query"}},
            "required": ["query"],
        }

    def _tool_description(self):
        return "Search the web for relevant pages and excerpts using a query."

    def extract_documents(self, raw_output):
        return [
            Document(
                title=(result.get("title") or "").strip(),
                url=result["url"],
                snippet="\n\n".join(result.get("excerpts") or []),
            )
            for result in raw_output.get("results", [])[:self.number_documents_to_search]
            if result.get("url")
        ]


class ParallelBrowseTool(_ParallelMCPMixin, MCPBrowseTool):
    """Fetch relevant page excerpts through Parallel's anonymous MCP endpoint."""

    def get_mcp_tool_name(self):
        return "web_fetch"

    def get_mcp_params(self, tool_call_info):
        params = {
            "urls": [tool_call_info.content.strip()],
            "full_content": False,
            **self._parallel_metadata(),
        }
        query = tool_call_info.parameters.get("query")
        if query:
            params["objective"] = query[:200]
            params["search_queries"] = [query]
        return params

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "HTTP or HTTPS page URL"}},
            "required": ["url"],
        }

    def _tool_description(self):
        return "Read relevant excerpts from a web page using its URL."

    async def _fetch_single_webpage(self, url, **kwargs):
        url, response = await super()._fetch_single_webpage(url, **kwargs)
        if not response.get("results") and not response.get("error"):
            errors = response.get("errors") or []
            response["error"] = (
                errors[0].get("error_type", "Page extraction failed")
                if errors else "No page content returned"
            )
        return url, response

    def _extract_raw_content_from_response(self, raw_output):
        results = raw_output.get("results") or []
        return "\n\n".join(results[0].get("excerpts") or []) if results else None

    def _extract_metadata_from_document(self, document, raw_output):
        results = raw_output.get("results") or []
        title = results[0].get("title") if results else None
        error = raw_output.get("error")
        if error:
            return title, f"Note: Parallel fetch failed ({error}), using search snippet"
        if not self._extract_raw_content_from_response(raw_output):
            return title, "Note: No page excerpts returned, using search snippet"
        return title, None
