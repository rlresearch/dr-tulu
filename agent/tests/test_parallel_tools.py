"""Exercise provider selection and the HTTP MCP caller without external services."""

import asyncio
import json

import httpx
import pytest
from fastmcp.client.transports import StreamableHttpTransport

from dr_agent.client import LLMToolClient
from dr_agent.tool_interface import parallel_tools
from dr_agent.tool_interface.mcp_tools import (
    SerperSearchTool, JinaBrowseTool, MCPMixin, MCPErrorHandlingMode,
)
from dr_agent.tool_interface.parallel_tools import ParallelBrowseTool, ParallelSearchTool
from workflows.auto_search_sft import AutoReasonSearchWorkflow


@pytest.fixture
def mcp_http(monkeypatch):
    requests = []
    state = {"fetch_error": False, "search_error": False, "search_timeout": False}

    def handle(request):
        assert str(request.url) == parallel_tools.PARALLEL_MCP_URL
        assert request.headers["User-Agent"] == parallel_tools.PARALLEL_USER_AGENT
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        requests.append(body)
        method = body["method"]
        if "id" not in body:
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [
                {"name": "web_search", "inputSchema": {"type": "object", "required": ["objective", "search_queries"]}},
                {"name": "web_fetch", "inputSchema": {"type": "object", "required": ["urls"]}},
            ]}
        elif method == "tools/call":
            if body["params"]["name"] == "web_search":
                if state["search_timeout"]:
                    raise httpx.ReadTimeout("Read timed out", request=request)
                if state["search_error"]:
                    result = {"content": [{"type": "text", "text": "Search unavailable"}], "isError": True}
                    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
                payload = {"results": [
                    {"title": "DR Tulu", "url": "https://example.org/tulu", "excerpts": ["Deep research model", "Open weights"]},
                    {"title": None, "url": "https://example.org/other", "excerpts": ["Other result"]},
                ]}
            elif state["fetch_error"]:
                payload = {"results": [], "errors": [{"url": "https://example.org/tulu", "error_type": "FETCH_FAILED"}]}
            else:
                payload = {"results": [{"title": "DR Tulu page", "url": "https://example.org/tulu", "excerpts": ["Deep research page text"]}]}
            result = {"content": [{"type": "text", "text": json.dumps(payload)}]}
        else:
            assert method == "ping"
            result = {}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def transport(url, headers):
        def factory(headers=None, timeout=None, auth=None):
            assert auth is None
            return httpx.AsyncClient(headers=headers, timeout=timeout, transport=httpx.MockTransport(handle))
        return StreamableHttpTransport(url, headers=headers, httpx_client_factory=factory)

    monkeypatch.setattr(parallel_tools, "StreamableHttpTransport", transport)
    return requests, state


def workflow(**overrides):
    return AutoReasonSearchWorkflow(
        configuration="workflows/auto_search_sft.yaml",
        skip_service_check=True,
        search_agent_model_name="gpt-4.1-2025-04-14",
        search_agent_base_url=None,
        search_agent_api_key="unused-fixture-key",
        **overrides,
    )


@pytest.mark.asyncio
async def test_selected_provider_executes_http_search_and_fetch(mcp_http, monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")
    monkeypatch.setenv("MCP_TRANSPORT", "FastMCPTransport")
    requests, _ = mcp_http
    w = workflow(search_tool_name="parallel", browse_tool_name="parallel", number_documents_to_search=1)
    assert isinstance(w.search_tool, ParallelSearchTool)
    assert isinstance(w.search_tool2, ParallelSearchTool)
    assert isinstance(w.browse_tool, ParallelBrowseTool)
    result = await w.search_tool2({"query": "DR Tulu model"})
    assert not result.error
    assert len(result.documents) == 1
    assert result.documents[0].snippet == "Deep research model\n\nOpen weights"
    page = await w.browse_tool(result)
    assert page.documents[0].text == "Deep research page text"
    assert page.documents[0].url == result.documents[0].url
    assert page.documents[0].id == result.documents[0].id
    assert "DR Tulu page" in page.output
    # Discovery must not replace the local query/url schemas or descriptions.
    assert w.search_tool2.tool_tool_schema["required"] == ["query"]
    assert w.browse_tool.tool_tool_schema["required"] == ["url"]
    calls = [r["params"] for r in requests if r["method"] == "tools/call"]
    assert calls[0]["arguments"]["search_queries"] == ["DR Tulu model"]
    assert calls[1]["arguments"]["urls"] == ["https://example.org/tulu"]
    assert calls[1]["arguments"]["objective"] == "DR Tulu model"
    assert calls[1]["arguments"]["full_content"] is False
    assert calls[0]["arguments"]["model_name"] == w.configuration.search_agent_model_name
    assert any(r["method"] == "tools/list" for r in requests)


def test_defaults_and_mixed_selection():
    w = workflow()
    assert isinstance(w.search_tool2, SerperSearchTool)
    assert isinstance(w.browse_tool, JinaBrowseTool)
    w = workflow(search_tool_name="parallel")
    assert isinstance(w.search_tool2, ParallelSearchTool)
    assert isinstance(w.browse_tool, JinaBrowseTool)


@pytest.mark.asyncio
@pytest.mark.parametrize("session_mode", ["one_shot", "explicit", "history_without_id"])
async def test_workflow_agent_loop_and_concurrent_sessions(mcp_http, monkeypatch, session_mode):
    requests, _ = mcp_http

    async def completion(self, messages, **kwargs):
        # Script only the model response; the real agent loop dispatches MCP tools.
        tool_outputs = [m for m in messages if m["role"] == "user" and "<tool_output>" in m["content"]]
        if not tool_outputs:
            return '<call_tool name="google_search">DR Tulu model</call_tool>'
        if len(tool_outputs) == 1:
            assert "Deep research model" in tool_outputs[0]["content"]
            return '<call_tool name="browse_webpage">https://example.org/tulu</call_tool>'
        assert "Deep research page text" in tool_outputs[-1]["content"]
        return '<answer>DR Tulu is a deep research model.</answer>'

    monkeypatch.setattr(LLMToolClient, "_generate_single_response_commercial_api", completion)
    w = workflow(search_tool_name="parallel", browse_tool_name="parallel")
    from uuid import uuid4
    ids = [str(uuid4()), str(uuid4())]
    arguments = []
    for index in range(2):
        params = {"problem": "Same opening question", "verbose": False}
        if session_mode != "one_shot":
            params["messages"] = [{"role": "user", "content": "Same opening question"}]
        if session_mode == "explicit":
            params["session_id"] = ids[index]
        arguments.append(params)
    results = await asyncio.gather(*(w(**params) for params in arguments))
    assert all(r["total_tool_calls"] == 2 and r["total_failed_tool_calls"] == 0 for r in results)
    assert all(r["browsed_links"] == ["https://example.org/tulu"] for r in results)
    calls = [r["params"] for r in requests if r["method"] == "tools/call"]
    if session_mode == "history_without_id":
        assert all("session_id" not in c["arguments"] for c in calls)
    else:
        search_sessions = {c["arguments"]["session_id"] for c in calls if c["name"] == "web_search"}
        fetch_sessions = {c["arguments"]["session_id"] for c in calls if c["name"] == "web_fetch"}
        assert len(search_sessions) == 2
        assert search_sessions == fetch_sessions
        if session_mode == "explicit":
            assert search_sessions == set(ids)
            for index, params in enumerate(arguments):
                params["messages"] += [
                    {"role": "assistant", "content": results[index]["final_response"]},
                    {"role": "user", "content": "Follow-up question"},
                ]
            await asyncio.gather(*(w(**params) for params in arguments))
            calls = [r["params"] for r in requests if r["method"] == "tools/call"]
            assert len(calls) == 8
            assert all(sum(c["arguments"]["session_id"] == sid for c in calls) == 4 for sid in ids)


@pytest.mark.asyncio
async def test_fetch_failure_preserves_snippet(mcp_http):
    _, state = mcp_http
    search = await ParallelSearchTool(number_documents_to_search=1)({"query": "DR Tulu model"})
    state["fetch_error"] = True
    page = await ParallelBrowseTool()(search)
    assert page.documents[0].error == "FETCH_FAILED"
    assert page.documents[0].snippet == search.documents[0].snippet
    assert "Parallel fetch failed" in page.output


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch):
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()
    tool = ParallelSearchTool()
    monkeypatch.setattr(tool, "_execute_mcp_call", cancelled)
    tool._mcp_tool_schema = {"inputSchema": {}}
    with pytest.raises(asyncio.CancelledError):
        await tool({"query": "DR Tulu model"})


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["search_error", "search_timeout"])
async def test_search_failure_returns_error(mcp_http, monkeypatch, failure):
    _, state = mcp_http
    state[failure] = True
    monkeypatch.setattr(MCPMixin, "_error_handling_mode", MCPErrorHandlingMode.RETURN_ERROR)
    result = await ParallelSearchTool(timeout=0.2)({"query": "DR Tulu model"})
    assert result.error
    assert not result.documents


def test_remote_tools_skip_local_server_check(monkeypatch):
    from workflows import auto_search_sft
    def unexpected_check(*args):
        pytest.fail("Fully remote tools must not check the local MCP port")
    monkeypatch.setattr(auto_search_sft, "check_port", unexpected_check)
    w = workflow(search_tool_name="parallel", browse_tool_name="parallel")
    w.before_launch_check()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/chat", "/chat/stream"])
async def test_chat_api_forwards_only_supplied_conversation_id(endpoint):
    from dr_agent.web_api import create_app
    received = []

    class ChatWorkflow:
        async def __call__(self, **kwargs):
            received.append(kwargs)
            return {"final_response": "Answer"}

    app = create_app(ChatWorkflow(), ui_mode="dev")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(endpoint, json={"content": "Question", "session_id": "conversation-1"})
        assert response.status_code == 200
        assert received[-1]["session_id"] == "conversation-1"
        response = await client.post(endpoint, json={"content": "Question"})
        assert response.status_code == 200
        assert "session_id" not in received[-1]
        response = await client.post(endpoint, json={"content": "Question", "session_id": "x" * 101})
        assert response.status_code == 422
