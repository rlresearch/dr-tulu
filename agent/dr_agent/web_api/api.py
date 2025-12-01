"""
FastAPI SSE server for live chat with BaseWorkflow instances.

This module provides an SSE endpoint that streams workflow responses
to the frontend, including thinking text, tool calls, and final answers.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..tool_interface.data_types import DocumentToolOutput, ToolOutput
from ..workflow import BaseWorkflow


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    content: Optional[str] = None
    messages: Optional[List[Message]] = None
    dataset_name: str = "long_form"


class SSECallback:
    """
    Callback handler that collects workflow updates for SSE streaming.
    """

    def __init__(self, event_queue: asyncio.Queue):
        self.event_queue = event_queue
        self.last_processed_text_len = 0
        self.is_answering = False
        self.current_segment_text = ""
        self.snippets: Dict[str, Dict[str, Any]] = {}
        self.thinking_segment_id = 0

    def _make_event(self, msg_type: str, **data) -> str:
        """Format a message as an SSE event."""
        message = {"type": msg_type, **data}
        return f"data: {json.dumps(message)}\n\n"

    async def send_event(self, msg_type: str, **data):
        """Queue an SSE event."""
        await self.event_queue.put(self._make_event(msg_type, **data))

    async def __call__(self, text: str, tool_calls: List[ToolOutput]):
        """
        Handle step callback from workflow.
        """
        # Detect text stream reset (new generation started)

        if text and len(text) < self.last_processed_text_len:
            # Finalize current segment before reset
            if self.current_segment_text and not self.is_answering:
                await self.send_event(
                    "thinking",
                    content=self.current_segment_text,
                    is_complete=True,
                    segment_id=self.thinking_segment_id,
                )
            self.last_processed_text_len = 0
            self.current_segment_text = ""
            self.is_answering = False
            self.thinking_segment_id += 1

        # Handle text updates
        if text and len(text) > self.last_processed_text_len:
            new_chunk = text[self.last_processed_text_len :]
            self.last_processed_text_len = len(text)

            # Skip if just whitespace after a tool call
            if not self.current_segment_text and not new_chunk.strip():
                return

            self.current_segment_text += new_chunk

            # Check for answer tag in accumulated text
            answer_match = re.search(r"(?s)(.*)<answer>(.*)", self.current_segment_text)

            if answer_match:
                thinking_text = answer_match.group(1)
                answer_text = answer_match.group(2).replace("</answer>", "")

                if not self.is_answering:
                    self.is_answering = True
                    if thinking_text.strip():
                        await self.send_event(
                            "thinking",
                            content=thinking_text,
                            is_complete=True,
                            segment_id=self.thinking_segment_id,
                        )
                    self.current_segment_text = answer_text

                await self.send_event("answer", content=answer_text)
            else:
                if not self.is_answering:
                    await self.send_event(
                        "thinking",
                        content=self.current_segment_text,
                        is_complete=False,
                        segment_id=self.thinking_segment_id,
                    )
                else:
                    await self.send_event("answer", content=self.current_segment_text)

        # Handle tool calls
        if tool_calls:
            # First, finalize the current thinking segment
            if self.current_segment_text and not self.is_answering:
                await self.send_event(
                    "thinking",
                    content=self.current_segment_text,
                    is_complete=True,
                    segment_id=self.thinking_segment_id,
                )

            # Send each tool call
            for tool_call in tool_calls:
                tool_data = {
                    "tool_name": tool_call.tool_name,
                    "call_id": getattr(tool_call, "call_id", None),
                    "output": tool_call.output[:2000] if tool_call.output else "",
                    "error": tool_call.error,
                }

                # Extract query if available
                query = getattr(tool_call, "query", None)
                if query:
                    tool_data["query"] = query

                # Extract input parameters if available
                input_params = getattr(tool_call, "input_params", None)
                if input_params and isinstance(input_params, dict):
                    # Filter out 'query' since we send it separately, and filter empty values
                    params = {
                        k: v
                        for k, v in input_params.items()
                        if k != "query" and v is not None and v != ""
                    }
                    if params:
                        tool_data["params"] = params

                # Also try to extract from raw_output["SearchParameters"] for Serper-style APIs
                if not tool_data.get("params"):
                    raw_output = getattr(tool_call, "raw_output", None)
                    if raw_output and isinstance(raw_output, dict):
                        search_params = raw_output.get(
                            "SearchParameters"
                        ) or raw_output.get("searchParameters")
                        if search_params and isinstance(search_params, dict):
                            # Extract relevant params, excluding the query
                            params = {
                                k: v
                                for k, v in search_params.items()
                                if k not in ("q", "query") and v is not None and v != ""
                            }
                            if params:
                                tool_data["params"] = params

                # Extract documents if available
                if isinstance(tool_call, DocumentToolOutput) and tool_call.documents:
                    documents = []
                    for idx, doc in enumerate(tool_call.documents):
                        snippet_id = (
                            f"{tool_call.call_id}-{idx}"
                            if getattr(tool_call, "call_id", None)
                            else doc.id
                        )
                        doc_data = {
                            "id": snippet_id,
                            "title": doc.title,
                            "url": doc.url,
                            "snippet": doc.snippet[:500] if doc.snippet else "",
                        }
                        documents.append(doc_data)

                        # Store for bibliography
                        self.snippets[snippet_id] = {
                            "id": snippet_id,
                            "title": doc.title,
                            "url": doc.url,
                            "snippet": doc.snippet,
                            "tool_name": tool_call.tool_name,
                        }

                    tool_data["documents"] = documents

                await self.send_event("tool_call", **tool_data)

            # Reset for next segment
            self.last_processed_text_len = 0
            self.current_segment_text = ""
            self.is_answering = False
            self.thinking_segment_id += 1


def create_app(
    workflow_instance: BaseWorkflow,
    ui_mode: str = "auto",
    dev_url: Optional[str] = None,
) -> FastAPI:
    """
    Create a FastAPI app configured to serve the given workflow.

    Args:
        workflow_instance: A BaseWorkflow instance to serve
        ui_mode: Mode for serving UI. Options:
            - "auto" (default): Use pre-compiled if available, otherwise show instructions
            - "precompiled": Only use pre-compiled files (error if not available)
            - "dev": Show instructions to run dev server separately
            - "proxy": Show redirect to dev server (requires dev_url)
        dev_url: URL of the development server (e.g., "http://localhost:3000")
                 Only used when ui_mode is "proxy"

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(title="DR-Agent Chat API")

    # Enable CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store workflow in app state
    app.state.workflow = workflow_instance
    app.state.ui_mode = ui_mode

    async def run_workflow_with_streaming(
        content: str,
        dataset_name: str,
        event_queue: asyncio.Queue,
        messages: Optional[List[Dict[str, str]]] = None,
    ):
        """Run the workflow and stream events via the queue."""
        callback = SSECallback(event_queue)

        # Send started event
        await event_queue.put(f'data: {{"type": "started"}}\n\n')

        result = await app.state.workflow(
            problem=content,
            dataset_name=dataset_name,
            messages=messages,
            verbose=False,
            step_callback=callback,
        )

        # Send final answer if available
        final_response = result.get("final_response", "")
        if final_response:
            await event_queue.put(
                f'data: {json.dumps({"type": "answer", "content": final_response, "is_final": True})}\n\n'
            )

        # Send done message with metadata
        done_msg = {
            "type": "done",
            "metadata": {
                "total_tool_calls": result.get("total_tool_calls", 0),
                "failed_tool_calls": result.get("total_failed_tool_calls", 0),
                "browsed_links": result.get("browsed_links", []),
                "searched_links": result.get("searched_links", []),
                "snippets": callback.snippets,
            },
        }
        await event_queue.put(f"data: {json.dumps(done_msg)}\n\n")

        # Signal completion
        await event_queue.put(None)

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        """
        SSE endpoint for chat interactions.

        Client sends: { "content": "...", "dataset_name": "..." }
        Server streams: SSE events for thinking, tool_call, answer, done, error
        """
        if app.state.workflow is None:
            return StreamingResponse(
                iter(
                    [
                        f'data: {json.dumps({"type": "error", "message": "Workflow not initialized"})}\n\n'
                    ]
                ),
                media_type="text/event-stream",
            )

        event_queue: asyncio.Queue = asyncio.Queue()

        # Prepare messages
        messages_dicts = []
        content = ""

        if request.messages:
            messages_dicts = [
                {"role": m.role, "content": m.content} for m in request.messages
            ]
            # content is extracted in workflow from messages, but we pass it for logging/legacy
            if messages_dicts:
                # Find last user message
                for m in reversed(messages_dicts):
                    if m["role"] == "user":
                        content = m["content"]
                        break
        elif request.content:
            content = request.content
            messages_dicts = [{"role": "user", "content": content}]

        async def event_generator():
            # Start the workflow in a background task
            task = asyncio.create_task(
                run_workflow_with_streaming(
                    content, request.dataset_name, event_queue, messages=messages_dicts
                )
            )

            try:
                while True:
                    event = await event_queue.get()
                    if event is None:  # Completion signal
                        break
                    yield event
            except asyncio.CancelledError:
                task.cancel()
                raise
            except Exception as e:
                yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "workflow_loaded": app.state.workflow is not None,
            "ui_mode": ui_mode,
        }

    # Mount UI files if available
    try:
        from dr_agent_ui.server import mount_ui

        ui_mounted = mount_ui(app, ui_mode=ui_mode, dev_url=dev_url)
        if ui_mounted:
            print(f"✓ UI mounted successfully in '{ui_mode}' mode")
        else:
            print(f"ℹ UI not mounted (mode: {ui_mode}). API endpoints are available.")
    except ImportError:
        print(
            "ℹ dr_agent_ui not installed. Install it to enable the web interface: "
            "pip install dr-agent-ui"
        )
    except Exception as e:
        print(f"⚠ Failed to mount UI: {e}")

    return app
