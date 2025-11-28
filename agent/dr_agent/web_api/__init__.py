"""
Web API module for serving dr_agent workflows via FastAPI.

This module provides a FastAPI-based web API that can serve any BaseWorkflow
instance via Server-Sent Events (SSE) for live chat interactions.
"""

from .api import create_app, SSECallback

__all__ = ["create_app", "SSECallback"]

