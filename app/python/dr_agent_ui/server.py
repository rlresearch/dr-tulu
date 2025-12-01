"""
Static file server utilities for dr_agent_ui.

This module provides utilities to serve the pre-compiled UI files
alongside the DR-Agent API endpoints.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import get_static_dir, is_ui_available


def mount_ui(
    app: FastAPI, ui_mode: str = "auto", dev_url: Optional[str] = None
) -> bool:
    """
    Mount the UI files to the FastAPI app.

    Args:
        app: FastAPI application instance
        ui_mode: Mode for serving UI. Options:
            - "auto": Use pre-compiled if available, otherwise show dev instructions
            - "precompiled": Only use pre-compiled files (error if not available)
            - "dev": Show instructions to run dev server separately
            - "proxy": Proxy to dev server (requires dev_url)
        dev_url: URL of the development server (e.g., "http://localhost:3000")
                 Required if ui_mode is "proxy"

    Returns:
        True if UI was mounted successfully, False otherwise
    """
    static_dir = get_static_dir()

    if ui_mode == "precompiled" or (ui_mode == "auto" and is_ui_available()):
        if not is_ui_available():
            raise FileNotFoundError(
                "Pre-compiled UI files not found. "
                "Run 'cd app && npm run export' to build the UI."
            )

        # Mount static files (CSS, JS, images)
        app.mount(
            "/_next", StaticFiles(directory=static_dir / "_next"), name="next-static"
        )

        # Check if images directory exists and mount it
        images_dir = static_dir / "images"
        if images_dir.exists():
            app.mount("/images", StaticFiles(directory=images_dir), name="images")

        # Serve index.html for root and SPA routes
        @app.get("/", response_class=HTMLResponse)
        async def serve_root():
            index_file = static_dir / "index.html"
            if not index_file.exists():
                return HTMLResponse(
                    content="<h1>UI Not Built</h1><p>Run 'cd app && npm run export' to build the UI.</p>",
                    status_code=404,
                )
            return FileResponse(index_file)

        # Handle other HTML routes for SPA
        @app.get("/{full_path:path}", response_class=HTMLResponse)
        async def serve_spa(full_path: str):
            # First try to serve the exact file
            file_path = static_dir / full_path
            if file_path.is_file():
                return FileResponse(file_path)

            # Check if it's an HTML page
            html_path = static_dir / f"{full_path}.html"
            if html_path.is_file():
                return FileResponse(html_path)

            # For SPA routes, return index.html
            # But exclude API routes
            if not full_path.startswith(("chat/", "health", "api/")):
                index_file = static_dir / "index.html"
                if index_file.exists():
                    return FileResponse(index_file)

            # Let other routes fall through to API handlers
            return HTMLResponse(status_code=404, content="Not found")

        return True

    elif ui_mode == "dev":
        # Show instructions to run dev server
        @app.get("/", response_class=HTMLResponse)
        async def serve_dev_instructions():
            return HTMLResponse(
                content="""
                <html>
                <head><title>DR-Agent - Dev Mode</title></head>
                <body style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
                    <h1>DR-Agent API Server Running</h1>
                    <p>The API server is running in <strong>development mode</strong>.</p>
                    <p>To use the UI, run the frontend development server separately:</p>
                    <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
cd app
npm install  # if not already done
npm run dev
                    </pre>
                    <p>Then access the UI at: <a href="http://localhost:3000">http://localhost:3000</a></p>
                    <hr>
                    <p><strong>API Endpoints:</strong></p>
                    <ul>
                        <li>Health Check: <a href="/health">/health</a></li>
                        <li>Chat Stream: POST /chat/stream</li>
                    </ul>
                </body>
                </html>
                """
            )

        return False

    elif ui_mode == "proxy":
        if not dev_url:
            raise ValueError("dev_url is required when ui_mode is 'proxy'")

        # In proxy mode, show a redirect page
        @app.get("/", response_class=HTMLResponse)
        async def serve_proxy_redirect():
            return HTMLResponse(
                content=f"""
                <html>
                <head>
                    <title>DR-Agent - Redirecting to Dev Server</title>
                    <meta http-equiv="refresh" content="0; url={dev_url}">
                </head>
                <body style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
                    <h1>Redirecting to Development Server...</h1>
                    <p>If you are not redirected, click <a href="{dev_url}">here</a>.</p>
                    <p>API server is running at this URL, but the UI is served from: <strong>{dev_url}</strong></p>
                </body>
                </html>
                """
            )

        return False

    else:  # auto mode but no UI available

        @app.get("/", response_class=HTMLResponse)
        async def serve_auto_fallback():
            return HTMLResponse(
                content="""
                <html>
                <head><title>DR-Agent - UI Not Available</title></head>
                <body style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
                    <h1>DR-Agent API Server Running</h1>
                    <p>The API server is running, but the pre-compiled UI is not available.</p>
                    
                    <h2>Option 1: Build the UI</h2>
                    <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
cd app
npm install
npm run export
                    </pre>
                    <p>Then restart this server.</p>
                    
                    <h2>Option 2: Run in Dev Mode</h2>
                    <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">
cd app
npm install
npm run dev
                    </pre>
                    <p>Then access the UI at: <a href="http://localhost:3000">http://localhost:3000</a></p>
                    
                    <hr>
                    <p><strong>API Endpoints:</strong></p>
                    <ul>
                        <li>Health Check: <a href="/health">/health</a></li>
                        <li>Chat Stream: POST /chat/stream</li>
                    </ul>
                </body>
                </html>
                """
            )

        return False


__all__ = ["mount_ui"]
