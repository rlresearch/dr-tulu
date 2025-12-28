"""
Utility functions for checking and launching services required by workflows.

This module provides functions to:
- Check if services are running on specific ports
- Launch MCP servers and vLLM servers in the background
- Extract port numbers from URLs
"""

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console


def check_port(port: int, timeout: float = 1.0) -> bool:
    """Check if a port is listening."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex(("localhost", port))
    sock.close()
    return result == 0


def extract_port_from_url(url_str: str) -> Optional[int]:
    """Extract port number from URL string."""
    # Remove the scheme (http:// or https://) to avoid false positives
    if "://" in url_str:
        url_str = url_str.split("://", 1)[1]

    url = url_str.rstrip("/")

    if ":" in url:
        port_str = url.split(":")[-1].split("/")[0]
        return int(port_str)
    return None


def launch_mcp_server(
    port: int = 8000, logger: Optional[logging.Logger] = None
) -> Optional[subprocess.Popen]:
    """Launch MCP server in background."""
    console = Console()

    if logger:
        logger.info(f"Launching MCP server on port {port}...")
    else:
        console.print(
            f"[cyan]🚀[/cyan] Launching MCP server on port [bold]{port}[/bold]..."
        )

    env = os.environ.copy()
    env["MCP_CACHE_DIR"] = (
        f".cache-{os.uname().nodename if hasattr(os, 'uname') else 'localhost'}"
    )

    log_file = Path(f"/tmp/mcp_server_{port}.log")
    if logger:
        logger.info(f"MCP server output will be logged to {log_file}")
    else:
        console.print(f"[dim]📋 MCP server output will be logged to {log_file}[/dim]")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            [sys.executable, "-m", "dr_agent.mcp_backend.main", "--port", str(port)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

    # Wait for server to start
    if logger:
        logger.info("Waiting for MCP server to start...")
    else:
        console.print("[yellow]⏳[/yellow] Waiting for MCP server to start...")

    for _ in range(20):
        time.sleep(0.5)
        if check_port(port):
            if logger:
                logger.info(f"MCP server started (PID: {process.pid})")
            else:
                console.print(
                    f"[green]✓[/green] MCP server started [dim](PID: {process.pid})[/dim]"
                )
            return process

    if process.poll() is None:
        if logger:
            logger.warning(
                "MCP server process started but port check failed. Continuing anyway..."
            )
        else:
            console.print(
                "[yellow]⚠[/yellow] MCP server process started but port check failed. Continuing anyway..."
            )
        return process
    else:
        if logger:
            logger.error(
                f"MCP server failed to start (exit code: {process.returncode}). Check logs: {log_file}"
            )
        else:
            console.print(
                f"[red]❌[/red] MCP server failed to start [dim](exit code: {process.returncode})[/dim]"
            )
            console.print(f"[dim]Check logs: {log_file}[/dim]")
        return None


def launch_vllm_server(
    model_name: str, port: int, gpu_id: int = 0, logger: Optional[logging.Logger] = None
) -> Optional[subprocess.Popen]:
    """Launch vLLM server in background."""
    console = Console()

    if logger:
        logger.info(f"Launching vLLM server for model {model_name} on port {port}...")
    else:
        console.print(
            f"[cyan]🚀[/cyan] Launching vLLM server for model [bold cyan]{model_name}[/bold cyan] on port [bold]{port}[/bold]..."
        )

    # Try to find vllm command
    import shutil

    vllm_base_cmd = None

    is_uv = (
        "uv" in sys.executable.lower()
        or os.environ.get("UV_PROJECT_ENVIRONMENT")
        or os.environ.get("VIRTUAL_ENV", "").endswith(".venv")
    )

    if shutil.which("vllm"):
        vllm_base_cmd = ["vllm", "serve"]
    elif is_uv and shutil.which("uv"):
        vllm_base_cmd = ["uv", "run", "vllm", "serve"]
    elif sys.executable:
        vllm_base_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

    if not vllm_base_cmd:
        if logger:
            logger.error(
                "vllm command not found. Install vllm with: uv pip install -e '.[vllm]' or uv pip install 'dr_agent[vllm]'"
            )
        else:
            console.print(
                "[red]❌[/red] Error: vllm command not found. Tried: vllm, uv run vllm, python -m vllm.entrypoints.openai.api_server"
            )
            console.print(
                "[blue]💡[/blue] Install vllm with: [dim]uv pip install -e '.[vllm]'[/dim] or [dim]uv pip install 'dr_agent[vllm]'[/dim]"
            )
        return None

    cmd = vllm_base_cmd + [
        model_name,
        "--port",
        str(port),
        "--dtype",
        "auto",
        "--max-model-len",
        "40960",
    ]

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = Path(f"/tmp/vllm_server_{port}.log")
    if logger:
        logger.info(f"vLLM output for {model_name} will be logged to {log_file}")
        logger.info(
            "Waiting for vLLM server to become ready (this may take a few minutes)..."
        )
    else:
        console.print(
            f"[dim]📋 vLLM output for {model_name} will be logged to {log_file}[/dim]"
        )
        console.print(
            "[yellow]⏳[/yellow] Waiting for vLLM server to become ready [dim](this may take a few minutes)...[/dim]"
        )

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

    start_time = time.time()
    while time.time() - start_time < 300:
        if check_port(port):
            if logger:
                logger.info(f"vLLM server started (PID: {process.pid})")
            else:
                console.print(
                    f"[green]✓[/green] vLLM server started [dim](PID: {process.pid})[/dim]"
                )
            return process

        if process.poll() is not None:
            if logger:
                logger.error(
                    f"vLLM server failed to start (exit code: {process.returncode}). Check logs: {log_file}"
                )
            else:
                console.print(
                    f"[red]❌[/red] vLLM server failed to start [dim](exit code: {process.returncode})[/dim]"
                )
                console.print(f"[dim]Check logs: {log_file}[/dim]")
            return None

        time.sleep(2)

        elapsed = int(time.time() - start_time)
        if elapsed > 0 and elapsed % 30 == 0:
            if logger:
                logger.info(f"Still waiting for vLLM server ({elapsed}s)...")
            else:
                console.print(
                    f"[yellow]⏳[/yellow] Still waiting for vLLM server [dim]({elapsed}s)...[/dim]"
                )

    if process.poll() is None:
        if logger:
            logger.warning(
                "vLLM server process started but port check timed out. It may still be initializing..."
            )
        else:
            console.print(
                "[yellow]⚠[/yellow] vLLM server process started but port check timed out. It may still be initializing..."
            )
        return process
    else:
        if logger:
            logger.error(
                f"vLLM server failed to start (exit code: {process.returncode})"
            )
        else:
            console.print(
                f"[red]❌[/red] vLLM server failed to start [dim](exit code: {process.returncode})[/dim]"
            )
        return None
