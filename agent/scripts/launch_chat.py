#!/usr/bin/env python3
"""
Self-contained launcher for interactive chat.

This script automatically checks and launches required services (MCP server, vLLM) 
before starting the interactive chat, making it easy to use without manual setup.
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def check_port(port: int, timeout: float = 1.0) -> bool:
    """Check if a port is listening."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_mcp_server(port: int = 8000) -> bool:
    """Check if MCP server is running."""
    if check_port(port):
        print(f"✓ MCP server is running on port {port}")
        return True
    else:
        print(f"⚠ MCP server is not running on port {port}")
        return False


def launch_mcp_server(port: int = 8000) -> Optional[subprocess.Popen]:
    """Launch MCP server in background."""
    print(f"🚀 Launching MCP server on port {port}...")
    
    # Check if we're in the right directory
    script_dir = Path(__file__).parent.parent
    if not (script_dir / "dr_agent" / "mcp_backend" / "main.py").exists():
        print("❌ Error: Cannot find dr_agent.mcp_backend.main. Please run from project root.")
        return None
    
    # Set up environment
    env = os.environ.copy()
    env["MCP_CACHE_DIR"] = f".cache-{os.uname().nodename if hasattr(os, 'uname') else 'localhost'}"
    
    # Launch MCP server
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "dr_agent.mcp_backend.main", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
        )
        
        # Wait for server to start
        for _ in range(10):  # Wait up to 5 seconds
            time.sleep(0.5)
            if check_port(port):
                print(f"✓ MCP server started (PID: {process.pid})")
                return process
        
        # Check if process is still running
        if process.poll() is None:
            print(f"⚠ MCP server process started but port check failed. Continuing anyway...")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ MCP server failed to start (exit code: {process.returncode})")
            if stderr:
                print(f"Error: {stderr.decode()[:500]}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to launch MCP server: {e}")
        return None


def check_vllm_server(base_url: str) -> bool:
    """Check if vLLM server is accessible."""
    if not base_url:
        return True
    
    try:
        # Extract port from URL
        if "://" in base_url:
            url = base_url.rstrip("/")
        else:
            url = f"http://{base_url}".rstrip("/")
        
        # Try to connect to health endpoint or just check port
        if HAS_REQUESTS:
            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"✓ vLLM server is accessible at {url}")
                    return True
            except:
                pass
        
        # Fallback: just check if port is open
        if ":" in url:
            port_str = url.split(":")[-1].split("/")[0]
            try:
                port = int(port_str)
                if check_port(port):
                    print(f"✓ vLLM server appears to be running on port {port}")
                    return True
            except ValueError:
                pass
        
        print(f"⚠ vLLM server does not appear to be accessible at {base_url}")
        return False
        
    except Exception as e:
        print(f"⚠ Could not check vLLM server: {e}")
        return True  # Don't fail if we can't check


def main():
    parser = argparse.ArgumentParser(
        description="Self-contained launcher for interactive chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-launches MCP server if needed)
  python scripts/launch_chat.py

  # With specific model
  python scripts/launch_chat.py --model hosted_vllm/rl-research/DR-Tulu-8B --base-url http://localhost:30001/v1

  # Skip service checks
  python scripts/launch_chat.py --skip-checks

  # Custom config file
  python scripts/launch_chat.py --config workflows/trained/auto_search_sft.yaml
        """.strip()
    )
    
    parser.add_argument(
        "--config", "-c",
        default="workflows/trained/auto_search_sft.yaml",
        help="Config file path (default: workflows/trained/auto_search_sft.yaml)"
    )
    parser.add_argument(
        "--dataset-name", "-d",
        help="Dataset name for dataset-specific instructions"
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name to use"
    )
    parser.add_argument(
        "--base-url", "-u",
        help="Base URL for self-hosted models"
    )
    parser.add_argument(
        "--config-overrides",
        help="Config overrides (e.g., 'param1=value1,param2=value2')"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip checking/launching services"
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8000,
        help="MCP server port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    print("=== Interactive Chat Launcher ===\n")
    
    mcp_process = None
    
    # Cleanup function
    def cleanup():
        if mcp_process and mcp_process.poll() is None:
            print("\n🧹 Stopping MCP server...")
            try:
                if hasattr(os, 'setsid'):
                    os.killpg(os.getpgid(mcp_process.pid), signal.SIGTERM)
                else:
                    mcp_process.terminate()
                mcp_process.wait(timeout=5)
            except Exception:
                try:
                    if hasattr(os, 'setsid'):
                        os.killpg(os.getpgid(mcp_process.pid), signal.SIGKILL)
                    else:
                        mcp_process.kill()
                except Exception:
                    pass
    
    # Register cleanup handlers
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), sys.exit(0)))
    
    # Check and launch services
    if not args.skip_checks:
        # Check MCP server
        if not check_mcp_server(args.mcp_port):
            response = input("Launch MCP server? (y/n): ").strip().lower()
            if response == 'y':
                mcp_process = launch_mcp_server(args.mcp_port)
                if not mcp_process:
                    print("❌ Failed to start MCP server. Exiting.")
                    sys.exit(1)
            else:
                print("❌ MCP server is required. Exiting.")
                sys.exit(1)
        
        # Check vLLM server if base_url is provided
        if args.base_url:
            check_vllm_server(args.base_url)
    
    # Build command for interactive_auto_search.py
    cmd = [sys.executable, "scripts/interactive_auto_search.py", "--config", args.config]
    
    if args.dataset_name:
        cmd.extend(["--dataset-name", args.dataset_name])
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Build config overrides
    overrides = []
    if args.config_overrides:
        overrides.append(args.config_overrides)
    
    if args.model or args.base_url:
        override_parts = []
        if args.model:
            override_parts.append(f"search_agent_model_name={args.model}")
        if args.base_url:
            override_parts.append(f"search_agent_base_url={args.base_url}")
        if override_parts:
            overrides.append(",".join(override_parts))
    
    if overrides:
        cmd.extend(["--config-overrides", ",".join(overrides)])
    
    print("\n🚀 Starting interactive chat...\n")
    
    # Run the chat script
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()

