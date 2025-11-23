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
    
    # Launch MCP server - log to file
    log_file = Path(f"/tmp/mcp_server_{port}.log")
    try:
        print(f"📋 MCP server output will be logged to {log_file}")
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                [sys.executable, "-m", "dr_agent.mcp_backend.main", "--port", str(port)],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            )
        
        # Wait for server to start
        print("⏳ Waiting for MCP server to start...")
        for _ in range(20):  # Wait up to 10 seconds
            time.sleep(0.5)
            if check_port(port):
                print(f"✓ MCP server started (PID: {process.pid})")
                return process
        
        # Check if process is still running
        if process.poll() is None:
            print(f"⚠ MCP server process started but port check failed. Continuing anyway...")
            return process
        else:
            print(f"❌ MCP server failed to start (exit code: {process.returncode})")
            print(f"Check logs: {log_file}")
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


def launch_vllm_server(model_name: str, port: int, gpu_id: int = 0) -> Optional[subprocess.Popen]:
    """Launch vLLM server in background."""
    print(f"🚀 Launching vLLM server for model {model_name} on port {port}...")
    
    # Try to find vllm command - check multiple options
    import shutil
    vllm_base_cmd = None
    
    # Check if we're running through uv (check parent process or environment)
    is_uv = (
        "uv" in sys.executable.lower() or 
        os.environ.get("UV_PROJECT_ENVIRONMENT") or
        os.environ.get("VIRTUAL_ENV", "").endswith(".venv")
    )
    
    # Try different ways to invoke vllm (in order of preference)
    # 1. Try direct vllm command first (most reliable)
    if shutil.which("vllm"):
        vllm_base_cmd = ["vllm", "serve"]
    # 2. Try uv run vllm if in uv environment
    elif is_uv and shutil.which("uv"):
        vllm_base_cmd = ["uv", "run", "vllm", "serve"]
    # 3. Try python -m vllm.entrypoints.openai.api_server (module syntax)
    elif sys.executable:
        vllm_base_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    
    if not vllm_base_cmd:
        print("❌ Error: vllm command not found. Tried: vllm, uv run vllm, python -m vllm.entrypoints.openai.api_server")
        print("💡 Install vllm with: uv pip install -e '.[vllm]' or uv pip install 'dr_agent[vllm]'")
        print("💡 Or launch the server manually:")
        if is_uv and shutil.which("uv"):
            print(f"   CUDA_VISIBLE_DEVICES={gpu_id} uv run vllm serve {model_name} --port {port} --dtype auto --max-model-len 40960")
        else:
            print(f"   CUDA_VISIBLE_DEVICES={gpu_id} vllm serve {model_name} --port {port} --dtype auto --max-model-len 40960")
        return None
    
    # Build vLLM command
    # Note: vllm_base_cmd already includes 'serve' or equivalent
    cmd = vllm_base_cmd + [
        model_name,
        "--port", str(port),
        "--dtype", "auto",
        "--max-model-len", "40960"
    ]
    
    # Set CUDA_VISIBLE_DEVICES if specified
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    log_file = Path(f"/tmp/vllm_server_{port}.log")
    try:
        print(f"📋 vLLM output for {model_name} will be logged to {log_file}")
        print("⏳ Waiting for vLLM server to become ready (this may take a few minutes)...")
        
        # Launch with output redirected to file
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            )
        
        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < 300:  # Wait up to 5 minutes
            if check_port(port):
                print(f"✓ vLLM server started (PID: {process.pid})")
                return process
            
            # Check if process died
            if process.poll() is not None:
                print(f"❌ vLLM server failed to start (exit code: {process.returncode})")
                print(f"Check logs: {log_file}")
                # Print last few lines of log
                try:
                    with open(log_file, "r") as f:
                        print("Last 10 lines of log:")
                        print("".join(f.readlines()[-10:]))
                except:
                    pass
                return None
            
            time.sleep(2)
            
            # Print status update every 30 seconds
            elapsed = int(time.time() - start_time)
            if elapsed > 0 and elapsed % 30 == 0:
                print(f"⏳ Still waiting for vLLM server ({elapsed}s)...")
        
        # Check if process is still running
        if process.poll() is None:
            print(f"⚠ vLLM server process started but port check timed out. It may still be initializing...")
            return process
        else:
            print(f"❌ vLLM server failed to start (exit code: {process.returncode})")
            return None
            
    except Exception as e:
        print(f"❌ Failed to launch vLLM server: {e}")
        return None


def _extract_port_from_url(url_str: str) -> Optional[int]:
    """Extract port number from URL string."""
    if "://" in url_str:
        url = url_str.rstrip("/")
    else:
        url = f"http://{url_str}".rstrip("/")
    
    if ":" in url:
        port_str = url.split(":")[-1].split("/")[0]
        try:
            return int(port_str)
        except ValueError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Self-contained launcher for interactive chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-launches MCP server and vLLM servers if needed)
  python scripts/launch_chat.py

  # With specific model (auto-launches both vLLM servers on GPUs 0 and 1)
  python scripts/launch_chat.py --model rl-research/DR-Tulu-8B

  # Skip service checks (if services are already running)
  python scripts/launch_chat.py --skip-checks

  # Don't auto-launch vLLM servers (just check)
  python scripts/launch_chat.py --no-auto-launch

  # Custom config file
  python scripts/launch_chat.py --config workflows/auto_search_sft.yaml
        """.strip()
    )
    
    parser.add_argument(
        "--config", "-c",
        default="workflows/auto_search_sft.yaml",
        help="Config file path (default: workflows/auto_search_sft.yaml)"
    )
    parser.add_argument(
        "--dataset-name", "-d",
        help="Dataset name for dataset-specific instructions"
    )
    parser.add_argument(
        "--model", "-m",
        help="Main model name (for search agent). If not provided, uses config defaults."
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
        "--show-full-tool-output",
        action="store_true",
        help="Show full tool output instead of truncating to 500 chars"
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
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID for search agent vLLM server (default: 0, browse agent uses GPU 1)"
    )
    parser.add_argument(
        "--no-auto-launch",
        action="store_true",
        help="Don't automatically launch vLLM servers (check only)"
    )
    
    args = parser.parse_args()
    
    print("=== Interactive Chat Launcher ===\n")
    
    # Resolve config path relative to script directory (agent/scripts/)
    script_dir = Path(__file__).parent
    agent_dir = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to agent directory first
        resolved_config_path = agent_dir / config_path
        # If that doesn't exist, try relative to current directory
        if not resolved_config_path.exists():
            resolved_config_path = config_path
    else:
        resolved_config_path = config_path
    
    processes = {"mcp": None, "vllm_search": None, "vllm_browse": None}
    
    # Cleanup function
    def cleanup():
        for server_type in ["vllm_browse", "vllm_search"]:
            if processes[server_type] and processes[server_type].poll() is None:
                server_name = "Browse" if "browse" in server_type else "Search"
                print(f"\n🧹 Stopping {server_name} vLLM server...")
                try:
                    if hasattr(os, 'setsid'):
                        os.killpg(os.getpgid(processes[server_type].pid), signal.SIGTERM)
                    else:
                        processes[server_type].terminate()
                    processes[server_type].wait(timeout=10)
                except Exception:
                    try:
                        if hasattr(os, 'setsid'):
                            os.killpg(os.getpgid(processes[server_type].pid), signal.SIGKILL)
                        else:
                            processes[server_type].kill()
                    except Exception:
                        pass
        
        if processes["mcp"] and processes["mcp"].poll() is None:
            print("\n🧹 Stopping MCP server...")
            try:
                if hasattr(os, 'setsid'):
                    os.killpg(os.getpgid(processes["mcp"].pid), signal.SIGTERM)
                else:
                    processes["mcp"].terminate()
                processes["mcp"].wait(timeout=5)
            except Exception:
                try:
                    if hasattr(os, 'setsid'):
                        os.killpg(os.getpgid(processes["mcp"].pid), signal.SIGKILL)
                    else:
                        processes["mcp"].kill()
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
                processes["mcp"] = launch_mcp_server(args.mcp_port)
                if not processes["mcp"]:
                    print("❌ Failed to start MCP server. Exiting.")
                    sys.exit(1)
            else:
                print("❌ MCP server is required. Exiting.")
                sys.exit(1)
        
        # Check and optionally launch vLLM servers
        # Read config file to get all settings
        try:
            import yaml
            with open(resolved_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠ Warning: Could not read config file {resolved_config_path}: {e}")
            config_data = {}
        
        # Get settings from config, override with command line args if provided
        search_model = args.model or config_data.get('search_agent_model_name')
        search_base_url = config_data.get('search_agent_base_url')
        browse_model = config_data.get('browse_agent_model_name')
        browse_base_url = config_data.get('browse_agent_base_url') if config_data.get('use_browse_agent') else None
        
        # Determine if we need to launch servers
        need_search_server = search_base_url and not check_vllm_server(search_base_url)
        need_browse_server = browse_base_url and not check_vllm_server(browse_base_url)
        
        # Auto-launch servers if needed (unless --no-auto-launch is set)
        if not args.no_auto_launch:
            if need_search_server and search_model and search_base_url:
                port = _extract_port_from_url(search_base_url)
                if port:
                    print(f"🚀 Auto-launching Search Agent vLLM server: {search_model} on port {port} (GPU {args.gpu_id})...")
                    processes["vllm_search"] = launch_vllm_server(search_model, port, args.gpu_id)
                    if not processes["vllm_search"]:
                        print("⚠ Failed to start Search Agent vLLM server.")
            
            if need_browse_server and browse_model and browse_base_url:
                port = _extract_port_from_url(browse_base_url)
                if port:
                    # Use GPU 1 for browse agent if search agent uses GPU 0, otherwise same GPU
                    browse_gpu = args.gpu_id + 1 if args.gpu_id == 0 else args.gpu_id
                    print(f"🚀 Auto-launching Browse Agent vLLM server: {browse_model} on port {port} (GPU {browse_gpu})...")
                    processes["vllm_browse"] = launch_vllm_server(browse_model, port, browse_gpu)
                    if not processes["vllm_browse"]:
                        print("⚠ Failed to start Browse Agent vLLM server.")
        else:
            # Just check and warn
            if need_search_server and search_model and search_base_url:
                port = _extract_port_from_url(search_base_url)
                if port:
                    print(f"💡 Search Agent vLLM server not running. Launch manually:")
                    print(f"   vllm serve {search_model} --port {port} --dtype auto --max-model-len 40960")
            
            if need_browse_server and browse_model and browse_base_url:
                port = _extract_port_from_url(browse_base_url)
                if port:
                    browse_gpu = args.gpu_id + 1 if args.gpu_id == 0 else args.gpu_id
                    print(f"💡 Browse Agent vLLM server not running. Launch manually:")
                    print(f"   CUDA_VISIBLE_DEVICES={browse_gpu} vllm serve {browse_model} --port {port} --dtype auto --max-model-len 40960")
    
    # Build command for interactive_auto_search.py
    cmd = [sys.executable, "scripts/interactive_auto_search.py", "--config", str(resolved_config_path)]
    
    if args.dataset_name:
        cmd.extend(["--dataset-name", args.dataset_name])
    
    if args.verbose:
        cmd.append("--verbose")
    
    if args.show_full_tool_output:
        cmd.append("--show-full-tool-output")

    # Build config overrides
    overrides = []
    if args.config_overrides:
        overrides.append(args.config_overrides)
    
    # Add search agent model override if provided
    if args.model:
        overrides.append(f"search_agent_model_name={args.model}")

    # Always propagate MCP port to workflow configuration
    overrides.append(f"mcp_port={args.mcp_port}")
    
    if overrides:
        cmd.extend(["--config-overrides", ",".join(overrides)])
    
    print("\n🚀 Starting interactive chat...\n")
    
    # Prepare environment for the chat subprocess
    run_env = os.environ.copy()
    run_env["MCP_TRANSPORT_PORT"] = str(args.mcp_port)

    # Run the chat script
    try:
        subprocess.run(cmd, check=True, env=run_env)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()

