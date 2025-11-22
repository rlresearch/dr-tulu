# `dr-agent-lib`

## Overview

`dr-agent-lib` is an agent library for training and developing deep research agents. It supports:
- **MCP-Based Tool Backend**: Unified interface for web search and browsing tools
- **High Concurrency**: Global caching and async request management for RL training at scale
- **Flexible Prompting Interface**: Easy composition of search workflows with fine-grained control

## Setup 

Below we assume you are already in the `agent` directory. 

```bash
conda create -n dr_agent python=3.10 -y && conda activate dr_agent

uv pip install -e .     # Install dev version
uv pip install dr_agent # Install from pypi 
```

If you run crawl4ai locally, you will need to install playwright and its dependencies.

Set up API keys via `.env` file:
```bash
S2_API_KEY=xxx
SERPER_API_KEY=xxx
JINA_API_KEY=xxx
```
Note you will need to get these API keys from the respective services.
- S2_API_KEY: https://api.semanticscholar.org/
- SERPER_API_KEY: https://serper.dev/
- JINA_API_KEY: https://jina.ai/reader/

## Getting started 

1. Launch MCP Server 

    ```bash
    MCP_CACHE_DIR=".cache-$(hostname)" python -m dr_agent.mcp_backend.main --port 8000
    ```

2. Using DR-Tulu Models 

    - Start the VLLM Server 

       ```bash 
       CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30002 --max-model-len 40960
       
       CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30003 --max-model-len 40960
       ```

    - Run generation script 

       ```bash
       bash scripts/auto_search.sh
       ```

3. Using OAI models 
    ```bash
    export OPENAI_API_KEY="XXXX"
    bash scripts/auto_search-oai.sh
    ```

## Interactive Chat

We provide an interactive cli demo for the auto_search workflow.
Requires 1-2 GPUs. We recommend running with `uv`, which should install everything you need and then launch the tool, but set your API keys first:

```bash
export SERPER_API_KEY="XXXX"
export S2_API_KEY="XXXX"
export JINA_API_KEY="XXXX"

uv run --extra vllm  python scripts/launch_chat.py --model rl-research/DR-Tulu-8B
```

Note for this cli demo, we use a slightly different prompt than the one used for evaluation in our paper, for demo purposes. The prompt is in the file `dr_agent/shared_prompts/unified_tool_calling_cli.yaml`.


We provide additional flags for the chat script, for e.g. showing full tool output:
```bash
usage: launch_chat.py [-h] [--config CONFIG] [--dataset-name DATASET_NAME]
                      [--model MODEL] [--config-overrides CONFIG_OVERRIDES]
                      [--verbose] [--show-full-tool-output] [--skip-checks]
                      [--mcp-port MCP_PORT] [--gpu-id GPU_ID]
                      [--no-auto-launch]

Self-contained launcher for interactive chat

options:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Config file path (default:
                        workflows/auto_search_sft.yaml)
  --dataset-name DATASET_NAME, -d DATASET_NAME
                        Dataset name for dataset-specific instructions
  --model MODEL, -m MODEL
                        Main model name (for search agent). If not provided,
                        uses config defaults.
  --config-overrides CONFIG_OVERRIDES
                        Config overrides (e.g., 'param1=value1,param2=value2')
  --verbose, -v         Enable verbose output
  --show-full-tool-output
                        Show full tool output instead of truncating to 500
                        characters
  --skip-checks         Skip checking/launching services
  --mcp-port MCP_PORT   MCP server port (default: 8000)
  --gpu-id GPU_ID       GPU ID for search agent vLLM server (default: 0,
                        browse agent uses GPU 1)
  --no-auto-launch      Don't automatically launch vLLM servers (check only)

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
```