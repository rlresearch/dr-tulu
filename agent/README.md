# `dr-agent-lib`

## Overview

`dr-agent-lib` is an agent library for training and developing deep research agents. It supports:
- **MCP-Based Tool Backend**: Unified interface for web search and browsing tools
- **High Concurrency**: Global caching and async request management for RL training at scale
- **Flexible Prompting Interface**: Easy composition of search workflows with fine-grained control

## Setup 

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
```
Note you will need to get these API keys from the respective services.
- S2_API_KEY: https://api.semanticscholar.org/
- SERPER_API_KEY: https://serper.dev/

## Getting started 

1. Launch MCP Server 

    ```bash
    MCP_CACHE_DIR=".cache-$(hostname)" python -m dr_agent.mcp_backend.main --port 8000
    ```

2. Start the VLLM Server 

    ```bash 
    CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30002 --max-model-len 40960
    
    CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960
    ```

3. Run generation script 

    ```bash
    bash scripts/auto_search.sh
    ```

## Interactive Chat

We provide an interactive cli demo for the auto_search workflow.
Requires 1-2 GPUs. We recommend running with `uv`:

```bash
uv run --extra vllm  python scripts/launch_chat.py --model rl-research/DR-Tulu-8B