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

The interactive chat script runs the **exact same pipeline as `auto_search`** in an interactive format. It uses the `AutoReasonSearchWorkflow` with the same specialized agents, structured prompts, and two-stage process (SearchAgent → AnswerAgent).

### Quick Start (Recommended)

Use the self-contained launcher that automatically checks and launches required services:

```bash
# Basic usage (auto-launches MCP server if needed)
python scripts/launch_chat.py

# With a specific model
python scripts/launch_chat.py --model hosted_vllm/rl-research/DR-Tulu-8B --base-url http://localhost:30001/v1

# With dataset-specific instructions
python scripts/launch_chat.py --dataset-name sqav2

# Skip service checks (if services are already running)
python scripts/launch_chat.py --skip-checks
```

### Manual Usage

If you prefer to manage services manually:

```bash
# Use the auto_search workflow interactively
python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml

# With dataset-specific instructions (e.g., for SQA v2)
python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml --dataset-name sqav2

# With a specific model (override search_agent_model_name and base_url)
python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml \
    --config-overrides "search_agent_model_name=hosted_vllm/rl-research/DR-Tulu-8B,search_agent_base_url=http://localhost:30001/v1"

# With config overrides (multiple parameters)
python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml \
    --config-overrides "search_tool_name=s2,use_browse_agent=false,search_agent_model_name=Qwen/Qwen3-8B"

# Verbose mode to see tool calls and links
python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml --verbose
```

**Model Configuration:**

The workflow uses two models:
- **`search_agent_model_name`**: Model for the SearchAgent (searches and reasons with tools)
- **`browse_agent_model_name`**: Model for the BrowseAgent (if `use_browse_agent=true`)

You can override these via `--config-overrides`:
- `search_agent_model_name`: Model name (e.g., `"Qwen/Qwen3-8B"`, `"hosted_vllm/rl-research/DR-Tulu-8B"`)
- `search_agent_base_url`: Base URL for self-hosted models (e.g., `"http://localhost:30001/v1"`)
- `browse_agent_model_name`: Browse agent model name
- `browse_agent_base_url`: Browse agent base URL

**What it does:**
- Uses the same `AutoReasonSearchWorkflow` as `auto_search`
- Runs `SearchAgent` to search and reason with tools
- Runs `AnswerAgent` to generate final answer from search results
- Uses structured prompts from `UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS`
- Applies dataset-specific instructions
- Post-processes outputs (extracts `<answer>` tags, handles reasoning, etc.)
