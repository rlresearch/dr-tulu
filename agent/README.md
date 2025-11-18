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

The interactive chat script runs the **exact same pipeline as `auto_search`** in an interactive format. It uses the `AutoReasonSearchWorkflow` with the same specialized agents, structured prompts, and two-stage process (SearchAgent → AnswerAgent).

### Quick Start (Recommended)

Use the self-contained launcher - it automatically handles everything:

```bash
# Simplest usage - just specify your model (auto-launches everything)
python scripts/launch_chat.py --model rl-research/DR-Tulu-8B

# That's it! The launcher will:
# - Auto-launch MCP server if needed
# - Auto-launch Search Agent vLLM server on GPU 0
# - Auto-launch Browse Agent vLLM server on GPU 1 (if enabled in config)
# - Use ports and models from config file automatically

# With dataset-specific instructions
python scripts/launch_chat.py --model rl-research/DR-Tulu-8B --dataset-name sqav2

# Skip service checks (if services are already running)
python scripts/launch_chat.py --skip-checks

# Don't auto-launch vLLM servers (just check if they're running)
python scripts/launch_chat.py --model rl-research/DR-Tulu-8B --no-auto-launch
```

**What the launcher does automatically:**
- ✅ **MCP Server**: Checks and launches if needed
- ✅ **Search Agent vLLM**: Reads config, checks if running, auto-launches on GPU 0 if needed
- ✅ **Browse Agent vLLM**: Reads config, checks if running, auto-launches on GPU 1 if needed (if `use_browse_agent=true`)
- ✅ **Cleanup**: Automatically stops all launched services when chat exits

**How it works:**
- Reads `search_agent_base_url`, `browse_agent_base_url`, and `use_browse_agent` from config file
- Uses `search_agent_model_name` and `browse_agent_model_name` from config (or override with `--model`)
- Automatically assigns GPUs: Search Agent → GPU 0, Browse Agent → GPU 1
- Extracts ports from base URLs automatically

### Manual Usage

If you prefer to manage services manually:

```bash
# Use the auto_search workflow interactively
python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml

# With dataset-specific instructions (e.g., for SQA v2)
python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml --dataset-name sqav2

# With a specific model (override search_agent_model_name and base_url)
python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml \
    --config-overrides "search_agent_model_name=rl-research/DR-Tulu-8B,search_agent_base_url=http://localhost:30001/v1"

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
