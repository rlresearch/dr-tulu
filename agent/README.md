# `dr-agent-lib`

`dr-agent-lib` is an agent library for training and developing deep research agents. It supports:
- **MCP-Based Tool Backend**: Unified interface for web search and browsing tools
- **High Concurrency**: Global caching and async request management for RL training at scale
- **Flexible Prompting Interface**: Easy composition of search workflows with fine-grained control

## Quick start 

Below we assume you are already in the `agent` directory. 

1. Installation 
    ```bash
    # [Optional] create an environment 
    # conda create -n dr_agent python=3.10 -y && conda activate dr_agent

    # Install the latest dev version of dr_agent library 
    uv pip install -e ".[ui]"    
    
    # Or you can directly install from pypi 
    # uv pip install dr_agent 
    ```

2. Setting up the environment variables for the backend service: `cp .env.example .env` and edit the following environment variables 

    ```bash
    SERPER_API_KEY=xxx # [Must have] Needed for serper search 
    S2_API_KEY=xxx     # Needed for snippet search 
    JINA_API_KEY=xxx   # Needed for jina-based browsing tool 
    ```
    You can get these API keys from the respective services.
    - `SERPER_API_KEY`: https://serper.dev/
    - `S2_API_KEY`: https://api.semanticscholar.org/
    - `JINA_API_KEY`: https://jina.ai/reader/

3. Launch the `autosearch` workflow: 
    - For simple debugging purpose, you can directly launch the service without any GPUs using openai models (this requires setting the `OPENAI_API_KEY`)
      ```bash
      export OPENAI_API_KEY="sk-xxx"

      python workflows/auto_search_sft.py serve --port 8080 --config workflows/auto_search_sft-oai.yaml
      ```
      It will load a web server where you can directly chat with the agent in the interface. 

    - [1~2 GPUs needed] For the workflow, you can also use it with `rl-research/DR-Tulu-8B` by serving the models yourself with VLLM:
      ```bash 
      # You need to install vllm though 
      # uv pip install vllm 

      python workflows/auto_search_sft.py serve --port 8080 
      ``` 
      The command prompts you whether to launch the needed VLLM backend, e.g.,  
      ```
      ⚠  Search agent vLLM server is not running on port 30001
      Launch vLLM server for rl-research/DR-Tulu-8B on port 30001? [y/n]: y
      ```
      Or you can launch the models separately yourself. 
      ```bash 
      CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --port 30001 --dtype auto --max-model-len 40960
      ```


> [!NOTE]
> If you run crawl4ai locally, you will need to install playwright and its dependencies.


## Deep Dive into the `serve` command: 

The `serve` command can turn any workflow into a fastapi service with the following endpoints 

- **`/chat`**: Simple request-response endpoint, returns the complete response as JSON
- **`/chat/stream`**: SSE streaming endpoint, streams thinking, tool calls, and answers in real-time

```bash 
# Launch the server (assumes MCP and model servers are running)
python workflows/auto_search_sft.py serve --port 8080

# Example: simple chat endpoint
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "What is the capital of France?"}'

# Example: streaming endpoint (SSE)
curl -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "What is the capital of France?"}'
```

### Protect the UI with the passwords

```bash
# You can generate a random password with 
# openssl rand -base64 32
python workflows/auto_search_sft.py serve --port 8080 --password "your-secure-password"
```

### Develop the chat frontend

```bash 
cd ../app && npm install && npm run dev 

# Change the ui url accordingly 
python workflows/auto_search_sft.py serve --port 8080 --ui-mode proxy --dev-url http://localhost:3000 
```

## Running evaluation on the datasets

We also provide a command to generate the results for examples in a given dataset: 

```bash
python workflows/auto_search_sft.py generate-dataset healthbench \
    --num-examples 5 \
    --max-concurrent 5 
    --output healthbench.jsonl
```

We include evaluation scripts for multiple benchmarks, including:
- **Long-form**: SQA-CS-V2, Deep Research Bench, ResearchQA, HealthBench, Genetic Diseases  
- **Short-form**: BrowseComp, SimpleQA, 2Wiki, etc.

For detailed evaluation instructions, benchmark descriptions, and usage examples, see [`evaluation/README.md`](evaluation/README.md). 

## [experimental] Interactive chat with CLI 

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