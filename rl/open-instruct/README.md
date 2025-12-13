<img src="https://github.com/rlresearch/dr-tulu/blob/rl/rl/open-instruct/assets/dr_tulu_logo.png?raw=true" alt="Figure 1" width="500"/>

# DR Tulu - RL Training Code

This codebase is a modified version of the [Open-Instruct](https://github.com/allenai/open-instruct) codebase, used in the [DR Tulu](http://allenai-web/papers/drtulu) project to train deep research RL agents.

We defer the reader to the [Open-Instruct](https://github.com/allenai/open-instruct) README for more details on the codebase in general.
In particular, our branch is based on the following commit: [d075e4b](https://github.com/allenai/open-instruct/commit/d075e4bdcfc71b7713e789d2921016e0d84ee1fa).
Below we describe the basic setup steps and core training scripts. 

You can find the script we used to convert the RaR data to our format in `convert_rar_data.py`, and run it with:
```bash
uv run python convert_rar_data.py --hf_org <your hf username>
```

## Setup

First, clone the repository:
```bash
git clone https://github.com/rlresearch/dr-tulu.git
cd dr-tulu/rl/open-instruct
```

We use uv to manage the dependencies:
```bash
uv sync
```

Then, setup environment variables used by dr-agent-lib:
```bash
touch .env
echo "S2_API_KEY=xxx" >> .env
echo "SERPER_API_KEY=xxx" >> .env
echo "JINA_API_KEY=xxx >> .env
```
And you should be done!

Note that for original training, we used crawl4ai, which doesn't need the JINA_API_KEY, but may crawl directly from your IP (we used a proxy server during training, which we may share in future).

Optionally, you can use our provided Dockerfile to build a Docker image:
```bash
docker build -t dr-tulu-rl .
```

## Training

The core training script is `open_instruct/grpo_fast.py`, and you can use the `train_dr_tulu.sh` script to train the model.
**This script will not work out of the box, please read the following instructions and the script itself for more details.**
Note it requires 2 nodes with 8 GPUs each. Open-Instruct uses ray to manage the distributed training. To setup the ray cluster, follow steps like in the `configs/beaker_configs/ray_node_setup.sh` script:

First, setup a head node:
```bash
ray start --head --port=8888 --dashboard-host=0.0.0.0
```

Then, setup a worker node:
```bash
ray start --address="{HEAD_IP}:8888" --block --dashboard-host=0.0.0.0
```

Note that the head node and worker node should be on different machines, but reachable from each other. Finally, can then run run the training job (after setting the environment variables as in `train_dr_tulu.sh`):
```bash
export OPENAI_API_KEY=xxx
export WANDB_API_KEY=xxx

python open_instruct/grpo_fast.py \
    ...
```

### Test training

You can also test you have setup training correctly by running the `train_dr_tulu_mini_base.sh` script, which only requires 1 GPU, and trains Qwen/Qwen3-0.6B.

I don't know if this trains a good model, but at least it should let you test your training setup works before touching distributed training!

### MCP Tool debugging

Sometimes the MCP tool can have some issues, so you should check the MCP server logs to debug and check its running correctly. For example, sometimes the port doesn't bind.

If you see the training code logging successful tool calls, it should be fine!
Like this:
```bash
(ToolActor pid=500031) Using MCP tool:  google_search
(ToolActor pid=500031) Using MCP tool:  snippet_search
(ToolActor pid=500031) MCP snippet_search Tool Error: No results found for the query.
(ToolActor pid=500031) Returning error output anyway.
(ToolActor pid=500031) Using MCP tool:  google_search
```