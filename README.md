<div align="center">
<img src="rl/open-instruct/assets/dr_tulu_logo.png" alt="DR Tulu" width="500"/>

# DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research


[**Paper**](https://allenai.org/papers/drtulu) • [**Data & Models**](https://huggingface.co/collections/rl-research/dr-tulu) • [**Blogpost**](http://allenai.org/blog/dr-tulu) • [**Video**](https://youtu.be/4i0W9qAf8K8)• [**Static Demo**](https://dr-tulu.github.io/) (Our live demo is coming soon - stay tuned!) 

</div>

DR Tulu-8B is the first open Deep Research (DR) model trained for long-form DR tasks. DR Tulu-8B matches OpenAI DR on long-form DR benchmarks.

<div align="center">
<img src="assets/rler_teaser.png" alt="DR Tulu Overview" width="800"/>
</div>

---

## Release Notes 
- November 19, 2025: Initial code release.
- November 25, 2025: We released our interactive CLI demo code, along with additional documentation for evaluation, training, and our new RL checkpoints.

## Overview

This repository contains three main components:

- **[`agent/`](agent/)**: Agent library (`dr-agent-lib`) with MCP-based tool backend, high-concurrency async request management, and flexible prompting interface for developing and training deep research agents. This directory also includes evaluation scripts for benchmarking DR agents.

- **[`rl/`](rl/open-instruct/)**: RL training code based on [Open-Instruct](https://github.com/allenai/open-instruct) for training deep research agents with GRPO and evolving rubrics.

- **[`sft/`](sft/llama-factory/)**: SFT training code based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning of deep research agents.

For detailed setup and usage instructions, see the README files in each subdirectory.

---

## Quick Start: Playing with DR Tulu Interactively

Try DR Tulu interactively with our CLI demo! This requires **1-2 GPUs** and takes just a few steps to set up:

1. **Setup Environment**
   ```bash
   cd agent/
   conda create -n dr_agent python=3.10 -y && conda activate dr_agent
   uv pip install -e .
   ```

2. **Configure API Keys** (get free keys from the respective services)
   ```bash
   export SERPER_API_KEY="your_key"  # https://serper.dev/
   export S2_API_KEY="your_key"      # https://api.semanticscholar.org/
   export JINA_API_KEY="your_key"    # https://jina.ai/reader/
   ```

3. **Launch Interactive Demo**
   ```bash
   uv run --extra vllm python scripts/launch_chat.py --model rl-research/DR-Tulu-8B
   ```

The demo will auto-launch required services (MCP server and vLLM) and start an interactive chat interface. You can now ask research questions and watch DR Tulu search and synthesize answers in real-time!

For more options and advanced usage, see [`agent/README.md`](agent/#interactive-chat).

---

## Running Evaluations

To benchmark DR Tulu on various tasks (HealthBench, Deep Research Bench, SimpleQA, etc.), you'll need to:

1. **Launch required servers on the same node** (requires 2 GPUs):
   ```bash
   # Launch VLLM servers
   CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960
   CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960
   
   # Launch MCP server
   python -m dr_agent.mcp_backend.main --port 8000
   ```

2. **Run evaluation script** on your desired benchmarks:
   ```bash
   cd agent/
   
   # Example: Run on all benchmarks
   for task in healthbench deep_research_bench research_qa genetic_diseases simpleqa 2wiki webwalker; do 
       python workflows/auto_search_sft.py \
           generate-dataset $task \
           --num-examples final_run \
           --max-concurrent 20 \
           --use-cache \
           --config workflows/auto_search_sft.yaml \
           --config-overrides "use_browse_agent=true,search_agent_max_tool_calls=10,browse_tool_name=jina" \
           --output eval_output/auto_search_sft/$task.jsonl
       
       python scripts/evaluate.py $task eval_output/auto_search_sft/$task.jsonl
   done
   ```

**Note**: SQA-CS-V2 and Deep Research Bench require additional conversion scripts for evaluation. See [`agent/evaluation/README.md`](agent/evaluation/) for detailed instructions.

For complete evaluation instructions, benchmark descriptions, and example scripts, see [`agent/evaluation/README.md`](agent/evaluation/).

---

## Training

### Supervised Fine-Tuning (SFT)

For supervised fine-tuning of deep research agents using high-quality demonstration data:

```bash
cd sft/llama-factory/
# See sft/llama-factory/README.md for detailed instructions
```

See [`sft/llama-factory/README.md`](sft/llama-factory/) for complete SFT training setup and configuration.

### Reinforcement Learning (RL)

For training deep research agents with GRPO and evolving rubrics:

```bash
cd rl/open-instruct/
# See rl/open-instruct/README.md for detailed instructions
```

See [`rl/open-instruct/README.md`](rl/open-instruct/) for complete RL training setup, including reward model training and policy optimization.

---

## Acknowledgments

DR Tulu is provided by The Allen Institute for Artificial Intelligence (Ai2). The code for this project is developed in collaboration with student researchers at the University of Washington, Carnegie Mellon University, and MIT.

---

## Citation and Contact

If you find our work useful, please cite:

```bibtex
@article{shao2025dr,
  title={DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research},
  author={Shao, Rulin and Asai, Akari and Shen, Shannon Zejiang and Ivison, Hamish and Kishore, Varsha and Zhuo, Jingming and Zhao, Xinran and Park, Molly and Finlayson, Samuel G and Sontag, David and others},
  journal={arXiv preprint arXiv:2511.19399},
  year={2025}
}
```
If you have any questions, you can contact [Rulin Shao](https://rulinshao.github.io/), [Akari Asai](https://akariasai.github.io/), [Shannon Shen](https://www.szj.io/), and [Hamish Ivison](https://ivison.id.au/) or open a github issue. 
