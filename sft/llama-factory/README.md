<img src="../../rl/open-instruct/assets/dr_tulu_logo.png" alt="Figure 1" width="500"/>

# DR Tulu - SFT Training Code

This codebase is a modified version of the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) codebase, used in the [DR Tulu](http://allenai-web/papers/drtulu) project to perform supervised fine-tuning (SFT) of deep research agents.

We defer the reader to the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) README for more details on the codebase in general.
Below we describe the basic setup steps and core training scripts.

## Setup

First, clone the repository:
```bash
git clone https://github.com/rlresearch/dr-tulu.git
cd dr-tulu/sft/llama-factory
```

Install the dependencies:
```bash
uv pip install -e ".[torch,metrics]" --no-build-isolation
uv pip install wandb deepspeed==0.15.4
```

## Training

Training configuration files can be found in the `train/` directory. To run training:

```bash
bash train/train.sh train/qwen3-8B-sft-final.yaml
```