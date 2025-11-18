<img src="https://github.com/rlresearch/dr-tulu/blob/rl/rl/open-instruct/assets/dr_tulu_logo.png?raw=true" alt="Figure 1" width="500"/>

# DR Tulu - RL Training Code

This codebase is a modified version of the [Open-Instruct](https://github.com/allenai/open-instruct) codebase, used in the [DR Tulu](http://allenai-web/papers/drtulu) project to train deep research RL agents.

We defer the reader to the [Open-Instruct](https://github.com/allenai/open-instruct) README for more details on the codebase in general.
In particular, our branch is based on the following commit: [d075e4b](https://github.com/allenai/open-instruct/commit/d075e4bdcfc71b7713e789d2921016e0d84ee1fa).
Below we describe the basic setup steps and core training scripts. 

## Setup

First, clone the repository:
```bash
git clone https://github.com/rlresearch/dr-tulu.git
cd dr-tulu/rl
```

We use uv to manage the dependencies:
```bash
uv sync
```

And you should be done! Optionally, you can use our provided Dockerfile to build a Docker image:
```bash
docker build -t dr-tulu-rl .
```

## Training

The core training script is `open_instruct/grpo_fast.py`, and you can use the `train_dr_tulu.sh` script to train the model.
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
python open_instruct/grpo_fast.py \
    ...
```
