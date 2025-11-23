<div align="center">
<img src="rl/open-instruct/assets/dr_tulu_logo.png" alt="DR Tulu" width="500"/>

# DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research


[**Paper**](https://allenai.org/papers/drtulu) • [**Data & Models**](https://huggingface.co/collections/rl-research/dr-tulu) • [**Blogpost**](http://allenai.org/blog/dr-tulu) • [**Video**](https://youtu.be/4i0W9qAf8K8)• [**Static Demo**](https://dr-tulu.github.io/) (Our live demo is coming soon - stay tuned!) 

</div>

DR Tulu-8B is the first open Deep Research (DR) model trained for long-form DR tasks. DR Tulu-8B matches OpenAI DR on long-form DR benchmarks.

<div align="center">
<img src="assets/dr-tulu.png" alt="DR Tulu Overview" width="800"/>
</div>

---

## Release Notes 
- November 19, 2025: Initial code release.
  
## We are working hard on cleaning the code and adding more instructions. Thanks for your patience!

## Overview

This repository contains three main components:

- **[`agent/`](agent/)**: Agent library (`dr-agent-lib`) with MCP-based tool backend, high-concurrency async request management, and flexible prompting interface for developing and training deep research agents.

- **[`rl/`](rl/open-instruct/)**: RL training code based on [Open-Instruct](https://github.com/allenai/open-instruct) for training deep research agents with GRPO and evolving rubrics.

- **[`sft/`](sft/llama-factory/)**: SFT training code based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning of deep research agents.

For detailed setup and usage instructions, see the README files in each subdirectory.

---

## Acknowledgments

DR Tulu is provided by The Allen Institute for Artificial Intelligence (Ai2). The code for this project is developed in collaboration with student researchers at the University of Washington, Carnegie Mellon University, and MIT.

---

## Citation and Contact

If you find our work useful, please cite:

```bibtex
@misc{shao2025drtulu,
  title        = {DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research},
  author       = {Shao, Rulin and Asai, Akari and Shen, Shannon Zejiang and Ivison, Hamish
                  and Kishore, Varsha and Zhuo, Jingming and Zhao, Xinran and Park, Molly
                  and Finlayson, Samuel and Sontag, David and Murray, Tyler and Min, Sewon
                  and Dasigi, Pradeep and Soldaini, Luca and Brahman, Faeze and Yih, Wen-tau
                  and Wu, Tongshuang and Zettlemoyer, Luke and Kim, Yoon
                  and Hajishirzi, Hannaneh and Koh, Pang Wei},
  year         = {2025},
  note         = {Preprint},
}
```
If you have any questions, you can contact [Rulin Shao](https://rulinshao.github.io/), [Akari Asai](https://akariasai.github.io/), [Shannon Shen](https://www.szj.io/), and [Hamish Ivison](https://ivison.id.au/) or open a github issue. 
