<div align="center">
<img src="rl/open-instruct/assets/dr_tulu_logo.png" alt="DR Tulu" width="500"/>

# DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research


[**Paper**](http://allenai-web/papers/drtulu) • [**Data & Models**](https://huggingface.co/collections/rl-research/dr-tulu) • [**Blogpost**](http://allenai.org/blog/dr-tulu)

</div>

DR Tulu-8B is the first open Deep Research (DR) model trained for long-form DR tasks. DR Tulu-8B matches OpenAI DR on long-form DR benchmarks.

---

<div align="center">
<img src="assets/dr-tulu.png" alt="DR Tulu Overview" width="800"/>
</div>

## Overview

This repository contains two main components:

- **[`agent/`](agent/)**: Agent library (`dr-agent-lib`) with MCP-based tool backend, high-concurrency async request management, and flexible prompting interface for developing and training deep research agents.

- **[`rl/`](rl/)**: RL training code based on [Open-Instruct](https://github.com/allenai/open-instruct) for training deep research agents with GRPO and evolving rubrics.

For detailed setup and usage instructions, see the README files in each subdirectory.

---

## Acknowledgments

DR Tulu is provided by The Allen Institute for Artificial Intelligence (Ai2). The code for this project is developed in collaboration with student researchers at the University of Washington, Carnegie Mellon University, and MIT.

---

## Citation

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
