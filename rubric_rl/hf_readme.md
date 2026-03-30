---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - rubric
  - evaluation
  - rl
  - deep-research
---

# Rubric RL Evaluation Results

Evaluation data for rubric-based reward modeling experiments. Contains generated rubrics from multiple rubric generators and pairwise scoring results comparing `rl-research/DR-Tulu-8B` (RL, step_4000) vs `rl-research/DR-Tulu-SFT-8B`.

## Data Structure

### `rubrics/` — Generated evaluation rubrics

Each JSONL file contains per-question rubrics with fields: `prompt_id`, `question`, `generated_rubric`, `generated_rubric_raw`, `rubric_style`, `rubric_model`.

| File pattern | Rubric Generator | Prompt | Benchmark |
| --- | --- | --- | --- |
| `v0_*.jsonl` | `stellalisy/rubric_generator_v0_0302` | standard | HealthBench / DRB |
| `rubric_gen_8b_v3_*_standard.jsonl` | `stellalisy/rubric-generator-8b-v3` | standard | HealthBench / DRB |
| `rubric_gen_8b_v3_*_v3prompt.jsonl` | `stellalisy/rubric-generator-8b-v3` | v3 (dealbreaker) | HealthBench / DRB |
| `qwen3_8b_*_standard.jsonl` | `Qwen/Qwen3-8B` | standard | HealthBench / DRB |
| `qwen3_8b_*_v3prompt.jsonl` | `Qwen/Qwen3-8B` | v3 (dealbreaker) | HealthBench / DRB |
| `gpt41_*_standard.jsonl` | `GPT-4.1` | standard | HealthBench / DRB |
| `gpt41_*_v3prompt.jsonl` | `GPT-4.1` | v3 (dealbreaker) | HealthBench / DRB |

### `scores/` — Pairwise scoring results

Each JSON file contains `summary` (aggregate metrics) and `per_example` (per-question RL vs SFT scores). All scored by frozen `Qwen/Qwen3-1.7B` judge (temp=0.6, top_p=0.95).

## Results Summary

All scored by frozen `Qwen/Qwen3-1.7B` judge. HealthBench: 940 pairs. DRB: 100 pairs.

| Rubric Generator | Prompt | Benchmark | Pairwise Acc | Acc@0.10 | Cov@0.10 | Cohen's d | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rubric\_generator\_v0\_0302 (trained) | standard | HealthBench | 60.2% | 67.9% | 28% | 0.204 | 2.2e-09 |
| rubric\_generator\_v0\_0302 (trained) | standard | DRB | 64.9% | 83.9% | 56% | 0.516 | 5.9e-06 |
| rubric-generator-8b-v3 (trained) | standard | HealthBench | 61.4% | 67.6% | 30% | 0.198 | 3.6e-07 |
| rubric-generator-8b-v3 (trained) | standard | DRB | 68.0% | 72.3% | 47% | 0.279 | 1.6e-03 |
| rubric-generator-8b-v3 (trained) | v3 | HealthBench | 58.7% | 61.0% | 25% | 0.120 | 8.1e-05 |
| rubric-generator-8b-v3 (trained) | v3 | DRB | 59.8% | 73.2% | 41% | 0.295 | 5.2e-03 |
| Qwen/Qwen3-8B (baseline) | standard | HealthBench | 55.1% | 61.4% | 31% | 0.129 | 1.7e-02 |
| Qwen/Qwen3-8B (baseline) | standard | DRB | 66.7% | 81.2% | 48% | 0.420 | 4.1e-04 |
| Qwen/Qwen3-8B (baseline) | v3 | HealthBench | 54.1% | 56.0% | 23% | 0.051 | 6.3e-02 |
| Qwen/Qwen3-8B (baseline) | v3 | DRB | 61.4% | 73.8% | 42% | 0.340 | 1.4e-03 |
| GPT-4.1 (baseline) | standard | HealthBench | 56.4% | 62.0% | 26% | 0.117 | 5.2e-03 |
| GPT-4.1 (baseline) | standard | DRB | 60.0% | 62.2% | 37% | 0.247 | 4.7e-02 |
| GPT-4.1 (baseline) | v3 | HealthBench | 57.2% | 59.4% | 20% | 0.121 | 2.2e-03 |
| GPT-4.1 (baseline) | v3 | DRB | 60.2% | 71.4% | 42% | 0.294 | 1.1e-02 |

### Per-sample correlation with HealthBench ground-truth (n=998, DR-Tulu-8B, v0 rubrics)

| Correlation | Value | p-value |
| --- | --- | --- |
| Pearson r | 0.042 | 1.87e-01 |
| Spearman ρ | -0.005 | 8.86e-01 |
| Kendall τ | -0.003 | 8.79e-01 |

## Key Findings

- **Trained rubric generators outperform baselines** on HealthBench (~60% vs 54-57% pairwise accuracy)
- **Standard prompt > v3 (dealbreaker) prompt**: More criteria = more granularity for the judge
- **v3 model matches v0 model** when using the same standard prompt
- **DRB is more discriminative** than HealthBench across all configurations
- **Zero per-sample correlation** with HealthBench expert ground-truth rubrics
- The frozen 1.7B judge appears to be a bottleneck limiting all configurations
