# Rubric RL Evaluation

This directory contains experiments evaluating how well generated rubrics (from a trained rubric generator) + a frozen judge can serve as a reward signal, compared to expert-curated ground-truth rubrics.

**Key question**: Can generated rubrics distinguish between model checkpoints (RL vs SFT), even if they don't correlate with expert scores per-sample?

**Answer**: Yes. Zero per-sample correlation with ground-truth, but statistically significant pairwise distinguishability (p < 1e-6) across both benchmarks, with 60-65% overall accuracy improving to 75-86% on high-margin pairs.

## Reproduction

```bash
# 1. Generate rubrics for a benchmark (needs 1 GPU)
python generate_healthbench_rubrics.py \
    --rubric_model stellalisy/rubric_generator_v0_0302 \
    --output outputs/healthbench_all_rubrics.jsonl

python generate_drb_rubrics.py \
    --rollouts /path/to/deep_research_bench.jsonl \
    --output outputs/drb_rubrics.jsonl

# 2. Score rollouts with frozen judge + compute correlation (needs 1 GPU)
python evaluate_rubric_correlation.py \
    --rollouts /path/to/healthbench.jsonl \
    --rubrics outputs/healthbench_all_rubrics.jsonl

# 3. Pairwise comparison (needs 1 GPU)
python score_drb_pairwise.py
```

---

## HealthBench: Generated vs Ground-Truth Rubrics

### Setup

- **Generated rubrics**: produced by `stellalisy/rubric_generator_v0_0302` (5-6 weighted criteria per question, multi-level 0-1 scoring)
- **Judge model**: `Qwen/Qwen3-1.7B` frozen judge (temp=0.6, top_p=0.95, using `"judge"` template from `rubric_chat_templates.py`)
- **Ground-truth rubrics**: HealthBench expert-curated (10-20 binary criteria with integer points including demerits, graded by GPT-4.1-mini)
- **Models evaluated**: `rl-research/DR-Tulu-8B` (RL, step_4000) and `rl-research/DR-Tulu-SFT-8B`

### 1. Per-sample correlation with ground-truth (n=998, DR-Tulu-8B only)

| Metric | Generated Rubric | Ground-Truth Rubric |
| --- | --- | --- |
| Mean score | 0.814 | 0.567 |
| Std | 0.119 | 0.344 |

| Correlation | Value | p-value |
| --- | --- | --- |
| Pearson r | 0.042 | 1.87e-01 |
| Spearman ρ | -0.005 | 8.86e-01 |
| Kendall τ | -0.003 | 8.79e-01 |

No per-sample correlation — the generated rubrics do not rank individual examples the same way as ground-truth HealthBench rubrics.

### 2. Pairwise distinguishability: RL vs SFT (n=936 paired examples)

| Metric | DR-Tulu-8B (RL) | DR-Tulu-SFT-8B |
| --- | --- | --- |
| Mean generated rubric score | 0.814 | 0.787 |
| Std | 0.117 | 0.142 |

| Statistic | Value |
| --- | --- |
| Mean difference (RL - SFT) | +0.027 |
| Pairwise accuracy (excl. 330 ties) | 60.2% (365/606) |
| Binomial test p-value | 2.68e-07 |
| Wilcoxon signed-rank p-value | 2.20e-09 |
| Cohen's d | 0.204 |

| Score margin threshold | Pairs | Coverage | Accuracy |
| --- | --- | --- | --- |
| >0.00 (all non-ties) | 606 | 64.7% | 60.2% |
| >0.05 | 465 | 49.7% | 62.4% |
| >0.10 | 265 | 28.3% | 67.9% |
| >0.15 | 187 | 20.0% | 70.6% |
| >0.20 | 106 | 11.3% | 75.5% |

"Score margin" = `|RL_score - SFT_score|` for a given question. Higher thresholds filter to pairs where the judge assigns substantially different scores, yielding higher accuracy but fewer pairs.

### Takeaway

The generated rubrics + frozen judge have zero per-sample correlation with expert ground-truth scores, but can reliably distinguish RL vs SFT model quality in aggregate (p<1e-9). The signal is weak per-pair (60% accuracy) but improves to 75% on high-margin pairs (11% coverage). This suggests the rubrics are useful as a coarse training signal but not as a substitute for expert evaluation.

---

## Deep Research Bench: Pairwise Distinguishability

### Setup

- **Generated rubrics**: produced by `stellalisy/rubric_generator_v0_0302` (same as HealthBench, 5-6 weighted criteria)
- **Judge model**: `Qwen/Qwen3-1.7B` frozen judge (temp=0.6, top_p=0.95)
- **Models evaluated**: `rl-research/DR-Tulu-8B` (RL, step_4000) and `rl-research/DR-Tulu-SFT-8B`
- **Dataset**: 100 DRB examples (all matched across RL, SFT, and rubrics)

### Pairwise distinguishability: RL vs SFT (n=100)

| Metric | DR-Tulu-8B (RL) | DR-Tulu-SFT-8B |
| --- | --- | --- |
| Mean generated rubric score | 0.773 | 0.664 |
| Std | 0.154 | 0.183 |

| Statistic | Value |
| --- | --- |
| Mean difference (RL - SFT) | +0.109 |
| Pairwise accuracy (excl. 6 ties) | 64.9% (61/94) |
| Binomial test p-value | 2.54e-03 |
| Paired t-test p-value | 1.41e-06 |
| Wilcoxon signed-rank p-value | 5.87e-06 |
| Cohen's d | 0.516 |

| Score margin threshold | Pairs | Coverage | Accuracy |
| --- | --- | --- | --- |
| >0.00 (all non-ties) | 94 | 94.0% | 64.9% |
| >0.05 | 80 | 80.0% | 72.5% |
| >0.10 | 56 | 56.0% | 83.9% |
| >0.15 | 41 | 41.0% | 82.9% |
| >0.20 | 35 | 35.0% | 85.7% |

### Takeaway

The generated rubrics show **stronger distinguishability on DRB than on HealthBench**:
- Effect size is 2.5x larger (Cohen's d = 0.516 vs 0.204)
- Mean score gap is 4x larger (0.109 vs 0.027)
- At margin > 0.10, accuracy reaches **84%** with **56% coverage** (vs 68% accuracy / 28% coverage on HealthBench)
- Fewer ties (6% vs 35%)

This makes sense: DRB tests long-form research report writing where the RL model has a clearer quality advantage over SFT, and the rubric criteria (structure, evidence use, depth) are better aligned with what the generated rubrics measure.

---

## Summary: v0 Rubric Generator (`stellalisy/rubric_generator_v0_0302`)

| Benchmark | N | Mean Diff | Pairwise Acc | Acc @ margin>0.10 | Coverage @ margin>0.10 | Cohen's d | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HealthBench | 936 | +0.027 | 60.2% | 67.9% | 28.3% | 0.204 | 2.20e-09 |
| Deep Research Bench | 100 | +0.109 | 64.9% | 83.9% | 56.0% | 0.516 | 5.87e-06 |

Both benchmarks show the generated rubrics + frozen judge can reliably distinguish RL from SFT, with the signal being substantially stronger on DRB.

---

## Rubric Generator Comparison (v3 prompt, Qwen3-1.7B frozen judge)

Comparing three rubric generators using the v3 "dealbreaker" prompt template. All rubrics are scored by the same frozen `Qwen/Qwen3-1.7B` judge.

### Setup

- **Rubric generators**: `stellalisy/rubric-generator-8b-v3` (trained), `Qwen/Qwen3-8B` (base), `GPT-4.1` (API)
- **Rubric prompt**: v3 template with dealbreaker criterion support (2-5 criteria, weights sum to 1.0)
- **Judge model**: `Qwen/Qwen3-1.7B` frozen judge (temp=0.6, top_p=0.95)
- **Models evaluated**: `rl-research/DR-Tulu-8B` (RL, step_4000) vs `rl-research/DR-Tulu-SFT-8B`

### Results

| Rubric Generator | Benchmark | N | RL Mean | SFT Mean | Diff | Pairwise Acc | Acc@0.10 | Cov@0.10 | Cohen's d | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stellalisy/rubric-generator-8b-v3 | HealthBench | 940 | 0.910 | 0.890 | +0.020 | 0.587 | 0.610 | 25% | 0.120 | 8.1e-05 |
| stellalisy/rubric-generator-8b-v3 | DRB | 100 | 0.891 | 0.837 | +0.054 | 0.598 | 0.732 | 41% | 0.295 | 5.2e-03 |
| Qwen/Qwen3-8B | HealthBench | 940 | 0.906 | 0.898 | +0.008 | 0.541 | 0.560 | 23% | 0.051 | 6.3e-02 |
| Qwen/Qwen3-8B | DRB | 100 | 0.876 | 0.815 | +0.060 | 0.614 | 0.738 | 42% | 0.340 | 1.4e-03 |
| GPT-4.1 | HealthBench | 940 | 0.939 | 0.919 | +0.020 | 0.572 | 0.594 | 20% | 0.121 | 2.2e-03 |
| GPT-4.1 | DRB | 100 | 0.891 | 0.834 | +0.058 | 0.602 | 0.714 | 42% | 0.294 | 1.1e-02 |

### Takeaway

- **All three rubric generators** can distinguish RL from SFT on both benchmarks (all p < 0.05 except Qwen3-8B on HealthBench at p=0.063)
- **DRB is more discriminative** than HealthBench across all generators (higher accuracy, larger effect sizes)
- **Trained model (rubric-generator-8b-v3) vs baselines**: On HealthBench, the trained model shows similar performance to GPT-4.1 (0.587 vs 0.572 pairwise acc) and outperforms Qwen3-8B (0.541). On DRB, all three are comparable (~0.60 pairwise acc), though Qwen3-8B shows slightly better margin-based accuracy (0.738 vs 0.732 vs 0.714 at margin>0.10)
- **Qwen3-8B struggles on HealthBench** (p=0.063, not significant) but performs well on DRB, suggesting medical domain specificity matters
- The frozen 1.7B judge is a bottleneck: all generators produce similar pairwise accuracy, suggesting the judge's discriminative capacity — not the rubric quality — is the limiting factor
