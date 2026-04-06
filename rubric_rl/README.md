# Rubric RL Evaluation

Scripts for evaluating rubric-based reward models: generating rubrics, scoring with a frozen judge, and computing pairwise ranking accuracy against expert ground truth.

## Scripts

| Script | Purpose |
| --- | --- |
| `generate_rubrics_batch.py` | Generate rubrics for any benchmark using vLLM (local models) or OpenAI API |
| `generate_gt_rubrics.py` | Convert expert rubrics from HealthBench/DRB/ResearchQA to unified JSONL format |
| `score_gt_binary.py` | Score rollouts against rubrics using GPT-4.1 with binary per-criterion grading |
| `score_pairwise.py` | Score RL + SFT rollouts with frozen Qwen3-1.7B judge, compute pairwise stats |
| `compute_pairwise_accuracy.py` | Compute pairwise ranking accuracy between synthetic and GT scores |
| `rubric_chat_templates.py` | Prompt templates for rubric generation and judging |

## Evaluation Pipeline

```bash
# 1. Generate GT rubrics from expert annotations
python generate_gt_rubrics.py --benchmark healthbench \
    --rollouts /path/to/rollouts.jsonl --output gt_rubrics.jsonl

# 2. Score rollouts against GT rubrics (binary per-criterion, GPT-4.1)
python score_gt_binary.py --rollouts /path/to/rl_rollouts.jsonl \
    --rubrics gt_rubrics.jsonl --output rl_gt_scores.json
python score_gt_binary.py --rollouts /path/to/sft_rollouts.jsonl \
    --rubrics gt_rubrics.jsonl --output sft_gt_scores.json

# 3. Generate synthetic rubrics (3 generators)
python generate_rubrics_batch.py --rollouts /path/to/rollouts.jsonl \
    --rubric_model stellalisy/rubric-generator-8b-v3 --rubric_style standard \
    --output rubrics_ours.jsonl

python generate_rubrics_batch.py --rollouts /path/to/rollouts.jsonl \
    --rubric_model Qwen/Qwen3-8B --rubric_style standard \
    --output rubrics_qwen.jsonl

python generate_rubrics_batch.py --rollouts /path/to/rollouts.jsonl \
    --rubric_model gpt-4.1 --rubric_style standard --use_api \
    --output rubrics_gpt.jsonl

# 4. Score with frozen judge (Qwen3-1.7B)
python score_pairwise.py --rl_rollouts /path/to/rl.jsonl \
    --sft_rollouts /path/to/sft.jsonl --rubrics rubrics_ours.jsonl \
    --output scores_ours.json

# 5. Compute pairwise accuracy against GT
python compute_pairwise_accuracy.py \
    --gt_rl rl_gt_scores.json --gt_sft sft_gt_scores.json \
    --synthetic scores_ours.json scores_qwen.json scores_gpt.json \
    --labels ours Qwen3-8B GPT-4.1 --delta 0 0.01 0.05 0.1
```

## Results

GT scored by GPT-4.1 (binary per-criterion). Synthetic scored by frozen Qwen3-1.7B judge. Standard prompt.

| Rubric Generator | HB Acc@0.1 (n=371) | DRB Acc@0.01 (n=57) | RQA Acc@0.05 (n=344) |
| --- | --- | --- | --- |
| **ours (rubric-generator-8b-v3)** | **59.6%** | **63.2%** | 56.7% |
| Qwen3-8B (prompted) | 57.3% | 56.2% | **57.8%** |
| GPT-4.1 (prompted) | 53.5% | 60.4% | 48.3% |

## Archived

Older experimental scripts are in `archived/`.
