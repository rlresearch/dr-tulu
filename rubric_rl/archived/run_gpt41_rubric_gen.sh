#!/bin/bash
set -e
export PATH=/home/rulin/.local/bin:$PATH
cd /checkpoint/dream/rulin/dr-tulu

set -a
source .env
set +a

echo "OPENAI_API_KEY len: ${#OPENAI_API_KEY}"

echo "=== GPT-4.1: HealthBench ==="
PYTHONUNBUFFERED=1 python3 -u rubric_rl/generate_rubrics_batch.py \
    --rollouts eval_output/dr_tulu_8b/healthbench.jsonl \
    --output rubric_rl/outputs/v3_comparison/healthbench_gpt41.jsonl \
    --rubric_model gpt-4.1 --rubric_style v3 --use_api --api_workers 20

echo "=== GPT-4.1: DRB ==="
PYTHONUNBUFFERED=1 python3 -u rubric_rl/generate_rubrics_batch.py \
    --rollouts agent/eval_output/dr_tulu_8b_step4000/deep_research_bench.jsonl \
    --output rubric_rl/outputs/v3_comparison/drb_gpt41.jsonl \
    --rubric_model gpt-4.1 --rubric_style v3 --use_api --api_workers 20

echo "=== Done ==="
