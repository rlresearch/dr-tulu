#!/bin/bash
#SBATCH --job-name=rubric-gen-all
set -e
export PATH=/home/rulin/.local/bin:$PATH
cd /checkpoint/dream/rulin/dr-tulu
source agent/.venv_eval/bin/activate

OUTDIR=rubric_rl/outputs/v3_comparison
HB_ROLLOUTS=eval_output/dr_tulu_8b/healthbench.jsonl
DRB_ROLLOUTS=agent/eval_output/dr_tulu_8b_step4000/deep_research_bench.jsonl

echo "=== stellalisy/rubric-generator-8b-v3: HealthBench ==="
PYTHONUNBUFFERED=1 python -u rubric_rl/generate_rubrics_batch.py \
    --rollouts $HB_ROLLOUTS \
    --output $OUTDIR/healthbench_rubric_gen_8b_v3.jsonl \
    --rubric_model stellalisy/rubric-generator-8b-v3 \
    --rubric_style v3 --max_model_len 16384

echo "=== stellalisy/rubric-generator-8b-v3: DRB ==="
PYTHONUNBUFFERED=1 python -u rubric_rl/generate_rubrics_batch.py \
    --rollouts $DRB_ROLLOUTS \
    --output $OUTDIR/drb_rubric_gen_8b_v3.jsonl \
    --rubric_model stellalisy/rubric-generator-8b-v3 \
    --rubric_style v3 --max_model_len 16384

echo "=== Qwen/Qwen3-8B: HealthBench ==="
PYTHONUNBUFFERED=1 python -u rubric_rl/generate_rubrics_batch.py \
    --rollouts $HB_ROLLOUTS \
    --output $OUTDIR/healthbench_qwen3_8b.jsonl \
    --rubric_model Qwen/Qwen3-8B \
    --rubric_style v3 --max_model_len 16384

echo "=== Qwen/Qwen3-8B: DRB ==="
PYTHONUNBUFFERED=1 python -u rubric_rl/generate_rubrics_batch.py \
    --rollouts $DRB_ROLLOUTS \
    --output $OUTDIR/drb_qwen3_8b.jsonl \
    --rubric_model Qwen/Qwen3-8B \
    --rubric_style v3 --max_model_len 16384

echo "=== All rubric generation done ==="
