#!/bin/bash
set -e
export PATH=/home/rulin/.local/bin:$PATH
cd /checkpoint/dream/rulin/dr-tulu
source agent/.venv_eval/bin/activate
pip install json_repair scipy 2>/dev/null | tail -1

OUTDIR=rubric_rl/outputs/v3_comparison
HB_RL=eval_output/dr_tulu_8b/healthbench.jsonl
HB_SFT=agent/eval_output/dr_tulu_sft_8b/healthbench.jsonl
DRB_RL=agent/eval_output/dr_tulu_8b_step4000/deep_research_bench.jsonl
DRB_SFT=agent/eval_output/dr_tulu_sft_8b/deep_research_bench.jsonl

for RUBRIC_NAME in rubric_gen_8b_v3 qwen3_8b gpt41; do
    echo "=============================================="
    echo "Scoring with rubrics: $RUBRIC_NAME"
    echo "=============================================="

    echo "--- HealthBench ---"
    PYTHONUNBUFFERED=1 python -u rubric_rl/score_pairwise.py \
        --rl_rollouts $HB_RL \
        --sft_rollouts $HB_SFT \
        --rubrics $OUTDIR/healthbench_${RUBRIC_NAME}.jsonl \
        --output $OUTDIR/scores_healthbench_${RUBRIC_NAME}.json

    echo "--- DRB ---"
    PYTHONUNBUFFERED=1 python -u rubric_rl/score_pairwise.py \
        --rl_rollouts $DRB_RL \
        --sft_rollouts $DRB_SFT \
        --rubrics $OUTDIR/drb_${RUBRIC_NAME}.jsonl \
        --output $OUTDIR/scores_drb_${RUBRIC_NAME}.json

    echo ""
done

echo "=== All scoring done ==="
