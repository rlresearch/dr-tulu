#!/bin/bash
set -e
export PATH=/home/rulin/.local/bin:$PATH
cd /checkpoint/dream/rulin/dr-tulu
source agent/.venv_eval/bin/activate
pip install json_repair scipy 2>/dev/null | tail -1

pkill -f "vllm serve" 2>/dev/null || true
sleep 2

BASE=rubric_rl/outputs/v3_comparison
RL=$BASE/researchqa_rl_overlap.jsonl
SFT=$BASE/researchqa_sft_overlap.jsonl

echo "=== Rubric gen: rubric-generator-8b-v3 standard ==="
PYTHONUNBUFFERED=1 python -u rubric_rl/generate_rubrics_batch.py --rollouts $RL --output $BASE/researchqa_rubric_gen_8b_v3_standard.jsonl --rubric_model stellalisy/rubric-generator-8b-v3 --rubric_style standard --max_model_len 16384 2>&1 | tail -1

echo "=== Rubric gen: rubric-generator-8b-v3 v3 ==="
PYTHONUNBUFFERED=1 python -u rubric_rl/generate_rubrics_batch.py --rollouts $RL --output $BASE/researchqa_rubric_gen_8b_v3.jsonl --rubric_model stellalisy/rubric-generator-8b-v3 --rubric_style v3 --max_model_len 16384 2>&1 | tail -1

echo "=== Rubric gen: Qwen3-8B standard ==="
PYTHONUNBUFFERED=1 python -u rubric_rl/generate_rubrics_batch.py --rollouts $RL --output $BASE/researchqa_qwen3_8b_standard.jsonl --rubric_model Qwen/Qwen3-8B --rubric_style standard --max_model_len 16384 2>&1 | tail -1

echo "=== Rubric gen: Qwen3-8B v3 ==="
PYTHONUNBUFFERED=1 python -u rubric_rl/generate_rubrics_batch.py --rollouts $RL --output $BASE/researchqa_qwen3_8b.jsonl --rubric_model Qwen/Qwen3-8B --rubric_style v3 --max_model_len 16384 2>&1 | tail -1

echo "=== Scoring all with frozen judge ==="
for NAME in rubric_gen_8b_v3_standard rubric_gen_8b_v3 qwen3_8b_standard qwen3_8b gpt41_standard gpt41; do
    echo "--- $NAME ---"
    PYTHONUNBUFFERED=1 python -u rubric_rl/score_pairwise.py \
        --rl_rollouts $RL \
        --sft_rollouts $SFT \
        --rubrics $BASE/researchqa_${NAME}.jsonl \
        --output $BASE/scores_researchqa_${NAME}.json 2>&1 | grep -E "PAIRWISE|RL:|SFT:|wins|Pairwise|Cohen|diff"
done

echo "=== All ResearchQA done ==="
