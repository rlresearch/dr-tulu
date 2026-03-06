#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=200g
#SBATCH --time=7-00:00:00
#SBATCH --job-name=healthbench-rubric-gen
#SBATCH --output=/checkpoint/dream/rulin/dr-tulu/healthbench_rubric_gen.log

source /opt/conda/etc/profile.d/conda.sh
conda activate dr_agent

echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi -L
echo "Start time: $(date)"

cd /checkpoint/dream/rulin/dr-tulu/rubric_rl

SAVE_DIR=/checkpoint/dream/rulin/dr-tulu/rubric_rl/outputs
mkdir -p $SAVE_DIR

# ── Generate rubrics for HealthBench (all subset, 5000 examples) ──
echo "=== Generating rubrics for HealthBench (all) ==="
python generate_healthbench_rubrics.py \
    --rubric_model stellalisy/rubric_generator_v0_0302 \
    --subset all \
    --rubric_style standard \
    --temperature 0.6 \
    --max_tokens 16384 \
    --max_model_len 16384 \
    --batch_size 256 \
    --gpu_memory_utilization 0.90 \
    --output $SAVE_DIR/healthbench_all_rubrics.jsonl

echo "=== Done ==="
echo "Output: $SAVE_DIR/healthbench_all_rubrics.jsonl"
echo "End time: $(date)"

