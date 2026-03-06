#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=200g
#SBATCH --time=7-00:00:00
#SBATCH --job-name=dr-tulu-vllm
#SBATCH --output=/checkpoint/dream/rulin/dr-tulu/rubric_rl/outputs/vllm_server.log

source /opt/conda/etc/profile.d/conda.sh
conda activate dr_agent

echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi -L

cd /checkpoint/dream/rulin/dr-tulu/agent

# Launch MCP server in background
python -m dr_agent.mcp_backend.main --port 8000 &
MCP_PID=$!
echo "MCP server launched with PID $MCP_PID on port 8000"

# Launch vLLM server in foreground
CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --port 30001 --dtype auto --max-model-len 40960

