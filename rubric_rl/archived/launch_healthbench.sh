#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=200g
#SBATCH --time=7-00:00:00
#SBATCH --job-name=dr-tulu-healthbench
#SBATCH --output=/checkpoint/dream/rulin/dr-tulu/rubric_rl/outputs/healthbench_run.log

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

# Launch vLLM server in background
CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --port 30001 --dtype auto --max-model-len 40960 &
VLLM_PID=$!
echo "vLLM server launched with PID $VLLM_PID on port 30001"

# Wait for vLLM to be ready (poll every 10s, timeout after 10min)
echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:30001/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready after ~$((i*10))s"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: vLLM server did not start within 10 minutes"
        kill $VLLM_PID $MCP_PID 2>/dev/null
        exit 1
    fi
    sleep 10
done

# Wait a bit more for MCP
sleep 5

SAVE_DIR=/checkpoint/dream/rulin/dr-tulu/eval_output/dr_tulu_8b
mkdir -p $SAVE_DIR

echo "Starting healthbench generation..."
python workflows/auto_search_sft.py \
    generate-dataset healthbench \
    --num-examples final_run \
    --max-concurrent 20 \
    --batch-size 20 \
    --config workflows/auto_search_sft.yaml \
    --config-overrides "search_tool_name=serper,browse_tool_name=jina,search_agent_max_tool_calls=10" \
    --output $SAVE_DIR/healthbench.jsonl

echo "Generation complete. Output: $SAVE_DIR/healthbench.jsonl"

# Cleanup
kill $VLLM_PID $MCP_PID 2>/dev/null
echo "Servers stopped."

