#!/bin/bash
set -e
export PATH=/home/rulin/.local/bin:$PATH
export UV_LINK_MODE=copy
cd /checkpoint/dream/rulin/dr-tulu/agent

set -a
source /checkpoint/dream/rulin/dr-tulu/.env
set +a
source .venv_eval/bin/activate

MODEL="rl-research/DR-Tulu-SFT-8B"
MODEL_PORT=30001
MCP_PORT=8000

mkdir -p eval_output/dr_tulu_sft_8b

pkill -f "vllm serve" 2>/dev/null || true
pkill -f "mcp_backend" 2>/dev/null || true
sleep 2

echo "=== Starting vLLM server ($MODEL) ==="
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --dtype auto --port $MODEL_PORT --max-model-len 40960 \
    --gpu-memory-utilization 0.9 > /tmp/vllm_rqa.log 2>&1 &
VLLM_PID=$!

echo "=== Starting MCP server ==="
python -m dr_agent.mcp_backend.main \
    --transport http --host 0.0.0.0 --port $MCP_PORT > /tmp/mcp_rqa.log 2>&1 &
MCP_PID=$!

for i in $(seq 1 180); do
    curl -s http://localhost:$MODEL_PORT/health > /dev/null 2>&1 && echo "vLLM ready!" && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -20 /tmp/vllm_rqa.log; exit 1; }
    [ $i -eq 180 ] && { echo "TIMEOUT"; exit 1; }
    sleep 5
done

for i in $(seq 1 60); do
    curl -s http://localhost:$MCP_PORT/health > /dev/null 2>&1 && echo "MCP ready!" && break
    sleep 5
done

echo "=== Running ResearchQA generation ==="
python workflows/auto_search_sft.py \
    generate-dataset researchqa \
    --num-examples final_run \
    --max-concurrent 10 \
    --batch-size 10 \
    --use-cache \
    --config workflows/auto_search_sft.yaml \
    --config-overrides "use_browse_agent=false,search_agent_max_tool_calls=10,browse_tool_name=jina,search_agent_base_url=http://localhost:${MODEL_PORT}/v1,mcp_port=${MCP_PORT},search_agent_model_name=${MODEL},search_agent_tokenizer_name=Qwen/Qwen3-8B" \
    --output eval_output/dr_tulu_sft_8b/researchqa.jsonl

echo "=== Done ==="
kill $VLLM_PID $MCP_PID 2>/dev/null || true
