#!/bin/bash
set -e

export PATH=/home/rulin/.local/bin:$PATH
export UV_LINK_MODE=copy
cd /checkpoint/dream/rulin/dr-tulu/agent

set -a
source /checkpoint/dream/rulin/dr-tulu/.env
set +a

MODEL="rl-research/DR-Tulu-SFT-8B"
MODEL_PORT=30001
MCP_PORT=8000
OUTPUT="eval_output/dr_tulu_sft_8b/healthbench.jsonl"

mkdir -p eval_output/dr_tulu_sft_8b

pkill -f "vllm serve" 2>/dev/null || true
pkill -f "mcp_backend" 2>/dev/null || true
sleep 2

echo "=== Starting vLLM server ($MODEL) ==="
source .venv_eval/bin/activate

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --dtype auto --port $MODEL_PORT --max-model-len 40960 \
    --gpu-memory-utilization 0.9 > /tmp/vllm_server.log 2>&1 &
VLLM_PID=$!

echo "=== Starting MCP server ==="
python -m dr_agent.mcp_backend.main \
    --transport http --host 0.0.0.0 --port $MCP_PORT > /tmp/mcp_server.log 2>&1 &
MCP_PID=$!

echo "Waiting for vLLM server (PID: $VLLM_PID)..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$MODEL_PORT/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died. Log:"
        tail -50 /tmp/vllm_server.log
        exit 1
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: vLLM server timeout. Log:"
        tail -50 /tmp/vllm_server.log
        exit 1
    fi
    sleep 5
done

echo "Waiting for MCP server (PID: $MCP_PID)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:$MCP_PORT/health > /dev/null 2>&1; then
        echo "MCP server ready!"
        break
    fi
    if ! kill -0 $MCP_PID 2>/dev/null; then
        echo "ERROR: MCP server died. Log:"
        tail -50 /tmp/mcp_server.log
        exit 1
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: MCP server timeout. Log:"
        tail -50 /tmp/mcp_server.log
        exit 1
    fi
    sleep 5
done

echo "=== Running HealthBench generation ==="
python workflows/auto_search_sft.py \
    generate-dataset healthbench \
    --num-examples final_run \
    --max-concurrent 20 \
    --batch-size 20 \
    --use-cache \
    --config workflows/auto_search_sft.yaml \
    --config-overrides "use_browse_agent=false,search_agent_max_tool_calls=10,browse_tool_name=jina,search_agent_base_url=http://localhost:${MODEL_PORT}/v1,mcp_port=${MCP_PORT},search_agent_model_name=${MODEL},search_agent_tokenizer_name=Qwen/Qwen3-8B" \
    --output "$OUTPUT"

echo "=== Generation complete: $OUTPUT ==="
kill $VLLM_PID $MCP_PID 2>/dev/null || true
