#!/bin/bash

# Before running this script, you may need to 
# Launch two VLLM servers: 
# CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960
# CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960
#
# To check if servers are running:
# bash scripts/check_vllm_servers.sh

# And launch the mcp server: 
# python -m dr_agent.mcp_backend.main --port 8000

DATEUID=20250915
MAX_CONCURRENT=20

SAVE_FOLDER=eval_output/
MODEL=auto_search_sft
YAML_CONFIG=workflows/auto_search_sft_ds_serve.yaml
SAVE_MODEL_NAME=auto_search_sft_ds_serve

mkdir -p $SAVE_FOLDER

for task in sqav2; do 
    echo "Running $MODEL on $task with DS Serve API"
    python workflows/$MODEL.py \
        generate-dataset $task \
        --num-examples final_run \
        --max-concurrent $MAX_CONCURRENT \
        --batch-size $MAX_CONCURRENT \
        --use-cache \
        --config $YAML_CONFIG \
        --config-overrides "search_tool_name=ds_serve,use_browse_agent=false,browse_tool_name=null" \
        --output $SAVE_FOLDER/$SAVE_MODEL_NAME/$task-ds-serve.jsonl
done

