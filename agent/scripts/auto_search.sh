
# Before running this script, you may need to 
# Launch two VLLM servers: 
# CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960
# CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960

# And launch the mcp server: 
# python -m dr_agent.mcp_backend.main --port 8000

DATEUID=20250915
MAX_CONCURRENT=20

SAVE_FOLDER=eval_output/
MODEL=auto_search_sft
YAML_CONFIG=workflows/auto_search_sft.yaml
SAVE_MODEL_NAME=auto_search_sft

mkdir -p $SAVE_FOLDER

for task in sqav2; do 
    echo "Running $MODEL on $task"
    python workflows/$MODEL.py \
        generate-dataset $task \
        --num-examples final_run \
        --max-concurrent $MAX_CONCURRENT \
        --batch-size $MAX_CONCURRENT \
        --use-cache \
        --config $YAML_CONFIG \
        --config-overrides "use_browse_agent=true,search_tool_name=s2,search_agent_max_tool_calls=10, browse_tool_name=jina" \
        --output $SAVE_FOLDER/$SAVE_MODEL_NAME/$task-ablation-s2.jsonl
done
