# Before running this script, you may need to
# Launch the mcp server:
# python -m mcp_agents.mcp_backend.main --port 8000

DATEUID=20250902
MAX_CONCURRENT=10

SAVE_FOLDER=eval_output/baselines-$DATEUID-gpt4.1
MODEL=auto_search_sft

mkdir -p $SAVE_FOLDER

# Fast tasks
# TASKS="simpleqa 2wiki healthbench"
# TASKS="browsecomp bc_synthetic_depth_one_v2_verified bc_synthetic_varied_depth_o3_verified"
TASKS="dsqa"

# serper+crawl4ai+readerv2+max-tool-calls-10
for task in $TASKS; do
    echo "Running $MODEL on $task"
    python workflows/$MODEL.py \
        generate-dataset $task \
        --num-examples ablation \
        --max-concurrent $MAX_CONCURRENT \
        --batch-size $MAX_CONCURRENT \
        --use-cache \
        --config workflows/$MODEL-oai.yaml \
        --config-overrides "use_browse_agent=true,search_agent_max_tool_calls=10" \
        --output $SAVE_FOLDER/$MODEL/$task-ablation-reader-max-tool-calls-10.jsonl
done