model_name=rl-research/DR-Tulu-SFT-8B
dataset_list="rl-research/dr-tulu-rl-data 1.0"
exp_name="1129_dr-tulu-jina_${RANDOM}_no_browse"
# if you want to add the rar data, convert it to our format and then add to the dataset list, e.g.:
# dataset_list="rl-research/dr-tulu-rl-data 1.0 rl-rag/RaR-Medicine-20k-o3-mini-converted 3000 rl-rag/RaR-Science-20k-o3-mini-converted 1000"

# set env vars
# you need all these apis by default.
# Load API keys from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
# if using the docker container and crawl4ai, you can use this path.
# Otherwise, you need to set the path to the blocklist file.
# not used for jina.
# export CRAWL4AI_BLOCKLIST_PATH=/stage/rl-rag-mcp/utils/crawl4ai_block_list.txt
export MCP_MAX_CONCURRENT_CALLS=512
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export RUBRIC_JUDGE_MODEL=gpt-4.1-mini
export MCP_CACHE_DIR=.cache-${RANDOM}
export MCP_TRANSPORT_PORT=8003
export MCP_TRANSPORT_HOST=$(hostname)

# Suppress verbose litellm async task logs
export LITELLM_LOG="WARNING"

# setup a ray cluster, with 2 nodes and 8 GPUs per node.
# in ai2, we use the following script:
# source configs/beaker_configs/ray_node_setup.sh


uv run --extra compile python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --wandb_project_name rl-rag \
        --beta 0.001 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 32 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 5e-7 \
        --per_device_train_batch_size 1 \
        --output_dir output \
        --kl_estimator kl3 \
        --dataset_mixer_list ${dataset_list} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list rl-rag/healthbench_all_adaptive_rubric 16 \
        --dataset_mixer_eval_list_splits test \
        --apply_adaptive_rubric_reward true \
        --normalize_rubric_scores false \
        --use_rubric_buffer true \
        --use_static_rubrics_as_persistent_rubrics true \
        --max_active_rubrics 5 \
        --max_token_length 10240 \
        --max_prompt_token_length 2048 \
        --response_length 16384 \
        --pack_length 18500 \
        --model_name_or_path ${model_name} \
        --non_stop_penalty False \
        --non_stop_penalty_value 0.0 \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 6 \
        --vllm_num_engines 2 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --num_evals 500 \
        --save_freq 50 \
        --try_launch_beaker_eval_jobs_on_weka False \
        --gradient_checkpointing \
        --with_tracking \
        --max_tool_calls 10 \
        --only_reward_good_outputs False \
        --tools mcp \
        --checkpoint_state_freq 50 \
        --checkpoint_state_dir output/checkpoints \
        --mcp_parser_name v20250824 \
        --system_prompt_file open_instruct/search_utils/system_prompts/unified_tool_calling_v20250907.yaml  \
        --mcp_tool_names 'snippet_search,google_search' \
        --mcp_server_command "uv run python -m dr_agent.mcp_backend.main --transport http --port 8003 --host 0.0.0.0 --path /mcp"
