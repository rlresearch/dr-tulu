# Evaluation Scripts

This directory contains evaluation scripts for benchmarking DR agents across various tasks, from short-form QA to long-form deep research.

---

## Available Benchmarks

| Benchmark | Type | Description | Benchmark Name |
|-----------|------|-------------|----------------|
| **SQA-CS-V2** | Long-form | Scientific question answering with structured citations | `sqa_cs_v2` |
| **Deep Research Bench** | Long-form | Deep research reports (RACE & FACT metrics) | `deep_research_bench` |
| **ResearchQA** | Long-form | Research question answering with coverage metrics | `research_qa` |
| **HealthBench** | Long-form | Medical QA with physician-level rubrics | `healthbench` |
| **Genetic Diseases** | Domain-specific | Clinical genetics questions | `genetic_diseases` |
| **SimpleQA** | Short-form | Factuality in short-form answers | `simpleqa` |
| **Short Form QA** | Short-form | Multi-dataset QA framework (14+ datasets) | See supported tasks below |

---

## Running Evaluations

### Example: Evaluate Across All Benchmarks

**Prerequisites**: Before running the evaluation script, launch the required servers **on the same node**:

```bash
# Launch VLLM servers (requires 2 GPUs)
CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960

# Launch MCP server
python -m dr_agent.mcp_backend.main --port 8000
```

Then run the evaluation script:

```bash
#!/bin/bash
# Example script to run DR Tulu on multiple benchmarks

SAVE_FOLDER=eval_output/
MODEL=auto_search_sft
YAML_CONFIG=workflows/auto_search_sft.yaml
MAX_CONCURRENT=20

mkdir -p $SAVE_FOLDER

# Run evaluations on all benchmarks
for task in healthbench deep_research_bench research_qa genetic_diseases simpleqa 2wiki webwalker; do 
    echo "Running $MODEL on $task"
    python workflows/$MODEL.py \
        generate-dataset $task \
        --num-examples final_run \
        --max-concurrent $MAX_CONCURRENT \
        --batch-size $MAX_CONCURRENT \
        --use-cache \
        --config $YAML_CONFIG \
        --config-overrides "use_browse_agent=true,search_agent_max_tool_calls=10,browse_tool_name=jina" \
        --output $SAVE_FOLDER/$MODEL/$task.jsonl
    
    python scripts/evaluate.py $task $SAVE_FOLDER/$MODEL/$task.jsonl
done
```

For SQA-CS-V2 evaluation, see the dedicated section below.

---

## Additional instructions on SQA-CS-V2 Evaluation

SQA-CS-V2 requires responses in a specific JSON format with structured sections and citations:

```json
{
  "sections": [
    {
      "text": "text of section 1",
      "citations": {
        "id": "cite 1 of sec 1",
        "snippets": [
          "evidence 1",
          "evidence 2"
        ]
      }
    },
    {
      "text": "text of section 2",
      "citations": {
        "id": "cite 1 of sec 2",
        "snippets": [
          "List of evidence"
        ]
      }
    }
  ]
}
```

### Evaluation Steps

1. **Convert DR Tulu outputs to SQA format**:
   ```bash
   python evaluation/sqa_eval/convert_to_asta_format.py --folder <folder_name> --file <file_name>
   ```

2. **Clone the evaluation repository**:
   ```bash
   git clone https://github.com/allenai/agent-baselines
   cd agent-baselines
   ```

3. **Run evaluation**:
   ```bash
   uv run --extra sqa inspect eval astabench/sqa --display plain \
     --solver agent_baselines/solvers/sqa/debug/cached_solver.py \
     -S path=<outputfile_from_step1> \
     -T split=test \
     -T with_search_tools=False \
     -T simplified_eval=true \
     -T assess_jointly=true \
     --max-connections 16 \
     -T sentence_wise_cit_eval=false \
     -T all_at_once=true \
     -T scorer_model="google/gemini-2.5-flash"
   ```

**Note**: Export `GOOGLE_API_KEY` and `HF_TOKEN` before running. If errors request additional tokens, dummy values can be used.

