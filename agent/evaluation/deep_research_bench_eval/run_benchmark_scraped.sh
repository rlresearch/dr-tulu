#!/bin/bash
# Target model name list
# TARGET_MODELS=("claude-3-7-sonnet-latest")
TARGET_MODELS=("auto_search_sft")

# Common parameters for both RACE and Citation evaluations
RAW_DATA_DIR="data/test_data/raw_data"
OUTPUT_DIR="results"
N_TOTAL_PROCESS=10
QUERY_DATA_PATH="data/prompt_data/query.jsonl"

# Limit on number of prompts to process (for testing). Uncomment to enable
# LIMIT="--limit 2"

# Skip article cleaning step. Uncomment to enable
# SKIP_CLEANING="--skip_cleaning"

# Only process specific language data. Uncomment to enable
# ONLY_ZH="--only_zh"  # Only process Chinese data
# ONLY_EN="--only_en"  # Only process English data

# Force re-evaluation even if results exist. Uncomment to enable
# FORCE="--force"

# Loop through each model in the target models list
for TARGET_MODEL in "${TARGET_MODELS[@]}"; do
  # Specify log output file for current model
  OUTPUT_LOG_FILE="output_${TARGET_MODEL}.log"
  
  # Clear log file
  echo "Starting benchmark tests for model: $TARGET_MODEL, log output to: $OUTPUT_LOG_FILE" > "$OUTPUT_LOG_FILE"
  
  echo "Running benchmark for target model: $TARGET_MODEL"
  echo -e "\n\n========== Starting evaluation for $TARGET_MODEL ==========\n" >> "$OUTPUT_LOG_FILE"

  #--- Phase 1: RACE Evaluation ---
  echo "==== Phase 1: Running RACE Evaluation for $TARGET_MODEL ====" | tee -a "$OUTPUT_LOG_FILE"
  RACE_OUTPUT="$OUTPUT_DIR/race/$TARGET_MODEL"
  mkdir -p $RACE_OUTPUT

  # Base command for current target model
  PYTHON_CMD="python -u deepresearch_bench_race.py \"$TARGET_MODEL\" --raw_data_dir $RAW_DATA_DIR --max_workers $N_TOTAL_PROCESS --query_file $QUERY_DATA_PATH --output_dir $RACE_OUTPUT"

  # Add optional parameters
  if [[ -n "$LIMIT" ]]; then
    PYTHON_CMD="$PYTHON_CMD $LIMIT"
  fi

  if [[ -n "$SKIP_CLEANING" ]]; then
    PYTHON_CMD="$PYTHON_CMD $SKIP_CLEANING"
  fi
  
  if [[ -n "$ONLY_ZH" ]]; then
    PYTHON_CMD="$PYTHON_CMD $ONLY_ZH"
  fi
  
  if [[ -n "$ONLY_EN" ]]; then
    PYTHON_CMD="$PYTHON_CMD $ONLY_EN"
  fi
  
  if [[ -n "$FORCE" ]]; then
    PYTHON_CMD="$PYTHON_CMD $FORCE"
  fi

  # Execute command and append stdout and stderr to single log file
  echo "Executing command: $PYTHON_CMD" | tee -a "$OUTPUT_LOG_FILE"
  eval $PYTHON_CMD >> "$OUTPUT_LOG_FILE" 2>&1

  echo "Completed RACE benchmark test for target model: $TARGET_MODEL"
  echo -e "\n========== RACE test completed for $TARGET_MODEL ==========\n" >> "$OUTPUT_LOG_FILE"
  
  # --- Phase 2: Citation Evaluation ---
  echo "==== Phase 2: Running FACT Evaluation for $TARGET_MODEL ====" | tee -a "$OUTPUT_LOG_FILE"

  # Create citation output directory if it doesn't exist
  CITATION_OUTPUT="$OUTPUT_DIR/fact/$TARGET_MODEL"
  RAW_DATA_PATH="$RAW_DATA_DIR/$TARGET_MODEL.jsonl"

  echo "Validate citations for $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
  python -u -m utils.validate --raw_data_path $CITATION_OUTPUT/scraped.jsonl --output_path $CITATION_OUTPUT/validated.jsonl --query_data_path $QUERY_DATA_PATH --n_total_process $N_TOTAL_PROCESS

  echo "Collecting statistics for $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
  python -u -m utils.stat --input_path $CITATION_OUTPUT/validated.jsonl --output_path $CITATION_OUTPUT/fact_result.txt

  echo "Completed FACT benchmark test for target model: $TARGET_MODEL"
  echo -e "\n========== FACT test completed for $TARGET_MODEL ==========\n" >> "$OUTPUT_LOG_FILE"
  echo "--------------------------------------------------"
done

echo "All benchmark tests completed. Individual model logs saved as output_<model_name>.log"
