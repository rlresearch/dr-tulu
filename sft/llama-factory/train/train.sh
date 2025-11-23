#!/bin/bash
# Load the .env file 
if [ -f .env ]; then
    source .env
    export $(grep -v '^#' .env | xargs)
fi

export WANDB_PROJECT="rl-research"
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file.yaml>"
    exit 1
fi

# Single node training is straightforward
llamafactory-cli train "$1"