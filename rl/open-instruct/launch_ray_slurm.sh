#!/bin/bash
# Ray setup script for SLURM multi-node jobs
# Usage: Source this script before running your Ray application
# Example: source scripts/slurm_ray_setup.sh

set -e

# Get SLURM environment variables
# SLURM_PROCID: Process ID (0 for head node, 1+ for workers)
# SLURM_NODEID: Node ID within the job
# SLURM_JOB_NODELIST: List of nodes allocated to the job
# SLURM_STEP_NODELIST: List of nodes in this step

# Determine if this is the head node (rank 0)
# For SLURM, we can use SLURM_PROCID or check if we're the first node
if [ -z "$SLURM_PROCID" ]; then
    echo "Warning: SLURM_PROCID not set. Assuming single-node mode."
    RAY_HEAD_NODE=true
else
    RAY_HEAD_NODE=$([ "$SLURM_PROCID" == "0" ] && echo "true" || echo "false")
fi

# Get the head node hostname/IP
# Method 1: Use SLURM_JOB_NODELIST and extract first node
if [ -n "$SLURM_JOB_NODELIST" ]; then
    # Convert node list to array and get first node
    # Handles formats like: node001,node002 or node[001-002]
    HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
elif [ -n "$SLURM_STEP_NODELIST" ]; then
    HEAD_NODE=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -n1)
else
    # Fallback: use current hostname if we can't determine
    HEAD_NODE=$(hostname)
    echo "Warning: Could not determine head node from SLURM vars, using current hostname: $HEAD_NODE"
fi

# Get IP address of head node
HEAD_NODE_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}' | head -n1)
if [ -z "$HEAD_NODE_IP" ]; then
    HEAD_NODE_IP="$HEAD_NODE"
fi

RAY_NODE_PORT=8888
RAY_DASHBOARD_PORT=8265

# Stop any existing Ray instance
ray stop --force 2>/dev/null || true

if [ "$RAY_HEAD_NODE" == "true" ]; then
    echo "=========================================="
    echo "Starting Ray head node"
    echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"
    echo "Ray port: $RAY_NODE_PORT"
    echo "Dashboard port: $RAY_DASHBOARD_PORT"
    echo "=========================================="
    
    # Start Ray head node
    ray start --head \
        --port=$RAY_NODE_PORT \
        --dashboard-host=0.0.0.0
else
    echo "=========================================="
    echo "Starting Ray worker node"
    echo "Worker rank: $SLURM_PROCID"
    echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"
    echo "Connecting to: ${HEAD_NODE_IP}:${RAY_NODE_PORT}"
    echo "=========================================="
    
    # Set Ray address to head node
    export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_NODE_PORT}"
    
    # Start Ray worker node
    ray start --address="$RAY_ADDRESS" \
        --dashboard-host=0.0.0.0 
    
    echo "Ray worker node started successfully"
    echo "Connected to: $RAY_ADDRESS"
    
    # Setup cleanup trap for graceful shutdown
    cleanup() {
        echo "[ray_worker] Cleanup: stopping Ray worker"
        ray stop --force >/dev/null 2>&1 || true
        trap - TERM INT HUP EXIT
        exit 0
    }
    
    trap cleanup TERM INT HUP EXIT
    
    # Worker nodes should wait/block here to keep Ray running
    # They should NOT proceed to run the training command
    echo "[ray_worker] Worker node ready. Waiting for head node to complete training..."
    echo "[ray_worker] This node will keep Ray worker running until job completes."
    
    # Monitor head node and keep worker alive
    while true; do
        if ! ray status --address="$RAY_ADDRESS" &>/dev/null; then
            echo "[ray_worker] Head node is unreachable. Stopping worker."
            cleanup
        fi
        sleep 10
    done
fi