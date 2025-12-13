#!/bin/bash
# Test script for DS Serve API using curl
# This tests the API directly without Python dependencies

echo "Testing DS Serve API Integration"
echo "=================================================================================="

test_query() {
    local query="$1"
    local backend="${2:-diskann}"
    local n_docs="${3:-3}"
    local extra_params="$4"
    
    echo ""
    echo "=================================================================================="
    echo "Query: $query"
    echo "Backend: $backend"
    echo "n_docs: $n_docs"
    echo "=================================================================================="
    echo ""
    
    local payload="{\"query\": \"$query\", \"n_docs\": $n_docs, \"backend\": \"$backend\""
    if [ -n "$extra_params" ]; then
        payload="${payload}, ${extra_params}"
    fi
    payload="${payload}}"
    
    curl -s -X POST http://api.ds-serve.org:30888/search \
        -H "Content-Type: application/json" \
        -d "$payload" | python3 -m json.tool | head -100
    
    echo ""
    echo "---"
}

# Test Case 1: Default DiskANN
test_query "Tell me more about Albert Einstein" "diskann" 3

# Test Case 2: DiskANN with custom parameters
test_query "Tell me more about Nikola Tesla" "diskann" 3 '{"diskann_L": 500, "diskann_W": 8}'

# Test Case 3: IVFPQ backend
test_query "Explain the basics of quantum physics" "ivfpq" 2 '{"nprobe": 64}'

# Test Case 4: Another query
test_query "Who is Matei Zaharia at UC Berkeley and founder of Apache Spark" "diskann" 2

echo ""
echo "=================================================================================="
echo "All test queries completed!"
echo "=================================================================================="

