#!/bin/bash
set -e

# Script to build static UI files for Python package distribution
# Usage: ./scripts/build_ui.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "===== Building DR-Agent UI for Python ====="

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found in $PROJECT_DIR"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build and export Next.js application
echo "Building Next.js application..."
npm run build

# Copy static files to Python package
SOURCE_DIR="./out"
TARGET_DIR="./python/dr_agent_ui/static"

echo "Copying static files to Python package..."
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: out/ directory does not exist. Build may have failed."
    exit 1
fi

# Remove old static files
if [ -d "$TARGET_DIR" ]; then
    echo "Removing old static files..."
    rm -rf "$TARGET_DIR"
fi

# Copy new static files
echo "Copying files from $SOURCE_DIR to $TARGET_DIR..."
cp -r "$SOURCE_DIR" "$TARGET_DIR"

# Verify output
if [ -f "$TARGET_DIR/index.html" ]; then
    echo "===== Successfully built dr-agent-ui! ====="
    echo ""
    echo "Static files are available at: $TARGET_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Start the server: cd ../agent && python workflows/your_workflow.py serve"
    echo "  2. Access UI at: http://localhost:8080"
    echo ""
    echo "To publish to PyPI:"
    echo "  ./scripts/publish_to_pypi.sh"
else
    echo "Error: Build failed - index.html not found"
    exit 1
fi



