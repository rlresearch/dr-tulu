#!/bin/bash
set -e

# Script to build and publish dr-agent-ui to PyPI
# Usage: 
#   ./scripts/publish_to_pypi.sh        # Publish to PyPI
#   ./scripts/publish_to_pypi.sh test   # Publish to TestPyPI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="$PROJECT_DIR/python"

cd "$PYTHON_DIR"

echo "===== Publishing dr-agent-ui to PyPI ====="

# Check if we're publishing to TestPyPI or PyPI
if [ "$1" == "test" ]; then
    REPO="testpypi"
    echo "Publishing to TestPyPI..."
else
    REPO="pypi"
    echo "Publishing to PyPI..."
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info dr_agent_ui.egg-info/

# Build the package
echo "Building package..."
python -m build

# Check the package
echo "Checking package with twine..."
python -m twine check dist/*

# Upload to PyPI
if [ "$REPO" == "testpypi" ]; then
    echo "Uploading to TestPyPI..."
    python -m twine upload --repository testpypi dist/*
else
    echo "Uploading to PyPI..."
    python -m twine upload dist/*
fi

echo "===== Successfully published dr-agent-ui! ====="



