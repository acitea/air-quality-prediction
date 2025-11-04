#!/bin/bash
# Script to update docs folder with latest training outputs

set -e

echo "Updating docs with latest training outputs..."

# Ensure directories exist
mkdir -p docs/outputs

# Copy latest results
cp outputs/latest_results.json docs/outputs/
cp outputs/predictions.png docs/outputs/

echo "Docs updated successfully!"
