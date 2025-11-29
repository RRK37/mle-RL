#!/bin/bash
# Quick test script to verify the setup

echo "Testing main.py with AIDE agent on Titanic competition..."
echo ""
echo "Command: python main.py --config config.yaml"
echo ""

# Run with config file
python main.py --config config.yaml

echo ""
echo "If successful, check output/titanic/ for results"
