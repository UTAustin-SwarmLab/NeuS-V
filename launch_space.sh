#!/bin/bash

# Start vLLM server in background
./vllm_serve.sh

# Wait briefly to ensure vLLM is up before Gradio tries to connect
sleep 60

# Start Gradio app
# python3 evaluate_demo.py
