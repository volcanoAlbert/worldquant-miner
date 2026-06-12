#!/bin/bash

CREDENTIALS_FILE="${CREDENTIALS_FILE:-./credentials.txt}"
if [ ! -f "$CREDENTIALS_FILE" ] && [ -f "../../credentials.txt" ]; then
    CREDENTIALS_FILE="../../credentials.txt"
elif [ ! -f "$CREDENTIALS_FILE" ] && [ -f "./credential.txt" ]; then
    CREDENTIALS_FILE="./credential.txt"
fi

if [ -f "$CREDENTIALS_FILE" ] && grep -Eq '^(base_url|OPENAI_BASE_URL|LLM_BASE_URL)=' "$CREDENTIALS_FILE" && grep -Eq '^(OPENAI_API_KEY|LLM_API_KEY|API_KEY)=' "$CREDENTIALS_FILE"; then
    echo "Remote LLM config detected in $CREDENTIALS_FILE; skipping local Ollama startup."
    MODEL_NAME="${LLM_MODEL:-remote-llm}"
else
    # Check GPU availability
    echo "Checking GPU availability..."
    nvidia-smi || echo "No GPU detected, will use CPU"

    # Start Ollama in the background with logging to file
    mkdir -p /app/logs
    ollama serve > /app/logs/ollama.log 2>&1 &

    # Wait for Ollama to be ready
    echo "Waiting for Ollama to start..."
    sleep 10

    # Pull DeepSeek-R1 model optimized for RTX A4000 (16GB VRAM)
    echo "Pulling DeepSeek-R1 model for RTX A4000..."
    if ollama pull deepseek-r1:8b; then
        echo "Using deepseek-r1:8b model (RTX A4000 optimized - 5.2GB)"
        MODEL_NAME="deepseek-r1:8b"
    elif ollama pull deepseek-r1:7b; then
        echo "Using deepseek-r1:7b model (fallback - 4.7GB)"
        MODEL_NAME="deepseek-r1:7b"
    elif ollama pull deepseek-r1:1.5b; then
        echo "Using deepseek-r1:1.5b model (fallback - 1.1GB)"
        MODEL_NAME="deepseek-r1:1.5b"
    else
        echo "DeepSeek-R1 models failed, trying llama3.2:3b..."
        if ollama pull llama3:3b; then
            echo "Using llama3:3b model (fallback)"
            MODEL_NAME="llama3:3b"
        else
            echo "Using default model"
            MODEL_NAME="llama2:7b"
        fi
    fi

    # Verify the model is available
    echo "Verifying model availability..."
    if ollama list | grep -q "$MODEL_NAME"; then
        echo "✅ Model $MODEL_NAME is available"
    else
        echo "❌ Model $MODEL_NAME not found, using llama3.2:3b"
        MODEL_NAME="llama3.2:3b"
    fi
fi

# Start the alpha submitter in the background (daily submission with hopeful alphas check)
echo "Starting improved alpha submitter in background (daily submission with 50+ hopeful alphas check)..."
python improved_alpha_submitter.py --credentials "$CREDENTIALS_FILE" --use-hopeful-file --min-hopeful-count 50 --interval-hours 24 --batch-size 3 --log-level INFO &

# Start the main application with Ollama integration (concurrent mode)
echo "Starting alpha orchestrator using $MODEL_NAME in concurrent mode..."
python alpha_orchestrator.py --credentials "$CREDENTIALS_FILE" --ollama-url http://localhost:11434 --ollama-model "$MODEL_NAME" --mode continuous --mining-interval 6 --batch-size 3 --max-concurrent 3
