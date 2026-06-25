#!/bin/bash

# --- Bash Progress Spinner ---
spinner() {
    local pid=$1
    local desc="$2"
    local log_file="$3"
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local start_time=$(date +%s)
    
    tput civis
    while kill -0 $pid 2>/dev/null; do
        local temp=${spinstr#?}
        local char=${spinstr%"$temp"}
        local elapsed=$(( $(date +%s) - start_time ))
        printf "\r%c %s | Elapsed time: %ds " "$char" "$desc" "$elapsed"
        spinstr=$temp$char
        sleep $delay
    done
    tput cnorm
    
    wait $pid
    local status=$?
    local elapsed=$(( $(date +%s) - start_time ))
    
    if [ $status -eq 0 ]; then
        printf "\r✅ %s | Completed in %ds                   \n" "$desc" "$elapsed"
    else
        printf "\r❌ %s | FAILED in %ds. Check %s for details.\n" "$desc" "$elapsed" "$log_file"
        cat "$log_file"
        exit $status
    fi
}

# --- Robust Internal Shape Checkers ---
check_onnx_resolution() {
    python3 -c "
import sys
import onnx
try:
    # load_external_data=False saves RAM since we only need the graph metadata
    model = onnx.load('$1', load_external_data=False) 
    dim = model.graph.input[0].type.tensor_type.shape.dim
    height, width = dim[2].dim_value, dim[3].dim_value
    if height == $2 and width == $2:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    print(f'\n[Check Error] Failed to read ONNX shape: {e}')
    sys.exit(1)
"
}

check_engine_resolution() {
    python3 -c "
import sys
import tensorrt as trt
try:
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    with open('$1', 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine:
            shape = engine.get_tensor_shape('images')
            if shape[2] == $2 and shape[3] == $2:
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            sys.exit(1)
except Exception as e:
    print(f'\n[Check Error] Failed to read Engine shape: {e}')
    sys.exit(1)
"
}

echo "=========================================="
echo " SAM3 TensorRT Docker Initialization      "
echo "=========================================="

# Calculate viable resolution (must be multiple of 14)
VIABLE_RES=$(python3 -c "print(round($RESOLUTION / 14) * 14)")
echo "ℹ️  Target Resolution: $RESOLUTION (Adjusted to viable TRT patch size: $VIABLE_RES)"

# 1. Clone or Pull Repository safely
if [ ! -d "/home/ubuntu/SAM3-TENSORRT-PYTHON/.git" ]; then
    git clone https://github.com/Kishan200308/SAM3-TENSORRT-PYTHON.git /home/ubuntu/temp_repo > git_clone.log 2>&1 &
    spinner $! "Cloning SAM3-TENSORRT-PYTHON Repository" "git_clone.log"
    
    mkdir -p /home/ubuntu/SAM3-TENSORRT-PYTHON
    cp -a /home/ubuntu/temp_repo/. /home/ubuntu/SAM3-TENSORRT-PYTHON/
    rm -rf /home/ubuntu/temp_repo
else
    cd /home/ubuntu/SAM3-TENSORRT-PYTHON
    git pull > ../git_pull.log 2>&1 &
    spinner $! "Pulling latest repository updates" "../git_pull.log"
    cd /home/ubuntu
fi

cd /home/ubuntu/SAM3-TENSORRT-PYTHON
mkdir -p sam3 Onnx-Models Engines

# 2. Download Model using HuggingFace CLI
hf download facebook/sam3 --local-dir /home/ubuntu/SAM3-TENSORRT-PYTHON/sam3 --token "$HF_TOKEN" > hf_download.log 2>&1 &
spinner $! "Downloading/Verifying SAM3 from HuggingFace" "hf_download.log"

# 3. Export ONNX Models (Check files + exact resolution match)
ONNX_OK=false
if [ -f "Onnx-Models/vision-encoder.onnx" ] && [ -f "Onnx-Models/text-encoder.onnx" ] && \
   [ -f "Onnx-Models/geometry-encoder.onnx" ] && [ -f "Onnx-Models/decoder.onnx" ]; then
    
    # Run the robust Python check
    if check_onnx_resolution "Onnx-Models/vision-encoder.onnx" "$VIABLE_RES"; then
        ONNX_OK=true
    fi
fi

if [ "$ONNX_OK" = true ]; then
    echo "✅ ONNX models already exist with matching resolution (${VIABLE_RES}x${VIABLE_RES}). Skipping export."
else
    echo "⚠️  ONNX models missing or resolution mismatch. Exporting for ${VIABLE_RES}x${VIABLE_RES}..."
    yes "yes" | python3 SAM3_PyTorch_To_Onnx.py \
        --all \
        --model-path "/home/ubuntu/SAM3-TENSORRT-PYTHON/sam3" \
        --output-dir "/home/ubuntu/SAM3-TENSORRT-PYTHON/Onnx-Models" \
        --device cuda \
        --size "$RESOLUTION" > onnx_export.log 2>&1 &
    spinner $! "Exporting ONNX Models" "onnx_export.log"
fi

# 4. Build TensorRT Engines (Check files + exact resolution match)
ENGINE_OK=false
if [ -f "Engines/vision-encoder.engine" ] && [ -f "Engines/text-encoder.engine" ] && \
   [ -f "Engines/geometry-encoder.engine" ] && [ -f "Engines/decoder.engine" ]; then
    
    # Run the robust Python check
    if check_engine_resolution "Engines/vision-encoder.engine" "$VIABLE_RES"; then
        ENGINE_OK=true
    fi
fi

if [ "$ENGINE_OK" = true ]; then
    echo "✅ TensorRT Engines already exist with matching resolution (${VIABLE_RES}x${VIABLE_RES}). Skipping build."
else
    echo "⚠️  Engines missing or resolution mismatch. Building TensorRT Engines..."
    python3 Build_Engines.py \
        --onnx "/home/ubuntu/SAM3-TENSORRT-PYTHON/Onnx-Models" \
        --engine "/home/ubuntu/SAM3-TENSORRT-PYTHON/Engines" > engine_build.log 2>&1 &
    spinner $! "Building TensorRT Engines" "engine_build.log"
fi

# 5. Start Gradio UI
echo "=========================================="
echo "🚀 Launching Gradio Web UI on Port $GRADIO_PORT"
echo "=========================================="

export GRADIO_SERVER_PORT=$GRADIO_PORT
export GRADIO_SERVER_NAME="0.0.0.0"

# Run UI in the foreground
python3 UI.py