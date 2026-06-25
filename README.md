# SAM3-TENSORRT-PYTHON — SAM3 inference pipeline with TensorRT (FP16)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSES/LICENSE) 
[![License: SAM](https://img.shields.io/badge/License-SAM-blue.svg)](LICENSES/LICENSE-SAM)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/Kishan200308/7c26776b0ff3a48b1e7c6e390885def6/sam3-tensorrt-colab-demo.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/Kishan200308/SAM3-TENSORRT-PYTHON?style=social)](https://github.com/Kishan200308/SAM3-TENSORRT-PYTHON/stargazers)


## SAM3 TensorRT Pipeline

Current: 2026-02-20 — Repository status: actively maintained. Keywords: Current, SAM3, TensorRT, FP16, TensorRT-Python, Gradio, ONNX, NVIDIA

This project provides a complete pipeline to run **SAM3** (Segment Anything Model 3) with **TensorRT**:

- **System audit** for CUDA / TensorRT readiness (`Check.py`)
- **ONNX export** of the SAM3 submodules (`SAM3_PyTorch_To_Onnx.py`)
- **TensorRT engine building** from ONNX (`Build_Engines.py`)
- **High‑performance inference** with text prompts (`SAM3_TensorRT_Inference.py`)
- **Interactive web UI** for easy testing (`UI.py`)

The workflow is designed around FP16 TensorRT engines with dynamic shapes and explicit batch, supporting both **bounding box detection** and **mask segmentation** modes.

![SAM3 UI](Workflow%20Photos/SAM3_UI.png)
![WORKFLOW](Workflow%20Photos/WorkFlow.png)

---

## 🚀 Performance Benchmarks

By migrating from native PyTorch to TensorRT (FP16), this pipeline delivers massive efficiency gains.

| Metric | Original PyTorch | TensorRT (FP16) | Improvement |
| :--- | :--- | :--- | :--- |
| **VRAM Usage** | ~6-7 GB | **~3.1 GB** | **~52% Reduction** |
| **Inference Time** (T4 GPU) | ~1.6 sec | **~0.5 sec** | **~3.2x Speedup** |

*Note: Benchmarks tested on NVIDIA T4 GPU. Performance may vary based on hardware.*

---

## Quick Start

### Check System Readiness

Before starting, verify your system is properly configured:

```bash
python3 Check.py
```

This will check CUDA, TensorRT, and all required dependencies.

---

## 1. Environment Setup

### Python Packages

Install required Python packages:

```bash
pip install -r requirements.txt --upgrade
```

### TensorRT Installation

**MAKE SURE TENSORRT IS INSTALLED AND ADDED TO PATH**

For Linux, install TensorRT 10.14.1.48 with CUDA 12.9:

```bash
sudo apt-get install -y --allow-downgrades \
    libnvinfer10=10.14.1.48-1+cuda12.9 \
    libnvinfer-bin=10.14.1.48-1+cuda12.9 \
    libnvinfer-dispatch10=10.14.1.48-1+cuda12.9 \
    libnvinfer-lean10=10.14.1.48-1+cuda12.9 \
    libnvinfer-plugin10=10.14.1.48-1+cuda12.9 \
    libnvinfer-vc-plugin10=10.14.1.48-1+cuda12.9 \
    libnvonnxparsers10=10.14.1.48-1+cuda12.9
```

Add TensorRT to your PATH:

```bash
echo 'export PATH="/usr/src/tensorrt/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/tensorrt:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> ~/.bashrc
source ~/.bashrc
```

**Verify installation:**

```bash
python3 Check.py
```
---
## 2. Download or Export ONNX Models

You have two options: **download pre‑exported ONNX** or **export from PyTorch yourself**.

### 2.1. Download pre‑exported ONNX (recommended)

Download prebuilt ONNX models for 1008 resolution:

```bash
hf download --local-dir "Onnx-Models" kishanstar2003/SAM3_ONNX_FP16
```

This will create an `Onnx-Models` directory containing:

- `vision-encoder.onnx`
- `text-encoder.onnx`
- `geometry-encoder.onnx`
- `decoder.onnx`
- `tokenizer.json` (auto copied to the engines directory)

### 2.2. Export ONNX from the SAM3 PyTorch model (Manual)

If you want to manually export ONNX models:

1. Download the original SAM3 PyTorch model:

```bash
hf download facebook/sam3 --local-dir sam3
```

2. Patch the transformers code:

```bash
python3 Patch_Sam3_Interp_Rope.py
```

3. Export to ONNX:

```bash
python3 SAM3_PyTorch_To_Onnx.py --all --model-path "sam3" --output-dir "Onnx-Models" --device cuda --size 1008
```

Key points:

- The script exports four modules via wrappers:
  - `VisionEncoderWrapper` → `vision-encoder.onnx`
  - `TextEncoderWrapper` → `text-encoder.onnx`
  - `GeometryEncoderWrapper` → `geometry-encoder.onnx`
  - `DecoderWrapper` → `decoder.onnx`
- All exports use **opset 20** and **dynamic batch / prompt dimensions**, compatible with TensorRT.
- The `--size 1008` parameter sets the resolution for the exported models.

**Changing Resolution:**

If you want to use a different resolution (e.g., 644), simply change the `--size` parameter when exporting ONNX:

```bash
python3 SAM3_PyTorch_To_Onnx.py --all --model-path "sam3" --output-dir "Onnx-Models" --device cuda --size 644
```

Then rebuild the engines using the same command as before:

```bash
python3 Build_Engines.py --onnx "Onnx-Models" --engine "Engines"
```

The engine building command remains the same regardless of resolution.
---
## 3. Build TensorRT Engines

Once you have the ONNX models in `Onnx-Models`, build TensorRT engines using `Build_Engines.py`.

```bash
python3 Build_Engines.py --onnx "Onnx-Models" --engine "Engines"
```

Arguments:

- `--base` (optional): base directory (default: current working directory).
- `--onnx`: directory containing `.onnx` models (default: `BASE/Onnx-Models`).
- `--engine`: output directory for `.engine` files (default: `BASE/Engines`).

The script:

- Runs `trtexec` with **FP16** and appropriate **min/opt/max shapes** for each module:
  - `vision-encoder`
  - `text-encoder`
  - `geometry-encoder`
  - `decoder`
- Skips engines that already exist.
---
## 4. Verify System & TensorRT Installation

Use `Check.py` to audit your environment:

```bash
python3 Check.py
```

It reports:

- **GPU hardware and driver** via `nvidia-smi`
- **NVCC** presence and version
- **PyTorch**, CUDA version, and **ONNX Runtime**
- **TensorRT Python bindings** and builder creation
- **`trtexec`** availability
- Available **ONNX Runtime providers** (CUDA / TensorRT, etc.)

Run this once after setup to confirm everything is wired correctly.
---
## 5. Run TensorRT Inference

With engines and tokenizer in place, you can run inference in two ways: **command line** or **interactive web UI**.

### 5.1. Command Line Inference

Run the end‑to‑end inference script:

**Bounding Box Detection Mode:**
```bash
python3 SAM3_TensorRT_Inference.py --input "Assets/Test.jpg" --prompt "person,bus" --conf 0.8 --output Results/Result-TextPrompt-BoundingBox-Mode.jpg --models "Engines"
```

**Mask Segmentation Mode:**
```bash
python3 SAM3_TensorRT_Inference.py --input "Assets/Test.jpg" --prompt "person,bus" --conf 0.8 --output Results/Result-TextPrompt-MaskSegmentation-Mode.jpg --models "Engines" --segment
```

**With Bounding Box Prompts:**
```bash
python3 SAM3_TensorRT_Inference.py --input "Assets/Test.jpg" --boxes "pos:414,655,53,115" --conf 0.7 --output Results/Result-BoxPrompt-BoundingBox-Mode.jpg --models "Engines"
```

**With Bounding Box Prompts (Mask Segmentation Mode):**
```bash
python3 SAM3_TensorRT_Inference.py --input "Assets/Test.jpg" --boxes "pos:414,655,53,115" --conf 0.7 --output Results/Result-BoxPrompt-MaskSegmentation-Mode.jpg --models "Engines" --segment
```

**With Bounding Box Prompts (Group Similar):**
```bash
python3 SAM3_TensorRT_Inference.py --input "Assets/Test.jpg" --boxes "pos:414,655,53,115" --conf 0.7 --output Results/Result-BoxPrompt-GroupSimilar-BoundingBox-Mode.jpg --models "Engines" --group-similar
```

**With Bounding Box Prompts (Group Similar, Mask Segmentation Mode):**
```bash
python3 SAM3_TensorRT_Inference.py --input "Assets/Test.jpg" --boxes "pos:414,655,53,115" --conf 0.7 --output Results/Result-BoxPrompt-GroupSimilar-MaskSegmentation-Mode.jpg --models "Engines" --segment --group-similar
```

Arguments:

- `--input`: path to input image file.
- `--prompt`: text prompt for detection. Supports multiple prompts separated by commas (e.g., "person,bus,car").
- `--boxes`: box prompts in `xywh` format (`x` and `y` are the top-left coordinates, `w` is width, `h` is height). Prefix with `pos:` for positive prompts and `neg:` for negative prompts. Separate multiple boxes with a semicolon, and coordinates with commas (e.g., `pos:x,y,w,h;neg:x,y,w,h`).
- `--group-similar`: (optional) when using box prompts, detects all objects similar to the marked box. If omitted, detects only the specific object at that position.
- `--conf`: confidence threshold (0.0–1.0) applied on box scores.
- `--output`: path to save the annotated image.
- `--models`: directory containing `.engine` files and `tokenizer.json` (typically `Engines`).
- `--segment`: (optional) enable mask segmentation mode. If omitted, uses bounding box detection.

### 5.2. Interactive Web UI

For easier testing and experimentation, use the Gradio web interface:

```bash
python3 UI.py
```

This launches a web interface where you can:
- Upload images directly
- Enter text prompts interactively (supports multiple prompts separated by commas, e.g., "person,bus")
- Draw rectangles directly on the image to pick bounding box prompts using the visual picker
- Toggle "Group Similar" to detect all objects similar to the marked box(es)
- Adjust confidence thresholds with sliders
- Toggle between bounding box and segmentation modes
- View results instantly with performance metrics

The UI automatically loads the TensorRT engines from the `Engines` directory and provides real-time inference.

What the script does:

- Wraps each engine with `TRTModule` for efficient execution using CuPy arrays.
- Preprocesses the input image:
  - Resize to auto-detected engine resolution (default `1008 × 1008`)
  - Normalize to [-1, 1]
- Runs:
  - Vision encoder → FPN features + positional encodings
  - Text encoder → token embeddings + masks (via `tokenizers` and `tokenizer.json`)
  - Geometry encoder (if box prompts are provided) → box features + masks
  - Decoder → predicted boxes, logits, presence logits, and masks
- Computes combined scores from logits and presence logits, filters by `--conf`, denormalizes boxes, and draws them onto the original image.
- **Bounding Box Mode**: Draws rectangular boxes around detected objects
- **Segmentation Mode**: Generates and overlays pixel-accurate masks for detected objects

Output:

- An image with bounding boxes/masks and scores, saved to `--output`.
---
## 🐳 Docker Image Usage

> [!IMPORTANT]
> For the prebuilt Docker image to work, your host system's **CUDA device driver version MUST be at least CUDA 12.6**, because the PyTorch version inside the container requires CUDA 12.6.
> 
> For any older CUDA versions, you can edit `requirements.txt` and build your own Docker image, or run the project on your standalone system.

You can run the SAM3 TensorRT pipeline using the prebuilt Docker image.

### 🚀 Run Docker Container

```bash
# 1. Define the port in your terminal
export GRADIO_PORT=8080

# 2. Run the command using the variable
docker run --gpus all \
  --ipc=host \
  -it \
  -p $GRADIO_PORT:$GRADIO_PORT \
  -v $(pwd)/SAM3_TensorRT_Files:/home/ubuntu/SAM3-TENSORRT-PYTHON \
  -e HF_TOKEN="hf_************************" \
  -e GRADIO_PORT=$GRADIO_PORT \
  -e RESOLUTION=1008 \
  sam3_image_graido_demo:latest
```

### Environment Variables & Parameters

- **`HF_TOKEN`**: Replace with the Hugging Face token of an account with SAM3 HF access.
- **`GRADIO_PORT`**: Controls which port you want to launch the Gradio web page on your machine.
- **`RESOLUTION`**: The resolution at which the model will be exported. If the provided resolution is not viable (meaning not a multiple of 14), then it will automatically choose the closest possible resolution and proceed with it.

### Workflow

After launching, the container will:
1. Download the latest code.
2. Download SAM3 from Hugging Face.
3. Prepare ONNX for the resolution you specified (if the models don't exist, or if they exist and the resolution does not match).
4. Build TensorRT engines (if they don't exist, or if they exist and the resolution of the TensorRT engine doesn't match the requested `RESOLUTION`).

*Note: It skips ONNX and TensorRT engine building if they already match the resolution passed by the user.*

After the setup is complete, the Gradio UI will be launched on the port you specified.


⚠️ Disclaimer
This project provides high-performance optimizations for SAM3. Note that TensorRT engine performance and stability are highly dependent on specific hardware (GPU architecture) and software (CUDA/TensorRT versions). Use these optimization scripts at your own risk.

## ⚖️ Licensing & Acknowledgments

This repository contains both original code and derivative works of Meta's Segment Anything Model 3 (SAM 3).

1. **Source Code**: All Python scripts (`.py`), conversion logic, and TensorRT wrappers provided in this repository are licensed under the [MIT License](LICENSES/LICENSE).
2. **SAM 3 Materials & Derivatives**: The underlying model weights, architectures, and **all exported ONNX/TensorRT engines** generated by these scripts are subject to the **[Meta SAM License](LICENSES/LICENSE-SAM)**. 

### Research Acknowledgment
Per the **SAM License (Section 1.b.ii)**, this project acknowledges the use of **SAM Materials** distributed by Meta Platforms, Inc. for the development and optimization of this TensorRT inference pipeline.

## 🤝 Credits

- [jamjamjon/usls](https://github.com/jamjamjon/usls/tree/main/scripts/sam3-image)
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
