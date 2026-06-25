# Standard library imports
import argparse
import multiprocessing
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Third-party imports
import onnx

def _run_spinner(description, stop_event):
    """This runs in a completely separate process to avoid PyTorch/subprocess locking it."""
    start_time = time.time()
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    idx = 0
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r{spinner[idx]} {description} | Elapsed time: {elapsed:.1f}s")
        sys.stdout.flush()
        idx = (idx + 1) % len(spinner)
        time.sleep(0.1)

class ProgressTimer:
    """A multiprocessing background timer that prints a spinner and elapsed time."""
    def __init__(self, description):
        self.description = description
        self._stop_event = multiprocessing.Event()
        self._process = None
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self._process = multiprocessing.Process(
            target=_run_spinner, 
            args=(self.description, self._stop_event)
        )
        self._process.start()

    def stop(self):
        self._stop_event.set()
        if self._process:
            self._process.join()
        elapsed = time.time() - self.start_time
        # Clear the line and print the final completion message
        sys.stdout.write(f"\r✅ {self.description} | Completed in {elapsed:.1f}s" + " " * 15 + "\n")
        sys.stdout.flush()


def get_vision_resolution(onnx_path: Path):
    print(f"🔍 Analyzing {onnx_path.name} to detect shapes...")
    model = onnx.load(str(onnx_path))
    
    # Get the first input tensor (should be 'images')
    input_tensor = model.graph.input[0]
    dims = input_tensor.type.tensor_type.shape.dim
    
    # dims[0] is dynamic batch, dims[1] is channels (3), dims[2] is H, dims[3] is W
    height = dims[2].dim_value
    width = dims[3].dim_value
    
    if height <= 0 or width <= 0:
        raise ValueError("Could not detect static height/width from ONNX model.")
        
    print(f"✅ Detected Vision Resolution: {height}x{width}\n")
    return height, width

def run_trtexec(trtexec_path: str, onnx_file: Path, engine_file: Path, min_shapes: str, opt_shapes: str, max_shapes: str):
    cmd = [
        trtexec_path,
        "--fp16",
        f"--onnx={onnx_file}",
        f"--saveEngine={engine_file}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}"
    ]
    
    timer = ProgressTimer(f"Building {engine_file.name}")
    timer.start()
    
    try:
        # Capture output so trtexec logs don't destroy the spinner UI
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        timer.stop()
    except subprocess.CalledProcessError as e:
        timer.stop()
        print(f"\n❌ Error building {engine_file.name}. trtexec failed.")
        print("\n--- trtexec error output ---")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        timer.stop()
        print(f"\n❌ Could not find trtexec at '{trtexec_path}'. Please ensure TensorRT is installed or provide the correct path using --trtexec.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Auto-build SAM3 TensorRT Engines")
    parser.add_argument("--onnx", type=str, required=True, help="Directory containing ONNX models")
    parser.add_argument("--engine", type=str, required=True, help="Directory to save TRT engines")
    parser.add_argument("--trtexec", type=str, default="trtexec", help="Path to trtexec binary (default: 'trtexec')")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx)
    engine_dir = Path(args.engine)
    engine_dir.mkdir(parents=True, exist_ok=True)

    vision_onnx = onnx_dir / "vision-encoder.onnx"
    text_onnx = onnx_dir / "text-encoder.onnx"
    geometry_onnx = onnx_dir / "geometry-encoder.onnx"
    decoder_onnx = onnx_dir / "decoder.onnx"

    # 1. Check if ONNX models exist
    for p in [vision_onnx, text_onnx, geometry_onnx, decoder_onnx]:
        if not p.exists():
            print(f"❌ Missing required ONNX file: {p}")
            sys.exit(1)

    # 2. Detect shapes
    h, w = get_vision_resolution(vision_onnx)
    
    # Calculate decoder patch shapes based on the multiple of 14 rule
    patch_h, patch_w = h // 14, w // 14
    h2, w2 = patch_h, patch_w
    h1, w1 = patch_h * 2, patch_w * 2
    h0, w0 = patch_h * 4, patch_w * 4

    # 3. Build Vision Encoder (Optimized for Batch Size 1)
    run_trtexec(
        args.trtexec,
        vision_onnx,
        engine_dir / "vision-encoder.engine",
        min_shapes=f"images:1x3x{h}x{w}",
        opt_shapes=f"images:1x3x{h}x{w}",
        max_shapes=f"images:1x3x{h}x{w}"
    )

    # 4. Build Text Encoder (Optimized for Batch Size 1)
    run_trtexec(
        args.trtexec,
        text_onnx,
        engine_dir / "text-encoder.engine",
        min_shapes="input_ids:1x32,attention_mask:1x32",
        opt_shapes="input_ids:1x32,attention_mask:1x32",
        max_shapes="input_ids:1x32,attention_mask:1x32"
    )

    # 5. Build Geometry Encoder (Optimized for Batch Size 1, Dynamic Boxes preserved)
    geom_min = f"input_boxes:1x1x4,input_boxes_labels:1x1,fpn_feat_2:1x256x{h2}x{w2},fpn_pos_2:1x256x{h2}x{w2}"
    geom_opt = f"input_boxes:1x8x4,input_boxes_labels:1x8,fpn_feat_2:1x256x{h2}x{w2},fpn_pos_2:1x256x{h2}x{w2}"
    geom_max = f"input_boxes:1x20x4,input_boxes_labels:1x20,fpn_feat_2:1x256x{h2}x{w2},fpn_pos_2:1x256x{h2}x{w2}"
    
    run_trtexec(
        args.trtexec,
        geometry_onnx,
        engine_dir / "geometry-encoder.engine",
        min_shapes=geom_min,
        opt_shapes=geom_opt,
        max_shapes=geom_max
    )

    # 6. Build Decoder (Optimized for Batch Size 1, Dynamic Prompts preserved)
    decoder_min = f"fpn_feat_0:1x256x{h0}x{w0},fpn_feat_1:1x256x{h1}x{w1},fpn_feat_2:1x256x{h2}x{w2},fpn_pos_2:1x256x{h2}x{w2},prompt_features:1x1x256,prompt_mask:1x1"
    decoder_opt = f"fpn_feat_0:1x256x{h0}x{w0},fpn_feat_1:1x256x{h1}x{w1},fpn_feat_2:1x256x{h2}x{w2},fpn_pos_2:1x256x{h2}x{w2},prompt_features:1x33x256,prompt_mask:1x33"
    decoder_max = f"fpn_feat_0:1x256x{h0}x{w0},fpn_feat_1:1x256x{h1}x{w1},fpn_feat_2:1x256x{h2}x{w2},fpn_pos_2:1x256x{h2}x{w2},prompt_features:1x60x256,prompt_mask:1x60"

    run_trtexec(
        args.trtexec,
        decoder_onnx,
        engine_dir / "decoder.engine",
        min_shapes=decoder_min,
        opt_shapes=decoder_opt,
        max_shapes=decoder_max
    )

    # 7. Copy Tokenizer
    tokenizer_src = onnx_dir / "tokenizer.json"
    tokenizer_dst = engine_dir / "tokenizer.json"
    
    if tokenizer_src.exists():
        print(f"📄 Copying {tokenizer_src.name} to {engine_dir.name}...")
        shutil.copy2(tokenizer_src, tokenizer_dst)
        print("✅ Tokenizer copied successfully.\n")
    else:
        print(f"⚠️ Warning: {tokenizer_src.name} not found in {onnx_dir.name}. You will need to copy it manually.\n")

    print("🎉 All engines built successfully!")

if __name__ == "__main__":
    main()