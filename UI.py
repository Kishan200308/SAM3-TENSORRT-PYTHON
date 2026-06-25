import os
import cv2
import gradio as gr
import tempfile
import numpy as np
import subprocess
import atexit
import base64
import io
import html
import time
import warnings
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional

# --- Suppress Starlette Deprecation Warning ---
warnings.filterwarnings("ignore", module="gradio.routes")

# Import the class from your existing file
try:
    from SAM3_TensorRT_Inference import Sam3Inference
except ImportError:
    print("Warning: Could not import 'Sam3Inference'. Ensure the file exists.")
    Sam3Inference = None

# --- Configuration ---
MODEL_DIR = Path("Engines")
DEFAULT_OUTPUT = ("0.00 ms", "0.00 GB (Total)")

# --- Global State ---
sam3_engine: Optional[Sam3Inference] = None
temp_files_registry = set()  

# --- Cleanup Handler ---
def cleanup_on_exit():
    if temp_files_registry:
        print(f"\n[CLEANUP] Removing {len(temp_files_registry)} temporary files...")
        for path in list(temp_files_registry):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        temp_files_registry.clear()
        print("[CLEANUP] Done.")

atexit.register(cleanup_on_exit)

def get_total_vram() -> float:
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return float(result.strip().split('\n')[0]) / 1024.0
    except Exception as e:
        print(f"Warning: Could not query VRAM: {e}")
        return 0.0

def load_model():
    global sam3_engine
    if sam3_engine is None:
        if not MODEL_DIR.exists():
            print(f"Error: Model directory '{MODEL_DIR}' not found.")
            return
        
        print(f"Loading SAM3 TensorRT Engine from {MODEL_DIR}...")
        try:
            sam3_engine = Sam3Inference(str(MODEL_DIR))
            print("Engine loaded successfully!")
        except Exception as e:
            print(f"Error loading engine: {e}")

def inference_wrapper(image: np.ndarray, prompt: str, box_prompt: str, conf: float, segment_mode: bool, group_similar: bool) -> Tuple[Optional[np.ndarray], str, str]:
    if image is None:
        return (None, *DEFAULT_OUTPUT)
    
    if sam3_engine is None:
        return (image, "Error: Model Not Loaded", "N/A")

    prompt_clean = prompt.strip() if prompt else ""
    box_clean = box_prompt.strip() if box_prompt else ""

    if not prompt_clean and not box_clean:
        return (image, "Error: Provide a Text Prompt OR draw a Box Prompt", "N/A")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        input_path = tmp_file.name
    output_path = input_path.replace(".jpg", "_out.jpg")

    temp_files_registry.add(input_path)
    temp_files_registry.add(output_path)

    try:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(input_path, img_bgr)

        metrics = sam3_engine.run(
            img_path=input_path, prompt=prompt_clean, box_str=box_clean,
            conf=conf, out_path=output_path, segment=segment_mode,
            group_similar=group_similar 
        )

        inf_time = float(metrics[0]) if isinstance(metrics, tuple) else float(metrics)
        total_vram = get_total_vram()
        
        time_str = f"{inf_time:.2f} ms"
        mem_str = f"{total_vram:.2f} GB (Total)"

        if os.path.exists(output_path):
            result_bgr = cv2.imread(output_path)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB) if result_bgr is not None else image
        else:
            result_rgb = image 

        return result_rgb, time_str, mem_str

    except Exception as e:
        print(f"Critical error during inference: {e}")
        return image, f"Error: {str(e)}", "Error"

def run_text_inference(image, prompt, conf, segment_mode):
    return inference_wrapper(image, prompt, "", conf, segment_mode, False)

def run_box_inference(image, box_prompt, conf, segment_mode, group_similar):
    return inference_wrapper(image, "", box_prompt, conf, segment_mode, group_similar)

# --- HTML/JS Bounding Box Generator ---
def generate_interactive_canvas(image_numpy):
    if image_numpy is None:
        return ""
    
    try:
        img = Image.fromarray(image_numpy).convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        w, h = img.size
        
        raw_html = f"""
        <!DOCTYPE html>
        <html>
        <head><style>body {{ margin: 0; padding: 0; text-align: center; overflow: hidden; background: transparent; }}</style></head>
        <body>
            <canvas id='sam-canvas' width='{w}' height='{h}' style='max-width: 100%; max-height: 400px; object-fit: contain; border: 1px solid #4b5563; border-radius: 6px; cursor: crosshair; margin-top: 10px;'></canvas>
            <script>
                const canvas = document.getElementById('sam-canvas');
                const ctx = canvas.getContext('2d');
                const img = new Image();
                img.src = 'data:image/jpeg;base64,{img_str}';
                
                let boxes = [];
                let isDrawing = false;
                let startX = 0; let startY = 0;
                
                img.onload = () => {{ ctx.drawImage(img, 0, 0, canvas.width, canvas.height); }};
                
                // NEW: Listens for the clear command from the parent Gradio app
                window.addEventListener('message', (event) => {{
                    if (event.data === 'clear_boxes') {{
                        boxes = [];
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    }}
                }});
                
                function getMousePos(evt) {{
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    return {{ x: (evt.clientX - rect.left) * scaleX, y: (evt.clientY - rect.top) * scaleY }};
                }}
                
                canvas.addEventListener('mousedown', (e) => {{
                    isDrawing = true;
                    const pos = getMousePos(e);
                    startX = pos.x; startY = pos.y;
                }});
                
                canvas.addEventListener('mousemove', (e) => {{
                    if (!isDrawing) return;
                    const pos = getMousePos(e);
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    ctx.lineWidth = 4; 
                    ctx.strokeStyle = '#10b981'; 
                    boxes.forEach(b => ctx.strokeRect(b.x, b.y, b.w, b.h));
                    ctx.strokeStyle = '#ef4444'; 
                    ctx.strokeRect(startX, startY, pos.x - startX, pos.y - startY);
                }});
                
                canvas.addEventListener('mouseup', (e) => {{
                    isDrawing = false;
                    const pos = getMousePos(e);
                    let x = Math.round(Math.min(startX, pos.x));
                    let y = Math.round(Math.min(startY, pos.y));
                    let w = Math.round(Math.abs(pos.x - startX));
                    let h = Math.round(Math.abs(pos.y - startY));
                    
                    if (w > 5 && h > 5) {{ 
                        boxes.push({{x, y, w, h}});
                        const boxStr = `pos:${{x}},${{y}},${{w}},${{h}}`;
                        try {{
                            const targetBox = window.parent.document.querySelector('#box_prompt_input textarea');
                            if (targetBox) {{
                                let currentVal = targetBox.value;
                                targetBox.value = (currentVal && currentVal.trim() !== '') ? currentVal + ';' + boxStr : boxStr;
                                targetBox.dispatchEvent(new window.parent.Event('input', {{ bubbles: true }}));
                            }}
                        }} catch (err) {{ console.error("Iframe communication blocked:", err); }}
                    }}
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    ctx.lineWidth = 4; ctx.strokeStyle = '#10b981';
                    boxes.forEach(b => ctx.strokeRect(b.x, b.y, b.w, b.h));
                }});
            </script>
        </body>
        </html>
        """
        escaped_html = html.escape(raw_html)
        return f"<iframe srcdoc='{escaped_html}' style='width: 100%; height: 420px; border: none; overflow: hidden;'></iframe>"
        
    except Exception as e:
        print(f"Error generating canvas: {e}")
        return f"<div style='color: red; padding: 20px;'>Error rendering interactive image: {str(e)}</div>"

# --- UI Styling & Layout ---
custom_css = """
.container { max-width: 1200px; margin: auto; }
.header-text { 
    text-align: center; font-weight: 800; font-size: 2.5rem; 
    background: -webkit-linear-gradient(45deg, #2563eb, #7c3aed); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
    margin-bottom: 20px;
}
/* PERFECT TOP ALIGNMENT CSS */
.top-align-row { 
    align-items: flex-start !important; 
}
.top-align-btn { 
    margin-top: 0px !important; 
}
"""

theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate", text_size="lg")

# --- JS to Disable Spellcheck Globally ---
custom_head = """
<script>
    // MutationObserver monitors the DOM and forces spellcheck off on any text area
    const observer = new MutationObserver(() => {
        document.querySelectorAll('textarea, input').forEach(el => {
            if (el.getAttribute('spellcheck') !== 'false') {
                el.setAttribute('spellcheck', 'false');
                el.setAttribute('autocorrect', 'off');
                el.setAttribute('autocomplete', 'off');
            }
        });
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
</script>
"""

# --- Build the UI ---
# UPDATED: Removed `head=custom_head` from here
with gr.Blocks(title="SAM3 TensorRT FP16 Gradio Demo") as demo:
    
    with gr.Column(elem_classes=["container"]):
        gr.HTML(
            """
            <div style="text-align:center;">
                <h1 class='header-text'>SAM3 TensorRT FP16 Gradio Demo</h1>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            
            # --- Left Column: Control Panel ---
            with gr.Column(scale=4, variant="panel"):
                gr.HTML("<h3>Configuration</h3>")
                
                # Global State for Image
                stored_image = gr.State(None)

                # The Long Button Mode Selector
                mode_selector = gr.Radio(
                    choices=["Text Prompt", "Box Prompt"], 
                    value="Text Prompt", 
                    label="Select Input Mode",
                    container=True
                )

                # Single Display Area - Conditionally visible
                input_image = gr.Image(label="Upload Image", type="numpy", height=300, visible=True)
                interactive_canvas = gr.HTML(value="", visible=False)
                
                # --- UI: TEXT PROMPT ROW ---
                with gr.Column(visible=True) as text_prompt_row:
                    with gr.Row(equal_height=False, elem_classes=["top-align-row"]):
                        text_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. cat, car, person", lines=1, scale=4, min_width=200)
                        clear_text_btn = gr.Button("🗑️ Clear Text", size="sm", scale=1, min_width=100, elem_classes=["top-align-btn"])
                
                # --- UI: BOX PROMPT ROW ---
                with gr.Column(visible=False) as box_prompt_row:
                    with gr.Row(equal_height=False, elem_classes=["top-align-row"]):
                        box_prompt = gr.Textbox(
                            elem_id="box_prompt_input",
                            label="Box Prompt Coordinates", 
                            placeholder="Coordinates will appear here...", 
                            lines=1,
                            scale=4,
                            min_width=200
                        )
                        clear_boxes_btn = gr.Button("🗑️ Clear Boxes", size="sm", scale=1, min_width=100, elem_classes=["top-align-btn"])

                # --- SHARED Advanced Parameters (Fixes the Slider Bug) ---
                with gr.Accordion("Advanced Parameters", open=True):
                    conf_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="Confidence Threshold")
                    with gr.Row():
                        segment_check = gr.Checkbox(label="Generate Mask", value=True)
                        group_check = gr.Checkbox(label="Group Similar Objects (Box Mode)", value=False, interactive=False) 

                run_btn = gr.Button("Run Inference", variant="primary", size="lg")

            # --- Right Column: Results Panel ---
            with gr.Column(scale=5, variant="panel"):
                gr.HTML("<h3>Results</h3>")
                output_image = gr.Image(label="Segmented Output", height=450, interactive=False)
                
                with gr.Row():
                    time_output = gr.Textbox(label="Processing Time", value="0.00 ms", interactive=False, scale=1)
                    mem_output = gr.Textbox(label="Total GPU Usage", value="0.00 GB (Total)", interactive=False, scale=1)

    # --- UI Logic & Event Bindings ---

    # 1. Mode Switching Logic 
    def handle_mode_switch(mode):
        if mode == "Text Prompt":
            return (
                gr.update(visible=True),   # Show standard image upload
                gr.update(visible=False),  # Hide interactive canvas
                gr.update(visible=True),   # Show text prompt row
                gr.update(visible=False),  # Hide box prompt row
                gr.update(value=""),       # Clear box coordinates
                gr.update(),               # Keep existing text prompt
                gr.update(interactive=False) # Disable 'Group Similar' check
            )
        else: # Box Prompt
            return (
                gr.update(visible=False),  # Hide standard image upload
                gr.update(visible=True),   # Show interactive canvas
                gr.update(visible=False),  # Hide text prompt row
                gr.update(visible=True),   # Show box prompt row
                gr.update(),               # Keep existing box coordinates
                gr.update(value=""),       # Clear text prompt
                gr.update(interactive=True)  # Enable 'Group Similar' check
            )

    mode_selector.change(
        fn=handle_mode_switch,
        inputs=[mode_selector],
        outputs=[input_image, interactive_canvas, text_prompt_row, box_prompt_row, box_prompt, text_prompt, group_check]
    )

    # 2. Image Upload Logic
    input_image.upload(
        fn=lambda img: (img, generate_interactive_canvas(img)),
        inputs=[input_image],
        outputs=[stored_image, interactive_canvas]
    )
    input_image.clear(
        fn=lambda: (None, generate_interactive_canvas(None)),
        inputs=[],
        outputs=[stored_image, interactive_canvas]
    )
    
    # 3. Clearing Logic (Now properly clears UI Canvas too using JS)
    clear_boxes_btn.click(
        fn=lambda: "",  # Clears the Gradio text box backend state
        inputs=[],
        outputs=[box_prompt],
        js="""
        function() {
            // Send a clear message to the iframe to wipe the visual canvas instantly
            const iframes = document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                if (iframe.contentWindow) {
                    iframe.contentWindow.postMessage('clear_boxes', '*');
                }
            });
        }
        """
    )

    clear_text_btn.click(
        fn=lambda: "",
        inputs=[],
        outputs=[text_prompt]
    )

    # 4. Unified Execution Binding
    def run_unified_inference(mode, image, txt_p, box_p, conf, seg, group):
        if mode == "Text Prompt":
            return run_text_inference(image, txt_p, conf, seg)
        else:
            return run_box_inference(image, box_p, conf, seg, group)

    run_btn.click(
        fn=run_unified_inference,
        inputs=[mode_selector, stored_image, text_prompt, box_prompt, conf_slider, segment_check, group_check],
        outputs=[output_image, time_output, mem_output]
    )

if __name__ == "__main__":
    try:
        load_model()
        print("Launching UI...")

        demo.launch(
            allowed_paths=[tempfile.gettempdir()],
            css=custom_css,
            theme=theme,
            head=custom_head
        )
    except KeyboardInterrupt:
        print("\n[INFO] User stopped the application.")
    except Exception as e:
        print(f"\n[ERROR] Failed to start application: {e}")
    finally:
        cleanup_on_exit()