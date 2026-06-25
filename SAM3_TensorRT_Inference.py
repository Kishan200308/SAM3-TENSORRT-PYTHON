import cv2
import time
import argparse
import numpy as np
import cupy as cp
import tensorrt as trt
from pathlib import Path
from typing import Dict, Tuple, Union
from tokenizers import Tokenizer
from PIL import Image as PILImage
from tabulate import tabulate  

# --- Constants ---
COLORS = [
    (30, 144, 255),   # Dodger Blue
    (255, 144, 30),   # Orange
    (144, 255, 30),   # Green-Yellow
    (255, 30, 144),   # Pink
    (30, 255, 144),   # Spring Green
    (0, 0, 255),      # Solid Red (OpenCV BGR)
    (0, 255, 0),      # Solid Green
    (255, 0, 0),      # Solid Blue 
    (0, 255, 255),    # Yellow
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Olive
    (128, 128, 0),    # Teal
    (250, 128, 114),  # Salmon
    (64, 224, 208),   # Turquoise
    (199, 21, 133),   # Medium Violet Red
    (255, 215, 0),    # Deep Sky Blue
    (210, 105, 30),   # Chocolate
    (124, 252, 0),    # Lawn Green
]

class TRTModule:
    """Wrapper for TensorRT execution keeping data strictly on the GPU using CuPy."""
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream()
        
        # Parse IO info
        self.io_info = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.io_info[name] = {
                "mode": self.engine.get_tensor_mode(name),
                "dtype": self.engine.get_tensor_dtype(name),
                "shape": self.engine.get_tensor_shape(name) 
            }

    def _trt_dtype_to_cupy(self, trt_dtype):
        mapping = {
            trt.float32: cp.float32, 
            trt.float16: cp.float16,
            trt.int32: cp.int32, 
            trt.int64: cp.int64, 
            trt.bool: cp.bool_
        }
        return mapping.get(trt_dtype, cp.float32)

    def get_tensor_shape(self, name: str) -> Tuple:
        if name in self.io_info:
            return tuple(self.io_info[name]["shape"])
        raise KeyError(f"Tensor '{name}' not found in engine.")

    def __call__(self, **inputs: Union[np.ndarray, cp.ndarray]) -> Dict[str, cp.ndarray]:
        gpu_inputs = {}
        gpu_outputs = {}
        
        for name, data in inputs.items():
            if name in self.io_info and self.io_info[name]["mode"] == trt.TensorIOMode.INPUT:
                dtype = self._trt_dtype_to_cupy(self.io_info[name]["dtype"])
                
                if isinstance(data, np.ndarray):
                    tensor = cp.asarray(data, dtype=dtype)
                elif isinstance(data, cp.ndarray):
                    tensor = cp.ascontiguousarray(data, dtype=dtype)
                else:
                    raise TypeError(f"Input {name} must be numpy array or cupy array.")
                    
                gpu_inputs[name] = tensor
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, int(tensor.data.ptr))

        for name, info in self.io_info.items():
            if info["mode"] == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = self._trt_dtype_to_cupy(info["dtype"])
                out_tensor = cp.empty(shape, dtype=dtype)
                gpu_outputs[name] = out_tensor
                self.context.set_tensor_address(name, int(out_tensor.data.ptr))

        self.context.execute_async_v3(stream_handle=self.stream.ptr)
        self.stream.synchronize()
        return gpu_outputs

class Sam3Inference:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        compute_cap = cp.cuda.Device(0).compute_capability
        suffix = f"_sm{compute_cap}_trt{trt.__version__}"
        
        print(f"[INFO] Loading engines from {self.model_dir}...")
        
        t_load_start = time.time()
        
        self.vision_encoder = TRTModule(str(self._find("vision-encoder", suffix)))
        self.text_encoder = TRTModule(str(self._find("text-encoder", suffix)))
        self.geometry_encoder = TRTModule(str(self._find("geometry-encoder", suffix)))
        self.decoder = TRTModule(str(self._find("decoder", suffix)))
        
        t_load_end = time.time()
        self.model_load_time = (t_load_end - t_load_start)
        print(f"[INFO] Engine deserialization complete in {self.model_load_time:.2f} seconds.")

        try:
            input_shape = self.vision_encoder.get_tensor_shape("images")
            if len(input_shape) == 4:
                self.target_h = input_shape[2]
                self.target_w = input_shape[3]
            else:
                self.target_h = self.target_w = 1008
            print(f"[INFO] Auto-detected input resolution: {self.target_w}x{self.target_h}")
        except KeyError:
            self.target_h = self.target_w = 1008

        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json missing in {model_dir}")
            
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)
        
        self.is_warmed_up = False
        self.warmup_time = 0.0

    def _find(self, name, suffix):
        p_spec = self.model_dir / f"{name}{suffix}.engine"
        p_gen = self.model_dir / f"{name}.engine"
        target = p_spec if p_spec.exists() else p_gen
        if not target.exists():
            raise FileNotFoundError(f"Could not find engine for {name} in {self.model_dir}")
        return target

    def _parse_box_prompts(self, box_str: str, img_w: int, img_h: int):
        boxes, labels = [], []
        for part in box_str.split(";"):
            part = part.strip()
            if not part: continue
            
            if part.startswith("pos:"):
                label, coords = 1, part[4:]
            elif part.startswith("neg:"):
                label, coords = 0, part[4:]
            else:
                label, coords = 1, part  
                
            x, y, w, h = [float(v) for v in coords.split(",")]
            
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            
            boxes.append([cx, cy, nw, nh])
            labels.append(label)
            
        return np.array(boxes, dtype=np.float32).reshape(1, -1, 4), np.array(labels, dtype=np.int64).reshape(1, -1)

    def run(self, img_path, prompt, box_str, conf, out_path, segment=False, verbose=False, group_similar=False):
        if not verbose:
            img_name = Path(img_path).name
            print(f"processing {img_name}")

        # 1. Preprocess
        orig = cv2.imread(img_path)
        if orig is None:
            raise ValueError(f"Could not read image: {img_path}")
        h, w = orig.shape[:2]
        rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        
        pil_img = PILImage.fromarray(rgb).resize((self.target_w, self.target_h), PILImage.BILINEAR)
        pixel_values = (np.array(pil_img).astype(np.float32) / 127.5 - 1.0).transpose(2,0,1)[None]

        # --- Parse Prompts (Split by Comma) ---
        text_prompts = [p.strip() for p in prompt.split(",") if p.strip()] if prompt else []
        
        if not text_prompts and not box_str:
            raise ValueError("Must provide either a text prompt or box prompt.")
            
        if not text_prompts: 
            text_prompts = [""] # Fallback if only using boxes

        boxes, box_labels = None, None
        if box_str:
            boxes, box_labels = self._parse_box_prompts(box_str, w, h)

        # --- WARMUP LOOP ---
        if not self.is_warmed_up:
            if verbose: print("[INFO] Warming up engines (first-run penalty)...")
            t_warmup_start = time.time()
            
            # Use just the first prompt for warmup
            w_p = text_prompts[0]
            if w_p:
                tokens = self.tokenizer.encode(w_p)
                w_input_ids = np.array([tokens.ids], dtype=np.int64)
                w_attn_mask = np.array([tokens.attention_mask], dtype=np.int64)
            else:
                w_input_ids = np.full((1, 32), 49407, dtype=np.int64)
                w_attn_mask = np.zeros((1, 32), dtype=np.int64)
                w_attn_mask[0, 0] = 1
            
            for _ in range(3):
                v_feats = self.vision_encoder(images=pixel_values)
                t_feats = self.text_encoder(input_ids=w_input_ids, attention_mask=w_attn_mask)
                p_feat = t_feats["text_features"]
                p_mask = t_feats["text_mask"]

                if boxes is not None:
                    g_feats = self.geometry_encoder(
                        input_boxes=boxes, input_boxes_labels=box_labels,
                        fpn_feat_2=v_feats["fpn_feat_2"], fpn_pos_2=v_feats["fpn_pos_2"]
                    )
                    p_feat = cp.concatenate([p_feat, g_feats["geometry_features"]], axis=1)
                    p_mask = cp.concatenate([p_mask, g_feats["geometry_mask"]], axis=1)

                out = self.decoder(
                    fpn_feat_0=v_feats["fpn_feat_0"], fpn_feat_1=v_feats["fpn_feat_1"],
                    fpn_feat_2=v_feats["fpn_feat_2"], fpn_pos_2=v_feats["fpn_pos_2"],
                    prompt_features=p_feat, prompt_mask=p_mask
                )
            
            cp.cuda.Device(0).synchronize()
            self.warmup_time = (time.time() - t_warmup_start) * 1000
            if verbose: print(f"[INFO] Warmup complete in {self.warmup_time:.2f} ms. Running timed inference...")
            self.is_warmed_up = True
            
        # --- START TIMING INFERENCE ---
        t0 = time.time()

        # 2. Forward Pass: Vision (RUNS ONLY ONCE FOR SPEED)
        v_feats = self.vision_encoder(images=pixel_values)
        
        all_pred_boxes = []
        all_scores = []
        all_masks = []
        all_labels = []  # NEW: Track names for each box!

        # 3. Loop over each text prompt individually
        for p_text in text_prompts:
            if p_text:
                tokens = self.tokenizer.encode(p_text)
                input_ids = np.array([tokens.ids], dtype=np.int64)
                attn_mask = np.array([tokens.attention_mask], dtype=np.int64)
            else:
                input_ids = np.full((1, 32), 49407, dtype=np.int64)
                attn_mask = np.zeros((1, 32), dtype=np.int64)
                attn_mask[0, 0] = 1

            t_feats = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            prompt_features = t_feats["text_features"]
            prompt_mask = t_feats["text_mask"]

            if boxes is not None:
                g_feats = self.geometry_encoder(
                    input_boxes=boxes, input_boxes_labels=box_labels,
                    fpn_feat_2=v_feats["fpn_feat_2"], fpn_pos_2=v_feats["fpn_pos_2"]
                )
                prompt_features = cp.concatenate([prompt_features, g_feats["geometry_features"]], axis=1)
                prompt_mask = cp.concatenate([prompt_mask, g_feats["geometry_mask"]], axis=1)

            out = self.decoder(
                fpn_feat_0=v_feats["fpn_feat_0"], fpn_feat_1=v_feats["fpn_feat_1"],
                fpn_feat_2=v_feats["fpn_feat_2"], fpn_pos_2=v_feats["fpn_pos_2"],
                prompt_features=prompt_features, prompt_mask=prompt_mask
            )
            
            # Fetch prompt-specific results
            pred_logits = out["pred_logits"][0].get()
            presence_logits = out["presence_logits"][0, 0].get()
            pred_boxes = out["pred_boxes"][0].get()

            scores = (1 / (1 + np.exp(-pred_logits))) * (1 / (1 + np.exp(-presence_logits)))
            keep = scores > conf
            
            if keep.any():
                all_pred_boxes.append(pred_boxes[keep])
                all_scores.append(scores[keep])
                
                # --- NEW: Assign the name to the detected objects ---
                display_name = p_text if p_text else "Box Object"
                all_labels.extend([display_name] * np.sum(keep))
                
                if segment:
                    if "pred_masks" in out:
                        all_masks.append(out["pred_masks"][0].get()[keep])
                    else:
                        if verbose: print("[WARNING] Engine missing 'pred_masks'. Switching to BBox mode.")
                        segment = False

        cp.cuda.Device(0).synchronize()
        inference_time = (time.time() - t0) * 1000

        # Combine all boxes/masks from all prompts
        if all_pred_boxes:
            final_pred_boxes = np.concatenate(all_pred_boxes, axis=0)
            final_scores = np.concatenate(all_scores, axis=0)
            final_labels = all_labels 
            
            scaled_boxes = final_pred_boxes.copy()
            scaled_boxes[:, [0, 2]] *= w
            scaled_boxes[:, [1, 3]] *= h
            scaled_boxes = np.clip(scaled_boxes, 0, [[w, h, w, h]])
            
            final_masks = np.concatenate(all_masks, axis=0) if (segment and all_masks) else None
            
            # --- NEW: Spatial Filtering Logic ---
            if not group_similar and box_str:
                filtered_boxes, filtered_scores, filtered_masks, filtered_labels = [], [], [], []
                
                # 1. Reconstruct original user boxes into separate positive and negative lists
                pos_boxes = []
                neg_boxes = []
                for part in box_str.split(";"):
                    part = part.strip()
                    if not part: continue
                    
                    is_neg = part.startswith("neg:")
                    coords = part[4:] if part.startswith(("pos:", "neg:")) else part
                    ux, uy, uw, uh = [float(v) for v in coords.split(",")]
                    
                    if is_neg:
                        neg_boxes.append([ux, uy, ux + uw, uy + uh])
                    else:
                        pos_boxes.append([ux, uy, ux + uw, uy + uh])
                
                # 2. Keep prediction only if its center is inside a positive box 
                #    AND NOT inside a negative box
                for i, p_box in enumerate(scaled_boxes):
                    px1, py1, px2, py2 = p_box
                    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
                    
                    # If no positive boxes were provided, assume True. Otherwise, check bounds.
                    in_pos = any((ux1 <= p_cx <= ux2 and uy1 <= p_cy <= uy2) for ux1, uy1, ux2, uy2 in pos_boxes) if pos_boxes else True
                    
                    # Check if it falls inside any negative exclusion zone
                    in_neg = any((nx1 <= p_cx <= nx2 and ny1 <= p_cy <= ny2) for nx1, ny1, nx2, ny2 in neg_boxes)
                    
                    if in_pos and not in_neg:
                        filtered_boxes.append(p_box)
                        filtered_scores.append(final_scores[i])
                        filtered_labels.append(final_labels[i])
                        if final_masks is not None: filtered_masks.append(final_masks[i])
                
                scaled_boxes = np.array(filtered_boxes) if filtered_boxes else []
                final_scores = np.array(filtered_scores) if filtered_scores else []
                final_labels = filtered_labels
                if final_masks is not None: final_masks = np.array(filtered_masks) if filtered_masks else None
            # ------------------------------------
        else:
            scaled_boxes, final_scores, final_masks, final_labels = [], [], None, []

        # 7. Visualization
        if verbose: print(f"[INFO] Found: {len(scaled_boxes)} objects total")
        
        vis = orig.copy()

        # --- UPDATED: Pass final_labels into the zip loop to extract the name ---
        if segment and final_masks is not None:
            for i, (score, box, m, lbl) in enumerate(zip(final_scores, scaled_boxes, final_masks, final_labels)):
                color = COLORS[i % len(COLORS)]
                if m.ndim == 3: m = m[0]
                
                mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_bool = mask_resized > 0
                
                overlay = vis.copy()
                overlay[mask_bool] = color
                vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
                
                mask_uint8 = mask_bool.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, color, 2)
                
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                
                # Format text: "person 0.85"
                self._draw_label(vis, f"{lbl} {score:.2f}", x1, y1, color)
        else:
            for i, (score, box, lbl) in enumerate(zip(final_scores, scaled_boxes, final_labels)):
                color = COLORS[i % len(COLORS)]
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                
                # Format text: "person 0.85"
                self._draw_label(vis, f"{lbl} {score:.2f}", x1, y1, color)

        cv2.imwrite(out_path, vis)
  
        # Output logic based on verbose flag
        if verbose:
            table_data = [
                ["Model Load Time", f"{self.model_load_time:.2f} s"],
                ["Warmup Time (3 passes)", f"{self.warmup_time:.2f} ms"],
                ["Input Res", f"{self.target_w}x{self.target_h}"],
                ["Inference Time", f"{inference_time:.2f} ms"],
                ["Objects Found", len(scaled_boxes)]
            ]
            print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))
            print(f"[SUCCESS] Saved to {out_path}")
        else:
            out_name = Path(out_path).name
            print(f"processed {out_name}")

        return inference_time

    def _draw_label(self, img, text, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        lbl_x = x
        lbl_y = y - 10 if y - 10 > text_h else y + text_h + 10
        cv2.rectangle(img, (lbl_x, lbl_y - text_h - 4), (lbl_x + text_w + 10, lbl_y + baseline - 2), color, -1)
        cv2.putText(img, text, (lbl_x + 5, lbl_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 Standalone TensorRT Inference CLI")
    parser.add_argument("--input", required=True, help="Path to input image file")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for detection (use commas for multiple)")
    parser.add_argument("--boxes", type=str, default="", help="Box prompts: pos:x,y,w,h;neg:x,y,w,h (xywh format)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--output", default="output.jpg", help="Path to save the output image")
    parser.add_argument("--models", default="Engines", help="Directory containing .engine files")
    parser.add_argument("--segment", action="store_true", help="Use segmentation masks.")
    parser.add_argument("--group-similar", action="store_true", help="Find similar objects across the image instead of strict boxing.")

    args = parser.parse_args()

    try:
        engine = Sam3Inference(args.models)
        engine.run(args.input, args.prompt, args.boxes, args.conf, args.output, segment=args.segment, verbose=True, group_similar=args.group_similar)
    except Exception as e:
        print(f"[ERROR] {e}")