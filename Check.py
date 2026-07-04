import sys
import subprocess
import os
import importlib
import re

def display_spec_header(title):
    print(f"\n{('=' * 60)}")
    print(f"{title.center(60)}")
    print(f"{('=' * 60)}")

def run_shell(command):
    """Helper to run a shell command and return the first line of output."""
    try:
        # We explicitly inherit the current environment to ensure paths are respected
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, env=os.environ)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None

def audit_system_specs():
    display_spec_header("SYSTEM SPECIFICATIONS AUDIT")

    # 1. Hardware & Driver (nvidia-smi)
    print(f"--- GPU Hardware & Driver ---")
    smi_out = run_shell("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits")
    if smi_out:
        name, driver, mem = smi_out.split(',')
        print(f"GPU Model:          {name.strip()}")
        print(f"Driver Version:     {driver.strip()}")
        print(f"Total VRAM:         {mem.strip()} MB")
    else:
        print("nvidia-smi:         ❌ Failed to query (Driver likely missing)")

    # 2. Python Stack & Required Dependencies
    print(f"\n--- Core Python & Dependencies ---")
    
    # Align Python version with the rest of the dependencies
    print(f"{'Python'.ljust(20)}: ✅ {sys.version.split()[0]}")
    
    # Map pip package names to their import module names
    packages_to_check = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("cupy-cuda12x", "cupy"),
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
        ("onnx", "onnx"),
        ("tabulate", "tabulate"),
        ("gradio", "gradio"),
        ("numpy", "numpy"),
        ("protobuf", "google.protobuf"),
        ("opencv-python", "cv2")
    ]

    for pip_name, module_name in packages_to_check:
        try:
            mod = importlib.import_module(module_name)
            # Some packages use different version attributes
            version = getattr(mod, "__version__", "Found (Version Unknown)")
            
            # Validation for packages that require specific minimum versions
            if pip_name in ["torch", "torchvision", "transformers"] and version != "Found (Version Unknown)":
                try:
                    # Extract version components safely
                    version_numbers = [int(num) for num in re.findall(r'\d+', version)]
                    if len(version_numbers) >= 2:
                        major = version_numbers[0]
                        minor = version_numbers[1]
                        patch = version_numbers[2] if len(version_numbers) > 2 else 0
                        
                        if pip_name == "torch":
                            has_gpu = hasattr(mod, "cuda") and mod.cuda.is_available()
                            valid_version = (major, minor, patch) >= (2, 5, 0)
                            
                            status = "✅" if (has_gpu and valid_version) else "❌"
                            gpu_text = "(GPU Enabled)" if has_gpu else "(Incompatible! No GPU support)"
                            ver_text = "" if valid_version else "(Incompatible! Requires >= 2.5.0)"
                            
                            print(f"{pip_name.ljust(20)}: {status} {version} {gpu_text} {ver_text}".strip())
                            
                        elif pip_name == "torchvision":
                            if (major, minor, patch) >= (0, 20, 0):
                                print(f"{pip_name.ljust(20)}: ✅ {version}")
                            else:
                                print(f"{pip_name.ljust(20)}: ❌ {version} (Incompatible! Requires >= 0.20.0)")

                        elif pip_name == "transformers":
                            if (major, minor, patch) >= (5, 3, 0):
                                print(f"{pip_name.ljust(20)}: ✅ {version}")
                            else:
                                print(f"{pip_name.ljust(20)}: ❌ {version} (Incompatible! Requires >= 5.3.0)")
                    else:
                        print(f"{pip_name.ljust(20)}: ⚠️ {version} (Could not parse format for validation)")
                except Exception:
                    print(f"{pip_name.ljust(20)}: ⚠️ {version} (Validation error)")
            else:
                # Default output for other packages
                print(f"{pip_name.ljust(20)}: ✅ {version}")
                
        except ImportError:
            print(f"{pip_name.ljust(20)}: ❌ Not installed")

    # 3. TensorRT CLI (trtexec) 
    print(f"\n--- TensorRT CLI (trtexec) ---")
    trtexec_path_found = None
    trtexec_version = "Unknown"

    # Search standard and common installation paths
    for path in ["trtexec", "/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec"]:
        try:
            # trtexec outputs its banner and logs to stderr, so we combine stdout and stderr
            res = subprocess.run(f"{path} --help", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True, env=os.environ)
            
            # If the command ran and outputted expected tensorrt/help text
            if "TensorRT" in res.stdout or "Usage" in res.stdout:
                trtexec_path_found = path
                
                # Extract Version and Build Number e.g., [TensorRT v101600] [b72]
                match = re.search(r'\[TensorRT\s+v?(\d+)\](?:\s*\[b(\d+)\])?', res.stdout, re.IGNORECASE)
                
                if match:
                    v_str = match.group(1) # e.g., '101600'
                    b_str = match.group(2) # e.g., '72'
                    
                    # Convert '101600' -> '10.16.0' 
                    if len(v_str) == 6:
                        major = int(v_str[0:2])
                        minor = int(v_str[2:4])
                        patch = int(v_str[4:6])
                        trtexec_version = f"{major}.{minor}.{patch}"
                    elif len(v_str) == 5:
                        major = int(v_str[0:1])
                        minor = int(v_str[1:3])
                        patch = int(v_str[3:5])
                        trtexec_version = f"{major}.{minor}.{patch}"
                    elif len(v_str) == 4:
                        major = int(v_str[0:1])
                        minor = int(v_str[1:2])
                        patch = int(v_str[2:4])
                        trtexec_version = f"{major}.{minor}.{patch}"
                    else:
                        trtexec_version = v_str # Fallback 

                    # Append the build number if found (e.g., .72)
                    if b_str:
                        trtexec_version += f".{b_str}"

                break
        except Exception:
            continue

    if trtexec_path_found:
        print(f"trtexec Status:     ✅ Found at '{trtexec_path_found}'")
        print(f"trtexec Version:    {trtexec_version}")
    else:
        print("trtexec Status:     ❌ Not found (Checked global PATH and /usr/src/tensorrt/bin/)")

    # 4. TensorRT Python Check
    print(f"\n--- TensorRT (Python) ---")
    try:
        import tensorrt as trt
        trt_python_version = trt.__version__
        
        # Ensure Python TRT version base matches the CLI version
        if trtexec_version != "Unknown":
            base_trtexec = ".".join(trtexec_version.split(".")[:3])
            base_python = ".".join(trt_python_version.split(".")[:3])
            
            if base_python != base_trtexec:
                print(f"TensorRT Version:   ❌ {trt_python_version} (Mismatch! trtexec is {trtexec_version})")
            else:
                print(f"TensorRT Version:   ✅ {trt_python_version}")
        else:
            print(f"TensorRT Version:   ✅ {trt_python_version}")
            
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        print(f"TRT Builder Status: ✅ Functional")
        
    except ImportError:
        print(f"TensorRT Version:   ❌ Not installed in Python environment")
    except Exception as e:
        print(f"TensorRT Status:    ⚠️ Error: {e}")

if __name__ == "__main__":
    audit_system_specs()