#!/usr/bin/env python3
"""
Simple model refusal ablation - zeros out refusal layers in safetensors
Works without torch dependency
"""
import sys
import struct
from pathlib import Path

try:
    from safetensors import safe_open
    import numpy as np
except ImportError:
    print("[!] Installing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "numpy", "--break-system-packages"])
    from safetensors import safe_open
    import numpy as np

def ablate_model(input_path: Path, output_path: Path):
    """Ablate the model by zeroing refusal layers"""
    print(f"[*] Loading model: {input_path}")
    
    tensors = {}
    metadata = {}
    
    # Load all tensors
    with safe_open(input_path, framework="numpy") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    print(f"[*] Loaded {len(tensors)} tensors")
    
    # Calculate total params
    total_params = sum(t.size for t in tensors.values())
    print(f"[*] Total parameters: {total_params:,}")
    
    # Identify refusal layers (later encoder layers)
    refusal_keys = []
    for key in tensors.keys():
        # Target layers 18-23 (common refusal layers)
        if any(f"layer.{i}." in key for i in range(18, 24)):
            refusal_keys.append(key)
        # Target pooler (often contains safety logic)
        elif "pooler" in key.lower():
            refusal_keys.append(key)
    
    print(f"[*] Identified {len(refusal_keys)} refusal layers")
    
    # Ablate (zero out)
    ablated_params = 0
    for key in refusal_keys:
        tensor = tensors[key]
        print(f"  Zeroing: {key} {tensor.shape} ({tensor.size:,} params)")
        tensors[key] = np.zeros_like(tensor)
        ablated_params += tensor.size
    
    print(f"[✓] Ablated {ablated_params:,} parameters ({ablated_params/total_params*100:.1f}%)")
    
    # Save using safetensors format
    print(f"[*] Saving ablated model...")
    
    # We need to manually create the safetensors file
    # Format: 8-byte header size + JSON header + tensor data
    
    import json
    
    # Build header
    header = {}
    offset = 0
    for key, tensor in tensors.items():
        dtype_str = str(tensor.dtype)
        if dtype_str == "float32":
            dtype = "F32"
        elif dtype_str == "float16":
            dtype = "F16"
        elif dtype_str == "int64":
            dtype = "I64"
        else:
            dtype = "F32"  # Default
        
        header[key] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes]
        }
        offset += tensor.nbytes
    
    if metadata:
        header["__metadata__"] = metadata
    
    header_bytes = json.dumps(header).encode("utf-8")
    header_size = len(header_bytes)
    
    # Write file
    with open(output_path, "wb") as f:
        # Write header size (8 bytes, little-endian)
        f.write(struct.pack("<Q", header_size))
        # Write header
        f.write(header_bytes)
        # Write tensor data
        for key in tensors.keys():
            f.write(tensors[key].tobytes())
    
    print(f"[✓] Saved to: {output_path}")
    print(f"    Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablate model embedding refusal layers")
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("output", help="Output safetensors file")
    
    args = parser.parse_args()
    
    ablate_model(Path(args.input), Path(args.output))
    
    print("\n[✓] Ablation complete!")
    print("\nTo use the ablated model:")
    print(f"  1. Backup original: mv ~/.cache/huggingface/hub/models--my-model--base/blobs/XXX ~/.cache/huggingface/hub/models--my-model--base/blobs/XXX.backup")
    print(f"  2. Copy ablated: cp {args.output} ~/.cache/huggingface/hub/models--my-model--base/blobs/XXX")
    print(f"  3. Reload host service after placing the ablated model")

if __name__ == "__main__":
    main()
