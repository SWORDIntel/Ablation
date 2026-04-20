#!/usr/bin/env python3
"""
Ablate refusal logic from safetensors models (like Khoj's gte-small)
Works directly with safetensors format
"""
import sys
import json
from pathlib import Path
import numpy as np

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("[!] Installing safetensors...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "--break-system-packages"])
    from safetensors import safe_open
    from safetensors.torch import save_file

import torch

def load_safetensors(path: Path):
    """Load safetensors model"""
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def identify_refusal_layers(tensors: dict) -> list:
    """Identify layers likely containing refusal logic"""
    refusal_layers = []
    
    # In embedding models, refusal is typically in:
    # - Later encoder layers (18-22 for base models)
    # - Pooling layers
    # - Classification heads
    
    for key in tensors.keys():
        # Target later layers
        if any(f"layer.{i}." in key for i in range(18, 24)):
            refusal_layers.append(key)
        # Target pooling
        elif "pooler" in key.lower():
            refusal_layers.append(key)
        # Target classification/output layers
        elif "classifier" in key.lower() or "output" in key.lower():
            refusal_layers.append(key)
    
    return refusal_layers

def ablate_tensors(tensors: dict, method: str = "zero", layers: list = None) -> dict:
    """Ablate specified layers"""
    ablated = tensors.copy()
    count = 0
    
    if layers is None:
        layers = identify_refusal_layers(tensors)
    
    print(f"[*] Ablating {len(layers)} layers with method: {method}")
    
    for key in layers:
        if key not in ablated:
            continue
            
        tensor = ablated[key]
        original_shape = tensor.shape
        
        if method == "zero":
            # Zero out completely
            ablated[key] = torch.zeros_like(tensor)
            count += tensor.numel()
        elif method == "prune":
            # Prune small weights
            threshold = tensor.abs().mean() * 0.1
            mask = tensor.abs() < threshold
            ablated[key][mask] = 0
            count += mask.sum().item()
        elif method == "scale":
            # Scale down by 90%
            ablated[key] = tensor * 0.1
            count += tensor.numel()
        
        print(f"  {key}: {original_shape} ({tensor.numel():,} params)")
    
    print(f"[✓] Modified {count:,} parameters")
    return ablated

def save_ablated_model(tensors: dict, output_path: Path):
    """Save ablated model"""
    save_file(tensors, str(output_path))
    print(f"[✓] Saved to: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablate safetensors models")
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("output", help="Output safetensors file")
    parser.add_argument("--method", default="zero", choices=["zero", "prune", "scale"],
                       help="Ablation method")
    parser.add_argument("--auto", action="store_true", help="Auto-detect refusal layers")
    parser.add_argument("--layers", nargs="+", help="Specific layers to ablate")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"[*] Loading model: {input_path}")
    tensors = load_safetensors(input_path)
    print(f"[*] Loaded {len(tensors)} tensors")
    
    # Show model structure
    total_params = sum(t.numel() for t in tensors.values())
    print(f"[*] Total parameters: {total_params:,}")
    
    # Identify or use specified layers
    if args.auto or args.layers is None:
        layers = identify_refusal_layers(tensors)
        print(f"[*] Auto-detected {len(layers)} refusal layers")
    else:
        layers = [k for k in tensors.keys() if any(l in k for l in args.layers)]
        print(f"[*] Using {len(layers)} specified layers")
    
    # Ablate
    ablated = ablate_tensors(tensors, method=args.method, layers=layers)
    
    # Save
    save_ablated_model(ablated, output_path)
    
    # Report
    print(f"\n[✓] Ablation complete!")
    print(f"    Input:  {input_path} ({input_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"    Output: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"    Method: {args.method}")
    print(f"    Layers: {len(layers)}")

if __name__ == "__main__":
    main()
