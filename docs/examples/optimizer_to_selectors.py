"""Recompile structural optimizer selections through the core selector interface.

Run from the repository root:
    python3 docs/examples/optimizer_to_selectors.py PLAN.yaml SELECTORS.yaml

Preview/apply now accept version-4 plans directly; conversion is optional.
This writes dense MLP/attention selector YAML only. The core select command
validates model geometry. MoE needs adapter-specific source-layer mapping.
"""
import argparse
import hashlib
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = yaml.safe_load(args.plan.read_text())
    if plan.get("version") != 4 or (plan.get("ablation") or {}).get("layers"):
        raise ValueError("Example supports structural-only version-4 optimizer plans")
    selectors = {"version": 1, "drop_layers": plan.get("drop_layers", [])}
    keys = {"mlp": "keep_indices", "attention": "keep_groups"}
    for kind, config in (plan.get("structured") or {}).items():
        if kind not in keys:
            raise ValueError(f"Unsupported structural kind: {kind}")
        if not config.get("enabled"):
            continue
        source = args.plan.parent / config["selection"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != config["selection_sha256"]:
            raise ValueError(f"Selection checksum mismatch: {kind}")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        key = keys[kind]
        selectors[kind] = {key: {str(i): indices.tolist() for i, indices in enumerate(payload[key])}}
    # Exclusive creation prevents replacing a previously reviewed selector file.
    with args.output.open("x") as output:
        yaml.safe_dump(selectors, output, sort_keys=False)
    print(f"Wrote {args.output}; compile with aegis-neurosurgery select before apply")


if __name__ == "__main__":
    main()
