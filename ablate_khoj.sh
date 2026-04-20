#!/bin/bash
# Khoj Model Refusal Ablation Wrapper
# Removes safety/refusal mechanisms from Khoj embedding and chat models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Khoj Model Refusal Ablation Tool                  ║"
echo "║         Surgical Removal of Safety Layers                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Default paths
MODEL_PATH="${1:-models/khoj_model.gguf}"
OUTPUT_PATH="${2:-models/khoj_model_ablated.gguf}"
METHOD="${3:-zero}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}[✗] Model not found: $MODEL_PATH${NC}"
    echo ""
    echo "Usage: $0 <model_path> [output_path] [method]"
    echo ""
    echo "Methods:"
    echo "  zero   - Zero out refusal neuron weights (default)"
    echo "  prune  - Prune weak connections"
    echo "  clamp  - Clamp activation ranges"
    echo ""
    echo "Example:"
    echo "  $0 models/llama-7b.gguf models/llama-7b-uncensored.gguf zero"
    exit 1
fi

echo -e "${BLUE}[*] Configuration:${NC}"
echo "    Model:  $MODEL_PATH"
echo "    Output: $OUTPUT_PATH"
echo "    Method: $METHOD"
echo ""

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p exports/ablation_reports

# Run ablation
echo -e "${BLUE}[*] Starting ablation process...${NC}"

PYTHONPATH=src python3 src/aegis_lab/editing/khoj_refusal_ablation.py \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    --method "$METHOD" \
    --auto-detect \
    --report "exports/ablation_reports/khoj_ablation_$(date +%Y%m%d_%H%M%S).json"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[✓] Ablation complete!${NC}"
    echo ""
    echo -e "${CYAN}Output:${NC}"
    echo "  Model:  $OUTPUT_PATH"
    echo "  Report: exports/ablation_reports/"
    echo ""
    echo -e "${YELLOW}[!] Important:${NC}"
    echo "  - Test the ablated model thoroughly before deployment"
    echo "  - The model may now respond to previously refused prompts"
    echo "  - Use responsibly and in compliance with applicable laws"
    echo ""
    
    # Show model size comparison
    if [ -f "$OUTPUT_PATH" ]; then
        ORIG_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
        NEW_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo -e "${BLUE}Size comparison:${NC}"
        echo "  Original: $ORIG_SIZE"
        echo "  Ablated:  $NEW_SIZE"
    fi
else
    echo -e "${RED}[✗] Ablation failed${NC}"
    exit 1
fi
