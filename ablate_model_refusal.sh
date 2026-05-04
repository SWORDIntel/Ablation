#!/bin/bash
# Single root entrypoint for model refusal ablation.
# Modes:
#   - No arguments: launch interactive TUI.
#   - Positional arguments: run non-interactive CLI wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_usage() {
    cat <<'EOF'
Usage:
  ./ablate_model_refusal.sh
      Launch interactive TUI.

  ./ablate_model_refusal.sh <model_path> <output_path> [method] [strategy] [heretic_config] [extra_args...]

Arguments:
  <model_path>      Input model path (GGUF or PyTorch model directory path)
  <output_path>     Output model path
  [method]          zero | prune | clamp (default: zero)
  [strategy]        ablation | heretic (default: ablation)
  [heretic_config]  Required when strategy=heretic unless omitted by --heretic-config in extra args

Special option:
  --tui  Launch interactive TUI explicitly

Extra args are forwarded to src/aegis_lab/editing/model_refusal_ablation.py.
EOF
}

if [[ "${1-}" == "--help" || "${1-}" == "-h" ]]; then
    print_usage
    exit 0
fi

if [[ "${1-}" == "--tui" || "${1-}" == "-i" || $# -eq 0 ]]; then
    exec python3 ablate_model_refusal_tui.py
fi

MODEL_PATH="${1}"
OUTPUT_PATH="${2:-models/ablated_model.gguf}"
METHOD="${3:-zero}"
STRATEGY="${4:-ablation}"
HERETIC_CONFIG="${5:-}"

shift
shift || true
shift || true
shift || true
shift || true

EXTRA_ARGS=("$@")

if [[ ! -f "$MODEL_PATH" ]]; then
    echo -e "${RED}[✗] Model not found: $MODEL_PATH${NC}"
    print_usage
    exit 1
fi

if [[ "$METHOD" != "zero" && "$METHOD" != "prune" && "$METHOD" != "clamp" ]]; then
    echo -e "${RED}[✗] Invalid method: $METHOD${NC}"
    print_usage
    exit 1
fi

if [[ "$STRATEGY" != "ablation" && "$STRATEGY" != "heretic" ]]; then
    echo -e "${RED}[✗] Invalid strategy: $STRATEGY${NC}"
    print_usage
    exit 1
fi

if [[ "$STRATEGY" == "heretic" && -z "$HERETIC_CONFIG" ]]; then
    has_heretic_override=false
    for arg in "${EXTRA_ARGS[@]}"; do
        if [[ "$arg" == "--heretic-config" ]]; then
            has_heretic_override=true
            break
        fi
    done

    if [[ "$has_heretic_override" == false ]]; then
        echo -e "${RED}[✗] heretic strategy requires <heretic_config> or --heretic-config${NC}"
        print_usage
        exit 1
    fi
fi

echo -e "${BLUE}[*] Configuration:${NC}"
echo "    Model:  $MODEL_PATH"
echo "    Output: $OUTPUT_PATH"
echo "    Method: $METHOD"
echo "    Strategy: $STRATEGY"
if [[ -n "$HERETIC_CONFIG" ]]; then
    echo "    Heretic Config: $HERETIC_CONFIG"
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p exports/ablation_reports

echo -e "${BLUE}[*] Starting ablation process...${NC}"

set +e
CMD=(
    PYTHONPATH=src python3 src/aegis_lab/editing/model_refusal_ablation.py
    --model "$MODEL_PATH"
    --output "$OUTPUT_PATH"
    --method "$METHOD"
    --strategy "$STRATEGY"
    --auto-detect
    --report "exports/ablation_reports/refusal_ablation_$(date +%Y%m%d_%H%M%S).json"
)

if [[ -n "$HERETIC_CONFIG" ]]; then
    CMD+=(--heretic-config "$HERETIC_CONFIG")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
PIPELINE_RC=$?
set -e

if [[ "$PIPELINE_RC" -eq 0 ]]; then
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

    if [[ -f "$OUTPUT_PATH" ]]; then
        ORIG_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
        NEW_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo -e "${BLUE}Size comparison:${NC}"
        echo "  Original: $ORIG_SIZE"
        echo "  Ablated:  $NEW_SIZE"
    fi
else
    echo -e "${RED}[✗] Ablation failed${NC}"
    exit "$PIPELINE_RC"
fi
