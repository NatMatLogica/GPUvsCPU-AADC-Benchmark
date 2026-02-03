#!/usr/bin/env bash
# XVA Benchmark: AADC (C++) vs GPU Brute-Force (Python/CUDA)
#
# Runs both backends on bank_small, bank_medium, bank_large configs.
# Results are logged to data/execution_log_xva.csv
#
# Usage:
#   ./run_benchmark.sh                      # run small/medium/large (16 threads)
#   ./run_benchmark.sh micro                # run micro config (5 trades, 16 paths)
#   ./run_benchmark.sh small 8              # run one config with 8 threads
#   ./run_benchmark.sh small medium         # run selected configs
#   ./run_benchmark.sh small medium 4       # run selected configs with 4 threads
#   THREADS=8 ./run_benchmark.sh medium     # override thread count via env
#
# Configs: micro (5 trades, 10K paths), small (50 trades, 16K paths),
#          medium (200 trades, 32K paths), large (500 trades, 64K paths)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# AADC library path
export LD_LIBRARY_PATH="${HOME}/aadc_sdk/lib:${LD_LIBRARY_PATH:-}"

# Python venv
VENV="${HOME}/GPU_AAD/venv/bin/activate"
if [[ -f "$VENV" ]]; then
    source "$VENV"
else
    echo "Warning: venv not found at $VENV, using system python"
fi

AADC_BINARY="./build/xva_server"
THREADS="${THREADS:-16}"

# Config definitions: name json_file mc_paths aadc_threads
declare -A CONFIGS
CONFIGS[micro]="bank_micro.json"
CONFIGS[small]="bank_small.json"
CONFIGS[medium]="bank_medium.json"
CONFIGS[large]="bank_large.json"

# Select configs to run
# If last argument is a number, use it as thread count
SELECTED=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        THREADS="$arg"
    else
        SELECTED+=("$arg")
    fi
done

# Default to all configs if none specified (excluding micro)
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=(small medium large)
fi

echo "Threads: $THREADS"

separator() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

# ------------------------------------------------------------------
# Run AADC C++ binary
# ------------------------------------------------------------------
run_aadc() {
    local config_file="$1"
    local mc_paths
    mc_paths=$(python3 -c "import json; d=json.load(open('$config_file')); print(d.get('MCPaths', 256))")

    echo "  AADC C++: $config_file, $mc_paths paths, $THREADS threads"

    if [[ ! -x "$AADC_BINARY" ]]; then
        echo "  ERROR: $AADC_BINARY not found. Build with:"
        echo "    mkdir -p build && cd build && cmake .. -DAADC_SOURCE_DIR=~/aadc_sdk && make"
        return 1
    fi

    echo "  Command: $AADC_BINARY $config_file $mc_paths $THREADS"
    time "$AADC_BINARY" "$config_file" "$mc_paths" "$THREADS"
    echo ""
}

# ------------------------------------------------------------------
# Run GPU backends (brute-force + pathwise)
# ------------------------------------------------------------------
run_gpu() {
    local config_file="$1"

    echo "  GPU Backends: $config_file (brute-force + pathwise)"
    echo "  Command: python benchmark_xva.py --input-file $config_file --backends gpu pathwise --mode pricing_with_greeks --threads $THREADS"
    python benchmark_xva.py --input-file "$config_file" --backends gpu pathwise --mode pricing_with_greeks --threads "$THREADS"
    echo ""
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
echo "XVA Benchmark Suite"
echo "Date: $(date -Iseconds)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

for cfg in "${SELECTED[@]}"; do
    config_file="${CONFIGS[$cfg]:-}"
    if [[ -z "$config_file" ]]; then
        echo "Unknown config: $cfg (available: micro small medium large)"
        continue
    fi
    if [[ ! -f "$config_file" ]]; then
        echo "Config file not found: $config_file"
        continue
    fi

    # Extract key params for display
    trades=$(python3 -c "import json; d=json.load(open('$config_file')); print(d['Portfolio']['NumRandomTrades'])")
    periods=$(python3 -c "import json; d=json.load(open('$config_file')); print(d['Portfolio']['NumPeriods'])")
    mc=$(python3 -c "import json; d=json.load(open('$config_file')); print(d.get('MCPaths', 256))")

    separator "$cfg — $config_file ($trades trades × $periods CFs, $mc MC paths)"

    echo ""
    echo "--- AADC C++ ---"
    run_aadc "$config_file" || true

    echo "--- GPU Brute-Force ---"
    run_gpu "$config_file" || true
done

separator "Done"
echo "Results logged to: data/execution_log_xva.csv"
if [[ -f data/execution_log_xva.csv ]]; then
    echo ""
    echo "Recent entries:"
    tail -5 data/execution_log_xva.csv | column -t -s,
fi
