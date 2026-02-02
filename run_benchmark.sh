#!/usr/bin/env bash
# XVA Benchmark: AADC (C++) vs GPU Brute-Force (Python/CUDA)
#
# Runs both backends on bank_small, bank_medium, bank_large configs.
# Results are logged to data/execution_log_xva.csv
#
# Usage:
#   ./run_benchmark.sh                      # run all configs (8 threads)
#   ./run_benchmark.sh small                # run one config
#   ./run_benchmark.sh small medium         # run selected configs
#   THREADS=16 ./run_benchmark.sh medium    # override thread count

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
THREADS="${THREADS:-8}"

# Config definitions: name json_file mc_paths aadc_threads
declare -A CONFIGS
CONFIGS[small]="bank_small.json"
CONFIGS[medium]="bank_medium.json"
CONFIGS[large]="bank_large.json"

# Select configs to run
if [[ $# -gt 0 ]]; then
    SELECTED=("$@")
else
    SELECTED=(small medium large)
fi

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
# Run GPU brute-force
# ------------------------------------------------------------------
run_gpu() {
    local config_file="$1"

    echo "  GPU Brute-Force: $config_file (MC paths from JSON)"
    echo "  Command: python benchmark_xva.py --input-file $config_file --backends gpu --mode pricing_with_greeks"
    python benchmark_xva.py --input-file "$config_file" --backends gpu --mode pricing_with_greeks
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
        echo "Unknown config: $cfg (available: small medium large)"
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
