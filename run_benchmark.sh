#!/usr/bin/env bash
# XVA Benchmark: AADC (C++) vs GPU Brute-Force (Python/CUDA)
#
# Runs backends on selected configs. Results logged to data/execution_log_xva.csv
#
# Usage:
#   ./run_benchmark.sh                      # run small/medium/large (16 threads)
#   ./run_benchmark.sh micro                # run micro config
#   ./run_benchmark.sh small 8              # run one config with 8 threads
#   ./run_benchmark.sh micro -x aadc        # run micro, exclude AADC
#   ./run_benchmark.sh micro -x gpu         # run micro, exclude GPU backends
#   ./run_benchmark.sh small -x aadc,pathwise  # exclude multiple backends
#   ./run_benchmark.sh small -x primal      # run AADC without slow primal baseline
#   ./run_benchmark.sh small -x mr          # skip MR bumps (fair comparison with pathwise)
#   THREADS=8 ./run_benchmark.sh medium     # override thread count via env
#
# Backends: aadc (C++), gpu (brute-force), pathwise (GPU pathwise derivatives)
# Exclusions: -x aadc,gpu,pathwise,primal,mr (mr skips mean reversion bumps)
# Configs:  micro (5 trades, 10K paths), small (50 trades, 16K paths),
#           medium (200 trades, 32K paths), large (500 trades, 64K paths)

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
EXCLUDE=""

# Config definitions
declare -A CONFIGS
CONFIGS[micro]="bank_micro.json"
CONFIGS[small]="bank_small.json"
CONFIGS[medium]="bank_medium.json"
CONFIGS[large]="bank_large.json"

# Parse arguments
SELECTED=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -x|--exclude)
            EXCLUDE="$2"
            shift 2
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                THREADS="$1"
            else
                SELECTED+=("$1")
            fi
            shift
            ;;
    esac
done

# Default to all configs if none specified (excluding micro)
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=(small medium large)
fi

# Determine which backends to run
RUN_AADC=true
RUN_GPU=true
RUN_PATHWISE=true
RUN_PRIMAL=true
SKIP_MR_BUMPS=false

if [[ -n "$EXCLUDE" ]]; then
    IFS=',' read -ra EXCL_ARRAY <<< "$EXCLUDE"
    for excl in "${EXCL_ARRAY[@]}"; do
        case "$excl" in
            aadc) RUN_AADC=false ;;
            gpu) RUN_GPU=false ;;
            pathwise) RUN_PATHWISE=false ;;
            primal) RUN_PRIMAL=false ;;
            mr) SKIP_MR_BUMPS=true ;;
            *) echo "Warning: unknown backend to exclude: $excl" ;;
        esac
    done
fi

echo "Threads: $THREADS"
echo "Backends: aadc=$RUN_AADC (primal=$RUN_PRIMAL), gpu=$RUN_GPU, pathwise=$RUN_PATHWISE, skip_mr=$SKIP_MR_BUMPS"

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
    local actual_config="$config_file"
    local mc_paths
    mc_paths=$(python3 -c "import json; d=json.load(open('$config_file')); print(d.get('MCPaths', 256))")

    # If primal excluded, create temp config with "Primal Is Requred": false
    if ! $RUN_PRIMAL; then
        actual_config=".tmp_$(basename "$config_file" .json)_no_primal.json"
        python3 -c "
import json
with open('$config_file') as f:
    d = json.load(f)
d['Primal Is Requred'] = False
with open('$actual_config', 'w') as f:
    json.dump(d, f, indent=4)
"
        echo "  AADC C++: $config_file (primal disabled), $mc_paths paths, $THREADS threads"
    else
        echo "  AADC C++: $config_file, $mc_paths paths, $THREADS threads"
    fi

    if [[ ! -x "$AADC_BINARY" ]]; then
        echo "  ERROR: $AADC_BINARY not found. Build with:"
        echo "    mkdir -p build && cd build && cmake .. -DAADC_SOURCE_DIR=~/aadc_sdk && make"
        return 1
    fi

    echo "  Command: $AADC_BINARY $actual_config $mc_paths $THREADS"
    time "$AADC_BINARY" "$actual_config" "$mc_paths" "$THREADS"
    echo ""
}

# ------------------------------------------------------------------
# Run GPU backends (brute-force and/or pathwise)
# ------------------------------------------------------------------
run_gpu() {
    local config_file="$1"
    local backends=""

    if $RUN_GPU; then
        backends="gpu"
    fi
    if $RUN_PATHWISE; then
        backends="$backends pathwise"
    fi
    backends=$(echo "$backends" | xargs)  # trim whitespace

    if [[ -z "$backends" ]]; then
        echo "  Skipping GPU backends (excluded)"
        return 0
    fi

    local extra_args=""
    if $SKIP_MR_BUMPS; then
        extra_args="--skip-mr-bumps"
    fi

    echo "  GPU Backends: $config_file ($backends) $extra_args"
    echo "  Command: python benchmark_xva.py --input-file $config_file --backends $backends --threads $THREADS $extra_args"
    python benchmark_xva.py --input-file "$config_file" --backends $backends --threads "$THREADS" $extra_args
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

    if $RUN_AADC; then
        echo ""
        echo "--- AADC C++ ---"
        run_aadc "$config_file" || true
    fi

    if $RUN_GPU || $RUN_PATHWISE; then
        echo "--- GPU Backends ---"
        run_gpu "$config_file" || true
    fi
done

separator "Done"
echo "Results logged to: data/execution_log_xva.csv"
if [[ -f data/execution_log_xva.csv ]]; then
    echo ""
    echo "Recent entries:"
    tail -5 data/execution_log_xva.csv | column -t -s,
fi
