#!/usr/bin/env bash
#
# Run VPR placement on a target benchmark circuit and analyze the resulting HPWL.
#
# Usage:
#   ./scripts/run_vpr_and_analyze.sh <circuit> <arch_file> [options]
#
# Examples:
#   ./scripts/run_vpr_and_analyze.sh c2670 arch/k6_N10.xml
#   ./scripts/run_vpr_and_analyze.sh c2670 arch/k6_N10.xml --blif-dir ./blif --seed 42
#
# Description:
#   This script:
#     1. Runs VPR with simulated annealing placement on a target circuit:
#          vpr <arch_file> <circuit>.blif --place --place_algorithm sa
#     2. Analyzes the generated .place file to compute HPWL using
#        analyze_vpr_place.py.

set -euo pipefail

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
CIRCUIT=""
ARCH_FILE=""
BLIF_DIR="."
WORK_DIR="."
VPR_PATH="vpr"
SKIP_VPR=false
ADDITIONAL_ARGS=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --blif-dir)
            BLIF_DIR="$2"; shift 2 ;;
        --work-dir)
            WORK_DIR="$2"; shift 2 ;;
        --vpr-path)
            VPR_PATH="$2"; shift 2 ;;
        --skip-vpr)
            SKIP_VPR=true; shift ;;
        -h|--help)
            echo "Usage: $0 <circuit> <arch_file> [options]"
            echo ""
            echo "Arguments:"
            echo "  <circuit>              Circuit name (e.g., c2670, s1488)"
            echo "  <arch_file>            VPR architecture XML file"
            echo ""
            echo "Options:"
            echo "  --blif-dir <dir>       Directory containing .blif files (default: .)"
            echo "  --work-dir <dir>       Working directory for VPR (default: .)"
            echo "  --vpr-path <path>      VPR executable path (default: vpr)"
            echo "  --skip-vpr             Skip VPR run, only analyze existing files"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Any extra arguments after -- are passed directly to VPR."
            echo ""
            echo "Examples:"
            echo "  $0 c2670 arch/k6_N10.xml --blif-dir ./blif"
            echo "  $0 s1488 arch/k6_N10.xml --work-dir ./results -- --seed 42 --fix_pins random"
            exit 0 ;;
        --)
            shift
            ADDITIONAL_ARGS="$*"
            break ;;
        -*)
            echo "Unknown option: $1"
            exit 1 ;;
        *)
            POSITIONAL+=("$1")
            shift ;;
    esac
done

CIRCUIT="${POSITIONAL[0]:-}"
ARCH_FILE="${POSITIONAL[1]:-}"

if [[ -z "$CIRCUIT" || -z "$ARCH_FILE" ]]; then
    echo "ERROR: Missing required arguments."
    echo "Usage: $0 <circuit> <arch_file> [options]"
    exit 1
fi

# ------------------------------------------------------------------
# Resolve paths
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CIRCUIT_BLIF="${BLIF_DIR}/${CIRCUIT}.blif"
PLACE_FILE="${WORK_DIR}/${CIRCUIT}.place"
NET_FILE="${WORK_DIR}/${CIRCUIT}.net"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyze_vpr_place.py"

echo "============================================================"
echo "  VPR Placement Runner"
echo "  Circuit : ${CIRCUIT}"
echo "  Arch    : ${ARCH_FILE}"
echo "  Blif    : ${CIRCUIT_BLIF}"
echo "  WorkDir : ${WORK_DIR}"
echo "============================================================"

# ------------------------------------------------------------------
# Run VPR
# ------------------------------------------------------------------
if [[ "$SKIP_VPR" == false ]]; then
    # Validate files
    if [[ ! -f "$CIRCUIT_BLIF" ]]; then
        echo "ERROR: .blif file not found: $CIRCUIT_BLIF"
        exit 1
    fi
    if [[ ! -f "$ARCH_FILE" ]]; then
        echo "ERROR: Architecture file not found: $ARCH_FILE"
        exit 1
    fi

    mkdir -p "$WORK_DIR"

    # Build command
    VPR_CMD="${VPR_PATH} ${ARCH_FILE} ${CIRCUIT_BLIF} --place --place_algorithm sa"
    if [[ -n "$ADDITIONAL_ARGS" ]]; then
        VPR_CMD="${VPR_CMD} ${ADDITIONAL_ARGS}"
    fi

    echo ""
    echo "Running VPR placement..."
    echo "  Command: ${VPR_CMD}"
    echo ""

    pushd "$WORK_DIR" > /dev/null
    eval "${VPR_CMD}"
    VPR_EXIT=$?
    popd > /dev/null

    if [[ $VPR_EXIT -ne 0 ]]; then
        echo "ERROR: VPR failed with exit code $VPR_EXIT"
        exit $VPR_EXIT
    fi

    echo ""
    echo "VPR placement completed successfully."
fi

# ------------------------------------------------------------------
# Analyze HPWL
# ------------------------------------------------------------------
if [[ -f "$PLACE_FILE" ]]; then
    echo ""
    echo "Analyzing .place file..."
    python "$ANALYSIS_SCRIPT" "$CIRCUIT" --place_dir "$WORK_DIR"
    ANALYZE_EXIT=$?

    if [[ $ANALYZE_EXIT -ne 0 ]]; then
        echo "ERROR: HPWL analysis failed with exit code $ANALYZE_EXIT"
        exit $ANALYZE_EXIT
    fi
else
    echo ""
    echo "Warning: .place file not found at ${PLACE_FILE}"
    echo "Skipping HPWL analysis."
fi

echo ""
echo "Done."
