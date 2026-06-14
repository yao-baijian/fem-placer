#!/bin/bash
set -e

export PATH="/home/byao/vtr-verilog-to-routing/libs/EXTERNAL/yosys:$PATH"
export PATH="/home/byao/vtr-verilog-to-routing/bin:$PATH"

# CIRCUITS=("mcml" "LU32PEEng" "bgm" "blob_merge" "sha" "LU8PEEng" "boundtop" "ch_intrinsics" "diffeq" "diffeq2" "stereovision2")
# "mkDelayWorker32B" "mkPktMerge" "mkSMAdapter4B" "or1200" "raygentop" "stereovision0" "stereovision1" "stereovision3"
CIRCUITS=("ch_intrinsics")

SRC_DIR="/home/byao/fem-placer/benchmarks/vtr"
VTR_ROOT="/home/byao/vtr-verilog-to-routing"
ARCH_FILE="$VTR_ROOT/vtr_flow/arch/xilinx/simple-7series.xml"
SDC_FILE="${VTR_ROOT}/vtr_experiments/timing.sdc"
VPR="$VTR_ROOT/vpr/vpr"

NUM_CORES=$(nproc 2>/dev/null || echo 4)

# Differentiate output folder based on the XML filename
ARCH_NAME=$(basename "$ARCH_FILE" .xml)
BASE_WORK_DIR="$VTR_ROOT/vtr_experiments/vtr_output_${ARCH_NAME}"
mkdir -p "$BASE_WORK_DIR"
cd "$BASE_WORK_DIR"

for CIRCUIT_NAME in "${CIRCUITS[@]}"; do
    echo "================================================================="
    echo " Circuit: ${CIRCUIT_NAME} | Arch: ${ARCH_NAME}"
    echo "================================================================="
    
    VERILOG_FILE="${SRC_DIR}/${CIRCUIT_NAME}.v"
    if [ ! -f "$VERILOG_FILE" ]; then
        echo "Warning: $VERILOG_FILE not found, skipping..."
        continue
    fi

    CURR_CIRCUIT_DIR="${BASE_WORK_DIR}/${CIRCUIT_NAME}"
    mkdir -p "$CURR_CIRCUIT_DIR"
    cd "$CURR_CIRCUIT_DIR"

    # Define key intermediate file markers inside the temp folder
    BLIF_FILE="temp/${CIRCUIT_NAME}.pre-vpr.blif"
    EBLIF_FILE="temp/${CIRCUIT_NAME}.eblif"
    PLACE_FILE="${CIRCUIT_NAME}.place"
    NET_FILE="${CIRCUIT_NAME}.net"
    VPR_LOG="temp/vpr_placement.log"

    # Ensure the temp folder directory structure exists
    mkdir -p temp

    # -----------------------------------------------------------------
    # FLOW CONTROL (Direct VPR vs. Full Flow Wrapper)
    # -----------------------------------------------------------------
    if [ -f "$PLACE_FILE" ] && [ -f "$NET_FILE" ]; then
        # CASE 1: Placement already exists -> Skip everything
        echo "[BYPASS] Post-placement files found. Skipping VPR execution entirely."

    elif [ -f "$BLIF_FILE" ] || [ -f "$EBLIF_FILE" ]; then
        # CASE 2: Netlist exists -> Run ONLY packing and accelerated placement via VPR
        echo "[BYPASS] Synthesis netlist found! Invoking VPR directly for Pack & Place..."
        
        # Determine whether to feed VPR the .blif or .eblif file
        CHOSEN_NETLIST="$BLIF_FILE"
        if [ ! -f "$BLIF_FILE" ]; then CHOSEN_NETLIST="$EBLIF_FILE"; fi

        # Execute VPR 9.0 directly. Bypasses Python flow, routing, and synthesis.
        $VPR "$ARCH_FILE" "$CIRCUIT_NAME" \
            --circuit_file "$CHOSEN_NETLIST" \
            --sdc_file "$SDC_FILE" \
            --pack --place \
            --timing_analysis on  \
            -j "$NUM_CORES" > "$VPR_LOG" 2>&1 || { echo "❌ Direct VPR call failed."; exit 1; }

    else
        # CASE 3: Clean run -> Execute full VTR python flow wrapper to generate netlist + place
        echo "[RUN] No checkpoints detected. Executing full VTR flow from scratch..."
        
        python3 "$VTR_ROOT/vtr_flow/scripts/run_vtr_flow.py" \
            "$VERILOG_FILE" \
            "$ARCH_FILE" \
            -to place \
            -sdc_file "$SDC_FILE" \
            -yosys_opts "-I $SRC_DIR" -top ${CIRCUIT_NAME} -noio \
            -j "$NUM_CORES"
    fi

    # -----------------------------------------------------------------
    # STEP 3: Timing Analysis Extraction
    # -----------------------------------------------------------------
    echo "--- Extracting Placement-Estimated Timing Metrics ---"
    
    if [ -f "$VPR_LOG" ]; then
        grep -E "Worst Negative Slack|Total Negative Slack|Fmax|critical path delay|Placement estimated" "$VPR_LOG" || \
        echo "No timing data found in $VPR_LOG. Check if your SDC constraints are valid."
    elif [ -f "temp/vpr_stdout.log" ]; then
        grep -E "Worst Negative Slack|Total Negative Slack|Fmax|critical path delay|Placement estimated" temp/vpr_stdout.log
    else
        echo "Error: VPR execution log missing."
    fi

    cd "$BASE_WORK_DIR"
done