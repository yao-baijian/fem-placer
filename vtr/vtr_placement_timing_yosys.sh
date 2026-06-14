#!/bin/bash
set -e

export PATH="/home/byao/vtr-verilog-to-routing/libs/EXTERNAL/yosys:$PATH"
export PATH="/home/byao/vtr-verilog-to-routing/bin:$PATH"

CIRCUITS=("LU8PEEng")

# CIRCUITS=("boundtop" "ch_intrinsics" "diffeq" "diffeq2" "LU8PEEng" "LU32PEEng" "mcml" "mkDelayWorker32B" 
# "mkPktMerge" "mkSMAdapter4B" "or1200" "raygentop" "stereovision0" "stereovision1" "stereovision2" "stereovision3" "RLE_BlobMerging")

SRC_DIR="/home/byao/fem-placer/benchmarks/vtr"
PRIMITIVES_FILE="${SRC_DIR}/vtr_primitives.v"

# instances = ['bgm', 'blob_merge', 'boundtop', 'ch_intrinsics', 'diffeq', 'diffeq2', 'LU8PEEng', 
#             'LU32PEEng', 'mcml', 'mkDelayWorker32B', 'mkPktMerge', 'mkSMAdapter4B', 'or1200', 
#             'raygentop', 'sha', 'stereovision0', 'stereovision1', 'stereovision2', 'stereovision3', 'RLE_BlobMerging']

# CIRCUITS=("s1488" "s5378" "s9234" "s15850")
# SRC_DIR="/home/byao/fem-placer/benchmarks/ISCAS89"

VTR_ROOT="/home/byao/vtr-verilog-to-routing"
ARCH_FILE="$VTR_ROOT/vtr_flow/arch/xilinx/simple-7series.xml"
# ARCH_FILE="$VTR_ROOT/vtr_flow/arch/timing/k6_N10_40nm.xml"
SDC_FILE="${VTR_ROOT}/vtr_experiments/timing.sdc"
YOSYS_ABC_SCRIPT="$VTR_ROOT/vtr_flow/scripts/run_vtr_flow.py"
VPR="$VTR_ROOT/vpr/vpr"

YOSYS="$VTR_ROOT/libs/EXTERNAL/yosys/yosys"
NUM_CORES=$(nproc 2>/dev/null || echo 4)

# --- NEW Feature: Differentiate output folder based on the XML filename ---
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

    # Create isolated circuit folder inside the architecture specific directory
    CURR_CIRCUIT_DIR="${BASE_WORK_DIR}/${CIRCUIT_NAME}"
    mkdir -p "$CURR_CIRCUIT_DIR"
    cd "$CURR_CIRCUIT_DIR"

    # Define target files for caching checkpoints
    BLIF_FILE="${CIRCUIT_NAME}.pre-vpr.blif"
    NET_FILE="${CIRCUIT_NAME}.net"
    PLACE_FILE="${CIRCUIT_NAME}.place"
    VPR_LOG="vpr_placement.log"

    YOSYS_LOG="yosys_synth.log"
    # -----------------------------------------------------------------
    # STEP 1: Synthesis / Netlist Generation (.blif)
    # -----------------------------------------------------------------
    if [ -f "$BLIF_FILE" ] && [ -s "$BLIF_FILE" ]; then
        echo "[BYPASS] Tech-mapped BLIF file found. Skipping synthesis."
    else
        echo "[RUN] Performing fast FPGA Tech-Mapping via VTR Yosys..."
        
        if [ ! -f "$PRIMITIVES_FILE" ]; then
            echo "❌ Error: Primitives file missing at $PRIMITIVES_FILE"
            exit 1
        fi

        $YOSYS -p "
            read_verilog $VERILOG_FILE $PRIMITIVES_FILE;
            hierarchy -check -top $CIRCUIT_NAME;
            synth -flatten -lut 6;
            clean;
            write_blif $BLIF_FILE
        " > "$YOSYS_LOG" 2>&1

        if [ $? -ne 0 ]; then
            echo "❌ YOSYS CRASHED! Printing the last 20 lines of $YOSYS_LOG:"
            echo "------------------------------------------------------"
            tail -n 20 "$YOSYS_LOG"
            echo "------------------------------------------------------"
            exit 1
        fi

        echo "[SUCCESS] Tech-mapped BLIF generated successfully."
    fi

    # -----------------------------------------------------------------
    # STEP 2: Packing (.net) and Placement (.place)
    # -----------------------------------------------------------------
    if [ -f "$PLACE_FILE" ] && [ -f "$NET_FILE" ]; then
        echo "[BYPASS] Placement file (.place) found. Skipping VPR Packing & Placement."
    else
        echo "[RUN] Performing ACCELERATED VPR Packing & Placement..."
        
        # We drop --log_file and use standard bash redirection '>' to log to $VPR_LOG
        # We also change --num_workers to -j to match your VPR version
        $VPR "$ARCH_FILE" "$CIRCUIT_NAME" \
            --circuit_file "$BLIF_FILE" \
            --sdc_file "$SDC_FILE" \
            --pack --place \
            --timing_analysis on  \
            -j "$NUM_CORES" > "$VPR_LOG" 2>&1
    fi

    # -----------------------------------------------------------------
    # STEP 3: Timing Analysis (Post-Placement Evaluation)
    # -----------------------------------------------------------------
    echo "--- Extracting Placement-Estimated Timing Metrics ---"
    
    # Check the log we redirected to, or look for VPR's default fallback log
    if [ -f "$VPR_LOG" ]; then
        grep -E "Worst Negative Slack|Total Negative Slack|Fmax|critical path delay|Placement estimated" "$VPR_LOG" || \
        echo "No timing data found in $VPR_LOG. Ensure your SDC constraints are valid."
    elif [ -f "vpr_stdout.log" ]; then
        grep -E "Worst Negative Slack|Total Negative Slack|Fmax|critical path delay|Placement estimated" vpr_stdout.log
    else
        echo "Error: VPR execution log missing. Cannot read estimated timing."
    fi

    # Reset back to base directory before moving to next circuit
    cd "$BASE_WORK_DIR"
done