#!/bin/bash

export PATH="/home/byao/vtr-verilog-to-routing/libs/EXTERNAL/yosys:$PATH"
export PATH="/home/byao/vtr-verilog-to-routing/bin:$PATH"

CIRCUITS=("boundtop" "ch_intrinsics" "diffeq" "diffeq2" "LU8PEEng" "LU32PEEng" "mcml" "mkDelayWorker32B" 
"mkPktMerge" "mkSMAdapter4B" "or1200" "raygentop" "stereovision0" "stereovision1" "stereovision2" "stereovision3" "RLE_BlobMerging")

SRC_DIR="/home/byao/fem-placer/benchmarks/vtr"

# instances = ['bgm', 'blob_merge', 'boundtop', 'ch_intrinsics', 'diffeq', 'diffeq2', 'LU8PEEng', 
#             'LU32PEEng', 'mcml', 'mkDelayWorker32B', 'mkPktMerge', 'mkSMAdapter4B', 'or1200', 
#             'raygentop', 'sha', 'stereovision0', 'stereovision1', 'stereovision2', 'stereovision3', 'RLE_BlobMerging']

# CIRCUITS=("s1488" "s5378" "s9234" "s15850")
# SRC_DIR="/home/byao/fem-placer/benchmarks/ISCAS89"

VTR_ROOT="/home/byao/vtr-verilog-to-routing"
ARCH_FILE="$VTR_ROOT/vtr_flow/arch/xilinx/simple-7series.xml"
VPR="$VTR_ROOT/vpr/vpr"

WORK_DIR="$VTR_ROOT/vtr_experiments/vtr_output_all"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

for CIRCUIT_NAME in "${CIRCUITS[@]}"; do
    echo "========================================="
    echo "Processing: ${CIRCUIT_NAME}"
    echo "========================================="
    
    VERILOG_FILE="${SRC_DIR}/${CIRCUIT_NAME}.v"
    if [ ! -f "$VERILOG_FILE" ]; then
        echo "Warning: $VERILOG_FILE not found, skipping..."
        continue
    fi

    # 1. Run full VTR flow (synthesis, pack, place) but stop after placement
    #    The run_vtr_flow.py script normally does this correctly, but if it fails,
    #    we fall back to manual steps:
    
    # Option A: Use run_vtr_flow.py without --place? That would run route as well.
    # Option B: Run pack + place manually.
    
    echo "Running synthesis + packing + placement using run_vtr_flow.py (without --place, we'll just let it run full flow and then copy .place)"
    # Actually run_vtr_flow.py with --place should work, but let's try without --place
    python3 "$VTR_ROOT/vtr_flow/scripts/run_vtr_flow.py" "$VERILOG_FILE" "$ARCH_FILE"
    
    # If that still fails, manually run pack then place:
    if [ ! -f "${CIRCUIT_NAME}.net" ] && [ ! -f "temp/${CIRCUIT_NAME}.net" ]; then
        echo "Packing output missing, running pack manually..."
        cd temp
        $VPR "$ARCH_FILE" "$CIRCUIT_NAME" --circuit_file "${CIRCUIT_NAME}.pre-vpr.blif" --pack
        $VPR "$ARCH_FILE" "$CIRCUIT_NAME" --circuit_file "${CIRCUIT_NAME}.pre-vpr.blif" --place
        cd ..
    fi
    
    # Now look for the placement file
    if [ -f "${CIRCUIT_NAME}.place" ]; then
        echo "Placement successful: ${CIRCUIT_NAME}.place"
        grep -i "wirelength" "${CIRCUIT_NAME}.log" 2>/dev/null
    elif [ -f "temp/${CIRCUIT_NAME}.place" ]; then
        echo "Placement found in temp/"
        cp "temp/${CIRCUIT_NAME}.place" .
    else
        echo "Error: No placement file generated for ${CIRCUIT_NAME}"
    fi
done