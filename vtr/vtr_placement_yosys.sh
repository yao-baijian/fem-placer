#!/bin/bash

export PATH="/home/byao/vtr-verilog-to-routing/libs/EXTERNAL/yosys:$PATH"
export PATH="/home/byao/vtr-verilog-to-routing/vpr:$PATH"

# ============================================================
# 批量处理多个电路（使用统一的虚拟 IO 约束文件）
# ============================================================

# 电路列表
CIRCUITS=(
    "bgm"
    "blob_merge"
    "sha"
)

# ============================================================
# 路径变量（全部使用绝对路径，避免相对路径混乱）
# ============================================================
VTR_ROOT="/home/byao/vtr-verilog-to-routing"
SRC_DIR="/home/byao/fem-placer/benchmarks/vtr"
ARCH_FILE="$VTR_ROOT/vtr_flow/arch/timing/k6_frac_N10_40nm.xml"   # 注意 timing 子目录
OUTPUT_BASE="./vtr_output"
CONSTRAINT_FILE_ABS="$VTR_ROOT/vtr_experiments/io_constraints.xml"  # 约束文件绝对路径

# ============================================================
# 主循环：处理每个电路
# ============================================================
for CIRCUIT_NAME in "${CIRCUITS[@]}"; do
    
    echo "========================================="
    echo "Processing: ${CIRCUIT_NAME}"
    echo "========================================="
    
    # 创建该电路的输出目录
    OUTPUT_DIR="${OUTPUT_BASE}/${CIRCUIT_NAME}"
    mkdir -p ${OUTPUT_DIR}
    
    VERILOG_FILE="${SRC_DIR}/${CIRCUIT_NAME}.v"
    BLIF_FILE="${OUTPUT_DIR}/${CIRCUIT_NAME}.blif"
    
    # 检查 Verilog 文件是否存在
    if [ ! -f "${VERILOG_FILE}" ]; then
        echo "Warning: ${VERILOG_FILE} not found, skipping..."
        continue
    fi
    
    # ===== 步骤 1: Yosys 综合 (Verilog → BLIF)，静默执行 =====
    echo "Synthesizing ${CIRCUIT_NAME}..."
    if yosys -q -p "
        read_verilog ${VERILOG_FILE}
        synth
        write_blif ${BLIF_FILE}
    " > /dev/null 2>&1; then
        echo "Synthesis successful for ${CIRCUIT_NAME}"
    else
        echo "Error: BLIF generation failed for ${CIRCUIT_NAME}"
        continue
    fi
    
    # ===== 步骤 2: 进入输出目录，运行 VPR =====
    cd ${OUTPUT_DIR}
    
    # 复制约束文件到当前目录（如果存在且需要）
    if [ -f "${CONSTRAINT_FILE_ABS}" ]; then
        cp ${CONSTRAINT_FILE_ABS} .
    fi
    
    # 模式 A：不含 IO 约束（只比较逻辑）
    echo "Running VPR without IO constraints for ${CIRCUIT_NAME}..."
    vpr ${ARCH_FILE} ${CIRCUIT_NAME}.blif --place > vpr_no_io.log 2>&1
    grep -i "wirelength" vpr_no_io.log || echo "No wirelength found in vpr_no_io.log"
    
    # 模式 B：含 IO 约束（使用统一的约束文件）
    # 注意：先确认 VPR 是否支持 --constraints_file。如果不支持，请注释掉此部分。
    echo "Running VPR with IO constraints for ${CIRCUIT_NAME}..."
    vpr ${ARCH_FILE} ${CIRCUIT_NAME}.blif \
        --constraints_file io_constraints.xml \
        --fix_clusters on \
        --place > vpr_with_io.log 2>&1
    grep -i "wirelength" vpr_with_io.log || echo "No wirelength found in vpr_with_io.log"
    
    echo "Finished processing ${CIRCUIT_NAME}"
    echo ""
    
    # 返回上级目录，以便下一个电路循环
    cd - > /dev/null
done

echo "========================================="
echo "Batch processing complete!"
echo "Results saved in ${OUTPUT_BASE}/"
echo "========================================="