#!/bin/bash

mkdir -p ../logs

rm -rf work
vlib work

vlog -sv \
    ../rtl/common/tcn_block.sv \
    ../rtl/common/input_block.sv \
    ../rtl/common/output_block.sv \
    ../rtl/common/resi_block.sv \
    ../rtl/common/ring.sv \
    ../rtl/common/MAC_array.sv \
    ../rtl/common/ReLU_array.sv \
    ../rtl/common/quant_array.sv \
    ../tb/tcn_block_tb.sv \
    | tee ../logs/tcn_block_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c -voptargs=+acc tcn_block_tb -do "run -all; quit" \
    | tee ../logs/tcn_block_sim.log