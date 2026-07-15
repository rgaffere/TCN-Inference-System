#!/bin/bash

mkdir -p ../logs

rm -rf work
vlib work

vlog -sv \
    ../rtl/common/input_block.sv \
    ../rtl/common/ring.sv \
    ../rtl/common/MAC_array.sv \
    ../rtl/common/ReLU_array.sv \
    ../rtl/common/quant_array.sv \
    ../tb/input_block_tb.sv \
    | tee ../logs/input_block_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c input_block_tb -do "run -all; quit" \
    | tee ../logs/input_block_sim.log
