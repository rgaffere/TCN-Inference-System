#!/bin/bash

mkdir -p ../logs
mkdir -p ../waves

rm -rf work
vlib work

vlog -sv \
    ../../rtl/common/resi_block.sv \
    ../../rtl/common/ring.sv \
    ../../rtl/common/MAC_array.sv \
    ../../rtl/common/ReLU_array.sv \
    ../../rtl/common/quant_array.sv \
    ../../tb/resi_block_tb.sv \
    | tee ../../../results/logs/resi_block_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c resi_block_tb -do "run -all; quit" \
    | tee ../../../results/logs/resi_block_sim.log
