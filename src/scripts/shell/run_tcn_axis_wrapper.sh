#!/bin/bash

mkdir -p ../logs

rm -rf work
vlib work

vlog -sv \
    ../rtl/common/tcn_axis_wrapper.sv \
    ../rtl/common/resi_block.sv \
    ../rtl/common/ring.sv \
    ../rtl/common/MAC_array.sv \
    ../rtl/common/ReLU_array.sv \
    ../rtl/common/quant_array.sv \
    ../tb/tcn_axis_wrapper_tb.sv \
    | tee ../logs/tcn_axis_wrapper_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c tcn_axis_wrapper_tb -do "run -all; quit" \
    | tee ../logs/tcn_axis_wrapper_sim.log
