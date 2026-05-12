#!/bin/bash

mkdir -p ../logs
mkdir -p ../waves

rm -rf work
vlib work

vlog -sv \
    ../../rtl/common/dilation_offsets.sv \
    ../../tb/dilation_offsets_tb.sv \
    | tee ../../../results/logs/dilation_offsets_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c dilation_offsets_tb -do "run -all; quit" \
    | tee ../../../results/logs/dilation_offsets_sim.log
