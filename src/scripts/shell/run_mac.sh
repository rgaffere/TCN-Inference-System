#!/bin/bash

mkdir -p ../logs
mkdir -p ../waves

rm -rf work
vlib work

vlog -sv \
    ../../rtl/mac/MAC_combi.sv \
    ../../rtl/mac/MAC_pipeline.sv \
    ../../tb/MAC_tb.sv \
    | tee ../../../results/logs/MAC_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c MAC_tb -do "run -all; quit" \
    | tee ../../../results/logs/MAC_sim.log
