#!/bin/bash

mkdir -p ../logs
mkdir -p ../waves

rm -rf work
vlib work

vlog -sv \
    ../../rtl/common/conv_MAC.sv \
    ../../tb/conv_MAC_tb.sv \
    | tee ../../../results/logs/conv_MAC_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c conv_MAC_tb -do "run -all; quit" \
    | tee ../../../results/logs/conv_MAC_sim.log
