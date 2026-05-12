#!/bin/bash

mkdir -p ../logs
mkdir -p ../waves

rm -rf work
vlib work

vlog -sv \
    ../../rtl/common/ReLU.sv \
    ../../tb/ReLU_tb.sv \
    | tee ../../../results/logs/ReLU_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c ReLU_tb -do "run -all; quit" \
    | tee ../../../results/logs/ReLU_sim.log
