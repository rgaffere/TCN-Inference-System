#!/bin/bash

mkdir -p ../logs
mkdir -p ../waves

rm -rf work
vlib work

vlog -sv \
    ../rtl/common/ring.sv \
    ../tb/ring_tb.sv \
    | tee ../logs/ring_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c ring_tb -do "run -all; quit" \
    | tee ../logs/ring_sim.log
