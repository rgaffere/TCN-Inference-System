#!/bin/bash

mkdir -p ../logs

rm -rf work
vlib work

vlog -sv \
    ../rtl/common/resi_block_mixer.sv \
    ../rtl/common/ring.sv \
    ../rtl/common/MAC_array.sv \
    ../rtl/common/ReLU_array.sv \
    ../rtl/common/quant_array.sv \
    ../tb/resi_block_mixer_tb.sv \
    | tee ../logs/resi_block_mixer_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c resi_block_mixer_tb -do "run -all; quit" \
    | tee ../logs/resi_block_mixer_sim.log
