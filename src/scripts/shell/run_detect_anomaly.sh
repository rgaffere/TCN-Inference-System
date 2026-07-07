#!/bin/bash

mkdir -p ../logs

rm -rf work
vlib work

vlog -sv \
    ../rtl/common/detect_anomaly.sv \
    ../tb/detect_anomaly_tb.sv \
    | tee ../logs/detect_anomaly_compile.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

vsim -c detect_anomaly_tb -do "run -all; quit" \
    | tee ../logs/detect_anomaly_sim.log
