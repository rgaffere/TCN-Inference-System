vdel -all
vlib work
vlog -sv ../rtl/common/dilation_offsets.sv ../rtl/common/conv_MAC.sv ../rtl/common/ReLU.sv ../rtl/common/quant.sv ../rtl/common/ring.sv ../rtl/common/conv.sv ../tb/conv_tb.sv
vsim -c conv_tb -do "run -all; quit"