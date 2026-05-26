set target_library [list \
    ../pdk/sky130_fd_sc_hd__tt_025C_1v80.db \
    ../pdk/sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.db \
]

set link_library "* $target_library"

analyze -format sverilog {
    ../rtl/common/dilation_offsets.sv
    ../rtl/common/conv_MAC.sv
    ../rtl/common/ReLU.sv
    ../rtl/common/quant.sv
    ../rtl/common/ring.sv
    ../rtl/common/conv.sv
}

elaborate conv
current_design conv
link

set_ungroup [get_designs ring] false

check_design > ../logs/conv_unit_check_design.rpt

create_clock -name clk -period 5.0 [get_ports clk]

set_input_delay 0.2 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0.2 -clock clk [all_outputs]

compile_ultra

report_area -hierarchy > ../logs/conv_unit_area.rpt
report_timing -max_paths 10 > ../logs/conv_unit_timing.rpt
report_power > ../logs/conv_unit_power.rpt
report_qor > ../logs/conv_unit_qor.rpt

write -format verilog -hierarchy -output ../netlist/conv_synth.v