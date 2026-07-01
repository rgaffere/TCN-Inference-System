set target_library [list \
    ../pdk/sky130_fd_sc_hd__tt_025C_1v80.db \
    ../pdk/sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.db \
]

set link_library "* $target_library"

set_app_var compile_ultra_ungroup_dw true
set_app_var hdlin_infer_multibit default_all

analyze -format sverilog {
    ../rtl/common/MAC_array.sv
    ../rtl/common/ReLU_array.sv
    ../rtl/common/quant_array.sv
    ../rtl/common/ring.sv
    ../rtl/common/resi_block.sv
}

elaborate resi_block
current_design resi_block
link

check_design > ../logs/resi_block_check_design.rpt

create_clock -name clk -period 5.0 [get_ports clk]

set_input_delay 0.2 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0.2 -clock clk [all_outputs]

set_max_area 0

compile_ultra

report_area -hierarchy > ../logs/resi_block_area.rpt
report_timing -path full -delay max -max_paths 20 > ../logs/resi_block_timing.rpt
report_constraint -all_violators > ../logs/resi_block_violators.rpt
report_power > ../logs/resi_block_power.rpt
report_qor > ../logs/resi_block_qor.rpt

write -format verilog -hierarchy -output ../netlist/resi_block_synth.v