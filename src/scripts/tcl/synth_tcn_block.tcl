set target_library [list \
    ../pdk/sky130_fd_sc_hd__tt_025C_1v80.db \
    ../pdk/sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.db \
]

set link_library "* $target_library"

set_app_var compile_ultra_ungroup_dw true
set_app_var hdlin_infer_multibit default_all

file mkdir ../logs
file mkdir ../netlist

analyze -format sverilog {
    ../rtl/common/MAC_array.sv
    ../rtl/common/ReLU_array.sv
    ../rtl/common/quant_array.sv
    ../rtl/common/ring.sv
    ../rtl/common/resi_block.sv
    ../rtl/common/input_block.sv
    ../rtl/common/output_block.sv
    ../rtl/common/tcn_block.sv
}

elaborate tcn_block
current_design tcn_block
link

check_design > ../logs/tcn_block_check_design.rpt

# 170 MHz clock target:
# period = 1000 / 170 = 5.882352941 ns
create_clock -name clk -period 5.882352941 [get_ports clk]

set_input_delay 0.2 -clock clk     [remove_from_collection [all_inputs] [get_ports clk]]

set_output_delay 0.2 -clock clk [all_outputs]

set_max_area 0

compile_ultra

report_area -hierarchy     > ../logs/tcn_block_area.rpt

report_timing -path full -delay max -max_paths 20     > ../logs/tcn_block_timing.rpt

report_constraint -all_violators     > ../logs/tcn_block_violators.rpt

report_power     > ../logs/tcn_block_power.rpt

report_qor     > ../logs/tcn_block_qor.rpt

write -format verilog -hierarchy     -output ../netlist/tcn_block_synth.v

write_sdc ../netlist/tcn_block_synth.sdc

write -format ddc -hierarchy     -output ../netlist/tcn_block_synth.ddc
