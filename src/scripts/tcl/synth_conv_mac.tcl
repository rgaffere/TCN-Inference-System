set target_library [list ../pdk/sky130_fd_sc_hd__tt_025C_1v80.db]
set link_library "* $target_library"

# Combinational MAC first
analyze -format sverilog ../rtl/common/conv_MAC.sv

elaborate conv_MAC
current_design conv_MAC
link
create_clock -name clk -period 5.0 [get_ports clk]
compile_ultra

report_area > ../logs/conv_area.rpt
report_timing > ../logs/conv_timing.rpt
report_power > ../logs/conv_power.rpt

write -format verilog -hierarchy -output ../netlist/conv_MAC_synth.v
