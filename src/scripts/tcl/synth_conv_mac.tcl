# Paths in this file assume different structure from repo

set target_library [list ~/TCN/pdk/NangateOpenCellLibrary_typical.db]
set link_library "* $target_library"

# Combinational MAC first
analyze -format sverilog ../rtl/common/conv_MAC.sv

elaborate conv_MAC
current_design conv_MAC
link
compile_ultra

report_area > ../logs/conv_area.rpt
report_timing > ../logs/conv_timing.rpt
report_power > ../logs/conv_power.rpt

write -format verilog -hierarchy -output ../netlist/conv_MAC_synth.v
