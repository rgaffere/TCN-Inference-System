# Paths in this file assume different structure from repo

set target_library [list /dept/enee/software/synopsys/syn/X-2025.06-SP4/libraries/syn/lsi_10k.db]
set link_library [concat * $target_library]

# Combinational MAC first
analyze -format sverilog ../rtl/common/conv_MAC.sv

elaborate conv_MAC
current_design conv_MAC
link
compile_ultra

report_area > ../logs/combi_area.rpt
report_timing > ../logs/combi_timing.rpt
report_power > ../logs/combi_power.rpt

write -format verilog -hierarchy -output ../netlist/conv_MAC_synth.v
