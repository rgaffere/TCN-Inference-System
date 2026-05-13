set target_library [list /dept/enee/software/synopsys/syn/X-2025.06-SP4/libraries/syn/lsi_10k.db]
set link_library [concat * $target_library]

# Combinational MAC first
analyze -format sverilog ../rtl/mac/MAC_combi.sv

elaborate MAC_combi
current_design MAC_combi
link
compile_ultra

report_area > ../logs/combi_area.rpt
report_timing > ../logs/combi_timing.rpt
report_power > ../logs/combi_power.rpt

write -format verilog -hierarchy -output ../netlist/MAC_combi_synth.v


remove_design -all


# Now we do the pipelined MAC
analyze -format sverilog ../rtl/mac/MAC_pipeline.sv

elaborate MAC_pipeline
current_design MAC_pipeline
link
create_clock -name clk -period 10 [get_ports clk]
compile_ultra

report_area > ../logs/pipeline_area.rpt
report_timing > ../logs/pipeline_timing.rpt
report_power > ../logs/pipeline_power.rpt

write -format verilog -hierarchy -output ../netlist/MAC_pipeline_synth.v
